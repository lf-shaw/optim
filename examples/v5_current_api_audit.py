#!/usr/bin/env python3
r"""用 v5 真实数据审计当前统一组合优化 API。

本脚本只是开发期范例，不定义 v5 文件格式的公共契约。v5 Excel/CSV 仍由历史 benchmark
适配器读取；进入 :class:`PortfolioProblem` 后，单期、多期、结果和诊断全部走当前公共 API。

默认使用零收益推进，以便仅依赖 ``tmp/v5`` 三份数据即可运行。传入 ``--price-hdf`` 时，
脚本会从指定 HDF 的 ``px.close`` 计算相邻调仓日 close-to-close 收益；当前示例使用的历史
HDF 是 fixed format，因此会一次读取完整 ``px`` 表。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks"
for import_root in (REPOSITORY_ROOT, BENCHMARK_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from frontend_prepare_v5_real import (  # noqa: E402
    Config,
    DataStore,
    _percent_risk_to_problem,
)
from optim import (  # noqa: E402
    AssetTradeConstraints,
    PortfolioOptimizer,
    SequencePolicy,
    SolverPolicy,
)


def _parser() -> argparse.ArgumentParser:
    """构造开发审计脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-model-xlsx",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/datayes_data_v5.xlsx",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/000852.SH_v5.csv",
    )
    parser.add_argument(
        "--alpha-csv",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/alpha_v5.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/optim_v5_current_api_audit_cache"),
    )
    parser.add_argument("--max-dates", type=int, default=5)
    parser.add_argument("--single-date", type=str)
    parser.add_argument("--single-repeats", type=int, default=5)
    parser.add_argument("--initial-top-n", type=int, default=500)
    parser.add_argument("--holding-missing-mass-tolerance", type=float, default=1e-5)
    parser.add_argument("--price-hdf", type=Path)
    parser.add_argument("--lp-prescreen", action="store_true")
    parser.add_argument("--on-failure", choices=("stop", "hold"), default="stop")
    parser.add_argument("--diagnostic-demo", action="store_true")
    parser.add_argument("--diagnose-chain-failure", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _business_arguments() -> SimpleNamespace:
    """返回本轮 v5 审计采用的业务参数，而不是求解器硬编码。"""

    return SimpleNamespace(
        asset_upper=1.0,
        active_upper=0.01,
        total_active=1.8,
        turnover=0.05,
        benchmark_member_lower=0.81,
        style_bound=0.60,
        industry_bound=0.05,
        tracking_error=0.02,
        risk_aversion=7500.0,
    )


def _legacy_loader_config(arguments: SimpleNamespace, initial_top_n: int) -> Config:
    """仅为 v5 历史文件读取器构造等价配置。"""

    return Config(
        turnover_limit=arguments.turnover,
        active_ub=arguments.active_upper,
        total_active_ub=arguments.total_active,
        benchmark_weight_lb=arguments.benchmark_member_lower,
        style_default_lb=-arguments.style_bound,
        style_default_ub=arguments.style_bound,
        size_lb=-arguments.style_bound,
        size_ub=arguments.style_bound,
        industry_lb=-arguments.industry_bound,
        industry_ub=arguments.industry_bound,
        risk_budget_pct=100.0 * arguments.tracking_error,
        common_risk_aversion=arguments.risk_aversion / 10_000.0,
        specific_risk_aversion=arguments.risk_aversion / 10_000.0,
        asset_ub=arguments.asset_upper,
        initial_top_n=initial_top_n,
    )


def _load_problem(
    store: DataStore,
    date: pd.Timestamp,
    *,
    initial: pd.Series,
    arguments: SimpleNamespace,
    loader_config: Config,
):
    """将一个 v5 日期转换为当前公共 :class:`PortfolioProblem`。"""

    day = store.prepare_day(date, initial, loader_config, "error")
    return _percent_risk_to_problem(
        day,
        case="factor_qcqp",
        args=arguments,
    )


def _holding_returns(
    problems: list[Any],
    price_hdf: Path | None,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[str, Any]]:
    """准备以当前调仓日为键的相邻调仓区间 C2C 收益。"""

    if len(problems) <= 1:
        return {}, {"source": "not_required", "missing_filled_zero": 0}
    if price_hdf is None:
        return (
            {
                current.data.date: pd.Series(
                    0.0,
                    index=previous.data.assets,
                    name="close_to_close_return",
                )
                for previous, current in zip(problems, problems[1:])
            },
            {"source": "explicit_zero_return_audit", "missing_filled_zero": 0},
        )

    read_started = time.perf_counter()
    price = pd.read_hdf(price_hdf, key="px")["close"]
    read_s = time.perf_counter() - read_started
    result: dict[pd.Timestamp, pd.Series] = {}
    missing = 0
    for previous, current in zip(problems, problems[1:]):
        previous_close = price.xs(previous.data.date, level="dt")
        current_close = price.xs(current.data.date, level="dt")
        assets = previous.data.assets
        interval_return = (
            current_close.reindex(assets) / previous_close.reindex(assets) - 1.0
        )
        invalid = ~np.isfinite(interval_return.to_numpy(float))
        missing += int(invalid.sum())
        interval_return.iloc[np.flatnonzero(invalid)] = 0.0
        interval_return.name = "close_to_close_return"
        result[current.data.date] = interval_return
    return result, {
        "source": str(price_hdf.resolve()),
        "hdf_read_s": read_s,
        "missing_filled_zero": missing,
    }


def _time_stats(values: list[float]) -> dict[str, float] | None:
    """汇总一组 wall-clock 秒数。"""

    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p95_s": float(np.quantile(array, 0.95)),
        "max_s": max(values),
    }


def _single_period_audit(
    optimizer: PortfolioOptimizer,
    problem: Any,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    """对同一单期问题热身一次，再统计 fresh solve 的端到端耗时。"""

    warmup = optimizer.optimize(
        data=problem.data,
        objective=problem.objective,
        constraints=problem.constraints,
    )
    warmup.require_weights()
    rows = []
    last = warmup
    for _ in range(repeats):
        wall_started = time.perf_counter()
        last = optimizer.optimize(
            data=problem.data,
            objective=problem.objective,
            constraints=problem.constraints,
        )
        wall_s = time.perf_counter() - wall_started
        last.require_weights()
        rows.append(
            {
                "wall_s": wall_s,
                "reported_total_s": last.timings.total_s,
                "prepare_s": last.timings.prepare_s,
                "backend_setup_s": last.timings.backend_setup_s,
                "backend_solve_s": last.timings.backend_solve_s,
                "validation_s": last.timings.validation_s,
            }
        )
    return {
        "date": problem.data.date.strftime("%Y-%m-%d"),
        "assets": len(problem.data.assets),
        "status": last.status.value,
        "backend": last.backend,
        "route": [attempt.backend for attempt in last.route],
        "wall": _time_stats([row["wall_s"] for row in rows]),
        "reported_total": _time_stats([row["reported_total_s"] for row in rows]),
        "last_timings": rows[-1],
        "tracking_error": last.metrics.tracking_error,
        "turnover_l1": last.metrics.turnover_l1,
        "certificate_gap": (
            None if last.certificate is None else last.certificate.absolute_gap
        ),
        "nonzero_weights": int((last.require_weights().abs() > 0.0).sum()),
    }, last


def _diagnostic_audit(
    optimizer: PortfolioOptimizer,
    problem: Any,
) -> dict[str, Any]:
    """故意制造换手率冲突，演示失败状态与显式 deep 诊断。"""

    initial = pd.Series(problem.data.initial_weight, index=problem.data.assets)
    held_assets = tuple(initial[initial > 0.0].index)
    failed_problem = replace(
        problem,
        constraints=replace(
            problem.constraints,
            tracking_error=None,
            asset_trade=AssetTradeConstraints(blacklist=held_assets),
        ),
    )
    failed_result = optimizer.solve(failed_problem)
    if failed_result.status.has_solution:
        raise RuntimeError("diagnostic demo unexpectedly produced a feasible portfolio")
    diagnostic_started = time.perf_counter()
    report = optimizer.diagnose(
        failed_problem,
        prior_result=failed_result,
        level="deep",
    )
    diagnostic_s = time.perf_counter() - diagnostic_started
    return {
        "status": failed_result.status.value,
        "message": failed_result.message,
        "route": [attempt.backend for attempt in failed_result.route],
        "diagnostic_s": diagnostic_s,
        "summary_text": report.summary_text,
        "linear_feasible": report.linear_feasible,
        "turnover_limit": report.turnover_limit,
        "turnover_linear_lower_bound": report.turnover_linear_lower_bound,
        "minimum_tracking_error": report.minimum_tracking_error,
        "top_relaxations": [
            {
                "constraint_id": item.constraint_id,
                "group": item.group,
                "side": item.side,
                "amount": item.amount,
                "key": item.key,
            }
            for item in report.relaxations[:10]
        ],
    }


def _diagnose_first_chain_failure(
    optimizer: PortfolioOptimizer,
    problems: list[Any],
    sequence: Any,
) -> dict[str, Any] | None:
    """重建首个链式失败日的准确问题，并显式运行 deep 诊断。"""

    failed_step = next(
        (step for step in sequence.steps if not step.result.status.has_solution),
        None,
    )
    if failed_step is None:
        return None
    position = next(
        index
        for index, problem in enumerate(problems)
        if problem.data.date == failed_step.date
    )
    if failed_step.pretrade_weight is None:
        raise RuntimeError("chain failure diagnosis requires stored pretrade weights")
    template = problems[position]
    failed_problem = replace(
        template,
        data=replace(
            template.data,
            initial_weight=failed_step.pretrade_weight.reindex(
                template.data.assets,
                fill_value=0.0,
            ).to_numpy(float),
        ),
    )
    started = time.perf_counter()
    report = optimizer.diagnose(
        failed_problem,
        prior_result=failed_step.result,
        level="deep",
    )
    return {
        "date": failed_step.date.strftime("%Y-%m-%d"),
        "diagnostic_s": time.perf_counter() - started,
        "summary_text": report.summary_text,
        "linear_feasible": report.linear_feasible,
        "turnover_limit": report.turnover_limit,
        "turnover_linear_lower_bound": report.turnover_linear_lower_bound,
        "minimum_tracking_error": report.minimum_tracking_error,
        "tracking_error_limit": report.tracking_error_limit,
        "top_relaxations": [
            {
                "constraint_id": item.constraint_id,
                "group": item.group,
                "side": item.side,
                "amount": item.amount,
                "key": item.key,
            }
            for item in report.relaxations[:10]
        ],
    }


def main() -> None:
    """运行 v5 短链、指定单日以及可选不可行诊断。"""

    args = _parser().parse_args()
    if args.max_dates <= 0 or args.single_repeats <= 0:
        raise ValueError("max-dates and single-repeats must be positive")
    arguments = _business_arguments()
    loader_config = _legacy_loader_config(arguments, args.initial_top_n)

    load_started = time.perf_counter()
    store = DataStore(
        args.risk_model_xlsx,
        args.benchmark_csv,
        args.alpha_csv,
        args.cache_dir,
    )
    data_store_wall_s = time.perf_counter() - load_started
    dates = list(store.dates[: args.max_dates])
    single_date = pd.Timestamp(args.single_date) if args.single_date else dates[0]
    if single_date not in store.dates:
        raise ValueError(f"single date {single_date.date()} is absent from v5 data")

    materialize_started = time.perf_counter()
    problems = []
    for date in dates:
        placeholder_initial = store.top_n_initial(args.initial_top_n, date)
        problems.append(
            _load_problem(
                store,
                date,
                initial=placeholder_initial,
                arguments=arguments,
                loader_config=loader_config,
            )
        )
    first_initial = store.top_n_initial(args.initial_top_n, dates[0])
    first_problem = replace(
        problems[0],
        data=replace(
            problems[0].data,
            initial_weight=first_initial.reindex(
                problems[0].data.assets,
                fill_value=0.0,
            ).to_numpy(float),
        ),
    )
    problems[0] = first_problem
    if single_date in dates:
        single_problem = problems[dates.index(single_date)]
    else:
        single_initial = store.top_n_initial(args.initial_top_n, single_date)
        single_problem = _load_problem(
            store,
            single_date,
            initial=single_initial,
            arguments=arguments,
            loader_config=loader_config,
        )
    materialize_s = time.perf_counter() - materialize_started

    returns, return_metadata = _holding_returns(problems, args.price_hdf)
    optimizer = PortfolioOptimizer(policy=SolverPolicy(lp_prescreen=args.lp_prescreen))

    chain_started = time.perf_counter()
    sequence = optimizer.solve_sequence(
        problems,
        holding_period_returns=returns,
        sequence_policy=SequencePolicy(
            mode="chained",
            on_failure=args.on_failure,
            theta_seed="auto",
            holding_missing_mass_tolerance=args.holding_missing_mass_tolerance,
            renormalize_missing_holdings=True,
            output_weights="sparse",
        ),
    )
    chain_wall_s = time.perf_counter() - chain_started
    daily_times = [step.result.timings.total_s for step in sequence.steps]
    drift_rows = []
    for previous_step, current_step in zip(sequence.steps, sequence.steps[1:]):
        previous_target = (
            previous_step.result.require_weights()
            if previous_step.result.status.has_solution
            else previous_step.pretrade_weight
        )
        pretrade = current_step.pretrade_weight
        if previous_target is None or pretrade is None:
            continue
        coordinate = previous_target.index.union(pretrade.index)
        drift_rows.append(
            {
                "date": current_step.date.strftime("%Y-%m-%d"),
                "l1": float(
                    (
                        pretrade.reindex(coordinate, fill_value=0.0)
                        - previous_target.reindex(coordinate, fill_value=0.0)
                    )
                    .abs()
                    .sum()
                ),
            }
        )
    chain = {
        "requested_dates": len(problems),
        "solved_dates": len(sequence.steps),
        "stopped_date": (
            None
            if sequence.stopped_date is None
            else sequence.stopped_date.strftime("%Y-%m-%d")
        ),
        "statuses": dict(Counter(step.result.status.value for step in sequence.steps)),
        "backends": dict(Counter(step.result.backend for step in sequence.steps)),
        "failures": [
            {
                "date": step.date.strftime("%Y-%m-%d"),
                "status": step.result.status.value,
                "message": step.result.message,
                "route": [
                    {
                        "backend": attempt.backend,
                        "status": attempt.status.value,
                        "reason": (
                            None if attempt.reason is None else attempt.reason.value
                        ),
                        "native_status": attempt.native_status,
                        "solve_s": attempt.solve_s,
                    }
                    for attempt in step.result.route
                ],
            }
            for step in sequence.steps
            if not step.result.status.has_solution
        ],
        "wall_s": chain_wall_s,
        "daily_reported_total": _time_stats(daily_times),
        "successful_daily_reported_total": _time_stats(
            [
                step.result.timings.total_s
                for step in sequence.steps
                if step.result.status.has_solution
            ]
        ),
        "theta_seeds": [step.theta_seed for step in sequence.steps],
        "tracking_error_max": max(
            step.result.metrics.tracking_error or 0.0 for step in sequence.steps
        ),
        "turnover_l1_max": max(
            step.result.metrics.turnover_l1 or 0.0 for step in sequence.steps
        ),
        "pretrade_drift_l1_max": max(
            (row["l1"] for row in drift_rows),
            default=0.0,
        ),
        "pretrade_drift": drift_rows,
        "return_input": return_metadata,
        "days": [
            {
                "date": step.date.strftime("%Y-%m-%d"),
                "status": step.result.status.value,
                "backend": step.result.backend,
                "total_s": step.result.timings.total_s,
                "prepare_s": step.result.timings.prepare_s,
                "backend_solve_s": step.result.timings.backend_solve_s,
                "tracking_error": step.result.metrics.tracking_error,
                "turnover_l1": step.result.metrics.turnover_l1,
                "theta_seed": step.theta_seed,
                "theta": next(
                    (
                        attempt.metadata.get("theta")
                        for attempt in reversed(step.result.route)
                        if attempt.metadata.get("theta") is not None
                    ),
                    None,
                ),
                "qp_solves": next(
                    (
                        attempt.metadata.get("qp_solves")
                        for attempt in reversed(step.result.route)
                        if attempt.metadata.get("qp_solves") is not None
                    ),
                    None,
                ),
            }
            for step in sequence.steps
        ],
    }

    single, _ = _single_period_audit(
        optimizer,
        single_problem,
        args.single_repeats,
    )
    diagnostic = (
        _diagnostic_audit(optimizer, single_problem) if args.diagnostic_demo else None
    )
    chain_failure_diagnostic = (
        _diagnose_first_chain_failure(optimizer, problems, sequence)
        if args.diagnose_chain_failure
        else None
    )
    payload = {
        "scope": "development audit example; not a public v5 file contract",
        "configuration": {
            "dates": [date.strftime("%Y-%m-%d") for date in dates],
            "single_date": single_date.strftime("%Y-%m-%d"),
            "initial": f"benchmark top {args.initial_top_n}, normalized",
            "lp_prescreen": args.lp_prescreen,
            "tracking_error_limit": arguments.tracking_error,
            "active_weight_limit": arguments.active_upper,
            "turnover_l1_limit": arguments.turnover,
            "holding_missing_mass_tolerance": args.holding_missing_mass_tolerance,
            "on_failure": args.on_failure,
        },
        "data_store_wall_s": data_store_wall_s,
        "problem_materialize_s": materialize_s,
        "chained": chain,
        "single_period": single,
        "diagnostic_demo": diagnostic,
        "chain_failure_diagnostic": chain_failure_diagnostic,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

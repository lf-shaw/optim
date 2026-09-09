"""通联落地优化器的隔离环境对照实验；不修改生产求解路线。

先在项目环境运行 prepare，再分别在项目与 DYOPT 环境运行 optim/dyopt。
交换文件仅含数值数组，不使用 pickle，不打包第三方代码或许可证。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def prepare(args):
    """将已有复现问题转换为跨 NumPy 版本的数值输入。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optim import load_repro

    p = load_repro(args.repro).problem
    d, r = p.data, p.data.risk_model
    np.savez_compressed(
        args.data,
        assets=np.array(d.assets, dtype=str),
        alpha=d.alpha,
        benchmark=d.benchmark,
        initial=d.initial_weight,
        tradable=d.tradable,
        exposure=r.exposure,
        covariance=r.covariance,
        specific_volatility=r.specific_volatility,
        factors=np.array(r.factor_names),
        types=np.array(r.factor_types),
        date=np.array(str(d.date.date())),
    )
    print(p.constraints)


def validate(d, w, case):
    """独立验算当前复现案例的原始权重；测试参数不进入产品代码。"""
    if w.shape != d["benchmark"].shape or not np.isfinite(w).all():
        return {"max_violation": float("inf"), "invalid_weights": True}
    b, x0 = d["benchmark"], d["initial"]
    a = w - b
    f = d["exposure"].T @ a
    te = float(
        np.sqrt(f @ d["covariance"] @ f + np.sum((d["specific_volatility"] * a) ** 2))
    )
    free = d["tradable"]
    violations = {
        "budget": abs(float(w.sum()) - 1),
        "long_only": max(0.0, float(-w.min())),
        "asset_upper": max(0.0, float(w.max()) - 1.0),
        "active": max(0.0, float(np.max(np.abs(a[free]))) - 0.01),
        "frozen": float(np.max(np.abs(w[~free] - x0[~free]))) if (~free).any() else 0.0,
        "total_active": max(0.0, float(np.abs(a).sum()) - 1.8),
        "benchmark_member": max(0.0, 0.81 - float(w[b > 0].sum())),
    }
    for name, kind, value in zip(d["factors"], d["types"], f):
        if kind in ("style", "industry"):
            bound = 0.05 if kind == "industry" else (0.3 if name == "size" else 0.6)
            violations[str(name)] = max(0.0, abs(float(value)) - bound)
    if case in ("original", "lp_turnover"):
        violations["turnover"] = max(0.0, float(np.abs(w - x0).sum()) - 0.05)
    if case in ("original", "te2", "te6"):
        violations["tracking_error"] = max(0.0, te - (0.06 if case == "te6" else 0.02))
    return {
        "alpha": float(d["alpha"] @ w),
        "te": te,
        "turnover_l1": float(np.abs(w - x0).sum()),
        "max_violation": max(violations.values()),
        "violations": violations,
    }


def run_dyopt(args, d):
    """仅调用供应商公共 API；CVXPY 包装仅记录计时和实际后端。"""
    os.environ["DY_OPT_LICENSE_PATH"] = str(Path(args.license_dir).resolve())
    sys.path.insert(0, str(Path(args.dyopt_path).resolve()))
    import pandas as pd
    import cvxpy as cp
    from DY_OPT.service import DataSpace
    from DY_OPT.opt.core.optimizer import Optimizer
    from DY_OPT.opt.core.objective import MaxExpReturn, MinVariance, ObjectivesSum
    from DY_OPT.opt.core.constraint import (
        AssetWeightBound,
        AssetInBenchmarkBound,
        AssetTWeightLimit,
        RMLStyleBound,
        RMLInduBound,
        RiskLimit,
        TurnoverLimit,
        Untrade,
    )

    calls = []
    original_solve = cp.Problem.solve

    def timed_solve(problem, *a, **kw):
        if args.ecos_gap is not None and kw.get("solver") == "ECOS":
            kw.update(abstol=args.ecos_gap, reltol=args.ecos_gap)
        start = time.perf_counter()
        record = {"requested_solver": str(kw.get("solver", "default"))}
        record["options"] = {
            k: v
            for k, v in kw.items()
            if k
            in ("abstol", "reltol", "feastol", "max_iters", "warm_start", "verbose")
        }
        try:
            return original_solve(problem, *a, **kw)
        finally:
            record.update(wall_s=time.perf_counter() - start, status=problem.status)
            if problem.solver_stats is not None:
                s = problem.solver_stats
                record.update(
                    solver=s.solver_name,
                    solve_s=s.solve_time,
                    setup_s=s.setup_time,
                    iterations=s.num_iters,
                )
            calls.append(record)

    cp.Problem.solve = timed_solve
    start = time.perf_counter()
    ds = DataSpace()
    dt = str(d["date"]).replace("-", "")
    assets = pd.Index(d["assets"])

    def series(x):
        return pd.Series(x, index=assets)

    ds.add_asset_set("test", {dt: list(assets)})
    ds.add_price({dt: series(np.ones(len(assets)))})
    ds.add_index_weight("test", {dt: series(d["benchmark"][...]).loc[lambda s: s > 0]})
    ds.add_halt_list({dt: list(assets[~d["tradable"]])})
    ds.add_asset_set("untrade", {dt: list(assets[~d["tradable"]])})
    names = d["factors"]
    ds.add_risk_model_schema(
        "test",
        industry_field=list(names[d["types"] == "industry"]),
        style_field=list(names[d["types"] == "style"]),
        country_field=list(names[d["types"] == "country"]),
    )
    ds.add_risk_model_exposure(
        "test", {dt: pd.DataFrame(d["exposure"], index=assets, columns=names)}
    )
    ds.add_risk_model_covariance(
        "test", {dt: pd.DataFrame(d["covariance"], index=names, columns=names)}
    )
    ds.add_risk_model_specificrisk("test", {dt: series(d["specific_volatility"])})
    opt = Optimizer(ds)
    prep_s = time.perf_counter() - start
    for repeat in range(args.repeats):
        calls.clear()
        constraints = [
            AssetWeightBound(bounds=[0.0, 1.0], w_type="absolute", name="absolute"),
            AssetWeightBound(bounds=[-0.01, 0.01], w_type="active", name="active"),
            AssetInBenchmarkBound(bounds=[0.81, None], name="membership"),
            AssetTWeightLimit(max_sum=1.8, name="total_active"),
            RMLStyleBound(
                bounds={"all": [-0.6, 0.6], "size": [-0.3, 0.3]},
                w_type="active",
                name="style",
            ),
            RMLInduBound(bounds=[-0.05, 0.05], w_type="active", name="industry"),
            Untrade(add_halt=True, name="freeze"),
        ]
        if args.case in ("original", "lp_turnover"):
            constraints.append(TurnoverLimit(max_turnover=0.05, name="turnover"))
        if args.case in ("original", "te2", "te6"):
            constraints.append(
                RiskLimit(
                    max_risk=0.06 if args.case == "te6" else 0.02,
                    w_type="active",
                    name="risk",
                )
            )
        objective = MaxExpReturn(coef=1, w_type="absolute")
        if args.case == "qp":
            objective = ObjectivesSum(
                [objective, MinVariance(coef=args.risk_aversion, w_type="active")]
            )
        start = time.perf_counter()
        try:
            result = opt.single_period_solve(
                dt,
                objective,
                constraints,
                soft_list=None,
                round_type=None,
                attribute=False,
                universe="test",
                alpha=series(d["alpha"]),
                benchmark="test",
                risk_model="test",
                pre_position=series(d["initial"] * 1e8),
                cash=0.0,
            )
            record = {
                "status": result["opt_status"],
                "wall_s": time.perf_counter() - start,
            }
            expected = result.get("expected_result") or {}
            w = expected.get("expected_portfolio_weight")
            if w is not None:
                w = w.reindex(assets, fill_value=0.0).to_numpy()
                record.update(validate(d, w, args.case))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            record = {"error": repr(exc), "wall_s": time.perf_counter() - start}
        record.update(
            repeat=repeat,
            backend="dyopt",
            case=args.case,
            data_prepare_s=prep_s,
            native_calls=list(calls),
            experimental_ecos_gap=args.ecos_gap,
        )
        emit(args, record)


def emit(args, record):
    """逐条落盘，长耗时或失败时仍保留已完成记录。"""
    record["initial_benchmark"] = args.initial_benchmark
    with open(args.output, "a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def run_optim(args, d):
    """当前实现的同输入对照；加载时间不计入单次 solve。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optim import load_repro, PortfolioOptimizer, SolverPolicy, TrackingErrorLimit

    p = load_repro(args.repro).problem
    if args.initial_benchmark:
        from dataclasses import replace

        p = replace(p, data=replace(p.data, initial_weight=p.data.benchmark))
    if args.case not in ("original", "lp_turnover"):
        p = p.with_constraints(turnover=None)
    if args.case in ("lp", "lp_turnover", "qp"):
        p = p.with_constraints(tracking_error=None)
    if args.case == "te6":
        p = p.with_constraints(tracking_error=TrackingErrorLimit(0.06))
    if args.case == "qp":
        from dataclasses import replace
        from optim import RiskAdjustedAlpha

        p = replace(
            p,
            objective=RiskAdjustedAlpha(
                factor_aversion=args.risk_aversion, specific_aversion=args.risk_aversion
            ),
        )
    opt = PortfolioOptimizer(SolverPolicy(backend=args.backend))
    for repeat in range(args.repeats):
        start = time.perf_counter()
        r = opt.solve(p)
        record = {
            "backend": args.backend,
            "case": args.case,
            "repeat": repeat,
            "status": r.status.value,
            "wall_s": time.perf_counter() - start,
            "solve_s": r.timings.backend_solve_s,
            "setup_s": r.timings.backend_setup_s,
        }
        if r.weights is not None:
            record.update(
                validate(
                    d,
                    r.weights.reindex(d["assets"], fill_value=0).to_numpy(),
                    args.case,
                )
            )
        emit(args, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "optim", "dyopt"])
    parser.add_argument("--repro", default="tmp/pp.tar.gz")
    parser.add_argument("--data", default="tmp/dyopt/pp_arrays.npz")
    parser.add_argument("--dyopt-path", default="tmp/dyopt/inspection/dyopt-1.0.12")
    parser.add_argument("--license-dir", default="tmp/dyopt")
    parser.add_argument("--output", default="tmp/dyopt/comparison.jsonl")
    parser.add_argument(
        "--case",
        choices=["original", "te2", "te6", "lp", "lp_turnover", "qp"],
        default="te2",
    )
    parser.add_argument("--backend", default="clarabel")
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument(
        "--ecos-gap",
        type=float,
        default=None,
        help="实验性覆盖 ECOS gap 容差；省略时保留供应商默认行为",
    )
    parser.add_argument(
        "--initial-benchmark",
        action="store_true",
        help="另一个可行情形：仅将初始持仓改成基准，保留全部约束",
    )
    parser.add_argument("--repeats", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.mode == "prepare":
        prepare(arguments)
    else:
        from threadpoolctl import threadpool_limits

        with (
            np.load(arguments.data, allow_pickle=False) as archive,
            threadpool_limits(limits=1),
        ):
            data = dict(archive)
            if arguments.initial_benchmark:
                data["initial"] = data["benchmark"].copy()
            (run_dyopt if arguments.mode == "dyopt" else run_optim)(arguments, data)

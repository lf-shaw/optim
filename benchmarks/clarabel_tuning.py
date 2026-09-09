"""针对复现包的 Clarabel 对照实验；仅进程内替换参数，不修改生产默认值。

运行：python benchmarks/clarabel_tuning.py --repro tmp/pp.tar.gz --rounds 3
输出不含持仓；每个候选仍经过公共优化器独立验收。
固定边界/消元变体仅用于性能探索，尚未实现诊断坐标逆映射，不能用于导出原生证书。
不应将这里的进程内 monkeypatch 用于并发任务或生产优化器。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clarabel
import numpy as np
import scipy.sparse as sp

from optim import PortfolioOptimizer, SolverPolicy, TrackingErrorLimit, load_repro
from optim._core.backends import clarabel as backend


def main():
    """交错执行等价表述及容差实验，逐次写出状态、验收值与耗时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro", default="tmp/pp.tar.gz")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", default="tmp/clarabel_tuning.json")
    parser.add_argument(
        "--variants",
        default="baseline,faer,gap6,full7,full6,step999,risk_normalized,fixed_equal,eliminate,alpha2,alpha002",
    )
    parser.add_argument(
        "--v5-dates",
        type=int,
        default=0,
        help="额外均匀抽取 v5 日期数，0 表示只测复现包",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.v5_dates < 0:
        parser.error("rounds 必须为正整数，v5-dates 不得为负")
    problem = load_repro(args.repro).problem
    problems = {
        "original": problem,
        "no_turnover": problem.with_constraints(turnover=None),
        "te6": problem.with_constraints(
            turnover=None, tracking_error=TrackingErrorLimit(0.06)
        ),
    }
    if args.v5_dates:
        from frontend_prepare_v5_real import (
            DataStore,
            Config,
            _parser,
            _percent_risk_to_problem,
        )

        config_args = _parser().parse_args([])
        store = DataStore(
            config_args.risk_model_xlsx,
            config_args.benchmark_csv,
            config_args.alpha_csv,
            config_args.cache_dir,
        )
        config = Config(
            turnover_limit=0.05,
            active_ub=0.01,
            total_active_ub=1.8,
            benchmark_weight_lb=0.81,
            style_default_lb=-0.6,
            style_default_ub=0.6,
            size_lb=-0.6,
            size_ub=0.6,
            industry_lb=-0.05,
            industry_ub=0.05,
            risk_budget_pct=2.0,
        )
        for index in np.unique(
            np.linspace(0, len(store.dates) - 1, args.v5_dates).astype(int)
        ):
            date = store.dates[index]
            initial, _ = store.random_top_range_initial(date, 500, 600, 20260826)
            day = store.prepare_day(date, initial, config, "error")
            p = _percent_risk_to_problem(day, case="factor_qcqp", args=config_args)
            for budget in (0.02, 0.06):
                problems[f"v5_{date.date()}_{budget}"] = p.with_constraints(
                    tracking_error=TrackingErrorLimit(budget)
                )
    build = backend._build_conic_data
    linear = backend._linear_cones
    variants = args.variants.split(",")
    allowed = {
        "baseline",
        "faer",
        "gap6",
        "full7",
        "full6",
        "step999",
        "risk_normalized",
        "fixed_equal",
        "eliminate",
        "alpha2",
        "alpha002",
        "chol",
    }
    if set(variants) - allowed:
        parser.error(f"未知实验变体：{set(variants) - allowed}")
    # 将导入和首次求解开销从对照中排除；同一随机种子给出可复现的交错顺序。
    PortfolioOptimizer(SolverPolicy(backend="clarabel")).solve(problems["no_turnover"])
    rng = np.random.default_rng(20260908)
    rows = []
    for round_index in range(args.rounds):
        for case, candidate in problems.items():
            for variant in rng.permutation(variants):
                built_data = []
                native_solvers = []

                def builder(model, options, module):
                    if variant in {"alpha2", "alpha002"}:
                        options = replace(
                            options,
                            objective_scale_target=2.0 if variant == "alpha2" else 0.02,
                        )
                    data = build(model, options, module)
                    if variant == "chol":
                        # 用三角平方根代替稠密特征平方根；半正定但非正定时保持原表示。
                        risk = model.risk_operator
                        try:
                            root = np.linalg.cholesky(
                                (risk.covariance + risk.covariance.T) * 0.5
                            ).T
                        except np.linalg.LinAlgError:
                            root = None
                        if root is not None:
                            nf = len(root)
                            start = len(data.b) - data.cones[-1].dim + 1
                            block = sp.hstack(
                                [
                                    sp.csc_matrix((nf, len(data.q) - nf)),
                                    sp.csc_matrix(-root),
                                ],
                                format="csc",
                            )
                            data = replace(
                                data,
                                A=sp.vstack(
                                    [data.A[:start], block, data.A[start + nf :]],
                                    format="csc",
                                ),
                            )
                    if variant == "risk_normalized":
                        n = data.cones[-1].dim
                        scale = 1.0 / model.risk_limit
                        diagonal = np.ones(len(data.b))
                        diagonal[-n:] = scale
                        data = replace(
                            data,
                            A=(sp.diags(diagonal) @ data.A).tocsc(),
                            b=data.b * diagonal,
                        )
                    built_data.append(data)
                    return data

                def linear_builder(domain, module):
                    if variant == "fixed_equal":
                        fixed = np.isfinite(domain.variable_lower) & (
                            domain.variable_lower == domain.variable_upper
                        )
                        indices = np.flatnonzero(fixed)
                        lower, upper = (
                            domain.variable_lower.copy(),
                            domain.variable_upper.copy(),
                        )
                        lower[fixed], upper[fixed] = -np.inf, np.inf
                        extra = sp.csc_matrix(
                            (np.ones(len(indices)), (np.arange(len(indices)), indices)),
                            shape=(len(indices), domain.n_variables),
                        )
                        domain = replace(
                            domain,
                            A=sp.vstack([domain.A, extra], format="csc"),
                            lower=np.r_[domain.lower, domain.variable_lower[fixed]],
                            upper=np.r_[domain.upper, domain.variable_upper[fixed]],
                            variable_lower=lower,
                            variable_upper=upper,
                        )
                    return linear(domain, module)

                solver_factory = clarabel.DefaultSolver

                def solver(P, q, A, b, cones, config):
                    if variant == "faer":
                        config.direct_solve_method = "faer"
                    if variant == "gap6":
                        config.tol_gap_abs = 1e-6
                        config.tol_gap_rel = 1e-6
                    if variant in {"full7", "full6"}:
                        config.tol_gap_abs = config.tol_gap_rel = config.tol_feas = (
                            10.0 ** -int(variant[-1])
                        )
                    if variant == "step999":
                        config.max_step_fraction = 0.999
                    if variant == "eliminate":
                        if P.nnz:
                            raise ValueError("消元实验仅支持零二次目标；不得套用于 QP")
                        domain = built_data[-1].domain
                        fixed = np.isfinite(domain.variable_lower) & (
                            domain.variable_lower == domain.variable_upper
                        )
                        free = ~fixed
                        values = domain.variable_lower[fixed]
                        original_q = q
                        b = b - A[:, fixed] @ values
                        A = A[:, free].tocsc()
                        q = q[free]
                        P = P[free][:, free].tocsc()
                        native = solver_factory(P, q, A, b, cones, config)
                        native_solvers.append(native)

                        class ExpandedSolver:
                            """实验中的固定变量回填；仅适用当前零二次目标 QCQP。"""

                            def solve(self):
                                result = native.solve()
                                full = np.zeros(len(free))
                                full[fixed], full[free] = values, result.x
                                offset = original_q[fixed] @ values

                                class ExpandedResult:
                                    x = full
                                    obj_val = result.obj_val + offset
                                    obj_val_dual = result.obj_val_dual + offset

                                    def __getattr__(self, name):
                                        return getattr(result, name)

                                return ExpandedResult()

                        return ExpandedSolver()
                    native = solver_factory(P, q, A, b, cones, config)
                    native_solvers.append(native)
                    return native

                started = time.perf_counter()
                with (
                    patch.object(backend, "_build_conic_data", builder),
                    patch.object(backend, "_linear_cones", linear_builder),
                    patch.object(clarabel, "DefaultSolver", solver),
                ):
                    result = PortfolioOptimizer(SolverPolicy(backend="clarabel")).solve(
                        candidate
                    )
                row = dict(
                    case=case,
                    variant=variant,
                    round=round_index,
                    wall_s=time.perf_counter() - started,
                    status=result.status.value,
                    objective=result.objective_value,
                )
                # 保存公开结果统计，避免依赖 dump 和持仓文件。
                row["setup_s"] = result.timings.backend_setup_s
                row["solve_s"] = result.timings.backend_solve_s
                row["gap"] = (
                    result.certificate.absolute_gap if result.certificate else None
                )
                row["max_violation"] = max(
                    (v.amount for v in result.violations), default=0
                )
                row["route"] = [
                    dict(backend=a.backend, status=a.native_status)
                    for a in result.route
                ]
                row["iterations"] = (
                    native_solvers[-1].get_info().iterations if native_solvers else None
                )
                row["diagnostic_mapping_valid"] = variant not in {
                    "eliminate",
                    "fixed_equal",
                }
                row["fixed_variables"] = (
                    int(
                        np.sum(
                            np.isfinite(built_data[-1].domain.variable_lower)
                            & (
                                built_data[-1].domain.variable_lower
                                == built_data[-1].domain.variable_upper
                            )
                        )
                    )
                    if built_data
                    else 0
                )
                rows.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "route"}),
                    flush=True,
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "clarabel": clarabel.__version__,
                },
                "configuration": vars(args),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

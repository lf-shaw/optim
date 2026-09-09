"""对照 total_active 保留/移除的成本；移除属于改变问题，不是等价优化。

仅保存统计值，不输出持仓。默认后端固定单线程，各次求解独立冷启动。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clarabel
import mosek.fusion as fusion
import numpy as np
import pandas as pd
import highspy  # 先加载 conda 数值栈，避免旧系统 libstdc++ 被此扩展提前驻留。

from optim import (
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioOptimizer,
    SolverPolicy,
    RiskAdjustedAlpha,
    TrackingErrorLimit,
    WeightBounds,
    load_repro,
)
from optim._core.backends.mosek import MosekBackend
from optim._core.backends.highs import HighsBackend
from optim._core.backends.piqp import PIQPBackend
from optim._impl.compiler import _DomainBuilder
from optim._impl import solver_adapter, compiler as compiler_module
from risk_budget_scan import risk_value
from sparse_active_prototype import (
    FullTurnoverBuilder,
    lift_active_weights,
    sparse_active_domain,
    split_active_domain,
)


def main():
    """交错运行，比较原始 L1 主动权重、TE、目标值和原生迭代数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="baseline,removed",
        help="baseline 指当前生产表达；生产已稀疏化时 sparse 不再二次裁剪",
    )
    parser.add_argument(
        "--turnover-forms",
        default="default",
        help="default,sparse,full；full 仅强制完整换手率表达，不改变只做多约束",
    )
    parser.add_argument("--repro", type=Path, help="指定单个复现包，替代内置的两个案例")
    parser.add_argument("--budgets", default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--backends", default="clarabel,mosek")
    parser.add_argument(
        "--scenario", choices=("socp", "lp", "qp", "min_te"), default="socp"
    )
    parser.add_argument(
        "--risk-aversion", type=float, default=0.75, help="QP 的因子/特异风险惩罚系数"
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="后端线程数，0 表示不覆盖生产设置"
    )
    parser.add_argument(
        "--active-limit", type=float, help="覆盖输入的 total_active，测试紧约束"
    )
    parser.add_argument(
        "--active-limits",
        help="逗号分隔的 total_active 扫描，不能与 --active-limit 同用",
    )
    parser.add_argument(
        "--short-lower", type=float, help="显式开启卖空并设置单股绝对下界，保留其余约束"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/total_active_ablation_20260909")
    )
    args = parser.parse_args()
    variants = args.variants.split(",")
    turnover_forms = args.turnover_forms.split(",")
    if set(turnover_forms) - {"default", "sparse", "full"}:
        parser.error("turnover-forms 仅支持 default,sparse,full")
    budgets = list(map(float, args.budgets.split(",")))
    backends = args.backends.split(",")
    if (
        args.rounds < 1
        or args.threads < 0
        or any(b <= 0 or not np.isfinite(b) for b in budgets)
    ):
        parser.error("rounds/预算须为正，threads 不能为负")
    if set(backends) - {"auto", "clarabel", "mosek", "highs", "piqp"}:
        parser.error("backends 仅支持 auto,clarabel,mosek,highs,piqp")
    if set(variants) - {"baseline", "removed", "sparse", "split"}:
        parser.error("variants 仅支持 baseline,removed,sparse,split")
    if args.active_limits is not None and args.active_limit is not None:
        parser.error("active-limit 与 active-limits 不能同用")
    active_limits = (
        list(map(float, args.active_limits.split(",")))
        if args.active_limits is not None
        else None
    )
    if active_limits is not None and any(
        x < 0 or not np.isfinite(x) for x in active_limits
    ):
        parser.error("active-limits 须为有限非负数")
    if args.short_lower is not None and (
        not np.isfinite(args.short_lower) or args.short_lower >= 0
    ):
        parser.error("short-lower 须为有限负数")
    if not np.isfinite(args.risk_aversion) or args.risk_aversion <= 0:
        parser.error("risk-aversion 须为有限正数")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {
        "20260814": "tmp/pp_20260814.tar.gz",
        "20260824": "tmp/solver_threads_20260909/inputs/pp_te2.zip",
    }
    if args.repro is not None:
        sources = {args.repro.name: str(args.repro)}
    problems = {key: load_repro(path).problem for key, path in sources.items()}
    if args.short_lower is not None:
        problems = {
            key: replace(
                p,
                constraints=replace(
                    p.constraints,
                    long_only=False,
                    asset_weight=WeightBounds(
                        args.short_lower, p.constraints.asset_weight.upper
                    ),
                ),
            )
            for key, p in problems.items()
        }
    if args.active_limit is not None:
        problems = {
            key: p.with_constraints(total_active=args.active_limit)
            for key, p in problems.items()
        }
    if args.scenario != "socp":
        objective = {
            "lp": MaximizeAlpha(),
            "qp": RiskAdjustedAlpha(args.risk_aversion, args.risk_aversion),
            "min_te": MinimizeTrackingError(),
        }[args.scenario]
        problems = {
            key: replace(
                p,
                objective=objective,
                constraints=replace(p.constraints, tracking_error=None),
            )
            for key, p in problems.items()
        }
    (args.output / "input.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in (
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "BLIS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
                "problems": {
                    key: {
                        "date": str(p.data.date),
                        "assets": len(p.data.assets),
                        "benchmark_support": int(np.count_nonzero(p.data.benchmark)),
                        "initial_support": int(np.count_nonzero(p.data.initial_weight))
                        if p.data.initial_weight is not None
                        else None,
                        "factor_count": len(p.data.risk_model.factor_names),
                        "risk_provenance": repr(p.data.risk_model.provenance),
                        "constraints": repr(p.constraints),
                        "objective": repr(p.objective),
                    }
                    for key, p in problems.items()
                },
            },
            default=str,
            indent=2,
        )
    )
    observed = {}
    factory, model_init = clarabel.DefaultSolver, fusion.Model.__init__
    native_mosek = MosekBackend._solve_factor_qcqp
    native_mosek_solve = MosekBackend.solve
    native_highs, native_piqp = HighsBackend.solve, PIQPBackend.solve
    highs_factory = highspy.Highs
    finish = _DomainBuilder.finish

    def make_solver(P, q, A, b, cones, settings):
        if args.threads:
            settings.max_threads = args.threads
        native = factory(P, q, A, b, cones, settings)
        observed.update(native=native, rows=A.shape[0], columns=A.shape[1], nnz=A.nnz)
        return native

    def init_model(model, *a, **kw):
        model_init(model, *a, **kw)
        if args.threads:
            model.setSolverParam("numThreads", args.threads)

    def solve_mosek(self, model, options):
        result = native_mosek(self, model, options)
        observed["iterations"] = result.iterations
        return result

    def make_highs(*a, **kw):
        solver = highs_factory(*a, **kw)
        if args.threads:
            solver.setOptionValue("threads", args.threads)
        return solver

    def observe_backend(method):
        def solve(self, model, options):
            result = method(self, model, options)
            observed.update(
                iterations=result.iterations,
                rows=model.domain.A.shape[0],
                columns=model.domain.n_variables,
                nnz=model.domain.A.nnz,
            )
            observed.setdefault("native_attempts", []).append(
                {
                    "backend": result.backend,
                    "status": result.status.value,
                    "iterations": result.iterations,
                    "solve_s": result.solve_s,
                }
            )
            return result

        return solve

    records = []
    jobs = [
        (date, budget, active_limit, variant, backend, turnover_form)
        for date in sources
        for budget in (budgets if args.scenario == "socp" else [None])
        for active_limit in (
            active_limits
            if active_limits is not None
            else [problems[date].constraints.total_active]
        )
        for variant in variants
        for backend in backends
        for turnover_form in turnover_forms
    ]
    rng = np.random.default_rng(20260909)
    with ExitStack() as stack:
        if args.scenario != "socp":
            stack.enter_context(
                patch.object(MosekBackend, "solve", observe_backend(native_mosek_solve))
            )
        stack.enter_context(patch.object(highspy, "Highs", make_highs))
        stack.enter_context(
            patch.object(HighsBackend, "solve", observe_backend(native_highs))
        )
        stack.enter_context(
            patch.object(PIQPBackend, "solve", observe_backend(native_piqp))
        )
        stack.enter_context(
            patch.object(solver_adapter, "lift_weights", lift_active_weights)
        )
        stack.enter_context(patch.object(clarabel, "DefaultSolver", make_solver))
        stack.enter_context(patch.object(fusion.Model, "__init__", init_model))
        stack.enter_context(
            patch.object(MosekBackend, "_solve_factor_qcqp", solve_mosek)
        )
        for backend in backends:
            PortfolioOptimizer(SolverPolicy(backend=backend)).solve(
                next(iter(problems.values()))
            )
        for repeat in range(args.rounds):
            for i in rng.permutation(len(jobs)):
                date, budget, active_limit, variant, backend, turnover_form = jobs[i]
                base = problems[date].with_constraints(total_active=active_limit)
                problem = base.with_constraints(
                    tracking_error=TrackingErrorLimit(budget)
                    if budget is not None
                    else None,
                    total_active=base.constraints.total_active
                    if variant != "removed"
                    else None,
                )
                observed.clear()
                start = time.perf_counter()
                cpu_start = time.process_time()

                def finish_domain(builder):
                    if turnover_form == "sparse" and not builder.layout.sparse_turnover:
                        raise ValueError("该问题不满足稀疏换手率表达的前提")
                    observed["sparse_turnover"] = builder.layout.sparse_turnover
                    observed["compiler_optimizations"] = list(builder.optimizations)
                    domain = finish(builder)
                    if variant == "sparse":
                        return sparse_active_domain(builder, domain)
                    if variant == "split":
                        return split_active_domain(builder, domain)
                    return domain

                with (
                    patch.object(
                        compiler_module,
                        "_DomainBuilder",
                        FullTurnoverBuilder
                        if turnover_form == "full"
                        else _DomainBuilder,
                    ),
                    patch.object(_DomainBuilder, "finish", finish_domain),
                ):
                    result = PortfolioOptimizer(SolverPolicy(backend=backend)).solve(
                        problem
                    )
                wall = time.perf_counter() - start
                cpu = time.process_time() - cpu_start
                native = observed.pop("native", None)
                if native is not None:
                    observed["iterations"] = native.get_info().iterations
                w = (
                    result.require_weights().to_numpy()
                    if result.status.has_solution
                    else None
                )
                active = (
                    float(np.abs(w - base.data.benchmark).sum())
                    if w is not None
                    else None
                )
                te = risk_value(base, w) if w is not None else None
                turnover = (
                    float(np.abs(w - base.data.initial_weight).sum())
                    if w is not None and base.data.initial_weight is not None
                    else None
                )
                row = dict(
                    date=date,
                    budget=budget,
                    variant=variant,
                    turnover_form=turnover_form,
                    semantic_hash=result.fingerprint.semantic_hash
                    if result.fingerprint is not None
                    else None,
                    backend=backend,
                    actual_backend=result.backend,
                    scenario=args.scenario,
                    risk_aversion=args.risk_aversion if args.scenario == "qp" else None,
                    repeat=repeat,
                    wall_s=wall,
                    cpu_s=cpu,
                    solve_s=result.timings.backend_solve_s,
                    setup_s=result.timings.backend_setup_s,
                    status=result.status.value,
                    objective=result.objective_value,
                    total_active=active,
                    original_limit=base.constraints.total_active,
                    original_active_violation=max(
                        0.0, active - base.constraints.total_active
                    )
                    if active is not None and base.constraints.total_active is not None
                    else None,
                    te=te,
                    risk_violation=max(0.0, te - budget)
                    if te is not None and budget is not None
                    else 0.0,
                    budget_violation=abs(float(w.sum()) - base.constraints.budget)
                    if w is not None
                    else None,
                    turnover=turnover,
                    turnover_violation=max(
                        0.0, turnover - base.constraints.turnover.l1_limit
                    )
                    if turnover is not None and base.constraints.turnover is not None
                    else 0.0,
                    short_gross=float(np.maximum(-w, 0.0).sum())
                    if w is not None
                    else None,
                    short_names=int(np.count_nonzero(w < -1e-8))
                    if w is not None
                    else None,
                    message=result.message,
                    max_violation=max(
                        (v.amount for v in result.violations), default=0.0
                    ),
                    **observed,
                )
                records.append(row)
                with (args.output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    frame = pd.DataFrame(records)
    frame.groupby(
        ["date", "budget", "original_limit", "variant", "backend", "turnover_form"],
        dropna=False,
    )[
        [
            "wall_s",
            "cpu_s",
            "solve_s",
            "setup_s",
            "iterations",
            "objective",
            "total_active",
            "te",
            "original_active_violation",
            "max_violation",
            "risk_violation",
            "budget_violation",
            "turnover_violation",
            "short_gross",
            "short_names",
        ]
    ].median().to_csv(args.output / "summary.csv")


if __name__ == "__main__":
    main()

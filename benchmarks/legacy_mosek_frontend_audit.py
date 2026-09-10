"""真实 v5 多日 LP：固定 MOSEK、相同输入，对照历史 linopt 与当前前端。

收益暂用显式零收益控制变量，不冒充生产回测。旧模块只从 Git 历史读取到独立命名空间，
不恢复废弃代码、不覆盖当前包。旧输出清理前的原始权重单独捕获以比较目标和可行性。
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
import types
import platform

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from frontend_prepare_v5_real import (  # noqa: E402
    DataStore,
    Config,
    _parser,
    _percent_risk_to_problem,
)
from optim import ExposureBounds, PortfolioOptimizer, SolverPolicy  # noqa: E402
from optim.data import FactorRiskFrames, InMemoryDataSource, PortfolioSchedule  # noqa: E402
from optim.model import compile_problem  # noqa: E402
from optim._impl.solution import evaluate_solution, lift_weights  # noqa: E402


def legacy_modules(ref):
    """只读导出历史模块，保留旧实现行为。"""
    package = types.ModuleType("_legacy_optim_audit")
    package.__path__ = []
    sys.modules[package.__name__] = package
    result = []
    for name in ("solver", "linopt"):
        module = types.ModuleType(f"{package.__name__}.{name}")
        module.__package__ = package.__name__
        sys.modules[module.__name__] = module
        source = subprocess.check_output(
            ["git", "show", f"{ref}:optim/{name}.py"], cwd=ROOT, text=True
        )
        exec(compile(source, f"{ref}/optim/{name}.py", "exec"), module.__dict__)
        result.append(module)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-ref", default="f5013b1^")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MOSEKLM_LICENSE_FILE", "/home/shao/.mosek/mosek.lic")
    config = _parser().parse_args([])
    store = DataStore(
        config.risk_model_xlsx, config.benchmark_csv, config.alpha_csv, config.cache_dir
    )
    dates = store.dates[: args.dates]
    problems = []
    for date in dates:
        day = store.prepare_day(date, store.benchmark_for_date(date), Config(), "error")
        problem = _percent_risk_to_problem(day, case="lp", args=config)
        from optim import SymmetricBound

        problem = replace(
            problem,
            constraints=replace(
                problem.constraints,
                active_weight=SymmetricBound(0.004),
                benchmark_member_weight=None,
                style=ExposureBounds(
                    default=(-0.3, 0.3), overrides={"size": (-0.2, 0.2)}
                ),
                industry=ExposureBounds(default=(-0.01, 0.01)),
            ),
        )
        problems.append(problem)
    exposure, covariance, specific, benchmark, universe = [], [], [], [], []
    for p in problems:
        d, r = p.data, p.data.risk_model
        idx = pd.MultiIndex.from_product([[d.date], d.assets], names=["dt", "sid"])
        exposure.append(pd.DataFrame(r.exposure, index=idx, columns=r.factor_names))
        covariance.append(
            pd.DataFrame(
                r.covariance,
                index=pd.MultiIndex.from_product(
                    [[d.date], r.factor_names], names=["dt", "factor"]
                ),
                columns=r.factor_names,
            )
        )
        specific.append(pd.Series(r.specific_volatility, index=idx))
        benchmark.append(pd.Series(d.benchmark, index=idx))
        universe.append(
            pd.DataFrame(
                {"alpha": d.alpha, "tradable": True, "member": True}, index=idx
            )
        )
    risk = FactorRiskFrames(
        exposure=pd.concat(exposure),
        covariance=pd.concat(covariance),
        specific_volatility=pd.concat(specific),
        factor_types=dict(zip(r.factor_names, r.factor_types)),
    )
    source = InMemoryDataSource(risk_data=risk, benchmark=pd.concat(benchmark))
    schedule = PortfolioSchedule(pd.concat(universe))
    started = time.perf_counter()
    run = source.prepare_run(
        schedule,
        objective=problems[0].objective,
        constraints=problems[0].constraints,
        alpha_spec=problems[0].data.alpha_spec,
        initial_weight=pd.Series(
            problems[0].data.benchmark, index=problems[0].data.assets
        ),
    )
    run.validation.raise_for_errors()
    prepare_run_s = time.perf_counter() - started
    solver, linopt = legacy_modules(args.legacy_ref)
    original = solver.Solver.solve
    raw = {}

    def capture(self, *a, **kw):
        started = time.perf_counter()
        raw["weights"] = original(self, *a, **kw)
        raw["solve_s"] = time.perf_counter() - started
        return raw["weights"]

    solver.Solver.solve = capture
    optimizer = PortfolioOptimizer(policy=SolverPolicy(backend="mosek"))
    # 两种后端都预热；不把首次导入/授权建立计入逐日期均值。
    optimizer.solve(problems[0])
    profile = cProfile.Profile()
    rows = []
    previous = None
    for i, template in enumerate(problems):
        started = time.perf_counter()
        p = run.problem_at(template.data.date)
        omitted_mass = 0.0
        if previous is not None:
            omitted_mass = float(
                previous.loc[~previous.index.isin(p.data.assets)].abs().sum()
            )
            initial = previous.reindex(p.data.assets, fill_value=0.0).to_numpy()
            initial /= initial.sum()
            p = replace(p, data=replace(p.data, initial_weight=initial))
        if i == 0:
            p = replace(p, constraints=replace(p.constraints, turnover=None))
        materialize_s = time.perf_counter() - started
        d, r = p.data, p.data.risk_model
        idx = universe[i].index
        style_names = [
            name
            for name, kind in zip(r.factor_names, r.factor_types)
            if kind == "style"
        ]
        industry_names = [
            name
            for name, kind in zip(r.factor_names, r.factor_types)
            if kind == "industry"
        ]
        kwargs = dict(
            styles=exposure[i][style_names],
            industries=exposure[i][industry_names],
            init_portfolio=pd.Series(d.initial_weight, index=d.assets),
            constraint_style={"all": (-0.3, 0.3), "size": (-0.2, 0.2)},
            constraint_industry={"all": (-0.01, 0.01)},
            turnover_limit=None if i == 0 else 0.05,
            active_ub=0.004,
            total_active_ub=1.8,
            enable_parameter_validation=False,
        )

        def old_solve():
            return linopt.optimize(
                d.date,
                universe[i],
                pd.DataFrame({"weight": d.benchmark}, index=idx),
                **kwargs,
            )

        if i == 0:
            old_solve()
        row = {
            "date": str(d.date.date()),
            "assets": len(d.assets),
            "materialize_s": materialize_s,
            "omitted_initial_mass": omitted_mass,
        }

        # 交替次序，减小缓存/温度对单边结果的偏置。
        def new_solve():
            started = time.perf_counter()
            result = optimizer._solve_prevalidated(p)
            row.update(
                new_s=time.perf_counter() - started,
                status=result.status.value,
                **asdict(result.timings),
            )
            return result

        def old_timed():
            started = time.perf_counter()
            try:
                old_solve()
                row.update(
                    old_s=time.perf_counter() - started,
                    old_solve_s=raw["solve_s"],
                    old_status="optimal",
                )
            except solver.ProblemInfeasible:
                row.update(old_s=time.perf_counter() - started, old_status="infeasible")

        if i % 2:
            old_timed()
            result = new_solve()
        else:
            result = new_solve()
            old_timed()
        if result.status.has_solution:
            previous = result.require_weights()
            if row["old_status"] == "optimal":
                row["objective_delta_raw"] = float(
                    d.alpha @ raw["weights"] - result.metrics.objective
                )
                compiled = compile_problem(p, validate=False)
                _, _, violation = evaluate_solution(
                    p, compiled, lift_weights(p, compiled, raw["weights"])
                )
                row["old_max_constraint_violation"] = violation
        rows.append(row)
        print(json.dumps(row), flush=True)
    # 单独 profiling，不将插桩时间混入前面的正常计时。
    profile.runcall(optimizer._solve_prevalidated, p)
    profile.dump_stats(str(args.output / "new_last.prof"))
    with (args.output / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(45)
    import mosek
    import optim._impl.compiler as compiler_module

    report = {
        "legacy_ref": args.legacy_ref,
        "legacy_commit": subprocess.check_output(
            ["git", "rev-parse", args.legacy_ref], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "mosek": mosek.Env.getversion(),
        "compiler_module": compiler_module.__file__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "benchmark": "v5 CSI1000, not production CSI300",
        "returns": "explicit zero-return controlled chain",
        "prepare_run_s": prepare_run_s,
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

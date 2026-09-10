"""仅测试当前版本：完整 optimize_range 墙钟、循环耗时及真实 repro。

v5 为中证1000多日真实风险/alpha/基准，收益明确设为零用于控制实验；不冒充生产收益。
不调用任何旧优化器，不替换公共求解入口。阶段计时只做只读计数，模型参数不变。
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import cProfile
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import mosek

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from frontend_prepare_v5_real import (  # noqa: E402
    DataStore, Config, _parser, _percent_risk_to_problem,
)
from optim import (  # noqa: E402
    AlphaSpec, ExposureBounds, MaximizeAlpha, PortfolioConstraints, PortfolioOptimizer,
    SequencePolicy, SolverPolicy, SymmetricBound, TurnoverLimit, load_repro,
)
from optim.data import FactorRiskFrames, InMemoryDataSource, PortfolioSchedule  # noqa: E402
import optim._impl.compiler as compiler_module  # noqa: E402
import optim._core.backends.mosek as backend_module  # noqa: E402


def constraints():
    """用户提供的纯 LP 约束，保持未显式配置项的公共默认值。"""
    return PortfolioConstraints(
        active_weight=SymmetricBound(.004), total_active=1.8, turnover=TurnoverLimit(.05),
        style=ExposureBounds(default=(-.3, .3), overrides={"size": (-.2, .2)}),
        industry=ExposureBounds(default=(-.01, .01)),
    )


def load_v5(count):
    """加载真实 v5，并构造公共内存数据源；加载成本单独报告。"""
    args = _parser().parse_args([])
    started = time.perf_counter()
    store = DataStore(args.risk_model_xlsx, args.benchmark_csv, args.alpha_csv, args.cache_dir)
    problems = []
    exposure, covariance, specific, benchmark, universe = [], [], [], [], []
    for date in store.dates[:count]:
        day = store.prepare_day(date, store.benchmark_for_date(date), Config(), "error")
        p = _percent_risk_to_problem(day, case="lp", args=args)
        problems.append(p)
        d, r = p.data, p.data.risk_model
        idx = pd.MultiIndex.from_product([[d.date], d.assets], names=["dt", "sid"])
        exposure.append(pd.DataFrame(r.exposure, index=idx, columns=r.factor_names))
        covariance.append(pd.DataFrame(r.covariance, index=pd.MultiIndex.from_product([[d.date], r.factor_names], names=["dt", "factor"]), columns=r.factor_names))
        specific.append(pd.Series(r.specific_volatility, index=idx))
        benchmark.append(pd.Series(d.benchmark, index=idx))
        universe.append(pd.DataFrame({"alpha": d.alpha, "tradable": d.tradable}, index=idx))
    frames = FactorRiskFrames(exposure=pd.concat(exposure), covariance=pd.concat(covariance), specific_volatility=pd.concat(specific), factor_types=dict(zip(r.factor_names, r.factor_types)))
    returns = {p.data.date: pd.Series(0.0, index=previous.data.assets) for previous, p in zip(problems, problems[1:])}
    return frames, pd.concat(benchmark), pd.concat(universe), returns, time.perf_counter()-started


class MeasuredOptimizer(PortfolioOptimizer):
    """仍由公共 optimize_range 驱动，只测量完整循环和逐期调用边界。"""
    def __init__(self, backend="mosek"):
        super().__init__(SolverPolicy(backend=backend))
        self.in_sequence = False
        self.sequence_wall_s = 0.0
        self.rows = []
        self.native = []

    def solve_sequence(self, *args, **kwargs):
        self.in_sequence = True
        started = time.perf_counter()
        try:
            return super().solve_sequence(*args, **kwargs)
        finally:
            self.sequence_wall_s = time.perf_counter()-started
            self.in_sequence = False

    def _solve_prevalidated(self, problem):
        self.native = []
        started = time.perf_counter()
        result = super()._solve_prevalidated(problem)
        wall = time.perf_counter()-started
        self.rows.append({
            "date": str(problem.data.date.date()), "call_wall_s": wall, "status": result.status.value,
            "backend": result.backend,
            "max_violation": max((v.amount for v in result.violations), default=0.0),
            "assets": len(problem.data.assets), "benchmark_members": int(np.sum(problem.data.benchmark > 0)),
            "nontradable": int(np.sum(~problem.data.tradable)),
            "turnover_limit": None if problem.constraints.turnover is None else problem.constraints.turnover.limit,
            "timings": asdict(result.timings), "native": list(self.native),
        })
        return result


def run_range(inputs, show_progress, on_failure, threads, backend="mosek"):
    """计时包含完整公共入口；独立报告预检与数据物化，不按成功结果筛选计数。"""
    frames, benchmark, universe, returns, _ = inputs
    optimizer = MeasuredOptimizer(backend)
    materialize = {"preflight_s": 0.0, "loop_s": 0.0, "preflight_calls": 0, "loop_calls": 0}
    original_materialize = InMemoryDataSource._materialize_problem
    original_optimize = mosek.Task.optimize

    def timed_materialize(self, *args, **kwargs):
        stage = "loop" if optimizer.in_sequence else "preflight"
        started = time.perf_counter()
        try:
            return original_materialize(self, *args, **kwargs)
        finally:
            materialize[f"{stage}_s"] += time.perf_counter()-started
            materialize[f"{stage}_calls"] += 1

    def native_optimize(task, *args, **kwargs):
        if threads is not None:
            task.putintparam(mosek.iparam.num_threads, threads)
        started = time.perf_counter()
        result = original_optimize(task, *args, **kwargs)
        optimizer.native.append({"wall_s": time.perf_counter()-started,
            "threads": task.getintinf(mosek.iinfitem.intpnt_num_threads),
            "iterations": task.getintinf(mosek.iinfitem.intpnt_iter)})
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(InMemoryDataSource, "_materialize_problem", timed_materialize))
        stack.enter_context(patch.object(mosek.Task, "optimize", native_optimize))
        started = time.perf_counter()
        result = optimizer.optimize_range(
            data_source=InMemoryDataSource(risk_data=replace(frames), benchmark=benchmark),
            schedule=PortfolioSchedule(universe), constraints=constraints(), objective=MaximizeAlpha(),
            alpha_spec=AlphaSpec(), show_progress=show_progress,
            holding_period_returns=returns,
            sequence_policy=SequencePolicy(on_failure=on_failure),
        )
        wall = time.perf_counter()-started
    completed = len(result.steps)
    call_sum = sum(r["call_wall_s"] for r in optimizer.rows)
    return {"backend_policy": backend, "show_progress": show_progress, "on_failure": on_failure, "threads_requested": threads,
        "scheduled": len(result.steps) if result.stopped_date is None else len(returns)+1,
        "attempted": completed, "statuses": dict(Counter(s.result.status.value for s in result.steps)),
        "stopped_date": str(result.stopped_date) if result.stopped_date is not None else None,
        "public_wall_s": wall, "sequence_wall_s": optimizer.sequence_wall_s,
        "public_periods_per_s": completed/wall, "sequence_periods_per_s": completed/optimizer.sequence_wall_s,
        "prepare_wall_s": wall-optimizer.sequence_wall_s,
        "reported_schedule_prepare_s": result.schedule_prepare_s,
        "inner_calls_wall_s": call_sum, "loop_outside_calls_s": optimizer.sequence_wall_s-call_sum,
        "materialize": materialize, "rows": optimizer.rows}


def repro_runs(repeats, backend="mosek"):
    """真实300/1000基准复现包，改为用户这次 LP 约束，仅测当前公共入口。"""
    rows = []
    optimizer = PortfolioOptimizer(SolverPolicy(backend=backend))
    for name in ("pp_20260814.tar.gz", "pp_20260814_csi1000.tar.gz"):
        p = replace(load_repro(ROOT / "tmp" / name).problem, constraints=constraints(), objective=MaximizeAlpha())
        for mode in ("original_initial", "first_build"):
            for repeat in range(repeats+1):
                started = time.perf_counter()
                if mode == "original_initial":
                    result = optimizer.solve(p)
                else:
                    result = optimizer.solve_sequence([replace(p, data=replace(p.data, initial_weight=None))]).steps[0].result
                row = {"sample": name, "mode": mode, "repeat": repeat, "wall_s": time.perf_counter()-started, "status": result.status.value, "timings": asdict(result.timings)}
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mosek-threads", type=int)
    parser.add_argument("--backend", choices=("mosek", "auto"), default="mosek")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = load_v5(args.dates)
    report = {"python": sys.version, "platform": platform.platform(), "mosek": mosek.Env.getversion(),
        "compiler_module": compiler_module.__file__, "backend_module": backend_module.__file__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "returns": "explicit zero-return experiment, not production daily returns",
        "input_load_and_assembly_s": inputs[-1], "runs": []}
    # 首轮冷启动单独保留，之后交错开/关进度条。全部调用都是完整公共 optimize_range。
    for repeat in range(args.repeats+1):
        for progress in ((False, True) if repeat % 2 == 0 else (True, False)):
            row = run_range(inputs, progress, "stop", args.mosek_threads, args.backend)
            row["repeat"] = repeat
            report["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "rows"}), flush=True)
    # 另测显式 hold 的整段35日，不把它与生产默认 stop 混为一谈。
    report["hold_run"] = run_range(inputs, False, "hold", args.mosek_threads, args.backend)
    report["repro_runs"] = repro_runs(args.repeats, args.backend)
    if args.profile:
        profile = cProfile.Profile()
        profile.runcall(run_range, inputs, False, "stop", args.mosek_threads, args.backend)
        profile.dump_stats(str(args.output / "full_range.prof"))
        with (args.output / "profile.txt").open("w") as stream:
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(65)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""真实 SOCP 的线程对照；独立进程隔离线程池，不修改产品配置。

运行 ``python benchmarks/solver_threads.py``。默认包括 pp 的三个变体和五个 v5
日期各两种风险预算。相同问题保持相同初始持仓，不把后端差异传播到下一期。
数据加载、进程启动和预热不计入单次优化耗时；输出统计不包含权重。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(folder: Path, count: int) -> None:
    """一次性准备固定输入；旧 pp 仅在实验内校验哈希后提取业务问题。

    不恢复旧求解策略和 theta，不放宽产品 load_repro 的版本检查。
    本地 inputs 包含敏感原始数据，不应随统计文件发送。
    """
    import numpy as np
    from optim import PortfolioOptimizer, TrackingErrorLimit, export_repro
    from optim.repro import _decode
    from frontend_prepare_v5_real import (
        Config,
        DataStore,
        _parser,
        _percent_risk_to_problem,
    )

    with ZipFile(ROOT / "tmp/pp.tar.gz") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {name: archive.read(name) for name in manifest}
        assert all(
            hashlib.sha256(v).hexdigest() == manifest[k] for k, v in members.items()
        )
        payload = json.loads(members["payload.json"])
        if payload["format_version"] != 1:
            raise ValueError("实验迁移仅接受已知 v1 pp；其他输入请使用正式 load_repro")
        problem = _decode(payload["problem"], members)
    problems = {
        "pp_infeasible": problem,
        "pp_te2": problem.with_constraints(turnover=None),
        "pp_te6": problem.with_constraints(
            turnover=None, tracking_error=TrackingErrorLimit(0.06)
        ),
    }
    if count:
        args = _parser().parse_args([])
        store = DataStore(
            args.risk_model_xlsx, args.benchmark_csv, args.alpha_csv, args.cache_dir
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
        for i in np.unique(np.linspace(0, len(store.dates) - 1, count).astype(int)):
            date = store.dates[i]
            initial, _ = store.random_top_range_initial(date, 500, 600, 20260826)
            day = store.prepare_day(date, initial, config, "error")
            p = _percent_risk_to_problem(day, case="factor_qcqp", args=args)
            for budget in (0.02, 0.06):
                problems[f"v5_{date.date()}_te{int(budget * 100)}"] = (
                    p.with_constraints(tracking_error=TrackingErrorLimit(budget))
                )
    folder.mkdir(parents=True, exist_ok=True)
    for name, problem in problems.items():
        export_repro(folder / f"{name}.zip", result=PortfolioOptimizer().solve(problem))


def worker(args) -> None:
    """在独立进程中设定原生参数，仍通过公共优化器及其验收路径求解。"""
    import clarabel
    import numpy as np
    import mosek.fusion as fusion
    from optim import PortfolioOptimizer, SolverPolicy, load_repro
    from optim._core.backends.mosek import MosekBackend

    method, threads_text = args.worker.split(":")
    threads = int(threads_text)
    problems = {
        p.stem: load_repro(p).problem
        for p in sorted((args.output / "inputs").glob("*.zip"))
    }
    observed = {}
    factory = clarabel.DefaultSolver
    model_init = fusion.Model.__init__
    model_solve = fusion.Model.solve
    mosek_solve = MosekBackend._solve_factor_qcqp

    def clarabel_factory(P, q, A, b, cones, settings):
        settings.direct_solve_method = method
        settings.max_threads = threads
        solver = factory(P, q, A, b, cones, settings)
        observed["solver"] = solver
        return solver

    def fusion_init(model, *a, **kw):
        model_init(model, *a, **kw)
        model.setSolverParam("numThreads", threads)

    def fusion_solve(model, *a, **kw):
        value = model_solve(model, *a, **kw)
        observed["actual_threads"] = int(model.getSolverIntInfo("intpntNumThreads"))
        return value

    def capture_mosek(self, model, options):
        result = mosek_solve(self, model, options)
        observed["iterations"] = result.iterations
        return result

    with ExitStack() as stack:
        if method == "mosek":
            stack.enter_context(patch.object(fusion.Model, "__init__", fusion_init))
            stack.enter_context(patch.object(fusion.Model, "solve", fusion_solve))
            stack.enter_context(
                patch.object(MosekBackend, "_solve_factor_qcqp", capture_mosek)
            )
        else:
            stack.enter_context(
                patch.object(clarabel, "DefaultSolver", clarabel_factory)
            )
        optimizer = PortfolioOptimizer(
            SolverPolicy(backend="mosek" if method == "mosek" else "clarabel")
        )
        optimizer.solve(problems["pp_te2"])
        rng = np.random.default_rng(20260909)
        for repeat in range(args.rounds):
            for name in rng.permutation(list(problems)):
                observed.clear()
                cpu_start, start = time.process_time(), time.perf_counter()
                result = optimizer.solve(problems[name])
                wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
                if method != "mosek":
                    info = observed.pop("solver").get_info()
                    observed.update(
                        iterations=info.iterations,
                        actual_threads=info.linsolver.threads,
                    )
                row = dict(
                    config=args.worker,
                    case=str(name),
                    repeat=repeat,
                    wall_s=wall,
                    cpu_s=cpu,
                    cpu_to_wall=cpu / wall,
                    setup_s=result.timings.backend_setup_s,
                    solve_s=result.timings.backend_solve_s,
                    status=result.status.value,
                    objective=result.objective_value,
                    max_violation=max((v.amount for v in result.violations), default=0),
                    gap=result.certificate.absolute_gap if result.certificate else None,
                    **observed,
                )
                with (args.output / f"{method}_{threads}.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


def summarize(folder: Path) -> None:
    """汇总逐次耗时和同问题重复中位数；统计文件均不包含持仓。"""
    import pandas as pd

    rows = [
        json.loads(line)
        for path in sorted(folder.glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    frame = pd.DataFrame(rows)
    summary = frame.groupby("config").agg(
        runs=("wall_s", "size"),
        optimal=("status", lambda s: (s == "optimal").sum()),
        infeasible=("status", lambda s: (s == "infeasible").sum()),
        wall_mean_s=("wall_s", "mean"),
        wall_median_s=("wall_s", "median"),
        wall_p95_s=("wall_s", lambda s: s.quantile(0.95)),
        setup_mean_s=("setup_s", "mean"),
        solve_mean_s=("solve_s", "mean"),
        cpu_mean_s=("cpu_s", "mean"),
        max_violation=("max_violation", "max"),
        actual_threads=("actual_threads", "max"),
    )
    summary["cpu_to_wall"] = summary.cpu_mean_s / summary.wall_mean_s
    summary.to_csv(folder / "summary.csv")
    frame.groupby(["case", "config"]).agg(
        wall_median_s=("wall_s", "median"),
        solve_median_s=("solve_s", "median"),
        iterations=("iterations", "median"),
        objective=("objective", "median"),
    ).to_csv(folder / "cases.csv")
    print(summary.to_string(), flush=True)


def main() -> None:
    """串行调度各配置，限制无关 BLAS 线程并保存可核查的逐次结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tmp/solver_threads_20260909"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--v5-dates", type=int, default=5)
    parser.add_argument("--worker")
    parser.add_argument(
        "--configs",
        default="qdldl:1,faer:1,faer:2,faer:4,faer:8,faer:0,mosek:1,mosek:2,mosek:4,mosek:8,mosek:0",
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        parser.error("输出目录已存在，请指定新目录，避免混合实验结果")
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    # 准备也限制 BLAS，但不修改当前进程已初始化的线程池。
    os.environ.update(
        {
            k: env[k]
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        }
    )
    prepare(args.output / "inputs", args.v5_dates)
    import numpy as np
    from importlib.metadata import version

    metadata = dict(
        python=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        versions={k: version(k) for k in ("clarabel", "Mosek", "numpy", "scipy")},
        rounds=args.rounds,
        seed=20260909,
        blas_threads=1,
        configs=args.configs,
        scope="fixed-input independent solves; no warm start",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for config in np.random.default_rng(20260909).permutation(args.configs.split(",")):
        print(f"START {config}", flush=True)
        with (args.output / f"{config.replace(':', '_')}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    config,
                    "--output",
                    str(args.output),
                    "--rounds",
                    str(args.rounds),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        print(f"DONE {config}", flush=True)
    summarize(args.output)


if __name__ == "__main__":
    main()

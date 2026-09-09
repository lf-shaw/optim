"""固定输入扫描风险预算，定位耗时增长来自建模、迭代数还是单次迭代成本。

使用 solver_threads.py 生成的当前格式输入。只调整实验进程的线程数，生产配置不变。
所有问题独立冷启动；数值库应通过环境变量限制为单线程。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clarabel
import mosek.fusion as fusion
import numpy as np

from optim import PortfolioOptimizer, SolverPolicy, TrackingErrorLimit, load_repro
from optim._core.backends.mosek import MosekBackend


def risk_value(problem, weights):
    """在原始年化风险数据上独立计算 TE。"""
    risk = problem.data.risk_model
    active = weights - problem.data.benchmark
    factor = risk.exposure.T @ active
    return float(
        np.sqrt(
            factor @ risk.covariance @ factor
            + np.sum((risk.specific_volatility * active) ** 2)
        )
    )


def main():
    """交错求解预算变体，同时记录 Clarabel 逐迭代日志和矩阵指纹。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path("tmp/solver_threads_20260909/inputs")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/risk_budget_scan_20260909")
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repro", help="指定单个当前格式复现包，替代默认输入目录")
    parser.add_argument("--budgets", default="0.02,0.04,0.06,0.10")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，请指定新目录")
    args.output.mkdir(parents=True)
    problems = (
        {"pp_te2": load_repro(args.repro).problem}
        if args.repro
        else {
            p.stem: load_repro(p).problem for p in sorted(args.inputs.glob("*te2.zip"))
        }
    )
    lp_records = []
    for name, problem in problems.items():
        lp = PortfolioOptimizer().solve(problem.with_constraints(tracking_error=None))
        lp_records.append(
            dict(
                case=name,
                status=lp.status.value,
                te=risk_value(problem, lp.require_weights().to_numpy())
                if lp.status.has_solution
                else None,
                objective=lp.objective_value,
            )
        )
    (args.output / "lp_reference.json").write_text(json.dumps(lp_records, indent=2))
    factory = clarabel.DefaultSolver
    model_init = fusion.Model.__init__
    native_mosek = MosekBackend._solve_factor_qcqp
    observed = {}

    def make_solver(P, q, A, b, cones, settings):
        settings.max_threads = 1
        solver = factory(P, q, A, b, cones, settings)
        observed["native"] = solver
        observed["matrix_hash"] = hashlib.sha256(
            A.data.tobytes() + A.indices.tobytes() + A.indptr.tobytes() + q.tobytes()
        ).hexdigest()
        observed["shape"] = A.shape
        observed["nnz"] = A.nnz
        history = []
        observed["history"] = history

        def callback(info):
            history.append(
                {
                    key: float(getattr(info, key))
                    for key in (
                        "iterations",
                        "step_length",
                        "gap_abs",
                        "gap_rel",
                        "res_primal",
                        "res_dual",
                        "mu",
                        "ktratio",
                    )
                }
            )
            return False

        solver.set_termination_callback(callback)
        return solver

    def init_model(model, *a, **kw):
        model_init(model, *a, **kw)
        model.setSolverParam("numThreads", 1)

    def solve_mosek(self, model, options):
        result = native_mosek(self, model, options)
        observed["iterations"] = result.iterations
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(clarabel, "DefaultSolver", make_solver))
        stack.enter_context(patch.object(fusion.Model, "__init__", init_model))
        stack.enter_context(
            patch.object(MosekBackend, "_solve_factor_qcqp", solve_mosek)
        )
        for backend in ("clarabel", "mosek"):
            PortfolioOptimizer(SolverPolicy(backend=backend)).solve(problems["pp_te2"])
        rng = np.random.default_rng(20260909)
        work = [
            (name, budget, backend)
            for name in problems
            for budget in map(float, args.budgets.split(","))
            for backend in ("clarabel", "mosek")
        ]
        for repeat in range(args.rounds):
            for i in rng.permutation(len(work)):
                name, budget, backend = work[i]
                problem = problems[name].with_constraints(
                    tracking_error=TrackingErrorLimit(budget)
                )
                observed.clear()
                start = time.perf_counter()
                result = PortfolioOptimizer(SolverPolicy(backend=backend)).solve(
                    problem
                )
                wall = time.perf_counter() - start
                if backend == "clarabel":
                    native = observed.pop("native")
                    info = native.get_info()
                    observed["iterations"] = info.iterations
                iterations = observed["iterations"]
                w = (
                    result.require_weights().to_numpy()
                    if result.status.has_solution
                    else None
                )
                row = dict(
                    case=name,
                    budget=budget,
                    backend=backend,
                    repeat=repeat,
                    wall_s=wall,
                    setup_s=result.timings.backend_setup_s,
                    solve_s=result.timings.backend_solve_s,
                    per_iter_s=result.timings.backend_solve_s / max(iterations, 1),
                    status=result.status.value,
                    objective=result.objective_value,
                    te=risk_value(problem, w) if w is not None else None,
                    weights_near_zero=int(np.sum(np.abs(w) < 1e-7))
                    if w is not None
                    else None,
                    max_violation=max((v.amount for v in result.violations), default=0),
                    **observed,
                )
                with (args.output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps({k: v for k, v in row.items() if k != "history"}),
                    flush=True,
                )
    import pandas as pd

    frame = pd.DataFrame(
        json.loads(line)
        for line in (args.output / "runs.jsonl").read_text().splitlines()
    )
    frame.groupby(["case", "budget", "backend"])[
        [
            "iterations",
            "wall_s",
            "solve_s",
            "setup_s",
            "per_iter_s",
            "te",
            "max_violation",
        ]
    ].median().to_csv(args.output / "summary.csv")


if __name__ == "__main__":
    main()

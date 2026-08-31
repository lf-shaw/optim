#!/usr/bin/env python3
"""Compare sparse QP backends on the exact hard-turnover factor formulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from .benchmark import Config, DataStore, build_problem, solver_kwargs
    from .factor_qcqp import FactorQCQPSettings, _build_qp_data, _warm_start_vector
except ImportError:
    from benchmark import Config, DataStore, build_problem, solver_kwargs
    from factor_qcqp import FactorQCQPSettings, _build_qp_data, _warm_start_vector


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-model-xlsx", type=Path, required=True)
    ap.add_argument("--benchmark-csv", type=Path, required=True)
    ap.add_argument("--date-index", type=int, default=0)
    ap.add_argument("--thetas", default="1,4,16")
    ap.add_argument("--eps", type=float, default=1e-7)
    ap.add_argument("--max-iter", type=int, default=10_000)
    ap.add_argument("--backends", default="PIQP,PROXQP")
    ap.add_argument("--output-csv", type=Path)
    return ap


def get_number(obj: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def split_constraints(data: Any) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
    equality = (
        np.isfinite(data.lower)
        & np.isfinite(data.upper)
        & np.isclose(data.lower, data.upper, rtol=0.0, atol=1e-14)
    )
    return (
        data.A[equality].tocsc(),
        data.lower[equality].copy(),
        data.A[~equality].tocsc(),
        data.lower[~equality].copy(),
        data.upper[~equality].copy(),
    )


def linear_violation(data: Any, vector: np.ndarray) -> float:
    value = np.asarray(data.A @ vector).reshape(-1)
    lower = np.where(np.isfinite(data.lower), data.lower - value, -np.inf)
    upper = np.where(np.isfinite(data.upper), value - data.upper, -np.inf)
    return max(0.0, float(np.max(lower)), float(np.max(upper)))


def metrics(day: Any, data: Any, vector: np.ndarray) -> dict[str, float]:
    weight = vector[: data.n_assets]
    active = weight - np.asarray(day.benchmark)
    factor = np.asarray(day.exposure).T @ active
    variance = float(
        factor @ np.asarray(day.covariance) @ factor
        + np.sum(np.square(np.asarray(day.spec_risk) * active))
    )
    return {
        "max_linear_violation": linear_violation(data, vector),
        "turnover_l1": float(np.abs(weight - np.asarray(day.initial)).sum()),
        "risk_pct": float(np.sqrt(max(0.0, variance))),
        "alpha_value": float(np.asarray(day.alpha) @ weight),
    }


def q_for_theta(data: Any, theta: float) -> np.ndarray:
    q = data.q_base.copy()
    q[: data.n_assets] -= theta * data.alpha_solver
    return q


def main() -> int:
    args = parser().parse_args()
    thetas = [float(value) for value in args.thetas.split(",")]
    config = Config()
    store = DataStore(args.risk_model_xlsx, args.benchmark_csv)
    date = store.dates[args.date_index]
    initial = store.top_n_initial(config.initial_top_n)
    day = store.prepare_day(date, initial, config, "error")

    lp = build_problem(day, "lp", config, "optimized", objective_target=0.2)
    lp_started = time.perf_counter()
    lp.problem.solve(**solver_kwargs("HIGHS", None))
    lp_s = time.perf_counter() - lp_started
    if lp.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or lp.x.value is None:
        raise RuntimeError(f"HiGHS warm-start LP failed: {lp.problem.status}")

    data = _build_qp_data(day, config, FactorQCQPSettings())
    P = (data.P + sp.triu(data.P, k=1).T).tocsc()
    Aeq, beq, C, lower, upper = split_constraints(data)
    warm = _warm_start_vector(data, day, np.asarray(lp.x.value).reshape(-1))
    rows: list[dict[str, Any]] = []

    backends = {value.strip().upper() for value in args.backends.split(",") if value.strip()}
    if "PIQP" in backends:
        run_piqp(rows, args, date, day, data, P, Aeq, beq, C, lower, upper, thetas)
    if "PROXQP" in backends:
        run_proxqp(rows, args, date, day, data, P, Aeq, beq, C, lower, upper, warm, thetas)

    frame = pd.DataFrame(rows)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "date": date.strftime("%Y-%m-%d"),
                "data_load_s": store.load_elapsed_s,
                "lp_warm_start_s": lp_s,
                "warm_start_linear_violation": linear_violation(data, warm),
                "n_variables": data.n_variables,
                "n_equalities": Aeq.shape[0],
                "n_inequalities": C.shape[0],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def run_proxqp(
    rows: list[dict[str, Any]], args: argparse.Namespace, date: Any, day: Any, data: Any,
    P: sp.csc_matrix, Aeq: sp.csc_matrix, beq: np.ndarray, C: sp.csc_matrix,
    lower: np.ndarray, upper: np.ndarray, warm: np.ndarray, thetas: list[float],
) -> None:
    try:
        import proxsuite

        qp = proxsuite.proxqp.sparse.QP(data.n_variables, Aeq.shape[0], C.shape[0])
        qp.settings.eps_abs = args.eps
        qp.settings.eps_rel = args.eps
        qp.settings.max_iter = args.max_iter
        qp.settings.compute_timings = True
        qp.settings.check_duality_gap = True
        for index, theta in enumerate(thetas):
            q = q_for_theta(data, theta)
            setup_started = time.perf_counter()
            if index == 0:
                qp.init(P, q, Aeq, beq, C, lower, upper)
            else:
                qp.update(g=q)
            setup_s = time.perf_counter() - setup_started
            solve_started = time.perf_counter()
            if index == 0:
                qp.solve(warm, np.zeros(Aeq.shape[0]), np.zeros(C.shape[0]))
            else:
                qp.solve()
            solve_s = time.perf_counter() - solve_started
            vector = np.asarray(qp.results.x).reshape(-1)
            info = qp.results.info
            row = {
                "backend": "PROXQP_SPARSE",
                "date": date.strftime("%Y-%m-%d"),
                "theta": theta,
                "status": str(info.status),
                "iterations": int(info.iter),
                "setup_update_wall_s": setup_s,
                "solve_wall_s": solve_s,
                "solver_setup_s": get_number(info, "setup_time"),
                "solver_solve_s": get_number(info, "solve_time", "run_time"),
                **metrics(day, data, vector),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except Exception as exc:
        row = {"backend": "PROXQP_SPARSE", "status": "exception", "error": repr(exc)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


def run_piqp(
    rows: list[dict[str, Any]], args: argparse.Namespace, date: Any, day: Any, data: Any,
    P: sp.csc_matrix, Aeq: sp.csc_matrix, beq: np.ndarray, C: sp.csc_matrix,
    lower: np.ndarray, upper: np.ndarray, thetas: list[float],
) -> None:
    try:
        import piqp

        solver = piqp.SparseSolver()
        solver.settings.eps_abs = args.eps
        solver.settings.eps_rel = args.eps
        solver.settings.eps_duality_gap_abs = 1e-4
        solver.settings.eps_duality_gap_rel = args.eps
        solver.settings.max_iter = args.max_iter
        solver.settings.compute_timings = True
        for index, theta in enumerate(thetas):
            q = q_for_theta(data, theta)
            setup_started = time.perf_counter()
            if index == 0:
                solver.setup(P, q, Aeq, beq, C, lower, upper)
            else:
                solver.update(c=q)
            setup_s = time.perf_counter() - setup_started
            solve_started = time.perf_counter()
            status = solver.solve()
            solve_s = time.perf_counter() - solve_started
            result = solver.result
            vector = np.asarray(result.x).reshape(-1)
            info = result.info
            row = {
                "backend": "PIQP_SPARSE",
                "date": date.strftime("%Y-%m-%d"),
                "theta": theta,
                "status": str(status),
                "iterations": int(info.iter),
                "setup_update_wall_s": setup_s,
                "solve_wall_s": solve_s,
                "solver_setup_s": get_number(info, "setup_time"),
                "solver_solve_s": get_number(info, "solve_time", "run_time"),
                **metrics(day, data, vector),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except Exception as exc:
        row = {"backend": "PIQP_SPARSE", "status": "exception", "error": repr(exc)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

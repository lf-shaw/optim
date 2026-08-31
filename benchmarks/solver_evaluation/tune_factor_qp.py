#!/usr/bin/env python3
"""Tune native OSQP settings on one real factor-QP without reloading Excel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import osqp
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
    ap.add_argument("--max-iter", type=int, default=20_000)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--experiment",
        choices=["settings", "turnover-dual", "turnover-dual-warm"],
        default="settings",
    )
    ap.add_argument("--output-csv", type=Path)
    return ap


def main() -> int:
    args = parser().parse_args()
    config = Config()
    store = DataStore(args.risk_model_xlsx, args.benchmark_csv)
    date = store.dates[args.date_index]
    initial = store.top_n_initial(config.initial_top_n)
    day = store.prepare_day(date, initial, config, "error")

    # A HiGHS LP produces the same linearly feasible warm start for every OSQP
    # variant.  The LP risk is deliberately irrelevant here.
    lp = build_problem(day, "lp", config, "optimized", objective_target=0.2)
    lp_started = time.perf_counter()
    lp.problem.solve(**solver_kwargs("HIGHS", None))
    lp_s = time.perf_counter() - lp_started
    if lp.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or lp.x.value is None:
        raise RuntimeError(f"HiGHS warm-start LP failed: {lp.problem.status}")

    data = _build_qp_data(day, config, FactorQCQPSettings())
    q = data.q_base.copy()
    q[: data.n_assets] -= args.theta * data.alpha_solver
    warm = _warm_start_vector(data, day, np.asarray(lp.x.value).reshape(-1))
    warm_violation = linear_detail(data, warm)[0]

    if args.experiment == "settings":
        variants = [
            ("default", {}, 0.0, 1.0, None),
            ("turnover_scale_20", {}, 0.0, 20.0, None),
            ("turnover_scale_100", {}, 0.0, 100.0, None),
            ("turnover_scale_1000", {}, 0.0, 1000.0, None),
            ("scaled_termination", {"scaled_termination": True}, 0.0, 1.0, None),
            ("rho_10_fixed", {"rho": 10.0, "adaptive_rho": False}, 0.0, 1.0, None),
            ("rho_100_fixed", {"rho": 100.0, "adaptive_rho": False}, 0.0, 1.0, None),
            ("sigma_1e-3", {"sigma": 1e-3}, 0.0, 1.0, None),
            ("rho10_sigma1e-3", {"rho": 10.0, "sigma": 1e-3}, 0.0, 1.0, None),
            ("aux_reg_1e-4", {}, 1e-4, 1.0, None),
            ("aux_reg_1e-2", {}, 1e-2, 1.0, None),
        ]
    elif args.experiment == "turnover-dual":
        multipliers = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
        variants = [
            (f"turnover_dual_{value:g}", {}, 0.0, 1.0, value)
            for value in multipliers
        ]
    else:
        multipliers = [0.0, 100.0, 300.0, 200.0, 150.0, 175.0, 187.5, 193.75]
        variants = [
            (f"turnover_dual_warm_{value:g}", {}, 0.0, 1.0, value)
            for value in multipliers
        ]
    rows: list[dict[str, object]] = []
    shared_solver: osqp.OSQP | None = None
    for name, extra, aux_regularization, turnover_row_scale, trade_multiplier in variants:
        P = data.P
        if aux_regularization:
            diagonal = np.zeros(data.n_variables)
            diagonal[data.n_assets + data.n_factors :] = aux_regularization
            P = sp.triu(P + sp.diags(diagonal, format="csc"), format="csc")
        A = data.A
        lower = data.lower.copy()
        upper = data.upper.copy()
        q_variant = q.copy()
        if trade_multiplier is not None:
            upper[data.turnover_row_index] = np.inf
            q_variant[data.new_idx] += 2.0 * trade_multiplier
            q_variant[
                data.trade_offset : data.trade_offset + len(data.support_idx)
            ] += 2.0 * trade_multiplier
        if turnover_row_scale != 1.0:
            row_scale = np.ones(A.shape[0])
            row_scale[data.turnover_row_index] = turnover_row_scale
            A = sp.diags(row_scale, format="csc") @ A
            lower = lower * row_scale
            upper = upper * row_scale
        reuse_workspace = args.experiment == "turnover-dual-warm"
        solver = shared_solver if reuse_workspace else None
        if solver is None:
            solver = osqp.OSQP()
        settings = {
            "verbose": False,
            "warm_starting": True,
            "polishing": False,
            "eps_abs": args.eps,
            "eps_rel": args.eps,
            "max_iter": args.max_iter,
            "adaptive_rho": True,
            "check_termination": 25,
            **extra,
        }
        started = time.perf_counter()
        if shared_solver is None:
            solver.setup(P=P, q=q_variant, A=A, l=lower, u=upper, **settings)
            solver.warm_start(x=warm)
            if reuse_workspace:
                shared_solver = solver
        else:
            solver.update(q=q_variant)
        setup_s = time.perf_counter() - started
        solve_started = time.perf_counter()
        result = solver.solve(raise_error=False)
        solve_s = time.perf_counter() - solve_started
        vector = None if result.x is None else np.asarray(result.x).reshape(-1)
        violation, row_index = (
            (np.nan, None) if vector is None else linear_detail(data, vector)
        )
        if vector is None:
            turnover = np.nan
            other_violation = np.nan
        else:
            turnover = float(
                np.abs(vector[: data.n_assets] - np.asarray(day.initial)).sum()
            )
            other_violation = linear_detail(
                data, vector, ignore_row=data.turnover_row_index
            )[0]
        rows.append(
            {
                "variant": name,
                "date": date.strftime("%Y-%m-%d"),
                "status": str(result.info.status),
                "iterations": int(result.info.iter or 0),
                "setup_wall_s": setup_s,
                "solve_wall_s": solve_s,
                "solver_run_s": float(result.info.run_time or 0.0),
                "primal_residual": float(result.info.prim_res),
                "dual_residual": float(result.info.dual_res),
                "duality_gap": float(result.info.duality_gap),
                "max_linear_violation": violation,
                "max_violation_row": row_index,
                "rho_estimate": float(result.info.rho_estimate),
                "aux_regularization": aux_regularization,
                "turnover_row_scale": turnover_row_scale,
                "trade_multiplier": trade_multiplier,
                "turnover_l1": turnover,
                "max_other_linear_violation": other_violation,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "data_load_s": store.load_elapsed_s,
                "lp_warm_start_s": lp_s,
                "lp_status": lp.problem.status,
                "warm_start_linear_violation": warm_violation,
                "n_assets": data.n_assets,
                "n_factors": data.n_factors,
                "n_variables": data.n_variables,
                "n_constraints": data.A.shape[0],
            },
            ensure_ascii=False,
        )
    )
    return 0


def linear_detail(
    data: object,
    vector: np.ndarray,
    *,
    ignore_row: int | None = None,
) -> tuple[float, int]:
    value = np.asarray(data.A @ vector).reshape(-1)
    lower = np.where(np.isfinite(data.lower), data.lower - value, -np.inf)
    upper = np.where(np.isfinite(data.upper), value - data.upper, -np.inf)
    violation = np.maximum(lower, upper)
    if ignore_row is not None:
        violation[ignore_row] = -np.inf
    index = int(np.argmax(violation))
    return max(0.0, float(violation[index])), index


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze and repeatedly solve one direct-PIQP portfolio QP.

This utility is intentionally independent of a particular benchmark scenario
after ``snapshot`` has run.  The snapshot contains only the numerical QP fed
to PIQP, plus stable content hashes.  ``repeat`` can then distinguish:

* repeated fresh workspaces inside one Python process;
* repeated fresh Python processes;
* PIQP's compact double-sided inequalities from an equivalent one-sided form;
* the default preconditioner from ``preconditioner_iter = 0``; and
* setup with the base cost followed by ``update(c=...)`` from setup with the
  final cost directly.

Snapshots contain derived proprietary inputs and must not be distributed with
the source package.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_SCHEMA = 1


def _canonical_csc(matrix: sp.spmatrix) -> sp.csc_matrix:
    result = matrix.astype(np.float64).tocsc(copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"name": name, "shape": array.shape, "dtype": array.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(len(header).to_bytes(8, "little"))
    hasher.update(header)
    hasher.update(array.tobytes(order="C"))


def _hash_sparse(hasher: Any, name: str, matrix: sp.csc_matrix) -> None:
    shape = np.asarray(matrix.shape, dtype=np.int64)
    _hash_array(hasher, f"{name}.shape", shape)
    _hash_array(hasher, f"{name}.indptr", matrix.indptr)
    _hash_array(hasher, f"{name}.indices", matrix.indices)
    _hash_array(hasher, f"{name}.data", matrix.data)


def _content_hash(
    matrices: Iterable[tuple[str, sp.csc_matrix]],
    arrays: Iterable[tuple[str, np.ndarray]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"direct-piqp-snapshot-v{SNAPSHOT_SCHEMA}".encode("ascii"))
    for name, matrix in matrices:
        _hash_sparse(hasher, name, matrix)
    for name, array in arrays:
        _hash_array(hasher, name, array)
    return hasher.hexdigest()


def _sparse_payload(prefix: str, matrix: sp.csc_matrix) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int64),
        f"{prefix}_indptr": matrix.indptr,
        f"{prefix}_indices": matrix.indices,
        f"{prefix}_data": matrix.data,
    }


def _sparse_from_payload(payload: Any, prefix: str) -> sp.csc_matrix:
    shape = tuple(int(value) for value in payload[f"{prefix}_shape"])
    return _canonical_csc(
        sp.csc_matrix(
            (
                payload[f"{prefix}_data"],
                payload[f"{prefix}_indices"],
                payload[f"{prefix}_indptr"],
            ),
            shape=shape,
        )
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _info_value(info: Any, name: str) -> Any:
    return _json_safe(getattr(info, name, None))


def snapshot(args: argparse.Namespace) -> None:
    # These imports are deliberately delayed: solving a frozen snapshot does
    # not need pandas, CVXPY, the original workbook, or benchmark.py.
    sys.path.insert(0, str(SCRIPT_DIR))
    from benchmark import Config, DataStore, portfolio_fingerprint
    from factor_qcqp import FactorQCQPSettings, _build_qp_data

    import pandas as pd

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    date = pd.Timestamp(args.date).normalize()
    config = Config(
        turnover_limit=args.turnover_limit,
        active_ub=args.active_ub,
        total_active_ub=args.total_active_ub,
        benchmark_weight_lb=args.benchmark_weight_lb,
        style_default_lb=-args.style_bound,
        style_default_ub=args.style_bound,
        size_lb=-args.style_bound,
        size_ub=args.style_bound,
        industry_lb=-args.industry_bound,
        industry_ub=args.industry_bound,
        risk_budget_pct=args.risk_budget_pct,
    )
    settings = FactorQCQPSettings(
        backend="PIQP",
        alpha_target=args.alpha_target,
        intermediate_eps=args.eps,
        piqp_max_iter=args.max_iter,
        theta_initial=args.theta,
        turnover_row_scale=args.turnover_row_scale,
        piqp_inequality_form="compact",
    )
    store = DataStore(
        args.risk_model_xlsx.resolve(),
        args.benchmark_csv.resolve(),
        args.alpha_csv.resolve(),
        args.data_cache_dir.resolve() if args.data_cache_dir else None,
    )
    if date not in store.dates:
        raise ValueError(f"{date:%Y-%m-%d} is not an aligned risk date")
    initial, generated_n = store.random_top_range_initial(
        date, args.initial_top_n, args.initial_top_n_max, args.seed
    )
    day = store.prepare_day(date, initial, config, args.missing_holding_policy)
    data = _build_qp_data(day, config, settings)

    P = _canonical_csc(data.P + sp.triu(data.P, k=1).T)
    equality = (
        np.isfinite(data.lower)
        & np.isfinite(data.upper)
        & np.isclose(data.lower, data.upper, rtol=0.0, atol=1e-14)
    )
    A = _canonical_csc(data.A[equality])
    b = np.asarray(data.lower[equality], dtype=np.float64)
    G = _canonical_csc(data.A[~equality])
    h_l = np.asarray(data.lower[~equality], dtype=np.float64)
    h_u = np.asarray(data.upper[~equality], dtype=np.float64)
    q_base = np.asarray(data.q_base, dtype=np.float64)
    q_theta = q_base.copy()
    q_theta[: data.n_assets] -= args.theta * np.asarray(
        data.alpha_solver, dtype=np.float64
    )

    matrices = [("P", P), ("A", A), ("G", G)]
    arrays = [
        ("q_base", q_base),
        ("q_theta", q_theta),
        ("b", b),
        ("h_l", h_l),
        ("h_u", h_u),
    ]
    content_hash = _content_hash(matrices, arrays)
    payload: dict[str, np.ndarray] = {}
    for name, matrix in matrices:
        payload.update(_sparse_payload(name, matrix))
    payload.update({name: value for name, value in arrays})
    np.savez_compressed(output / "qp.npz", **payload)

    manifest = {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "content_sha256": content_hash,
        "date": date.strftime("%Y-%m-%d"),
        "theta": args.theta,
        "eps": args.eps,
        "max_iter": args.max_iter,
        "piqp_version_at_snapshot": _package_version("piqp"),
        "numpy_version_at_snapshot": _package_version("numpy"),
        "scipy_version_at_snapshot": _package_version("scipy"),
        "n_assets": data.n_assets,
        "n_factors": data.n_factors,
        "n_variables": data.n_variables,
        "n_equalities": A.shape[0],
        "n_inequalities": G.shape[0],
        "n_finite_lower": int(np.isfinite(h_l).sum()),
        "n_finite_upper": int(np.isfinite(h_u).sum()),
        "P_nnz": P.nnz,
        "A_nnz": A.nnz,
        "G_nnz": G.nnz,
        "generated_initial_n": generated_n,
        "initial_nonzero_count": int(np.count_nonzero(day.initial > config.weight_zero_tol)),
        "initial_fingerprint": portfolio_fingerprint(
            day.sids, day.initial, config.weight_zero_tol
        ),
        "benchmark_source_date": (
            day.benchmark_source_date.strftime("%Y-%m-%d")
            if day.benchmark_source_date is not None
            else None
        ),
        "alpha_source_date": day.alpha_source_date.strftime("%Y-%m-%d"),
        "config": asdict(config),
        "factor_settings": asdict(settings),
        "omitted_total_active": data.omitted_total_active,
        "omitted_benchmark_weight": data.omitted_benchmark_weight,
        "note": "Derived proprietary numerical QP; do not include in source package.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def load_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unsupported snapshot schema: {manifest.get('snapshot_schema')}")
    with np.load(path / "qp.npz", allow_pickle=False) as payload:
        data = {
            "P": _sparse_from_payload(payload, "P"),
            "A": _sparse_from_payload(payload, "A"),
            "G": _sparse_from_payload(payload, "G"),
            "q_base": np.asarray(payload["q_base"], dtype=np.float64),
            "q_theta": np.asarray(payload["q_theta"], dtype=np.float64),
            "b": np.asarray(payload["b"], dtype=np.float64),
            "h_l": np.asarray(payload["h_l"], dtype=np.float64),
            "h_u": np.asarray(payload["h_u"], dtype=np.float64),
        }
    observed_hash = _content_hash(
        [(name, data[name]) for name in ("P", "A", "G")],
        [(name, data[name]) for name in ("q_base", "q_theta", "b", "h_l", "h_u")],
    )
    if observed_hash != manifest["content_sha256"]:
        raise ValueError(
            f"snapshot hash mismatch: {observed_hash} != {manifest['content_sha256']}"
        )
    return manifest, data


def _one_sided(
    G: sp.csc_matrix, h_l: np.ndarray, h_u: np.ndarray
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    finite_upper = np.isfinite(h_u)
    finite_lower = np.isfinite(h_l)
    expanded = _canonical_csc(
        sp.vstack([G[finite_upper], -G[finite_lower]], format="csc")
    )
    upper = np.concatenate([h_u[finite_upper], -h_l[finite_lower]])
    lower = np.full(len(upper), -np.inf, dtype=np.float64)
    return expanded, lower, upper


def solve_once(
    manifest: dict[str, Any],
    data: dict[str, Any],
    *,
    variant: str,
    setup_cost: str,
    eps: float,
    max_iter: int,
    repeat_index: int,
    cost_multipliers: list[float] | None = None,
) -> dict[str, Any]:
    import piqp

    G = data["G"]
    h_l = data["h_l"]
    h_u = data["h_u"]
    if variant == "one_sided":
        G, h_l, h_u = _one_sided(G, h_l, h_u)

    solver = piqp.SparseSolver()
    solver.settings.verbose = False
    solver.settings.eps_abs = eps
    solver.settings.eps_rel = eps
    solver.settings.eps_duality_gap_abs = eps
    solver.settings.eps_duality_gap_rel = eps
    solver.settings.max_iter = max_iter
    solver.settings.compute_timings = True
    if variant == "no_preconditioner":
        solver.settings.preconditioner_iter = 0

    multipliers = cost_multipliers or [1.0]
    if not multipliers or any(not np.isfinite(value) for value in multipliers):
        raise ValueError("cost multipliers must be a non-empty finite sequence")

    cost_delta = data["q_theta"] - data["q_base"]

    def cost(multiplier: float) -> np.ndarray:
        return data["q_base"] + multiplier * cost_delta

    initial_q = data["q_base"] if setup_cost == "update" else cost(multipliers[0])
    setup_started = time.perf_counter()
    solver.setup(
        data["P"], initial_q, data["A"], data["b"], G, h_l, h_u
    )
    setup_wall_s = time.perf_counter() - setup_started
    update_wall_s = 0.0
    solve_wall_s = 0.0
    exception = None
    status = None
    result = None
    info = None
    x_value = None
    subproblems: list[dict[str, Any]] = []
    for position, multiplier in enumerate(multipliers):
        needs_update = setup_cost == "update" or position > 0
        one_update_s = 0.0
        if needs_update:
            update_started = time.perf_counter()
            solver.update(c=cost(multiplier))
            one_update_s = time.perf_counter() - update_started
            update_wall_s += one_update_s

        solve_started = time.perf_counter()
        try:
            status = solver.solve()
        except Exception as exc:  # retain unexpected binding/core failures as evidence
            status = None
            exception = f"{type(exc).__name__}: {exc}"
        one_solve_s = time.perf_counter() - solve_started
        solve_wall_s += one_solve_s
        result = getattr(solver, "result", None)
        info = getattr(result, "info", None)
        x_value = getattr(result, "x", None)
        one_success = (
            status is not None
            and "PIQP_SOLVED" in str(status)
            and x_value is not None
        )
        subproblems.append(
            {
                "position": position,
                "cost_multiplier": multiplier,
                "success": one_success,
                "status": str(status) if status is not None else None,
                "iterations": _info_value(info, "iter"),
                "update_wall_s": one_update_s,
                "solve_wall_s": one_solve_s,
                "rho": _info_value(info, "rho"),
                "delta": _info_value(info, "delta"),
                "mu": _info_value(info, "mu"),
            }
        )
        if not one_success:
            break
    success = len(subproblems) == len(multipliers) and all(
        item["success"] for item in subproblems
    )

    eq_violation = None
    ineq_violation = None
    objective = None
    x_sha256 = None
    if x_value is not None:
        x = np.asarray(x_value, dtype=np.float64).reshape(-1)
        eq_violation = float(np.max(np.abs(data["A"] @ x - data["b"])))
        gx = np.asarray(data["G"] @ x).reshape(-1)
        lower_violation = np.where(np.isfinite(data["h_l"]), data["h_l"] - gx, -np.inf)
        upper_violation = np.where(np.isfinite(data["h_u"]), gx - data["h_u"], -np.inf)
        ineq_violation = max(
            0.0, float(np.max(lower_violation)), float(np.max(upper_violation))
        )
        final_q = cost(multipliers[len(subproblems) - 1])
        objective = float(0.5 * x @ (data["P"] @ x) + final_q @ x)
        x_sha256 = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

    record = {
        "repeat": repeat_index,
        "pid": os.getpid(),
        "snapshot_sha256": manifest["content_sha256"],
        "variant": variant,
        "setup_cost": setup_cost,
        "piqp_version": _package_version("piqp"),
        "piqp_module": getattr(piqp, "__file__", None),
        "success": success,
        "status": str(status) if status is not None else None,
        "exception": exception,
        "iterations": _info_value(info, "iter"),
        "setup_wall_s": setup_wall_s,
        "update_wall_s": update_wall_s,
        "solve_wall_s": solve_wall_s,
        "solver_setup_s": _info_value(info, "setup_time"),
        "solver_update_s": _info_value(info, "update_time"),
        "solver_solve_s": _info_value(info, "solve_time"),
        "primal_residual": _info_value(info, "primal_residual"),
        "dual_residual": _info_value(info, "dual_residual"),
        "duality_gap": _info_value(info, "duality_gap"),
        "rho": _info_value(info, "rho"),
        "delta": _info_value(info, "delta"),
        "mu": _info_value(info, "mu"),
        "eq_violation": eq_violation,
        "ineq_violation": ineq_violation,
        "objective": objective,
        "x_sha256": x_sha256,
        "n_inequalities_effective": G.shape[0],
        "preconditioner_iter": solver.settings.preconditioner_iter,
        "eps": eps,
        "max_iter": max_iter,
        "cost_multipliers": ",".join(f"{value:g}" for value in multipliers),
        "subproblems_attempted": len(subproblems),
        "subproblems_solved": sum(bool(item["success"]) for item in subproblems),
        "subproblem_trace_json": json.dumps(subproblems, separators=(",", ":")),
    }
    del result, solver
    gc.collect()
    return record


def _write_records(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for record in records:
        for name in record:
            if name not in columns:
                columns.append(name)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    failures = [record for record in records if not record.get("success")]
    iterations = [float(record["iterations"]) for record in successful]
    solve_times = [float(record["solve_wall_s"]) for record in records]
    statuses: dict[str, int] = {}
    for record in records:
        key = str(record.get("status") or record.get("exception") or "unknown")
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "runs": len(records),
        "successes": len(successful),
        "failures": len(failures),
        "failure_rate": len(failures) / len(records) if records else None,
        "statuses": statuses,
        "iterations_min": min(iterations) if iterations else None,
        "iterations_median": float(np.median(iterations)) if iterations else None,
        "iterations_max": max(iterations) if iterations else None,
        "solve_wall_s_mean": float(np.mean(solve_times)) if solve_times else None,
        "solve_wall_s_max": max(solve_times) if solve_times else None,
        "unique_solution_hashes": len(
            {record["x_sha256"] for record in successful if record.get("x_sha256")}
        ),
    }


def repeat(args: argparse.Namespace) -> None:
    snapshot_path = args.snapshot.resolve()
    manifest, data = load_snapshot(snapshot_path)
    records: list[dict[str, Any]] = []
    if args.process_mode == "same":
        for index in range(args.repeats):
            record: dict[str, Any] | None = None
            for attempt in range(args.retries + 1):
                setup_cost = (
                    args.setup_cost
                    if attempt == 0 or args.retry_setup_cost == "same"
                    else args.retry_setup_cost
                )
                record = solve_once(
                    manifest,
                    data,
                    variant=args.variant,
                    setup_cost=setup_cost,
                    eps=args.eps,
                    max_iter=args.max_iter,
                    repeat_index=index,
                    cost_multipliers=args.cost_multipliers,
                )
                record["retry_attempt"] = attempt
                records.append(record)
                if record["success"]:
                    break
            assert record is not None
            if args.progress_every and (
                (index + 1) % args.progress_every == 0 or not record["success"]
            ):
                print(
                    f"[{index + 1}/{args.repeats}] attempt={record['retry_attempt']} "
                    f"{record['status']} "
                    f"iters={record['iterations']} solve={record['solve_wall_s']:.4f}s",
                    flush=True,
                )
    else:
        for index in range(args.repeats):
            record = None
            for attempt in range(args.retries + 1):
                setup_cost = (
                    args.setup_cost
                    if attempt == 0 or args.retry_setup_cost == "same"
                    else args.retry_setup_cost
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "solve-one",
                    "--snapshot",
                    str(snapshot_path),
                    "--variant",
                    args.variant,
                    "--setup-cost",
                    setup_cost,
                    "--eps",
                    str(args.eps),
                    "--max-iter",
                    str(args.max_iter),
                    "--repeat-index",
                    str(index),
                    "--cost-multipliers",
                    ",".join(f"{value:g}" for value in args.cost_multipliers),
                ]
                process_started = time.perf_counter()
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                process_wall_s = time.perf_counter() - process_started
                result_lines = [
                    line.removeprefix("RESULT_JSON=")
                    for line in completed.stdout.splitlines()
                    if line.startswith("RESULT_JSON=")
                ]
                if result_lines:
                    record = json.loads(result_lines[-1])
                else:
                    record = {
                        "repeat": index,
                        "snapshot_sha256": manifest["content_sha256"],
                        "variant": args.variant,
                        "setup_cost": setup_cost,
                        "success": False,
                        "status": "CHILD_PROCESS_FAILED",
                        "exception": completed.stderr[-4000:],
                    }
                record["retry_attempt"] = attempt
                record["child_returncode"] = completed.returncode
                record["process_wall_s"] = process_wall_s
                records.append(record)
                if record["success"]:
                    break
            assert record is not None
            if args.progress_every and (
                (index + 1) % args.progress_every == 0 or not record["success"]
            ):
                print(
                    f"[{index + 1}/{args.repeats}] attempt={record['retry_attempt']} "
                    f"{record.get('status')} "
                    f"iters={record.get('iterations')} process={process_wall_s:.4f}s",
                    flush=True,
                )

    _write_records(records, args.output.resolve())
    summary = {
        "snapshot": str(snapshot_path),
        "snapshot_sha256": manifest["content_sha256"],
        "process_mode": args.process_mode,
        "variant": args.variant,
        "setup_cost": args.setup_cost,
        "retries": args.retries,
        "retry_setup_cost": args.retry_setup_cost,
        "eps": args.eps,
        "max_iter": args.max_iter,
        **_summary(records),
    }
    final_attempts: dict[int, dict[str, Any]] = {}
    for record in records:
        final_attempts[int(record["repeat"])] = record
    summary.update(
        {
            "logical_requests": len(final_attempts),
            "logical_successes": sum(
                bool(record.get("success")) for record in final_attempts.values()
            ),
            "logical_failures": sum(
                not bool(record.get("success")) for record in final_attempts.values()
            ),
            "retry_attempts": sum(
                int(record.get("retry_attempt", 0) or 0) > 0 for record in records
            ),
        }
    )
    summary_path = args.output.resolve().with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def solve_one_command(args: argparse.Namespace) -> None:
    manifest, data = load_snapshot(args.snapshot.resolve())
    record = solve_once(
        manifest,
        data,
        variant=args.variant,
        setup_cost=args.setup_cost,
        eps=args.eps,
        max_iter=args.max_iter,
        repeat_index=args.repeat_index,
        cost_multipliers=args.cost_multipliers,
    )
    print("RESULT_JSON=" + json.dumps(record, separators=(",", ":")))


def _add_solve_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["double_sided", "one_sided", "no_preconditioner"],
        default="double_sided",
    )
    parser.add_argument(
        "--setup-cost",
        choices=["update", "direct"],
        default="update",
        help="faithful base-cost setup + update, or setup final theta cost directly",
    )
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--cost-multipliers",
        type=lambda value: [float(item) for item in value.split(",")],
        default=[1.0],
        help=(
            "comma-separated multiples of the frozen theta cost delta; values "
            "after the first are solved through cost updates in one workspace"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("snapshot", help="freeze one benchmark date to raw PIQP arrays")
    freeze.add_argument("--risk-model-xlsx", type=Path, required=True)
    freeze.add_argument("--benchmark-csv", type=Path, required=True)
    freeze.add_argument("--alpha-csv", type=Path, required=True)
    freeze.add_argument("--data-cache-dir", type=Path)
    freeze.add_argument("--date", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20260826)
    freeze.add_argument("--initial-top-n", type=int, default=500)
    freeze.add_argument("--initial-top-n-max", type=int, default=600)
    freeze.add_argument("--missing-holding-policy", choices=["error", "renormalize"], default="renormalize")
    freeze.add_argument("--active-ub", type=float, default=0.004)
    freeze.add_argument("--turnover-limit", type=float, default=0.05)
    freeze.add_argument("--total-active-ub", type=float, default=1.8)
    freeze.add_argument("--benchmark-weight-lb", type=float, default=0.81)
    freeze.add_argument("--style-bound", type=float, default=0.6)
    freeze.add_argument("--industry-bound", type=float, default=0.05)
    freeze.add_argument("--risk-budget-pct", type=float, default=6.0)
    freeze.add_argument("--alpha-target", type=float, default=0.2)
    freeze.add_argument("--theta", type=float, default=16384.0)
    freeze.add_argument("--turnover-row-scale", type=float, default=1.0)
    freeze.add_argument("--eps", type=float, default=1e-5)
    freeze.add_argument("--max-iter", type=int, default=1000)
    freeze.set_defaults(func=snapshot)

    repeated = subparsers.add_parser("repeat", help="repeat a frozen QP in same or fresh processes")
    _add_solve_options(repeated)
    repeated.add_argument("--process-mode", choices=["same", "fresh"], default="same")
    repeated.add_argument("--repeats", type=int, default=100)
    repeated.add_argument("--retries", type=int, default=0)
    repeated.add_argument(
        "--retry-setup-cost",
        choices=["same", "update", "direct"],
        default="same",
        help="cost lifecycle used by a rebuilt retry workspace",
    )
    repeated.add_argument("--progress-every", type=int, default=25)
    repeated.add_argument("--output", type=Path, required=True)
    repeated.set_defaults(func=repeat)

    child = subparsers.add_parser("solve-one", help=argparse.SUPPRESS)
    _add_solve_options(child)
    child.add_argument("--repeat-index", type=int, default=0)
    child.set_defaults(func=solve_one_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

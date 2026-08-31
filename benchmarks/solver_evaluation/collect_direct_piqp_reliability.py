#!/usr/bin/env python3
"""Collect direct-PIQP reliability runs and expand per-subproblem traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECT_CASES = {"FACTOR_PENALTY_QP_PIQP", "FACTOR_QP_PIQP"}
SUCCESS_STATUSES = {"optimal", "optimal_inaccurate"}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_root", type=Path)
    ap.add_argument("--constraint-tol", type=float, default=1e-5)
    ap.add_argument("--objective-gap-abs", type=float, default=1e-4)
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "return nonzero when a direct fallback, failed subproblem, "
            "certificate violation, or benchmark metadata error is found"
        ),
    )
    return ap


def scenario_fields(result_dir: Path, root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    relative = result_dir.relative_to(root)
    config = metadata.get("config", {})
    factor_settings = metadata.get("factor_qcqp_settings", {})
    parts = relative.parts
    return {
        "scenario": str(relative),
        "settings_profile": parts[0] if parts else "unknown",
        "active_ub": config.get("active_ub"),
        "risk_budget_pct": config.get("risk_budget_pct"),
        "factor_carry_theta": metadata.get("factor_carry_theta", False),
        "factor_final_eps": factor_settings.get("final_eps"),
        "factor_piqp_max_iter": factor_settings.get("piqp_max_iter"),
        "package_version": metadata.get("package_version"),
    }


def with_fields(frame: pd.DataFrame, fields: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    for name, value in reversed(list(fields.items())):
        frame.insert(0, name, value)
    if "model" in frame:
        frame.loc[frame["model"].astype(str).str.lower() != "socp", "risk_budget_pct"] = np.nan
    return frame


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def boolean_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[name]
    return values.eq(True) | values.astype(str).str.lower().eq("true")


def reliability_summary(runs: pd.DataFrame) -> pd.DataFrame:
    direct = runs[
        runs["case"].isin(DIRECT_CASES) & runs["status"].isin(SUCCESS_STATUSES)
    ].copy()
    if direct.empty:
        return pd.DataFrame()
    group_columns = [
        "settings_profile",
        "active_ub",
        "risk_budget_pct",
        "factor_carry_theta",
        "factor_final_eps",
        "factor_piqp_max_iter",
        "initial_mode",
        "model",
        "case",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in direct.groupby(group_columns, dropna=False, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        fallback_count = int(boolean_column(group, "factor_fallback_used").sum())
        solves = len(group)
        rss_start = pd.to_numeric(group["process_rss_before_mb"], errors="coerce").dropna()
        rss_end = pd.to_numeric(group["process_rss_after_mb"], errors="coerce").dropna()
        record.update(
            {
                "dates": solves,
                "optimal": int((group["status"] == "optimal").sum()),
                "optimal_inaccurate": int((group["status"] == "optimal_inaccurate").sum()),
                "direct_successes": solves - fallback_count,
                "fallbacks": fallback_count,
                "direct_success_rate": (solves - fallback_count) / solves if solves else np.nan,
                "optimization_mean_s": float(group["optimization_total_s"].mean()),
                "optimization_p50_s": percentile(group["optimization_total_s"], 0.50),
                "optimization_p95_s": percentile(group["optimization_total_s"], 0.95),
                "optimization_p99_s": percentile(group["optimization_total_s"], 0.99),
                "optimization_max_s": float(group["optimization_total_s"].max()),
                "qp_solves_mean": float(group["factor_qp_solves"].mean()),
                "qp_solves_max": float(group["factor_qp_solves"].max()),
                "qp_iters_mean": float(group["factor_qp_iters"].mean()),
                "qp_iters_max": float(group["factor_qp_iters"].max()),
                "objective_gap_max": float(
                    pd.to_numeric(group["factor_dual_gap_abs"], errors="coerce").max()
                ),
                "max_violation": float(group["max_violation"].max()),
                "max_turnover_violation": float(group["violation_turnover"].max()),
                "max_risk_violation": float(group["violation_risk"].max()),
                "rss_start_mb": float(rss_start.iloc[0]) if len(rss_start) else np.nan,
                "rss_end_mb": float(rss_end.iloc[-1]) if len(rss_end) else np.nan,
                "rss_net_change_mb": (
                    float(rss_end.iloc[-1] - rss_start.iloc[0])
                    if len(rss_start) and len(rss_end)
                    else np.nan
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def expand_traces(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity = [
        "scenario",
        "settings_profile",
        "active_ub",
        "risk_budget_pct",
        "factor_carry_theta",
        "factor_final_eps",
        "factor_piqp_max_iter",
        "initial_mode",
        "model",
        "case",
        "repeat",
        "date",
        "solver_path",
    ]
    for _, run in runs[runs["case"].isin(DIRECT_CASES)].iterrows():
        raw = run.get("factor_qp_trace_json", "[]")
        try:
            trace = json.loads(raw) if isinstance(raw, str) and raw else []
        except json.JSONDecodeError as exc:
            trace = [{"success": False, "exception": f"invalid trace JSON: {exc}"}]
        for index, item in enumerate(trace):
            rows.append(
                {
                    **{name: run.get(name) for name in identity},
                    "subproblem_index": index,
                    **item,
                }
            )
    return pd.DataFrame(rows)


def subproblem_summary(traces: pd.DataFrame) -> pd.DataFrame:
    if traces.empty:
        return pd.DataFrame()
    group_columns = [
        "settings_profile",
        "active_ub",
        "risk_budget_pct",
        "factor_carry_theta",
        "initial_mode",
        "model",
        "case",
        "phase",
        "final",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in traces.groupby(group_columns, dropna=False, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        success = boolean_column(group, "success")
        record.update(
            {
                "subproblems": len(group),
                "successes": int(success.sum()),
                "failures": int((~success).sum()),
                "wall_mean_s": float(pd.to_numeric(group["wall_time_s"], errors="coerce").mean()),
                "wall_p95_s": percentile(group["wall_time_s"], 0.95),
                "wall_max_s": float(pd.to_numeric(group["wall_time_s"], errors="coerce").max()),
                "iterations_mean": float(numeric_column(group, "iterations").mean()),
                "iterations_max": float(numeric_column(group, "iterations").max()),
                "primal_residual_max": float(
                    numeric_column(group, "primal_residual").max()
                ),
                "dual_residual_max": float(
                    numeric_column(group, "dual_residual").max()
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    args = parser().parse_args()
    root = args.output_root.resolve()
    if not root.is_dir():
        raise ValueError(f"output root does not exist: {root}")

    runs_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    metadata_errors: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        result_dir = metadata_path.parent
        runs_path = result_dir / "runs.csv"
        if not runs_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fields = scenario_fields(result_dir, root, metadata)
        run = with_fields(pd.read_csv(runs_path), fields)
        runs_frames.append(run)
        comparison_path = result_dir / "solution_comparisons.csv"
        if comparison_path.is_file():
            comparison_frames.append(with_fields(pd.read_csv(comparison_path), fields))
        errors = metadata.get("errors", [])
        metadata_errors.extend({**fields, **error} for error in errors)
        manifests.append(
            {
                **fields,
                "rows": len(run),
                "successful_rows": int(run["status"].isin(SUCCESS_STATUSES).sum()),
                "fallbacks": int(
                    boolean_column(run, "factor_fallback_used").sum()
                ),
                "metadata_errors": len(errors),
                "total_elapsed_s": metadata.get("total_elapsed_s"),
                "data_load_elapsed_s": metadata.get("data_load_elapsed_s"),
            }
        )

    if not runs_frames:
        raise ValueError(f"no reliability results found under {root}")

    runs = pd.concat(runs_frames, ignore_index=True, sort=False)
    traces = expand_traces(runs)
    summary = reliability_summary(runs)
    sub_summary = subproblem_summary(traces)
    failures = runs[
        runs["case"].isin(DIRECT_CASES)
        & boolean_column(runs, "factor_fallback_used")
    ].copy()
    failed_traces = (
        traces[~boolean_column(traces, "success")].copy()
        if not traces.empty
        else pd.DataFrame()
    )

    runs.to_csv(root / "all_runs.csv.gz", index=False, compression="gzip")
    summary.to_csv(root / "reliability_summary.csv", index=False)
    traces.to_csv(root / "piqp_subproblems.csv.gz", index=False, compression="gzip")
    sub_summary.to_csv(root / "piqp_subproblem_summary.csv", index=False)
    failures.to_csv(root / "direct_failures.csv", index=False)
    failed_traces.to_csv(root / "failed_subproblems.csv", index=False)
    pd.DataFrame(manifests).to_csv(root / "scenario_manifest.csv", index=False)
    pd.DataFrame(metadata_errors).to_csv(root / "metadata_errors.csv", index=False)
    if comparison_frames:
        pd.concat(comparison_frames, ignore_index=True, sort=False).to_csv(
            root / "solution_comparisons.csv.gz", index=False, compression="gzip"
        )

    direct_rows = runs[runs["case"].isin(DIRECT_CASES)].copy()
    direct = direct_rows[direct_rows["status"].isin(SUCCESS_STATUSES)].copy()
    skipped_count = int((direct_rows["status"] == "skipped_infeasible").sum())
    fallback_count = int(boolean_column(direct, "factor_fallback_used").sum())
    violation_count = int((pd.to_numeric(direct["max_violation"], errors="coerce") > args.constraint_tol).sum())
    socp = direct[direct["model"].astype(str).str.lower() == "socp"]
    gap_count = int(
        (
            pd.to_numeric(socp["factor_dual_gap_abs"], errors="coerce")
            > args.objective_gap_abs
        ).sum()
    )
    report = [
        "# Direct PIQP reliability audit",
        "",
        f"- Evaluated direct daily results: {len(direct)}",
        f"- Consistently skipped infeasible sequence rows: {skipped_count}",
        f"- Same-day fallbacks: {fallback_count}",
        f"- Returned solutions above constraint tolerance `{args.constraint_tol:g}`: {violation_count}",
        f"- Factor-QP results above objective-gap threshold `{args.objective_gap_abs:g}`: {gap_count}",
        f"- Failed PIQP subproblems: {len(failed_traces)}",
        f"- Benchmark-level metadata errors: {len(metadata_errors)}",
        "",
        "Use `reliability_summary.csv` for daily P50/P95/P99 and fallback rates,",
        "and `piqp_subproblems.csv.gz` to identify the exact date/theta/status of",
        "every frontier QP. A fallback is retained as a successful portfolio result",
        "but is not counted as a direct PIQP success.",
    ]
    (root / "RELIABILITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    strict_failure = bool(
        fallback_count
        or violation_count
        or gap_count
        or len(failed_traces)
        or len(metadata_errors)
    )
    return 1 if args.strict and strict_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

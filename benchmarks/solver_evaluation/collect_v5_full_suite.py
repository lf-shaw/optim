#!/usr/bin/env python3
"""Collect nested v5 full-suite outputs without mixing scenario identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_root", type=Path)
    ap.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="return a nonzero status after collecting when metadata contains errors",
    )
    return ap


def scenario_fields(result_dir: Path, root: Path, metadata: dict) -> dict:
    config = metadata.get("config", {})
    return {
        "scenario": str(result_dir.relative_to(root)),
        "package_version": metadata.get("package_version"),
        "active_ub": config.get("active_ub"),
        "risk_budget_pct": config.get("risk_budget_pct"),
        "common_risk_aversion": config.get("common_risk_aversion"),
        "specific_risk_aversion": config.get("specific_risk_aversion"),
        "factor_carry_theta": metadata.get("factor_carry_theta"),
        "configured_chained_state_weight_tol": metadata.get(
            "chained_state_weight_tol"
        ),
        "missing_holding_policy": metadata.get("missing_holding_policy"),
    }


def add_scenario_fields(frame: pd.DataFrame, fields: dict) -> pd.DataFrame:
    frame = frame.copy()
    for name, value in reversed(list(fields.items())):
        if name in frame.columns:
            frame[name] = value
        else:
            frame.insert(0, name, value)
    if "model" in frame.columns:
        model = frame["model"].astype(str).str.lower()
        frame.loc[model != "socp", "risk_budget_pct"] = float("nan")
        frame.loc[model != "qp", "common_risk_aversion"] = float("nan")
        frame.loc[model != "qp", "specific_risk_aversion"] = float("nan")
    return frame


def main() -> int:
    args = parser().parse_args()
    root = args.output_root.resolve()
    if not root.is_dir():
        raise ValueError(f"output root does not exist: {root}")

    summaries: list[pd.DataFrame] = []
    runs: list[pd.DataFrame] = []
    errors: list[dict] = []
    manifests: list[dict] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        result_dir = metadata_path.parent
        summary_path = result_dir / "summary.csv"
        runs_path = result_dir / "runs.csv"
        if not summary_path.is_file() or not runs_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fields = scenario_fields(result_dir, root, metadata)

        summary = add_scenario_fields(pd.read_csv(summary_path), fields)
        summaries.append(summary)

        run = add_scenario_fields(pd.read_csv(runs_path), fields)
        runs.append(run)

        scenario_errors = metadata.get("errors", [])
        for error in scenario_errors:
            errors.append({**fields, **error})
        manifests.append(
            {
                **fields,
                "result_dir": str(result_dir),
                "rows": len(run),
                "successful_rows": int(
                    run["status"].isin(["optimal", "optimal_inaccurate"]).sum()
                ),
                "skipped_infeasible_rows": int(
                    (run["status"] == "skipped_infeasible").sum()
                ),
                "errors": len(scenario_errors),
                "total_elapsed_s": metadata.get("total_elapsed_s"),
                "data_load_elapsed_s": metadata.get("data_load_elapsed_s"),
            }
        )

    if not summaries:
        raise ValueError(f"no benchmark result directories found under {root}")

    all_summary = pd.concat(summaries, ignore_index=True, sort=False)
    all_runs = pd.concat(runs, ignore_index=True, sort=False)
    all_errors = pd.DataFrame(errors)
    if all_errors.empty:
        all_errors = pd.DataFrame(
            columns=[
                "scenario",
                "model",
                "case",
                "solver",
                "initial_mode",
                "repeat",
                "error",
                "traceback",
            ]
        )
    manifest = pd.DataFrame(manifests)

    all_summary.to_csv(root / "all_summary.csv", index=False)
    all_runs.to_csv(root / "all_runs.csv.gz", index=False, compression="gzip")
    all_errors.to_csv(root / "all_errors.csv", index=False)
    manifest.to_csv(root / "scenario_manifest.csv", index=False)

    print(f"Collected {len(manifest)} scenarios under {root}")
    print(
        manifest[
            [
                "scenario",
                "rows",
                "successful_rows",
                "skipped_infeasible_rows",
                "errors",
                "total_elapsed_s",
            ]
        ].to_string(index=False)
    )
    return 1 if args.fail_on_errors and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and compare paired v5 active-weight benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUCCESS = {"optimal", "optimal_inaccurate"}
KEYS = ["initial_mode", "model", "case", "repeat", "date"]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("active_004", type=Path)
    ap.add_argument("active_010", type=Path)
    ap.add_argument("output_dir", type=Path)
    return ap


def load_result(path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    runs = pd.read_csv(path / "runs.csv", dtype={"initial_fingerprint": str})
    summary = pd.read_csv(path / "summary.csv")
    with (path / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    return runs, metadata, summary


def config_without_active(metadata: dict) -> dict:
    config = dict(metadata["config"])
    config.pop("active_ub", None)
    return config


def main() -> int:
    args = parser().parse_args()
    runs_004, meta_004, summary_004 = load_result(args.active_004)
    runs_010, meta_010, summary_010 = load_result(args.active_010)

    checks = {
        "dates_identical": meta_004["dates"] == meta_010["dates"],
        "seed_identical": (
            meta_004["initial_random_seed"] == meta_010["initial_random_seed"]
        ),
        "top_n_range_identical": (
            meta_004["initial_top_n_range"] == meta_010["initial_top_n_range"]
        ),
        "non_active_config_identical": (
            config_without_active(meta_004) == config_without_active(meta_010)
        ),
        "active_004_value": meta_004["config"]["active_ub"] == 0.004,
        "active_010_value": meta_010["config"]["active_ub"] == 0.01,
        "no_recorded_errors": not meta_004.get("errors") and not meta_010.get("errors"),
    }

    generated_004 = runs_004[runs_004["generated_initial_n"].notna()][
        KEYS + ["generated_initial_n", "initial_fingerprint"]
    ]
    generated_010 = runs_010[runs_010["generated_initial_n"].notna()][
        KEYS + ["generated_initial_n", "initial_fingerprint"]
    ]
    generated = generated_004.merge(
        generated_010,
        on=KEYS,
        how="outer",
        suffixes=("_004", "_010"),
        indicator=True,
    )
    checks["generated_initial_rows_paired"] = bool((generated["_merge"] == "both").all())
    paired_generated = generated[generated["_merge"] == "both"]
    checks["generated_n_identical"] = bool(
        (
            paired_generated["generated_initial_n_004"]
            == paired_generated["generated_initial_n_010"]
        ).all()
    )
    checks["generated_fingerprint_identical"] = bool(
        (
            paired_generated["initial_fingerprint_004"]
            == paired_generated["initial_fingerprint_010"]
        ).all()
    )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"paired-run validation failed: {failed}")

    summary_columns = [
        "initial_mode",
        "model",
        "case",
        "solves",
        "optimization_mean_s",
        "end_to_end_mean_s",
        "screen_accepts",
        "fallbacks",
        "factor_dual_gap_max",
        "max_violation",
        "te_median_pct",
        "te_max_pct",
    ]
    comparison = summary_004[summary_columns].merge(
        summary_010[summary_columns],
        on=["initial_mode", "model", "case"],
        how="outer",
        suffixes=("_004", "_010"),
        validate="one_to_one",
    )
    comparison["optimization_mean_ratio_010_over_004"] = (
        comparison["optimization_mean_s_010"]
        / comparison["optimization_mean_s_004"]
    )
    comparison["te_median_delta_pct"] = (
        comparison["te_median_pct_010"] - comparison["te_median_pct_004"]
    )
    comparison["te_max_delta_pct"] = (
        comparison["te_max_pct_010"] - comparison["te_max_pct_004"]
    )

    daily_columns = KEYS + [
        "status",
        "tracking_error_pct",
        "factor_risk_pct",
        "specific_risk_pct",
        "max_abs_style_exposure",
        "max_abs_industry_exposure",
        "optimization_total_s",
        "screen_accepted",
        "factor_fallback_used",
    ]
    daily = runs_004[daily_columns].merge(
        runs_010[daily_columns],
        on=KEYS,
        how="outer",
        suffixes=("_004", "_010"),
        validate="one_to_one",
    )
    for column in [
        "tracking_error_pct",
        "factor_risk_pct",
        "specific_risk_pct",
        "max_abs_style_exposure",
        "max_abs_industry_exposure",
        "optimization_total_s",
    ]:
        daily[f"{column}_delta_010_minus_004"] = (
            daily[f"{column}_010"] - daily[f"{column}_004"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "paired_summary.csv", index=False)
    daily.to_csv(args.output_dir / "paired_daily.csv", index=False)
    with (args.output_dir / "paired_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "checks": checks,
                "active_004": str(args.active_004.resolve()),
                "active_010": str(args.active_010.resolve()),
                "rows": len(daily),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Paired comparison: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

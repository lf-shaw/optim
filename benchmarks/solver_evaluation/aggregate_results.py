#!/usr/bin/env python3
"""Combine non-overlapping benchmark result directories and recompute summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .benchmark import compare_weights, summarize, summarize_screens
except ImportError:
    from benchmark import compare_weights, summarize, summarize_screens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", type=Path, action="append", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    source_metadata = []
    for path in args.result_dir:
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_metadata.append(metadata)
        if metadata.get("errors"):
            raise ValueError(
                f"source result contains skipped/failed cases: {metadata_path}"
            )

    runs = pd.concat(
        [pd.read_csv(path / "runs.csv") for path in args.result_dir],
        ignore_index=True,
    )
    key = ["initial_mode", "model", "case", "repeat", "date"]
    duplicates = runs.duplicated(key, keep=False)
    if duplicates.any():
        sample = runs.loc[duplicates, key].head().to_dict("records")
        raise ValueError(f"overlapping result rows: {sample}")

    weight_parts = []
    for path in args.result_dir:
        weights_path = path / "weights.csv.gz"
        if weights_path.exists():
            weight_parts.append(pd.read_csv(weights_path))
    weights = pd.concat(weight_parts, ignore_index=True) if weight_parts else pd.DataFrame()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = runs.sort_values(key, kind="mergesort")
    runs.to_csv(output / "runs.csv", index=False)
    summarize(runs).to_csv(output / "summary.csv", index=False)
    summarize(runs, by_month=True).to_csv(output / "summary_by_month.csv", index=False)
    summarize_screens(runs).to_csv(output / "screen_summary.csv", index=False)
    if not weights.empty:
        weights = weights.sort_values(key + ["sid"], kind="mergesort")
        weights.to_csv(output / "weights.csv.gz", index=False, compression="gzip")
        compare_weights(weights, runs).to_csv(
            output / "solution_comparisons.csv", index=False
        )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "source_result_dirs": [str(path.resolve()) for path in args.result_dir],
                "source_commands": [metadata.get("command") for metadata in source_metadata],
                "rows": len(runs),
                "weight_rows": len(weights),
                "date_min": str(runs["date"].min()),
                "date_max": str(runs["date"].max()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Combine independently launched solver/thread cases into one result set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import benchmark


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    runs: list[pd.DataFrame] = []
    weights: list[pd.DataFrame] = []
    metadata: list[dict] = []
    for input_dir in args.inputs:
        runs_path = input_dir / "runs.csv"
        if not runs_path.is_file():
            raise FileNotFoundError(runs_path)
        runs.append(pd.read_csv(runs_path))
        weights_path = input_dir / "weights.csv.gz"
        if weights_path.is_file():
            weights.append(pd.read_csv(weights_path))
        metadata_path = input_dir / "metadata.json"
        if metadata_path.is_file():
            metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    combined_runs = pd.concat(runs, ignore_index=True)
    keys = ["model", "case", "repeat", "date"]
    duplicated = combined_runs.duplicated(keys, keep=False)
    if duplicated.any():
        duplicates = combined_runs.loc[duplicated, keys].to_dict("records")
        raise ValueError(f"duplicate run keys: {duplicates[:10]}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_runs.to_csv(output_dir / "runs.csv", index=False)
    benchmark.summarize(combined_runs).to_csv(output_dir / "summary.csv", index=False)

    if weights:
        combined_weights = pd.concat(weights, ignore_index=True)
        combined_weights.to_csv(
            output_dir / "weights.csv.gz", index=False, compression="gzip"
        )
        benchmark.compare_weights(combined_weights, combined_runs).to_csv(
            output_dir / "solution_comparisons.csv", index=False
        )

    combined_metadata = {
        "package_version": benchmark.PACKAGE_VERSION,
        "kind": "pardiso_mkl_independent_process_sweep",
        "input_directories": [str(path.resolve()) for path in args.inputs],
        "component_metadata": metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(combined_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Combined results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

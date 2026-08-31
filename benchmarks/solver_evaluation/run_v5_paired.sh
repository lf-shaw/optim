#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_v5_paired.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_ROOT]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_root=${5:-results/v5_paired}
python_bin=${BENCH_PYTHON_BIN:-python3}

ACTIVE_UB=0.004 "$script_dir/run_v5.sh" \
    "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license" \
    "$output_root/active_004"

ACTIVE_UB=0.01 "$script_dir/run_v5.sh" \
    "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license" \
    "$output_root/active_010"

"$python_bin" "$script_dir/compare_v5_variants.py" \
    "$output_root/active_004" \
    "$output_root/active_010" \
    "$output_root/paired"

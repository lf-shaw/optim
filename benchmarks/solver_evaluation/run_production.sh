#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 RISK_MODEL_XLSX BENCHMARK_CSV MOSEK_LICENSE [OUTPUT_DIR]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
risk_model_xlsx=$1
benchmark_csv=$2
mosek_license=$3
output_dir=${4:-results/production_v2}
bench_python_bin=${BENCH_PYTHON_BIN:-python3}

"$bench_python_bin" "$script_dir/benchmark.py" \
    --risk-model-xlsx "$risk_model_xlsx" \
    --benchmark-csv "$benchmark_csv" \
    --mosek-license "$mosek_license" \
    --output-dir "$output_dir" \
    --models LP,QP,SOCP \
    --lp-cases HIGHS,MOSEK,CLARABEL \
    --qp-cases MOSEK,CLARABEL \
    --socp-cases MOSEK,CLARABEL,CLARABEL_SCALED,CLARABEL_SCREENED,CLARABEL_SCREENED_SCALED,CLARABEL_SCALED_FAER \
    --alpha-target 0.2 \
    --screen-margin-pct 0.000001 \
    --screen-feasibility-tol 0.0000001 \
    --save-weights

echo "Results written to: $output_dir"
echo "Please return summary.csv, runs.csv, solution_comparisons.csv, metadata.json, and weights.csv.gz."

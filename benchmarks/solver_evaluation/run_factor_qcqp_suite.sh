#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 JUNE_JULY_XLSX AUGUST_XLSX BENCHMARK_CSV MOSEK_LICENSE [OUTPUT_ROOT]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
june_july_xlsx=$1
august_xlsx=$2
benchmark_csv=$3
mosek_license=$4
output_root=${5:-results/factor_qcqp_v4}
bench_python_bin=${BENCH_PYTHON_BIN:-python3}

run_dataset() {
    dataset_label=$1
    risk_model_xlsx=$2
    output_dir="$output_root/$dataset_label"

    "$bench_python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model_xlsx" \
        --benchmark-csv "$benchmark_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$output_dir" \
        --models SOCP \
        --socp-cases MOSEK,CLARABEL_SCALED_FAER,CLARABEL_SCREENED_SCALED,FACTOR_QP_PIQP,FACTOR_QP_PIQP_SCREENED \
        --initial-modes chained,independent_top500 \
        --alpha-target 0.2 \
        --screen-time-limit-s 0.75 \
        --screen-margin-pct 0.000001 \
        --screen-feasibility-tol 0.0000001 \
        --factor-objective-gap-abs 0.0001 \
        --factor-risk-margin-pct 0.00001 \
        --factor-intermediate-eps 0.00001 \
        --factor-final-eps 0.00000001 \
        --factor-max-outer-iters 30 \
        --factor-theta-initial 65536 \
        --save-weights
}

run_dataset 202606_202607 "$june_july_xlsx"
run_dataset 202608 "$august_xlsx"

"$bench_python_bin" "$script_dir/aggregate_results.py" \
    --result-dir "$output_root/202606_202607" \
    --result-dir "$output_root/202608" \
    --output-dir "$output_root/combined"

echo "Results written under: $output_root"
echo "Return the combined directory, including summary_by_month.csv and weights.csv.gz."

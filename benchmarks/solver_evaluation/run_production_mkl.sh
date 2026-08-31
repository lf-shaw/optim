#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 RISK_MODEL_XLSX BENCHMARK_CSV MOSEK_LICENSE [OUTPUT_ROOT]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
risk_model_xlsx=$1
benchmark_csv=$2
mosek_license=$3
output_root=${4:-results/production_mkl_v3}
mkl_thread_list=${MKL_THREAD_LIST:-"1 4 8 16 32"}
faer_num_threads=${FAER_NUM_THREADS:-8}

if [ -n "${BENCH_PYTHON_BIN:-}" ]; then
    bench_python_bin=$BENCH_PYTHON_BIN
elif [ -x "$script_dir/.venv_mkl/bin/python" ]; then
    bench_python_bin="$script_dir/.venv_mkl/bin/python"
else
    bench_python_bin=python3
fi

python_prefix=$(
    "$bench_python_bin" -c 'import sys; print(sys.prefix)'
)
mkl_lib_dir=${MKL_LIB_DIR:-"$python_prefix/lib"}
dynamic_library_path="$mkl_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

env \
    LD_LIBRARY_PATH="$dynamic_library_path" \
    MKL_THREADING_LAYER=SEQUENTIAL \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    "$bench_python_bin" "$script_dir/verify_clarabel_mkl.py"

mkdir -p "$output_root/components"

env \
    LD_LIBRARY_PATH="$dynamic_library_path" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    RAYON_NUM_THREADS="$faer_num_threads" \
    "$bench_python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model_xlsx" \
        --benchmark-csv "$benchmark_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$output_root/components/reference" \
        --models SOCP \
        --socp-cases MOSEK,CLARABEL_SCALED_QDLDL,CLARABEL_SCALED_FAER \
        --alpha-target 0.2 \
        --save-weights \
        --fail-fast

component_dirs=("$output_root/components/reference")
failures=0

run_mkl_case() {
    local threads=$1
    local threading_layer=INTEL
    if [ "$threads" -eq 1 ]; then
        threading_layer=SEQUENTIAL
    fi
    local case_name="CLARABEL_SCALED_MKL_T${threads}"
    local case_dir="$output_root/components/mkl_t${threads}"
    if env \
        LD_LIBRARY_PATH="$dynamic_library_path" \
        MKL_THREADING_LAYER="$threading_layer" \
        MKL_DYNAMIC=FALSE \
        MKL_NUM_THREADS="$threads" \
        OMP_NUM_THREADS="$threads" \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        "$bench_python_bin" "$script_dir/benchmark.py" \
            --risk-model-xlsx "$risk_model_xlsx" \
            --benchmark-csv "$benchmark_csv" \
            --output-dir "$case_dir" \
            --models SOCP \
            --socp-cases "$case_name" \
            --alpha-target 0.2 \
            --save-weights \
            --fail-fast
    then
        component_dirs+=("$case_dir")
    else
        failures=$((failures + 1))
        echo "WARNING: $case_name failed; continuing the remaining sweep" >&2
    fi
}

for threads in $mkl_thread_list; do
    run_mkl_case "$threads"
done

"$bench_python_bin" "$script_dir/combine_results.py" \
    --output-dir "$output_root/combined" \
    --inputs "${component_dirs[@]}"

echo "Results written to: $output_root"
echo "Return the whole directory, including components and combined."
if [ "$failures" -ne 0 ]; then
    echo "$failures MKL thread case(s) failed" >&2
    exit 1
fi

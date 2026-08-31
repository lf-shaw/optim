#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_v5_lp_qp.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_DIR]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_dir=${5:-results/v5_lp_qp}
python_bin=${BENCH_PYTHON_BIN:-python3}
active_ub=${ACTIVE_UB:-0.004}
style_bound=${STYLE_BOUND:-0.6}
industry_bound=${INDUSTRY_BOUND:-0.05}
random_seed=${INITIAL_RANDOM_SEED:-20260826}
state_weight_tol=${CHAINED_STATE_WEIGHT_TOL:-1e-5}
missing_policy=${MISSING_HOLDING_POLICY:-renormalize}
common_risk_aversion=${COMMON_RISK_AVERSION:-0.75}
specific_risk_aversion=${SPECIFIC_RISK_AVERSION:-0.75}

for input_path in "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license"; do
    if [[ ! -f "$input_path" ]]; then
        echo "input file not found: $input_path" >&2
        exit 2
    fi
done

"$python_bin" "$script_dir/benchmark.py" \
    --risk-model-xlsx "$risk_model" \
    --benchmark-csv "$benchmark_csv" \
    --alpha-csv "$alpha_csv" \
    --mosek-license "$mosek_license" \
    --output-dir "$output_dir" \
    --models LP,QP \
    --lp-cases HIGHS,MOSEK,CLARABEL \
    --qp-cases FACTOR_PENALTY_QP_PIQP,PIQP,MOSEK,CLARABEL,CLARABEL_SCALED_FAER \
    --initial-modes chained_topn_random,independent_topn_random \
    --initial-top-n 500 \
    --initial-top-n-max 600 \
    --initial-random-seed "$random_seed" \
    --active-ub "$active_ub" \
    --style-bound "$style_bound" \
    --industry-bound "$industry_bound" \
    --common-risk-aversion "$common_risk_aversion" \
    --specific-risk-aversion "$specific_risk_aversion" \
    --chained-state-weight-tol "$state_weight_tol" \
    --missing-holding-policy "$missing_policy" \
    --skip-infeasible-rebalance-dates \
    --require-consistent-skips \
    --save-weights \
    --fail-fast

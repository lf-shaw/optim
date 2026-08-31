#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_v5.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_DIR]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_dir=${5:-results/v5}
python_bin=${BENCH_PYTHON_BIN:-python3}
random_seed=${INITIAL_RANDOM_SEED:-20260826}
active_ub=${ACTIVE_UB:-0.004}
style_bound=${STYLE_BOUND:-0.6}
industry_bound=${INDUSTRY_BOUND:-0.05}
factor_theta_initial=${FACTOR_THETA_INITIAL:-16384}
risk_budget_pct=${RISK_BUDGET_PCT:-6.0}
chained_state_weight_tol=${CHAINED_STATE_WEIGHT_TOL:-1e-5}
screen_max_initial_n=${SCREEN_MAX_INITIAL_N:-1200}

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
    --models SOCP \
    --socp-cases MOSEK,CLARABEL_SCALED_FAER,CLARABEL_SCREENED_SCALED,FACTOR_QP_PIQP,FACTOR_QP_PIQP_SCREENED \
    --initial-modes chained_topn_random,independent_topn_random \
    --initial-top-n 500 \
    --initial-top-n-max 600 \
    --initial-random-seed "$random_seed" \
    --active-ub "$active_ub" \
    --style-bound "$style_bound" \
    --industry-bound "$industry_bound" \
    --risk-budget-pct "$risk_budget_pct" \
    --screen-time-limit-s 0.75 \
    --screen-max-initial-n "$screen_max_initial_n" \
    --chained-state-weight-tol "$chained_state_weight_tol" \
    --factor-objective-gap-abs 1e-4 \
    --factor-theta-initial "$factor_theta_initial" \
    --skip-infeasible-rebalance-dates \
    --require-consistent-skips \
    --save-weights \
    --fail-fast

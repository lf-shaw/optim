#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_v5_direct_piqp.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_ROOT]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_root=${5:-results/v5_direct_piqp}
python_bin=${BENCH_PYTHON_BIN:-python3}
random_seed=${INITIAL_RANDOM_SEED:-20260826}
style_bound=${STYLE_BOUND:-0.6}
industry_bound=${INDUSTRY_BOUND:-0.05}
state_weight_tol=${CHAINED_STATE_WEIGHT_TOL:-1e-5}
common_risk_aversion=${COMMON_RISK_AVERSION:-0.75}
specific_risk_aversion=${SPECIFIC_RISK_AVERSION:-0.75}
risk_budget_pct=${RISK_BUDGET_PCT:-2.0}
factor_theta_initial=${FACTOR_THETA_INITIAL:-16384}

for input_path in "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license"; do
    if [[ ! -f "$input_path" ]]; then
        echo "input file not found: $input_path" >&2
        exit 2
    fi
done

for active_ub in 0.004 0.01; do
    case "$active_ub" in
        0.004) active_label=004 ;;
        0.01|0.010) active_label=010 ;;
        *) active_label=${active_ub//./p} ;;
    esac
    active_root="$output_root/active_$active_label"

    "$python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model" \
        --benchmark-csv "$benchmark_csv" \
        --alpha-csv "$alpha_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$active_root/qp" \
        --models QP \
        --qp-cases FACTOR_PENALTY_QP_PIQP,PIQP,MOSEK \
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
        --missing-holding-policy renormalize \
        --skip-infeasible-rebalance-dates \
        --require-consistent-skips \
        --save-weights \
        --fail-fast

    "$python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model" \
        --benchmark-csv "$benchmark_csv" \
        --alpha-csv "$alpha_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$active_root/socp_fixed_theta" \
        --models SOCP \
        --socp-cases FACTOR_QP_PIQP,MOSEK \
        --initial-modes chained_topn_random \
        --initial-top-n 500 \
        --initial-top-n-max 600 \
        --initial-random-seed "$random_seed" \
        --active-ub "$active_ub" \
        --style-bound "$style_bound" \
        --industry-bound "$industry_bound" \
        --risk-budget-pct "$risk_budget_pct" \
        --factor-objective-gap-abs 1e-4 \
        --factor-theta-initial "$factor_theta_initial" \
        --chained-state-weight-tol "$state_weight_tol" \
        --missing-holding-policy renormalize \
        --skip-infeasible-rebalance-dates \
        --require-consistent-skips \
        --save-weights \
        --fail-fast

    "$python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model" \
        --benchmark-csv "$benchmark_csv" \
        --alpha-csv "$alpha_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$active_root/socp_carry_theta" \
        --models SOCP \
        --socp-cases FACTOR_QP_PIQP \
        --initial-modes chained_topn_random \
        --initial-top-n 500 \
        --initial-top-n-max 600 \
        --initial-random-seed "$random_seed" \
        --active-ub "$active_ub" \
        --style-bound "$style_bound" \
        --industry-bound "$industry_bound" \
        --risk-budget-pct "$risk_budget_pct" \
        --factor-objective-gap-abs 1e-4 \
        --factor-theta-initial "$factor_theta_initial" \
        --factor-carry-theta \
        --chained-state-weight-tol "$state_weight_tol" \
        --missing-holding-policy renormalize \
        --skip-infeasible-rebalance-dates \
        --require-consistent-skips \
        --save-weights \
        --fail-fast
done

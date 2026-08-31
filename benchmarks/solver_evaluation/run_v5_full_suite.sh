#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_v5_full_suite.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_ROOT]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_root=${5:-results/v5_full_suite}
python_bin=${BENCH_PYTHON_BIN:-python3}
random_seed=${INITIAL_RANDOM_SEED:-20260826}
style_bound=${STYLE_BOUND:-0.6}
industry_bound=${INDUSTRY_BOUND:-0.05}
state_weight_tol=${CHAINED_STATE_WEIGHT_TOL:-1e-5}
common_risk_aversion=${COMMON_RISK_AVERSION:-0.75}
specific_risk_aversion=${SPECIFIC_RISK_AVERSION:-0.75}
factor_theta_initial=${FACTOR_THETA_INITIAL:-16384}
screen_time_limit_s=${SCREEN_TIME_LIMIT_S:-0.75}
screen_max_initial_n=${SCREEN_MAX_INITIAL_N:-1200}
active_ub_list=${ACTIVE_UB_LIST:-"0.004 0.01"}
risk_budget_list=${RISK_BUDGET_LIST:-"2 6"}
lp_cases=${LP_CASES:-"HIGHS,MOSEK,CLARABEL,CLARABEL_SCALED_FAER"}
qp_cases=${QP_CASES:-"FACTOR_PENALTY_QP_PIQP,PIQP,MOSEK,CLARABEL,CLARABEL_SCALED_FAER"}
socp_cases=${SOCP_CASES:-"MOSEK,CLARABEL,CLARABEL_SCALED_QDLDL,CLARABEL_SCALED_FAER,CLARABEL_SCREENED_SCALED,FACTOR_QP_PIQP,FACTOR_QP_PIQP_SCREENED,ECOS"}
carry_cases=${CARRY_CASES:-"FACTOR_QP_PIQP,FACTOR_QP_PIQP_SCREENED"}
max_dates=${MAX_DATES:-}
max_dates_args=()
if [[ -n "$max_dates" ]]; then
    max_dates_args=(--max-dates "$max_dates")
fi

for input_path in "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license"; do
    if [[ ! -f "$input_path" ]]; then
        echo "input file not found: $input_path" >&2
        exit 2
    fi
done

"$python_bin" "$script_dir/verify_piqp_install.py"

for active_ub in $active_ub_list; do
    case "$active_ub" in
        0.004) active_label=004 ;;
        0.01|0.010) active_label=010 ;;
        *) active_label=${active_ub//./p} ;;
    esac
    active_root="$output_root/active_$active_label"

    # LP is independent of any TE budget. QP is the independent fixed
    # risk-penalty model; it does not inherit the SOCP risk budget below.
    "$python_bin" "$script_dir/benchmark.py" \
        --risk-model-xlsx "$risk_model" \
        --benchmark-csv "$benchmark_csv" \
        --alpha-csv "$alpha_csv" \
        --mosek-license "$mosek_license" \
        --output-dir "$active_root/lp_qp" \
        --models LP,QP \
        --lp-cases "$lp_cases" \
        --qp-cases "$qp_cases" \
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
        "${max_dates_args[@]}" \
        --save-weights

    for risk_budget_pct in $risk_budget_list; do
        risk_label=${risk_budget_pct//./p}

        # Fixed-theta matrix: commercial baseline, free conic solvers,
        # direct factor-QP, and exact LP-screened paths. Both chained and
        # independent cold-start sequences are included.
        "$python_bin" "$script_dir/benchmark.py" \
            --risk-model-xlsx "$risk_model" \
            --benchmark-csv "$benchmark_csv" \
            --alpha-csv "$alpha_csv" \
            --mosek-license "$mosek_license" \
            --output-dir "$active_root/socp_te_$risk_label/fixed_theta" \
            --models SOCP \
            --socp-cases "$socp_cases" \
            --initial-modes chained_topn_random,independent_topn_random \
            --initial-top-n 500 \
            --initial-top-n-max 600 \
            --initial-random-seed "$random_seed" \
            --active-ub "$active_ub" \
            --style-bound "$style_bound" \
            --industry-bound "$industry_bound" \
            --risk-budget-pct "$risk_budget_pct" \
            --screen-time-limit-s "$screen_time_limit_s" \
            --screen-max-initial-n "$screen_max_initial_n" \
            --factor-objective-gap-abs 1e-4 \
            --factor-theta-initial "$factor_theta_initial" \
            --chained-state-weight-tol "$state_weight_tol" \
            --missing-holding-policy renormalize \
            --skip-infeasible-rebalance-dates \
            --require-consistent-skips \
            "${max_dates_args[@]}" \
            --save-weights

        # Theta continuation affects only chained factor-QP cases. Keep it in
        # a separate result directory so the policy/path dependency is explicit.
        "$python_bin" "$script_dir/benchmark.py" \
            --risk-model-xlsx "$risk_model" \
            --benchmark-csv "$benchmark_csv" \
            --alpha-csv "$alpha_csv" \
            --mosek-license "$mosek_license" \
            --output-dir "$active_root/socp_te_$risk_label/carry_theta" \
            --models SOCP \
            --socp-cases "$carry_cases" \
            --initial-modes chained_topn_random \
            --initial-top-n 500 \
            --initial-top-n-max 600 \
            --initial-random-seed "$random_seed" \
            --active-ub "$active_ub" \
            --style-bound "$style_bound" \
            --industry-bound "$industry_bound" \
            --risk-budget-pct "$risk_budget_pct" \
            --screen-time-limit-s "$screen_time_limit_s" \
            --screen-max-initial-n "$screen_max_initial_n" \
            --factor-objective-gap-abs 1e-4 \
            --factor-theta-initial "$factor_theta_initial" \
            --factor-carry-theta \
            --chained-state-weight-tol "$state_weight_tol" \
            --missing-holding-policy renormalize \
            --skip-infeasible-rebalance-dates \
            --require-consistent-skips \
            "${max_dates_args[@]}" \
            --save-weights
    done
done

"$python_bin" "$script_dir/collect_v5_full_suite.py" "$output_root"

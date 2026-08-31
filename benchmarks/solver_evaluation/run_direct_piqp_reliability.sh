#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
risk_model=${1:?"usage: run_direct_piqp_reliability.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv MOSEK.lic [OUTPUT_ROOT]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
mosek_license=${4:?"missing MOSEK license path"}
output_root=${5:-results/direct_piqp_reliability}

python_bin=${BENCH_PYTHON_BIN:-python3}
random_seed=${INITIAL_RANDOM_SEED:-20260826}
style_bound=${STYLE_BOUND:-0.6}
industry_bound=${INDUSTRY_BOUND:-0.05}
state_weight_tol=${CHAINED_STATE_WEIGHT_TOL:-1e-5}
common_risk_aversion=${COMMON_RISK_AVERSION:-0.75}
specific_risk_aversion=${SPECIFIC_RISK_AVERSION:-0.75}
theta_initial=${FACTOR_THETA_INITIAL:-16384}
active_ub_list=${ACTIVE_UB_LIST:-"0.004 0.01"}
risk_budget_list=${RISK_BUDGET_LIST:-"2 6"}
repeats=${REPEATS:-1}
max_dates=${MAX_DATES:-}
save_weights=${SAVE_WEIGHTS:-1}
run_robust_matrix=${RUN_ROBUST_MATRIX:-1}
smoke_only=${SMOKE_ONLY:-0}
data_cache_dir=${DATA_CACHE_DIR:-/tmp/optim_direct_piqp_data_cache_${EUID}}

for input_path in "$risk_model" "$benchmark_csv" "$alpha_csv" "$mosek_license"; do
    if [[ ! -f "$input_path" ]]; then
        echo "input file not found: $input_path" >&2
        exit 2
    fi
done

"$python_bin" "$script_dir/verify_piqp_install.py"

max_dates_args=()
if [[ -n "$max_dates" ]]; then
    max_dates_args=(--max-dates "$max_dates")
fi
save_weights_args=()
if [[ "$save_weights" == "1" ]]; then
    save_weights_args=(--save-weights)
fi

common_args=(
    --risk-model-xlsx "$risk_model"
    --benchmark-csv "$benchmark_csv"
    --alpha-csv "$alpha_csv"
    --mosek-license "$mosek_license"
    --data-cache-dir "$data_cache_dir"
    --initial-modes chained_topn_random,independent_topn_random
    --initial-top-n 500
    --initial-top-n-max 600
    --initial-random-seed "$random_seed"
    --style-bound "$style_bound"
    --industry-bound "$industry_bound"
    --chained-state-weight-tol "$state_weight_tol"
    --missing-holding-policy renormalize
    --skip-infeasible-rebalance-dates
    --require-consistent-skips
    --factor-objective-gap-abs 1e-4
    --factor-theta-initial "$theta_initial"
    --factor-diagnostics
    --factor-fallback-solver CLARABEL
    --factor-fallback-clarabel-method qdldl
    --direct-qp-fallback MOSEK
    --repeats "$repeats"
    "${max_dates_args[@]}"
    "${save_weights_args[@]}"
)

if [[ "$smoke_only" == "1" ]]; then
    smoke_active_ub=${SMOKE_ACTIVE_UB:-0.004}
    smoke_risk_budget=${SMOKE_RISK_BUDGET:-2}
    echo "[smoke] one process, one risk date; full input is parsed once and cached"
    "$python_bin" "$script_dir/benchmark.py" \
        "${common_args[@]}" \
        --output-dir "$output_root/smoke" \
        --models QP,SOCP \
        --qp-cases FACTOR_PENALTY_QP_PIQP,MOSEK \
        --socp-cases FACTOR_QP_PIQP,MOSEK \
        --active-ub "$smoke_active_ub" \
        --risk-budget-pct "$smoke_risk_budget" \
        --common-risk-aversion "$common_risk_aversion" \
        --specific-risk-aversion "$specific_risk_aversion" \
        --factor-final-eps 1e-8 \
        --factor-piqp-max-iter 1000 \
        --max-dates 1
    "$python_bin" "$script_dir/collect_direct_piqp_reliability.py" "$output_root"
    echo
    echo "Smoke results: $output_root"
    exit 0
fi

run_matrix() {
    local settings_label=$1
    local final_eps=$2
    local piqp_max_iter=$3
    local active_ub=$4
    local active_label=$5
    local include_references=$6
    local settings_root="$output_root/$settings_label/active_$active_label"
    local qp_cases=FACTOR_PENALTY_QP_PIQP
    local socp_cases=FACTOR_QP_PIQP
    if [[ "$include_references" == "1" ]]; then
        qp_cases=FACTOR_PENALTY_QP_PIQP,PIQP,MOSEK
        socp_cases=FACTOR_QP_PIQP,MOSEK
    fi

    "$python_bin" "$script_dir/benchmark.py" \
        "${common_args[@]}" \
        --output-dir "$settings_root/qp" \
        --models QP \
        --qp-cases "$qp_cases" \
        --active-ub "$active_ub" \
        --common-risk-aversion "$common_risk_aversion" \
        --specific-risk-aversion "$specific_risk_aversion" \
        --factor-final-eps "$final_eps" \
        --factor-piqp-max-iter "$piqp_max_iter"

    for risk_budget_pct in $risk_budget_list; do
        local risk_label=${risk_budget_pct//./p}
        local risk_root="$settings_root/socp_te_$risk_label"

        # LP screening is deliberately absent. Fixed theta is tested in both
        # sequence modes against a same-configuration MOSEK reference.
        "$python_bin" "$script_dir/benchmark.py" \
            "${common_args[@]}" \
            --output-dir "$risk_root/fixed_theta" \
            --models SOCP \
            --socp-cases "$socp_cases" \
            --active-ub "$active_ub" \
            --risk-budget-pct "$risk_budget_pct" \
            --factor-final-eps "$final_eps" \
            --factor-piqp-max-iter "$piqp_max_iter"

        # Carry-theta applies only to the chained PIQP path and is isolated so
        # its path dependence cannot be mistaken for a backend comparison.
        "$python_bin" "$script_dir/benchmark.py" \
            "${common_args[@]}" \
            --output-dir "$risk_root/carry_theta" \
            --models SOCP \
            --socp-cases FACTOR_QP_PIQP \
            --initial-modes chained_topn_random \
            --active-ub "$active_ub" \
            --risk-budget-pct "$risk_budget_pct" \
            --factor-final-eps "$final_eps" \
            --factor-piqp-max-iter "$piqp_max_iter" \
            --factor-carry-theta
    done
}

for active_ub in $active_ub_list; do
    case "$active_ub" in
        0.004) active_label=004 ;;
        0.01|0.010) active_label=010 ;;
        *) active_label=${active_ub//./p} ;;
    esac
    run_matrix baseline 1e-8 1000 "$active_ub" "$active_label" 1
    if [[ "$run_robust_matrix" == "1" ]]; then
        # This is a separate experiment, not an automatic production retry:
        # it measures whether 1e-7 final tolerance and a larger iteration cap
        # recover baseline PIQP failures without material runtime regression.
        run_matrix robust 1e-7 5000 "$active_ub" "$active_label" 0
    fi
done

"$python_bin" "$script_dir/collect_direct_piqp_reliability.py" "$output_root"

echo
echo "Direct PIQP reliability results: $output_root"
echo "Return the whole directory (tar.gz is recommended)."

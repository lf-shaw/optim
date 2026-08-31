#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
risk_model=${1:?"usage: run_piqp_patched_acceptance.sh RISK_MODEL.xlsx BENCHMARK.csv ALPHA.csv [OUTPUT_ROOT]"}
benchmark_csv=${2:?"missing dated benchmark CSV"}
alpha_csv=${3:?"missing dated alpha CSV"}
output_root=${4:-results/piqp_patched_acceptance}

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
data_cache_dir=${DATA_CACHE_DIR:-/tmp/optim_piqp_patched_acceptance_cache_${EUID}}

for input_path in "$risk_model" "$benchmark_csv" "$alpha_csv"; do
    if [[ ! -f "$input_path" ]]; then
        echo "input file not found: $input_path" >&2
        exit 2
    fi
done

# Fail before loading large inputs unless the supported upstream compact build
# is active. PIQP releases older than 0.6.4 are intentionally unsupported.
"$python_bin" "$script_dir/verify_piqp_install.py"

max_dates_args=()
if [[ -n "$max_dates" ]]; then
    max_dates_args=(--max-dates "$max_dates")
fi

common_args=(
    --risk-model-xlsx "$risk_model"
    --benchmark-csv "$benchmark_csv"
    --alpha-csv "$alpha_csv"
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
    --factor-objective-gap-abs 1e-4
    --factor-theta-initial "$theta_initial"
    --factor-intermediate-eps 1e-5
    --factor-final-eps 1e-8
    --factor-piqp-max-iter 1000
    --factor-piqp-inequality-form auto
    --factor-diagnostics
    --factor-fallback-solver CLARABEL
    --factor-fallback-clarabel-method qdldl
    --direct-qp-fallback CLARABEL
    --repeats "$repeats"
    "${max_dates_args[@]}"
)

for active_ub in $active_ub_list; do
    case "$active_ub" in
        0.004) active_label=004 ;;
        0.01|0.010) active_label=010 ;;
        *) active_label=${active_ub//./p} ;;
    esac
    active_root="$output_root/active_$active_label"

    # QP is independent of the SOCP risk-budget list and is solved once for
    # each active-weight configuration.
    "$python_bin" "$script_dir/benchmark.py" \
        "${common_args[@]}" \
        --output-dir "$active_root/qp" \
        --models QP \
        --qp-cases FACTOR_PENALTY_QP_PIQP \
        --active-ub "$active_ub" \
        --common-risk-aversion "$common_risk_aversion" \
        --specific-risk-aversion "$specific_risk_aversion"

    for risk_budget_pct in $risk_budget_list; do
        risk_label=${risk_budget_pct//./p}
        "$python_bin" "$script_dir/benchmark.py" \
            "${common_args[@]}" \
            --output-dir "$active_root/socp_te_$risk_label" \
            --models SOCP \
            --socp-cases FACTOR_QP_PIQP \
            --active-ub "$active_ub" \
            --risk-budget-pct "$risk_budget_pct"
    done
done

# No benchmark invocation above passes --save-weights. The collector writes
# only compressed runs/traces, summaries, failures, and environment metadata.
"$python_bin" "$script_dir/collect_direct_piqp_reliability.py" \
    "$output_root" --strict

echo
echo "Patched PIQP acceptance results: $output_root"
echo "No portfolio weights were written. Return this result directory as tar.gz."

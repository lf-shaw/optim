#!/usr/bin/env python3
r"""Exercise the unified optimizer on the existing v5 DataYes workbook.

The historical workbook stores annualized factor covariance in $\%^2$
and specific volatility in percent.  This adapter converts them explicitly to
the core ``annualized_decimal`` contract before compilation.  It intentionally
does not save portfolio weights.

The legacy ``DataStore`` lives in the old CVXPY benchmark module.  CVXPY is not
a dependency of the unified optimizer, so a two-constant import stub is used
when CVXPY is absent; none of the legacy modeling functions are invoked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "solver_evaluation"
for path in (REPOSITORY_ROOT, LEGACY_BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import cvxpy  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("cvxpy")
    setattr(stub, "OPTIMAL", "optimal")
    setattr(stub, "OPTIMAL_INACCURATE", "optimal_inaccurate")
    sys.modules["cvxpy"] = stub

from benchmark import Config, DataStore, STYLE_FACTORS  # noqa: E402
from optim import (  # noqa: E402
    AlphaSpec,
    DataProvenance,
    ExposureBounds,
    FactorRiskModel,
    LowerBound,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    RiskAdjustedAlpha,
    SolverPolicy,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-model-xlsx",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/datayes_data_v5.xlsx",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/000852.SH_v5.csv",
    )
    parser.add_argument(
        "--alpha-csv",
        type=Path,
        default=REPOSITORY_ROOT / "tmp/v5/alpha_v5.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/optim_v5_real_data_cache"),
    )
    parser.add_argument("--max-dates", type=int, default=0, help="0 means all dates")
    parser.add_argument(
        "--cases",
        default="lp,qp,factor_qcqp",
        help="comma-separated subset of lp,qp,factor_qcqp",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--initial-top-n", type=int, default=500)
    parser.add_argument("--initial-top-n-max", type=int, default=600)
    parser.add_argument("--initial-seed", type=int, default=20260826)
    parser.add_argument("--asset-upper", type=float, default=1.0)
    parser.add_argument("--active-upper", type=float, default=0.01)
    parser.add_argument("--total-active", type=float, default=1.8)
    parser.add_argument("--turnover", type=float, default=0.05)
    parser.add_argument("--benchmark-member-lower", type=float, default=0.81)
    parser.add_argument("--style-bound", type=float, default=0.60)
    parser.add_argument("--industry-bound", type=float, default=0.05)
    parser.add_argument("--tracking-error", type=float, default=0.02)
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=7500.0,
        help=(
            "coefficient on decimal variance; 7500 matches legacy 0.75 on "
            "DataYes percent-squared risk"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--lp-prescreen",
        action="store_true",
        help="explicitly enable the exact HiGHS LP certificate before factor-QCQP",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only configuration and aggregates; --output still keeps per-date rows",
    )
    return parser


def _percent_risk_to_problem(
    day: Any,
    *,
    case: str,
    args: argparse.Namespace,
) -> PortfolioProblem:
    if day.benchmark_stale_days or day.alpha_stale_days:
        raise ValueError(
            f"{day.date.date()}: benchmark/alpha must have exact-date data; "
            f"stale_days=({day.benchmark_stale_days}, {day.alpha_stale_days})"
        )
    if day.benchmark_missing_fraction > 1e-12:
        raise ValueError(
            f"{day.date.date()}: benchmark risk-universe missing mass is "
            f"{day.benchmark_missing_fraction:.6%}; explicit normalization was not enabled"
        )

    factors = tuple(str(name) for name in day.factors)
    factor_types = tuple(
        "style"
        if name in STYLE_FACTORS
        else "country"
        if name == "country"
        else "industry"
        for name in factors
    )
    risk_model = FactorRiskModel(
        asof=pd.Timestamp(day.date),
        exposure=np.asarray(day.exposure, dtype=float),
        covariance=np.asarray(day.covariance, dtype=float) / 10_000.0,
        specific_volatility=np.asarray(day.spec_risk, dtype=float) / 100.0,
        factor_names=factors,
        factor_types=factor_types,
        provenance=DataProvenance(
            source="datayes_v5_workbook",
            source_date=pd.Timestamp(day.date),
            metadata={
                "factor_covariance_input_unit": "annualized_percent_squared",
                "specific_volatility_input_unit": "annualized_percent",
            },
        ),
    )
    data = PortfolioData(
        date=pd.Timestamp(day.date),
        assets=pd.Index(day.sids.astype(str), name="sid"),
        alpha=np.asarray(day.alpha, dtype=float),
        benchmark=np.asarray(day.benchmark, dtype=float),
        initial_weight=np.asarray(day.initial, dtype=float),
        tradable=np.ones(len(day.sids), dtype=bool),
        risk_model=risk_model,
        alpha_spec=AlphaSpec(units="standardized_score", scale=1.0),
        provenance=DataProvenance(
            source="v5_benchmark_adapter",
            source_date=pd.Timestamp(day.date),
            metadata={
                "benchmark_source_date": day.benchmark_source_date,
                "alpha_source_date": day.alpha_source_date,
                "benchmark_missing_mass": day.benchmark_missing_fraction,
                "alpha_missing_filled_with_zero": day.alpha_missing_count,
            },
        ),
    )
    tracking_error = (
        TrackingErrorLimit(args.tracking_error) if case == "factor_qcqp" else None
    )
    constraints = PortfolioConstraints(
        asset_weight=WeightBounds(0.0, args.asset_upper),
        active_weight=SymmetricBound(args.active_upper),
        total_active=args.total_active,
        turnover=TurnoverLimit(args.turnover),
        benchmark_member_weight=LowerBound(args.benchmark_member_lower),
        style=ExposureBounds(default=(-args.style_bound, args.style_bound)),
        industry=ExposureBounds(default=(-args.industry_bound, args.industry_bound)),
        tracking_error=tracking_error,
    )
    objective = (
        RiskAdjustedAlpha(args.risk_aversion, args.risk_aversion)
        if case == "qp"
        else MaximizeAlpha()
    )
    return PortfolioProblem(data=data, objective=objective, constraints=constraints)


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for case in sorted({row["case"] for row in rows}):
        group = [row for row in rows if row["case"] == case]

        def stats(field: str) -> dict[str, float]:
            values = [float(row[field]) for row in group]
            return {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "max": max(values),
            }

        def optional_stats(field: str) -> dict[str, float] | None:
            values = [float(row[field]) for row in group if row[field] is not None]
            if not values:
                return None
            return {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "max": max(values),
            }

        result[case] = {
            "dates": len(group),
            "statuses": dict(Counter(row["status"] for row in group)),
            "backends": dict(Counter(row["backend"] for row in group)),
            "prepare_s": stats("prepare_s"),
            "backend_setup_s": stats("backend_setup_s"),
            "backend_solve_s": stats("backend_solve_s"),
            "validation_s": stats("validation_s"),
            "optimization_total_s": stats("optimization_total_s"),
            "wall_s": stats("wall_s"),
            "tracking_error_max": max(
                (row["tracking_error"] for row in group if row["tracking_error"] is not None),
                default=None,
            ),
            "certificate_gap_max": max(
                (
                    row["certificate_gap"]
                    for row in group
                    if row["certificate_gap"] is not None
                ),
                default=None,
            ),
            "maximum_reported_violation": max(row["max_violation"] for row in group),
            "qp_solves": optional_stats("qp_solves"),
            "workspace_rebuilds_total": sum(
                int(row["workspace_rebuilds"] or 0) for row in group
            ),
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    requested_cases = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    unknown = set(requested_cases) - {"lp", "qp", "factor_qcqp"}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    if args.max_dates < 0 or args.warmups < 0:
        raise ValueError("max-dates and warmups must be non-negative")
    if args.initial_top_n_max < args.initial_top_n:
        raise ValueError("initial-top-n-max must be >= initial-top-n")

    store = DataStore(
        args.risk_model_xlsx,
        args.benchmark_csv,
        args.alpha_csv,
        args.cache_dir,
    )
    dates = store.dates[: args.max_dates or None]
    legacy_config = Config(
        turnover_limit=args.turnover,
        active_ub=args.active_upper,
        total_active_ub=args.total_active,
        benchmark_weight_lb=args.benchmark_member_lower,
        style_default_lb=-args.style_bound,
        style_default_ub=args.style_bound,
        size_lb=-args.style_bound,
        size_ub=args.style_bound,
        industry_lb=-args.industry_bound,
        industry_ub=args.industry_bound,
        risk_budget_pct=100.0 * args.tracking_error,
        common_risk_aversion=args.risk_aversion / 10_000.0,
        specific_risk_aversion=args.risk_aversion / 10_000.0,
        asset_ub=args.asset_upper,
        initial_top_n=args.initial_top_n,
    )

    materialize_started = time.perf_counter()
    days = []
    for date in dates:
        initial, _ = store.random_top_range_initial(
            date,
            args.initial_top_n,
            args.initial_top_n_max,
            args.initial_seed,
        )
        days.append(store.prepare_day(date, initial, legacy_config, "error"))
    materialize_s = time.perf_counter() - materialize_started

    optimizer = PortfolioOptimizer(policy=SolverPolicy(lp_prescreen=args.lp_prescreen))
    problems = {
        case: [
            _percent_risk_to_problem(day, case=case, args=args) for day in days
        ]
        for case in requested_cases
    }
    for case in requested_cases:
        for _ in range(args.warmups):
            warm = optimizer.prepare(problems[case][0])
            warm.validation.raise_for_errors()
            warm_result = optimizer.solve_prepared(warm)
            if not warm_result.status.has_solution:
                raise RuntimeError(f"{case} warmup failed: {warm_result.message}")

    rows: list[dict[str, Any]] = []
    for case in requested_cases:
        for problem in problems[case]:
            wall_started = time.perf_counter()
            prepared = optimizer.prepare(problem)
            prepared.validation.raise_for_errors()
            result = optimizer.solve_prepared(prepared)
            wall_s = time.perf_counter() - wall_started
            risk_model = problem.data.risk_model
            assert isinstance(risk_model, FactorRiskModel)
            route_metadata = (
                {} if not result.route else result.route[-1].metadata
            )
            rows.append(
                {
                    "case": case,
                    "date": problem.data.date.strftime("%Y-%m-%d"),
                    "assets": len(problem.data.assets),
                    "factors": len(risk_model.factor_names),
                    "status": result.status.value,
                    "backend": result.backend,
                    "route": [attempt.backend for attempt in result.route],
                    "prepare_s": result.timings.prepare_s,
                    "backend_setup_s": result.timings.backend_setup_s,
                    "backend_solve_s": result.timings.backend_solve_s,
                    "validation_s": result.timings.validation_s,
                    "optimization_total_s": result.timings.total_s,
                    "wall_s": wall_s,
                    "tracking_error": result.metrics.tracking_error,
                    "objective": result.metrics.objective,
                    "turnover_l1": result.metrics.turnover_l1,
                    "total_active_l1": result.metrics.total_active_l1,
                    "benchmark_member_weight": result.metrics.benchmark_member_weight,
                    "max_active_weight": result.metrics.max_active_weight,
                    "certificate_gap": (
                        None
                        if result.certificate is None
                        else result.certificate.absolute_gap
                    ),
                    "certificate_kind": (
                        None if result.certificate is None else result.certificate.kind
                    ),
                    "certificate_proof_status": (
                        None
                        if result.certificate is None
                        else result.certificate.proof_status.value
                    ),
                    "max_violation": max(
                        (violation.amount for violation in result.violations),
                        default=0.0,
                    ),
                    "theta": route_metadata.get("theta"),
                    "theta_seed": route_metadata.get("theta_seed"),
                    "qp_solves": route_metadata.get("qp_solves"),
                    "outer_points": route_metadata.get("outer_points"),
                    "workspace_rebuilds": route_metadata.get("workspace_rebuilds"),
                    "objective_scale": route_metadata.get("objective_scale"),
                    "primal_residual": route_metadata.get("primal_residual"),
                    "dual_residual": route_metadata.get("dual_residual"),
                    "duality_gap": route_metadata.get("duality_gap"),
                    "fingerprint": {
                        "semantic_hash": result.fingerprint.semantic_hash,
                        "canonical_hash": result.fingerprint.canonical_hash,
                        "compiler_version": result.fingerprint.compiler_version,
                    },
                }
            )

    payload = {
        "configuration": {
            "risk_model_xlsx": str(args.risk_model_xlsx.resolve()),
            "dates": len(dates),
            "date_start": dates[0].strftime("%Y-%m-%d"),
            "date_end": dates[-1].strftime("%Y-%m-%d"),
            "cases": requested_cases,
            "warmups_per_case": args.warmups,
            "initial": (
                f"daily benchmark top-N, N uniform in "
                f"[{args.initial_top_n}, {args.initial_top_n_max}], "
                f"seed={args.initial_seed}"
            ),
            "risk_unit_conversion": {
                "factor_covariance": "divide by 10000",
                "specific_volatility": "divide by 100",
            },
            "risk_aversion_decimal_variance": args.risk_aversion,
            "lp_prescreen": args.lp_prescreen,
            "compiler_version": (
                None if not rows else rows[0]["fingerprint"]["compiler_version"]
            ),
            "weights_saved": False,
        },
        "data_store_load_s": store.load_elapsed_s,
        "all_dates_materialize_s": materialize_s,
        "summary": _summarize(rows),
        "rows": rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    printed = (
        {
            key: value
            for key, value in payload.items()
            if key != "rows"
        }
        if args.summary_only
        else payload
    )
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Public orchestration layer for the unified portfolio optimizer.

This module owns the lifecycle of one mathematical request: validate, compile,
route to a backend, independently audit the returned vector, optionally try
fallback backends, and construct :class:`OptimizationResult`.  Business model
construction stays in ``portfolio_types``/``model.compiler`` and native solver
details stay in ``backends``.  Keeping that boundary explicit is important:
a backend ``solved`` status is never sufficient to return portfolio weights.

The convenience single-period and sequence methods are stateless facades over
the same immutable :class:`PortfolioProblem`; operational lists supplied for a
live request are not retained on :class:`PortfolioOptimizer`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .backends import (
    BackendOptions,
    ClarabelBackend,
    HighsBackend,
    MosekBackend,
    PIQPBackend,
)
from .backends.base import BackendResult
from .factor_qcqp import solve_factor_qcqp
from .model import CompiledProblem, FactorQCQP, LinearProgram, QuadraticProgram, compile_problem
from .portfolio_types import (
    AlignmentReport,
    AssetTradeConstraints,
    ConstraintViolation,
    FailureReason,
    MaximizeAlpha,
    OptimalityCertificate,
    OptimizationResult,
    PortfolioMetrics,
    PortfolioConstraints,
    PortfolioData,
    PortfolioObjective,
    PortfolioProblem,
    ProofStatus,
    RiskAdjustedAlpha,
    SolveStatus,
    SolveTimings,
    SolverAttempt,
    SolverPolicy,
)
from .solution import evaluate_solution, lift_weights
from .validation import ValidationReport, validate_problem

if TYPE_CHECKING:
    from .data import InMemoryDataSource, PortfolioSchedule
    from .portfolio_types import AlphaSpec, SequencePolicy
    from .sequence import PortfolioSequenceResult


@dataclass(frozen=True)
class PreparedPortfolioProblem:
    """Validated problem plus its solver-independent canonical representation.

    Invalid input is retained with ``compiled=None`` so callers can inspect the
    aggregated :class:`ValidationReport` without paying backend setup cost.
    ``prepare_s`` includes static validation and canonical compilation.  The
    object currently exposes ``compiled`` for review and diagnostics; it is a
    candidate for an opaque handle in a future binary ``_core`` distribution.
    """

    problem: PortfolioProblem
    validation: ValidationReport
    compiled: CompiledProblem | None
    prepare_s: float


@dataclass(frozen=True)
class _Evaluation:
    """Internal result of auditing one backend attempt."""

    metrics: PortfolioMetrics
    violations: tuple
    max_violation: float
    accepted: bool
    validation_s: float


class PortfolioOptimizer:
    """Compile, route, solve and independently validate portfolio problems.

    The instance stores only an immutable solver policy.  It deliberately does
    not accumulate universe, benchmark, holdings, blacklists, workspaces or
    sequence state, so reusing one optimizer across requests is safe.
    """

    def __init__(self, policy: SolverPolicy | None = None):
        self.policy = SolverPolicy() if policy is None else policy

    def validate(self, problem: PortfolioProblem) -> ValidationReport:
        """Aggregate cheap input/model issues without compiling or solving."""

        return validate_problem(problem)

    def prepare(self, problem: PortfolioProblem) -> PreparedPortfolioProblem:
        """Validate and, if valid, compile a request before backend setup.

        Compilation is solver-independent and computes both semantic and
        canonical fingerprints.  Validation is not repeated inside
        ``compile_problem`` because this method has already produced the same
        report.
        """

        started = time.perf_counter()
        report = validate_problem(problem)
        compiled = None
        if report.is_valid:
            compiled = compile_problem(problem, validate=False)
        return PreparedPortfolioProblem(
            problem=problem,
            validation=report,
            compiled=compiled,
            prepare_s=time.perf_counter() - started,
        )

    def solve(
        self,
        problem: PortfolioProblem,
        *,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """Prepare and solve one immutable problem.

        ``theta_seed`` affects only the initial search point of the specialized
        factor-QCQP strategy; it does not alter the mathematical constraints or
        become persistent optimizer state.
        """

        return self.solve_prepared(self.prepare(problem), theta_seed=theta_seed)

    def optimize(
        self,
        *,
        data: PortfolioData,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints | None = None,
        asset_trade: AssetTradeConstraints | None = None,
        blacklist: Iterable[Any] | None = None,
        frozen: Iterable[Any] | None = None,
        not_buyable: Iterable[Any] | None = None,
        not_sellable: Iterable[Any] | None = None,
        weight_overrides: Mapping[Any, float | tuple[float, float]] | None = None,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """Solve one live-trading request without retaining one-off lists.

        ``solve(PortfolioProblem(...))`` remains the canonical low-level API.
        This convenience method makes the common single-period path explicit
        and translates asset lists into immutable problem-local constraints.
        """

        config = PortfolioConstraints() if constraints is None else constraints
        convenience_used = any(
            value is not None
            for value in (
                blacklist,
                frozen,
                not_buyable,
                not_sellable,
                weight_overrides,
            )
        )
        if asset_trade is not None and convenience_used:
            raise ValueError(
                "pass either asset_trade or the individual one-off lists, not both"
            )
        if config.asset_trade is not None and (asset_trade is not None or convenience_used):
            raise ValueError(
                "asset_trade is already present in constraints; do not provide it twice"
            )
        if convenience_used:
            asset_trade = AssetTradeConstraints(
                blacklist=() if blacklist is None else tuple(blacklist),
                frozen=() if frozen is None else tuple(frozen),
                not_buyable=() if not_buyable is None else tuple(not_buyable),
                not_sellable=() if not_sellable is None else tuple(not_sellable),
                weight_overrides={} if weight_overrides is None else weight_overrides,
            )
        if asset_trade is not None:
            config = replace(config, asset_trade=asset_trade)
        return self.solve(
            PortfolioProblem(data=data, objective=objective, constraints=config),
            theta_seed=theta_seed,
        )

    def optimize_range(
        self,
        *,
        data_source: "InMemoryDataSource",
        schedule: "PortfolioSchedule",
        objective: "PortfolioObjective",
        constraints: "PortfolioConstraints",
        alpha_spec: "AlphaSpec | None",
        initial_weight: pd.Series,
        holding_period_returns=None,
        sequence_policy: "SequencePolicy | None" = None,
        independent_initial_weights=None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> "PortfolioSequenceResult":
        """Strictly align an already-loaded schedule and solve it as a sequence.

        Static non-empty asset-trading instructions are rejected because lists
        such as blacklist/frozen are normally effective for one date only.  A
        strategy needing dated instructions must build dated problems rather
        than silently broadcasting one list across the whole range.
        """

        if constraints.asset_trade is not None and not constraints.asset_trade.is_empty:
            raise ValueError(
                "static asset_trade constraints are not accepted by optimize_range; "
                "construct dated PortfolioProblem objects for date-specific instructions"
            )

        prepared_run = data_source.prepare_run(
            schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=extra_attribute_columns,
        )
        return self.solve_sequence(
            prepared_run,
            holding_period_returns=holding_period_returns,
            sequence_policy=sequence_policy,
        )

    def solve_sequence(
        self,
        problems,
        *,
        holding_period_returns=None,
        sequence_policy=None,
    ):
        """Solve an ordered close-to-close portfolio sequence.

        Importing the sequence engine lazily keeps the single-period import path
        small.  Holding drift, failure continuation and theta propagation are
        sequence policies, not backend behavior.
        """

        from .sequence import solve_sequence

        return solve_sequence(
            self,
            problems,
            holding_period_returns=holding_period_returns,
            sequence_policy=sequence_policy,
        )

    def diagnose(
        self,
        problem: PortfolioProblem,
        *,
        prior_result: OptimizationResult | None = None,
        level: str = "deep",
    ):
        """Run an explicit expensive diagnostic for one selected problem.

        Diagnostics are never triggered automatically on the ordinary failure
        path.  When ``prior_result`` is supplied, the diagnostic layer verifies
        its problem fingerprint before using it as evidence.
        """

        from .diagnostics import diagnose_problem

        prepared = self.prepare(problem)
        prepared.validation.raise_for_errors()
        assert prepared.compiled is not None
        return diagnose_problem(
            problem,
            prepared.compiled,
            self.policy,
            prior_result=prior_result,
            level=level,
        )

    def solve_prepared(
        self,
        prepared: PreparedPortfolioProblem,
        *,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """Route one canonical model and audit every primary/fallback attempt.

        Current routing is intentionally explicit: LP to HiGHS, QP to PIQP,
        and factor-QCQP to the specialized frontier strategy.  A rejected QP or
        QCQP attempt first tries configured MOSEK and then Clarabel.  Each
        fallback solves the identical ``CompiledProblem`` and passes through the
        same independent audit; LP currently has no fallback route.
        """

        total_started = time.perf_counter()
        prepared.validation.raise_for_errors()
        if prepared.compiled is None:
            raise RuntimeError("valid prepared problem has no canonical model")
        compiled = prepared.compiled
        model = compiled.model
        options = self._backend_options()

        if isinstance(model, LinearProgram):
            primary = HighsBackend().solve(model, options)
        elif isinstance(model, QuadraticProgram):
            primary = PIQPBackend().solve(model, options)
        elif isinstance(model, FactorQCQP):
            assert prepared.problem.data.alpha_spec is not None
            primary = solve_factor_qcqp(
                model,
                self.policy,
                theta_seed=theta_seed,
                objective_tolerance=self.policy.objective_tolerance.raw_limit(
                    prepared.problem.data.alpha_spec
                ),
            )
        else:
            raise TypeError(f"unsupported canonical model: {type(model).__name__}")

        attempts: list[BackendResult] = []
        primary, evaluation = self._audit_backend_result(prepared, primary)
        attempts.append(primary)
        if not evaluation.accepted and isinstance(model, (QuadraticProgram, FactorQCQP)):
            if self.policy.licensed_fallback.lower() == "mosek":
                fallback = MosekBackend().solve(model, options)
                fallback, evaluation = self._audit_backend_result(prepared, fallback)
                attempts.append(fallback)
            if (
                not evaluation.accepted
                and self.policy.free_fallback.lower() in {"clarabel", "clarabel_qdldl"}
            ):
                fallback = ClarabelBackend().solve(model, options)
                fallback, evaluation = self._audit_backend_result(prepared, fallback)
                attempts.append(fallback)
        return self._result(prepared, attempts, evaluation, total_started)

    def _backend_options(self) -> BackendOptions:
        tuning = self.policy.tuning
        return BackendOptions(
            max_iter=tuning.piqp_max_iter,
            eps_abs=tuning.final_eps,
            eps_rel=tuning.final_eps,
            objective_scale_target=tuning.alpha_target,
            inequality_form=tuning.piqp_inequality_form,
        )

    def _audit_backend_result(
        self,
        prepared: PreparedPortfolioProblem,
        backend_result: BackendResult,
    ) -> tuple[BackendResult, _Evaluation]:
        """Convert a native solver claim into an independently accepted result.

        The audit recomputes the canonical rows, variable bounds, tracking risk
        and business metrics from the returned vector.  A feasible raw solution
        may then undergo the separately audited small-weight cleanup.  Numerical
        rejection is normalized to ``INVALID_NUMERICS`` so routing does not
        depend on backend-specific status vocabulary.
        """

        assert prepared.compiled is not None
        validation_started = time.perf_counter()
        metrics = PortfolioMetrics()
        violations: tuple[ConstraintViolation, ...] = ()
        max_violation = 0.0
        accepted = backend_result.status.has_solution and backend_result.primal is not None
        if accepted:
            assert backend_result.primal is not None
            metrics, violations, max_violation = evaluate_solution(
                prepared.problem,
                prepared.compiled,
                backend_result.primal,
            )
            accepted = max_violation <= self.policy.tuning.feasibility_tolerance
            if not accepted:
                backend_result = replace(
                    backend_result,
                    status=SolveStatus.NUMERICAL_ERROR,
                    reason=FailureReason.INVALID_NUMERICS,
                    message=(
                        f"backend solution failed independent validation: max violation "
                        f"{max_violation:.3e}"
                    ),
                )
            else:
                backend_result, metrics, violations, max_violation = self._clean_solution(
                    prepared,
                    backend_result,
                    metrics,
                    violations,
                    max_violation,
                )
        return backend_result, _Evaluation(
            metrics,
            violations,
            max_violation,
            accepted,
            time.perf_counter() - validation_started,
        )

    def _clean_solution(
        self,
        prepared: PreparedPortfolioProblem,
        backend_result: BackendResult,
        raw_metrics: PortfolioMetrics,
        raw_violations: tuple,
        raw_max_violation: float,
    ) -> tuple[BackendResult, PortfolioMetrics, tuple, float]:
        """Apply the 0.1bp output threshold only when certificates survive.

        Weights with magnitude below ``weight_zero_tolerance`` are set to zero
        and the remaining weights are rescaled to the configured budget.  All
        auxiliary variables are reconstructed from the modified weights before
        constraints and metrics are recomputed.  The cleaned vector is discarded
        if it is infeasible or if its objective loss would push an LP/frontier
        certificate over the caller's objective tolerance.
        """

        assert prepared.compiled is not None
        assert backend_result.primal is not None
        domain = prepared.compiled.model.domain
        raw_weight = backend_result.primal[domain.weight_indices]
        tolerance = self.policy.tuning.weight_zero_tolerance
        small = np.abs(raw_weight) < tolerance
        removed_l1 = float(np.abs(raw_weight[small]).sum())
        if not np.any(small) or removed_l1 == 0.0:
            return backend_result, raw_metrics, raw_violations, raw_max_violation
        cleaned = raw_weight.copy()
        cleaned[small] = 0.0
        cleaned_sum = float(cleaned.sum())
        if abs(cleaned_sum) <= 1e-15:
            return backend_result, raw_metrics, raw_violations, raw_max_violation
        cleaned *= prepared.problem.constraints.budget / cleaned_sum
        cleaned_vector = lift_weights(prepared.problem, prepared.compiled, cleaned)
        metrics, violations, max_violation = evaluate_solution(
            prepared.problem,
            prepared.compiled,
            cleaned_vector,
        )
        diagnostics = dict(backend_result.diagnostics)
        diagnostics.update(
            {
                "weight_cleanup_threshold": tolerance,
                "weight_cleanup_removed_l1": removed_l1,
                "weight_cleanup_count": int(small.sum()),
                "weight_cleanup_applied": max_violation
                <= self.policy.tuning.feasibility_tolerance,
            }
        )
        if max_violation > self.policy.tuning.feasibility_tolerance:
            return (
                replace(backend_result, diagnostics=diagnostics),
                raw_metrics,
                raw_violations,
                raw_max_violation,
            )
        cleanup_loss = 0.0
        if raw_metrics.objective is not None and metrics.objective is not None:
            if isinstance(prepared.problem.objective, (MaximizeAlpha, RiskAdjustedAlpha)):
                cleanup_loss = max(0.0, raw_metrics.objective - metrics.objective)
            else:
                cleanup_loss = max(0.0, metrics.objective - raw_metrics.objective)
        diagnostics["weight_cleanup_objective_loss"] = cleanup_loss
        certificate_sensitive = bool(
            "total_gap" in diagnostics
            or diagnostics.get("lp_prescreen_certified") is True
            or isinstance(prepared.compiled.model, LinearProgram)
        )
        if certificate_sensitive:
            prior_gap = float(diagnostics.get("total_gap", 0.0))
            certified_gap = prior_gap + cleanup_loss
            diagnostics["certified_objective_gap"] = certified_gap
            if "total_gap" in diagnostics:
                diagnostics["total_gap"] = certified_gap
            diagnostics["cleanup_objective_loss"] = cleanup_loss
            alpha_spec = prepared.problem.data.alpha_spec
            if alpha_spec is not None:
                accepted_gap = self.policy.objective_tolerance.raw_limit(alpha_spec)
                if certified_gap > accepted_gap:
                    diagnostics["weight_cleanup_applied"] = False
                    diagnostics["certified_objective_gap"] = prior_gap
                    if "total_gap" in diagnostics:
                        diagnostics["total_gap"] = prior_gap
                    return (
                        replace(backend_result, diagnostics=diagnostics),
                        raw_metrics,
                        raw_violations,
                        raw_max_violation,
                    )
        return (
            replace(backend_result, primal=cleaned_vector, diagnostics=diagnostics),
            metrics,
            violations,
            max_violation,
        )

    def _result(
        self,
        prepared: PreparedPortfolioProblem,
        backend_results: list[BackendResult],
        evaluation: _Evaluation,
        total_started: float,
    ) -> OptimizationResult:
        """Normalize the final attempt into the public immutable result contract.

        Certificate kind reflects the actual route: an LP-prescreen proof,
        factor-frontier Lagrangian estimate, conic primal/dual gap, exact LP
        objective, or a backend estimate.  High-volume theta traces stay out of
        the public route metadata, while aggregate diagnostics remain auditable.
        """

        assert prepared.compiled is not None
        compiled = prepared.compiled
        problem = prepared.problem
        backend_result = backend_results[-1]
        metrics = evaluation.metrics
        violations = evaluation.violations
        max_violation = evaluation.max_violation
        accepted = evaluation.accepted
        status = backend_result.status
        message = backend_result.message

        weights = None
        objective_value = None
        certificate = None
        if accepted and backend_result.primal is not None:
            weight = backend_result.primal[compiled.model.domain.weight_indices]
            weights = pd.Series(
                weight.copy(),
                index=pd.Index(compiled.model.domain.assets, name="sid"),
                name="weight",
            )
            objective_value = metrics.objective
            assert objective_value is not None
            if (
                isinstance(compiled.model, FactorQCQP)
                and backend_result.diagnostics.get("lp_prescreen_certified") is True
            ):
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                certified_gap = float(
                    backend_result.diagnostics.get("certified_objective_gap", 0.0)
                )
                certificate = OptimalityCertificate(
                    kind="lp_global_optimum_feasible_for_factor_qcqp",
                    proof_status=ProofStatus.VERIFIED,
                    primal_value=float(objective_value),
                    dual_bound=float(objective_value) + certified_gap,
                    absolute_gap=certified_gap,
                    normalized_gap=certified_gap / alpha_spec.scale,
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_spec.scale,
                    components={
                        "weight_cleanup_objective_loss": certified_gap,
                        "max_constraint_violation": max_violation,
                    },
                )
            elif isinstance(compiled.model, FactorQCQP) and "total_gap" in backend_result.diagnostics:
                total_gap = float(backend_result.diagnostics["total_gap"])
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                alpha_scale = alpha_spec.scale
                certificate = OptimalityCertificate(
                    kind="factor_qcqp_lagrangian",
                    proof_status=ProofStatus.NUMERICAL_ESTIMATE,
                    primal_value=float(objective_value),
                    dual_bound=float(objective_value) + total_gap,
                    absolute_gap=total_gap,
                    normalized_gap=total_gap / alpha_scale,
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_scale,
                    components={
                        "frontier_slack_gap": float(
                            backend_result.diagnostics["frontier_gap"]
                        ),
                        "qp_subproblem_gap": float(
                            backend_result.diagnostics["subproblem_gap"]
                        ),
                        "max_constraint_violation": max_violation,
                    },
                )
            elif isinstance(compiled.model, FactorQCQP):
                native_gap = backend_result.diagnostics.get("native_gap_unscaled")
                native_gap = None if native_gap is None else float(native_gap)
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                alpha_scale = alpha_spec.scale
                certificate = OptimalityCertificate(
                    kind="conic_primal_dual",
                    proof_status=(
                        ProofStatus.VERIFIED
                        if native_gap is not None
                        else ProofStatus.UNAVAILABLE
                    ),
                    primal_value=float(objective_value),
                    dual_bound=(
                        float(objective_value) + native_gap
                        if native_gap is not None
                        else None
                    ),
                    absolute_gap=native_gap,
                    normalized_gap=(
                        native_gap / alpha_scale if native_gap is not None else None
                    ),
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_scale,
                    components={"max_constraint_violation": max_violation},
                )
            else:
                proof_status = (
                    ProofStatus.VERIFIED
                    if isinstance(compiled.model, LinearProgram)
                    and backend_result.status == SolveStatus.OPTIMAL
                    else ProofStatus.NUMERICAL_ESTIMATE
                )
                certified_gap = float(
                    backend_result.diagnostics.get("certified_objective_gap", 0.0)
                )
                certificate = OptimalityCertificate(
                    kind=(
                        "backend_primal_dual"
                        if proof_status is ProofStatus.VERIFIED
                        else "backend_kkt"
                    ),
                    proof_status=proof_status,
                    primal_value=float(objective_value),
                    dual_bound=(
                        float(objective_value) + certified_gap
                        if proof_status is ProofStatus.VERIFIED
                        else None
                    ),
                    absolute_gap=(
                        certified_gap if proof_status is ProofStatus.VERIFIED else None
                    ),
                    normalized_gap=(
                        certified_gap / problem.data.alpha_spec.scale
                        if proof_status is ProofStatus.VERIFIED
                        and problem.data.alpha_spec is not None
                        else None
                    ),
                    objective_units=(
                        problem.data.alpha_spec.units
                        if problem.data.alpha_spec is not None
                        else "annualized_decimal"
                    ),
                    objective_scale=(
                        problem.data.alpha_spec.scale
                        if problem.data.alpha_spec is not None
                        else None
                    ),
                    components={"max_constraint_violation": max_violation},
                )

        route = tuple(
            SolverAttempt(
                backend=item.backend,
                status=item.status,
                reason=item.reason,
                native_status=item.native_status,
                message=item.message,
                solve_s=item.solve_s,
                recovered=bool(item.diagnostics.get("workspace_rebuilds", 0)),
                metadata={
                    key: value
                    for key, value in item.diagnostics.items()
                    if key != "trace"
                },
            )
            for item in backend_results
        )
        source_dates = {"portfolio": problem.data.date}
        if problem.data.provenance.source_date is not None:
            source_dates["portfolio_source"] = problem.data.provenance.source_date
        if problem.data.risk_model is not None:
            source_dates["risk_model"] = problem.data.risk_model.asof
        metadata = problem.data.provenance.metadata
        for field, label in (
            ("benchmark_source_date", "benchmark"),
            ("alpha_source_date", "alpha"),
        ):
            value = metadata.get(field)
            if value is not None:
                try:
                    source_dates[label] = pd.Timestamp(value)
                except (TypeError, ValueError):
                    pass
        timings = SolveTimings(
            prepare_s=prepared.prepare_s,
            backend_setup_s=sum(item.setup_s for item in backend_results),
            backend_solve_s=sum(item.solve_s for item in backend_results),
            validation_s=evaluation.validation_s,
            total_s=time.perf_counter() - total_started + prepared.prepare_s,
        )
        return OptimizationResult(
            status=status,
            weights=weights,
            objective_value=objective_value,
            backend=backend_result.backend if accepted else None,
            route=route,
            metrics=metrics,
            certificate=certificate,
            violations=violations,
            alignment=AlignmentReport(
                benchmark_missing_mass=_metadata_float(
                    metadata, "benchmark_missing_mass", 0.0
                ),
                benchmark_renormalization_factor=_metadata_float(
                    metadata, "benchmark_renormalization_factor", 1.0
                ),
                holding_missing_mass=_metadata_float(
                    metadata, "holding_missing_mass", 0.0
                ),
                source_dates=source_dates,
            ),
            diagnostics=None,
            timings=timings,
            fingerprint=compiled.fingerprint,
            message=message,
        )


def _metadata_float(metadata, key: str, default: float) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default

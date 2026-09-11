from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optim import (
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioOptimizer,
    PortfolioProblem,
    ProofStatus,
    RiskAdjustedAlpha,
    SolveStatus,
    SolverPolicy,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)
from optim import DataProvenance
from optim._core.backends.piqp import resolve_piqp_inequality_form
from optim._core.backends.base import BackendResult
from optim._core.factor_conic import extend_factor_domain
from optim.portfolio_types import FailureReason
from optim._core.canonical import FactorQCQP, LinearProgram, QuadraticProgram
from optim.model import compile_problem


def test_compiler_classifies_lp_qp_and_factor_qcqp(sample_data, sample_constraints):
    lp = compile_problem(
        PortfolioProblem(sample_data, MaximizeAlpha(), sample_constraints)
    )
    qp = compile_problem(
        PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    )
    qcqp_constraints = replace(
        sample_constraints,
        tracking_error=TrackingErrorLimit(annualized=0.20),
    )
    qcqp = compile_problem(
        PortfolioProblem(sample_data, MaximizeAlpha(), qcqp_constraints)
    )
    assert isinstance(lp.model, LinearProgram)
    assert isinstance(qp.model, QuadraticProgram)
    assert isinstance(qcqp.model, FactorQCQP)


def test_fingerprint_is_stable_and_sensitive_to_alpha(sample_lp_problem):
    first = compile_problem(sample_lp_problem)
    second = compile_problem(sample_lp_problem)
    changed = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=sample_lp_problem.data.alpha + 0.01),
    )
    third = compile_problem(changed)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.semantic_hash != third.fingerprint.semantic_hash
    assert first.fingerprint.canonical_hash != third.fingerprint.canonical_hash


def test_semantic_fingerprint_excludes_nonmathematical_provenance(sample_lp_problem):
    first = compile_problem(sample_lp_problem)
    changed = replace(
        sample_lp_problem,
        data=replace(
            sample_lp_problem.data,
            provenance=DataProvenance(source="tuda2", version="new"),
            risk_model=replace(
                sample_lp_problem.data.risk_model,
                provenance=DataProvenance(source="different_store", version="other"),
            ),
        ),
    )
    second = compile_problem(changed)
    assert first.fingerprint.semantic_hash == second.fingerprint.semantic_hash
    assert first.fingerprint.canonical_hash == second.fingerprint.canonical_hash


def test_highs_lp_end_to_end(sample_lp_problem):
    result = PortfolioOptimizer().solve(sample_lp_problem)
    assert result.status is SolveStatus.OPTIMAL
    weight = result.require_weights()
    assert np.isclose(weight.sum(), 1.0, atol=1e-9)
    assert result.metrics.turnover_l1 <= 0.50 + 1e-9
    assert result.metrics.max_active_weight <= 0.30 + 1e-9
    assert result.metrics.total_active_l1 <= 1.0 + 1e-9
    assert result.metrics.objective == np.dot(sample_lp_problem.data.alpha, weight)
    assert result.certificate is not None
    assert result.violations == ()
    assert result.timings.compile_s > 0.0
    assert result.timings.prepare_s >= 0.0
    assert result.timings.total_s >= (
        result.timings.prepare_s + result.timings.compile_s
    )


def test_direct_piqp_risk_adjusted_qp(sample_data, sample_constraints):
    problem = PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    result = PortfolioOptimizer().solve(problem)
    assert result.status is SolveStatus.OPTIMAL
    weight = result.require_weights()
    assert np.isclose(weight.sum(), 1.0, atol=1e-6)
    assert result.metrics.tracking_error is not None
    assert result.metrics.turnover_l1 <= 0.50 + 1e-6
    assert result.route[0].backend == "piqp"


def test_risk_adjusted_qp_centers_alpha_without_changing_the_portfolio(
    sample_data, sample_constraints
):
    objective = RiskAdjustedAlpha()
    original_problem = PortfolioProblem(sample_data, objective, sample_constraints)
    shifted_problem = PortfolioProblem(
        replace(sample_data, alpha=sample_data.alpha + 7.0),
        objective,
        sample_constraints,
    )
    original_model = compile_problem(original_problem).model
    shifted_model = compile_problem(shifted_problem).model
    assert isinstance(original_model, QuadraticProgram)
    assert isinstance(shifted_model, QuadraticProgram)
    np.testing.assert_allclose(original_model.P.toarray(), shifted_model.P.toarray())
    np.testing.assert_allclose(original_model.q, shifted_model.q)
    assert original_model.objective_scale_reference == pytest.approx(
        shifted_model.objective_scale_reference
    )

    optimizer = PortfolioOptimizer()
    original = optimizer.solve(original_problem)
    shifted = optimizer.solve(shifted_problem)
    np.testing.assert_allclose(
        original.require_weights(), shifted.require_weights(), atol=1e-8
    )
    assert shifted.metrics.objective == pytest.approx(original.metrics.objective + 7.0)


def test_factor_variables_remove_duplicate_dense_exposure_bound_rows(
    sample_data, sample_constraints
):
    qp_model = compile_problem(
        PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    ).model
    assert isinstance(qp_model, QuadraticProgram)
    factor_bounds = tuple(
        record
        for record in qp_model.domain.constraints
        if record.group in {"style", "industry"}
    )
    assert factor_bounds
    assert all(
        qp_model.domain.A.getrow(record.index).nnz == 1 for record in factor_bounds
    )

    qcqp_model = compile_problem(
        PortfolioProblem(
            sample_data,
            MaximizeAlpha(),
            replace(
                sample_constraints,
                tracking_error=TrackingErrorLimit(annualized=0.03),
            ),
        )
    ).model
    assert isinstance(qcqp_model, FactorQCQP)
    extended = extend_factor_domain(qcqp_model)
    extended_bounds = tuple(
        record
        for record in extended.constraints
        if record.group in {"style", "industry"}
    )
    assert all(extended.A.getrow(record.index).nnz == 1 for record in extended_bounds)
    factor_definition_keys = {
        record.key
        for record in extended.constraints
        if record.group == "factor_definition"
    }
    assert factor_definition_keys == set(sample_data.risk_model.factor_names)


def test_minimize_tracking_error_accepts_none_alpha(sample_data, sample_constraints):
    data = replace(sample_data, alpha=None, alpha_spec=None)
    problem = PortfolioProblem(data, MinimizeTrackingError(), sample_constraints)
    result = PortfolioOptimizer().solve(problem)
    assert result.status is SolveStatus.OPTIMAL
    assert result.metrics.tracking_error is not None
    assert result.metrics.tracking_error <= 1e-5


def test_supported_piqp_uses_compact_inequalities_by_default():
    assert resolve_piqp_inequality_form("auto") == "compact"
    assert resolve_piqp_inequality_form("compact") == "compact"
    assert resolve_piqp_inequality_form("one_sided") == "one_sided"


def test_factor_qcqp_clarabel_enforces_annualized_te_budget(
    sample_data, sample_constraints
):
    constraints = replace(
        sample_constraints,
        tracking_error=TrackingErrorLimit(annualized=0.03),
    )
    problem = PortfolioProblem(sample_data, MaximizeAlpha(), constraints)
    result = PortfolioOptimizer().solve(problem)
    assert result.status is SolveStatus.OPTIMAL
    assert result.route[0].backend == "clarabel_qdldl"
    assert result.metrics.tracking_error <= 0.03 + 1e-8
    assert result.certificate is not None
    assert result.certificate.kind == "conic_primal_dual"
    assert result.certificate.absolute_gap <= 1e-4
    # LP prescreen is deliberately disabled by default.
    assert len(result.route) == 1


def test_factor_qcqp_lp_prescreen_is_opt_in_and_returns_a_global_certificate(
    sample_data, sample_constraints
):
    constraints = replace(
        sample_constraints,
        tracking_error=TrackingErrorLimit(annualized=0.20),
    )
    optimizer = PortfolioOptimizer(policy=replace(SolverPolicy(), lp_prescreen=True))
    result = optimizer.solve(
        PortfolioProblem(sample_data, MaximizeAlpha(), constraints)
    )
    assert result.status is SolveStatus.OPTIMAL
    assert result.backend == "factor_qcqp_lp_prescreen_highs"
    assert result.route[0].metadata["lp_prescreen_certified"] is True
    assert result.certificate is not None
    assert result.certificate.kind == "lp_global_optimum_feasible_for_factor_qcqp"
    assert result.certificate.proof_status is ProofStatus.VERIFIED
    assert result.certificate.absolute_gap == 0.0


def test_risky_lp_prescreen_continues_to_clarabel(sample_data, sample_constraints):
    constraints = replace(
        sample_constraints,
        tracking_error=TrackingErrorLimit(annualized=0.03),
    )
    optimizer = PortfolioOptimizer(policy=replace(SolverPolicy(), lp_prescreen=True))
    result = optimizer.solve(
        PortfolioProblem(sample_data, MaximizeAlpha(), constraints)
    )
    assert result.status is SolveStatus.OPTIMAL
    assert result.backend == "clarabel_qdldl"
    assert result.route[0].metadata["lp_prescreen_enabled"] is True
    assert result.route[0].metadata["lp_prescreen_certified"] is False
    assert result.route[0].metadata["lp_prescreen_tracking_error"] > 0.03


def test_invalid_primary_solution_is_rejected_before_fallback(
    sample_data, sample_constraints, monkeypatch
):
    def invalid_piqp(_backend, model, _options):
        return BackendResult(
            backend="piqp",
            status=SolveStatus.OPTIMAL,
            primal=np.zeros(model.domain.n_variables),
            objective_value=0.0,
            native_status="forced_invalid_solution",
        )

    failed_mosek = BackendResult(
        backend="mosek",
        status=SolveStatus.SOLVER_ERROR,
        primal=None,
        objective_value=None,
        native_status="forced_no_license",
        reason=FailureReason.BACKEND_UNAVAILABLE,
    )
    monkeypatch.setattr("optim._core.engine.PIQPBackend.solve", invalid_piqp)
    monkeypatch.setattr(
        "optim._core.engine.MosekBackend.solve", lambda *args, **kwargs: failed_mosek
    )
    result = PortfolioOptimizer().solve(
        PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    )
    assert result.status is SolveStatus.OPTIMAL
    assert result.route[0].status is SolveStatus.NUMERICAL_ERROR
    assert result.route[0].reason is FailureReason.INVALID_NUMERICS
    assert result.backend == "clarabel_qdldl"


def test_sparse_turnover_reformulation_is_conditional_and_exact(
    sample_data, sample_constraints
):
    sparse_initial = np.array([0.5, 0.5, 0.0, 0.0])
    data = replace(sample_data, initial_weight=sparse_initial)
    constraints = replace(
        sample_constraints,
        total_active=None,
        benchmark_member_weight=None,
        turnover=TurnoverLimit(0.50),
    )
    problem = PortfolioProblem(data, MaximizeAlpha(), constraints)
    compiled = compile_problem(problem)
    assert "exact_sparse_turnover" in compiled.compiler_optimizations
    assert compiled.model.domain.n_variables == len(data.assets) + 2
    result = PortfolioOptimizer().solve(problem)
    assert result.status is SolveStatus.OPTIMAL
    assert (
        np.abs(result.require_weights().to_numpy() - sparse_initial).sum()
        <= 0.50 + 1e-9
    )

    short_constraints = replace(
        constraints,
        long_only=False,
        asset_weight=WeightBounds(lower=-1.0, upper=1.0),
    )
    short_compiled = compile_problem(
        PortfolioProblem(data, MaximizeAlpha(), short_constraints)
    )
    assert "exact_sparse_turnover" not in short_compiled.compiler_optimizations
    assert short_compiled.model.domain.n_variables == 2 * len(data.assets)

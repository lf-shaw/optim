"""显式选择只运行一个后端，保留数学问题和独立验收。"""

import pytest

from optim import (
    PortfolioOptimizer,
    SolverPolicy,
    RiskAdjustedAlpha,
    TrackingErrorLimit,
)
from optim._core.backends.base import BackendResult
from optim._core.contracts import CoreSolveStatus, CoreFailureReason
from optim._core import engine


@pytest.mark.parametrize(
    "backend,kind",
    [
        ("highs", "lp"),
        ("piqp", "qp"),
        ("clarabel", "lp"),
        ("clarabel", "qp"),
        ("clarabel", "qcqp"),
        ("mosek", "lp"),
        ("mosek", "qp"),
        ("mosek", "qcqp"),
    ],
)
def test_selected_backend_solves_same_problem(sample_lp_problem, backend, kind):
    problem = sample_lp_problem
    if kind == "qp":
        problem = problem.with_objective(RiskAdjustedAlpha())
    if kind == "qcqp":
        problem = problem.with_constraints(tracking_error=TrackingErrorLimit(0.025))
    baseline = PortfolioOptimizer().solve(problem)
    result = PortfolioOptimizer(SolverPolicy(backend=backend, lp_prescreen=True)).solve(
        problem
    )
    if backend == "mosek" and not result.status.has_solution:
        if (
            result.route[-1].reason is not None
            and "unavailable" in result.route[-1].reason.value
        ):
            pytest.skip("本机未安装 MOSEK 或缺少可用 license")
    assert result.status.has_solution, result.message
    assert result.fingerprint == baseline.fingerprint
    assert len(result.route) == 1
    assert result.backend == ("clarabel_qdldl" if backend == "clarabel" else backend)
    assert result.objective_value == pytest.approx(baseline.objective_value, abs=2e-4)


@pytest.mark.parametrize(
    "backend,kind", [("highs", "qp"), ("piqp", "lp"), ("piqp", "qcqp")]
)
def test_unsupported_model_rejected(sample_lp_problem, backend, kind):
    problem = sample_lp_problem
    if kind == "qp":
        problem = problem.with_objective(RiskAdjustedAlpha())
    if kind == "qcqp":
        problem = problem.with_constraints(tracking_error=TrackingErrorLimit(0.02))
    with pytest.raises(ValueError, match="only supports"):
        PortfolioOptimizer(SolverPolicy(backend=backend)).prepare(problem)


def test_selected_mosek_failure_never_falls_back(monkeypatch, sample_lp_problem):
    def fail(self, model, options):
        return BackendResult(
            backend="mosek",
            status=CoreSolveStatus.NUMERICAL_ERROR,
            primal=None,
            objective_value=None,
            native_status="test_failure",
            reason=CoreFailureReason.INVALID_NUMERICS,
            message="test failure",
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("显式后端不允许转入其他求解器")

    monkeypatch.setattr(engine.MosekBackend, "solve", fail)
    monkeypatch.setattr(engine.ClarabelBackend, "solve", forbidden)
    monkeypatch.setattr(engine.HighsBackend, "solve", forbidden)
    result = PortfolioOptimizer(SolverPolicy(backend="mosek")).solve(sample_lp_problem)
    assert not result.status.has_solution
    assert len(result.route) == 1
    assert result.route[0].backend == "mosek"


def test_diagnose_auxiliary_models_ignore_pinned_backend(sample_lp_problem):
    from optim import TurnoverLimit
    import numpy as np

    problem = (
        sample_lp_problem.with_objective(RiskAdjustedAlpha())
        .with_constraints(turnover=TurnoverLimit(0.0))
        .with_data(initial_weight=np.array([1.0, 0.0, 0.0, 0.0]))
    )
    optimizer = PortfolioOptimizer(SolverPolicy(backend="piqp"))
    result = optimizer.solve(problem)
    assert not result.status.has_solution
    report = optimizer.diagnose(result)
    assert report.linear_feasible is False
    assert report.attempts[0].backend == "highs"


def test_policy_rejects_ignored_selection():
    with pytest.raises(ValueError, match="backend must"):
        SolverPolicy(backend="unknown")
    with pytest.raises(ValueError, match="use backend"):
        SolverPolicy(qp="mosek")


@pytest.mark.parametrize("backend", ["clarabel", "mosek"])
def test_diagnostic_explicit_backend_covers_lp_and_risk_qp(sample_lp_problem, backend):
    import numpy as np
    from optim import TurnoverLimit

    problem = sample_lp_problem.with_constraints(
        turnover=TurnoverLimit(0.0), tracking_error=TrackingErrorLimit(0.001)
    ).with_data(initial_weight=np.array([0.5, 0.3, 0.1, 0.1]))
    optimizer = PortfolioOptimizer()
    report = optimizer.diagnose(problem, backend=backend)
    if backend == "mosek" and any(
        attempt.reason is not None and "unavailable" in attempt.reason.value
        for attempt in report.attempts
    ):
        pytest.skip("本机缺少 MOSEK 或可用 license")
    expected = "clarabel_qdldl" if backend == "clarabel" else backend
    assert all(attempt.backend == expected for attempt in report.attempts)
    assert len(report.attempts) >= 3
    assert report.linear_feasible is True
    assert report.minimum_tracking_error > 0.001
    assert report.turnover_linear_lower_bound is None


@pytest.mark.parametrize("backend", ["unknown", "piqp"])
def test_diagnostic_bad_backend_rejected_before_prepare(
    monkeypatch, sample_lp_problem, backend
):
    optimizer = PortfolioOptimizer()

    def forbidden(*args, **kwargs):
        raise AssertionError("非法选择不应进入编译")

    monkeypatch.setattr(optimizer, "prepare", forbidden)
    with pytest.raises(ValueError):
        optimizer.diagnose(sample_lp_problem, backend=backend)


def test_diagnostic_does_not_inherit_original_model_restriction(sample_lp_problem):
    optimizer = PortfolioOptimizer(SolverPolicy(backend="piqp"))
    report = optimizer.diagnose(sample_lp_problem, backend="highs")
    assert report.linear_feasible is True
    assert all(attempt.backend == "highs" for attempt in report.attempts)

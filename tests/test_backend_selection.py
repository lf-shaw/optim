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


@pytest.mark.parametrize("kind", ["lp", "qp", "qcqp"])
def test_auto_never_calls_mosek(monkeypatch, sample_lp_problem, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("auto 不得调用 MOSEK")

    monkeypatch.setattr(engine.MosekBackend, "solve", forbidden)
    p = sample_lp_problem
    if kind == "qp":
        p = p.with_objective(RiskAdjustedAlpha())
    elif kind == "qcqp":
        p = p.with_constraints(tracking_error=TrackingErrorLimit(0.03))
    r = PortfolioOptimizer().solve(p)
    assert r.status.has_solution
    assert r.backend == {"lp": "highs", "qp": "piqp", "qcqp": "clarabel_qdldl"}[kind]


@pytest.mark.parametrize("kind", ["lp", "qp", "qcqp"])
def test_missing_license_raises_native_error(
    tmp_path, monkeypatch, sample_lp_problem, kind
):
    pytest.importorskip("mosek")
    import subprocess
    import sys
    from optim import export_repro

    p = sample_lp_problem
    if kind == "qp":
        p = p.with_objective(RiskAdjustedAlpha())
    elif kind == "qcqp":
        p = p.with_constraints(tracking_error=TrackingErrorLimit(0.03))
    path = export_repro(tmp_path / "input.zip", result=PortfolioOptimizer().solve(p))
    # Fusion 会在进程内缓存已签出的授权；隔离进程才能真实检验首次缺授权。
    code = """
import sys
from pathlib import Path
from optim import load_repro, PortfolioOptimizer, SolverPolicy
from optim._core.backends import mosek as adapter
adapter._default_license_path = lambda: Path(sys.argv[1]).parent / "missing.lic"
try:
    PortfolioOptimizer(SolverPolicy(backend="mosek")).solve(load_repro(sys.argv[1]).problem)
except RuntimeError as exc:
    assert "MOSEK 不可用" in str(exc), str(exc)
else:
    raise AssertionError("缺授权必须抛异常")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_expired_license_native_code_and_wrapped_error():
    mosek = pytest.importorskip("mosek")
    from optim._core.backends.mosek import _exception_reason

    expired = mosek.Error(
        mosek.rescode.err_license_expired, "arbitrary localized message"
    )
    wrapped = RuntimeError("wrapper without license text")
    wrapped.__cause__ = expired
    assert _exception_reason(wrapped) is CoreFailureReason.BACKEND_UNAVAILABLE
    assert (
        _exception_reason(RuntimeError("license text alone"))
        is CoreFailureReason.NUMERICAL_FAILURE
    )


def test_removed_search_options_are_rejected(sample_lp_problem):
    from optim import SolverTuning, SequencePolicy

    with pytest.raises(TypeError):
        SolverTuning(theta_initial=1.0)
    with pytest.raises(TypeError):
        SequencePolicy(theta_seed="auto")
    with pytest.raises(TypeError):
        SolverPolicy(factor_qcqp_strategy="frontier")
    with pytest.raises(TypeError):
        PortfolioOptimizer().solve(sample_lp_problem, theta_seed=1.0)


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
    assert report.turnover_linear_lower_bound == pytest.approx(0.0, abs=1e-6)


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

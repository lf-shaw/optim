"""诊断下界的共享数学验收和跨后端一致性，不要求原生乘子逐项一致。"""

import numpy as np
import pytest
import scipy.sparse as sp

from optim import PortfolioOptimizer, TurnoverLimit
from optim._core.backends.base import BackendOptions
from optim._core.dual_bounds import lp_dual_bound_diagnostics
from optim._core.backends.highs import HighsBackend
from optim._core.backends.mosek import MosekBackend
from optim._core.backends.clarabel import ClarabelBackend
from optim._core.canonical import (
    CanonicalKind,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
)


def _lp():
    # min x + 7, x >= 2, x >= 0；最优值 9，上界仅由目标子水平集提供。
    domain = LinearDomain(
        sp.csc_matrix([[1.0]]),
        np.array([2.0]),
        np.array([np.inf]),
        np.array([0.0]),
        np.array([np.inf]),
        (),
        (),
        np.array([0]),
        ("x",),
    )
    return LinearProgram(CanonicalKind.LP, domain, np.array([1.0]), 7.0)


def _domain_lp(A, lower, upper, c):
    """构造自由变量测试模型，有限界必须从约束推导，而非依赖持仓假设。"""
    n = len(c)
    domain = LinearDomain(
        sp.csc_matrix(A),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        np.full(n, -np.inf),
        np.full(n, np.inf),
        (),
        (),
        np.arange(n),
        tuple(range(n)),
    )
    return LinearProgram(CanonicalKind.LP, domain, np.asarray(c, dtype=float))


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("sign", [-1, 1])
def test_free_variable_negative_coefficients_and_row_scaling(scale, sign):
    # x = -2，最小化 x；两种系数符号和极不相同的行量级给出同一最优值。
    a = sign * scale
    model = _domain_lp([[a]], [-2 * a], [-2 * a], [1.0])
    result = lp_dual_bound_diagnostics(model, [(1 + 1e-7) / a], [-2.0])
    assert result["dual_lower_bound"] is not None
    assert result["dual_lower_bound"] <= -2 + 1e-10
    assert result["dual_lower_bound"] == pytest.approx(-2.0, abs=1e-6)


def test_degenerate_duplicate_equalities():
    model = _domain_lp([[1.0], [-1.0]], [2, -2], [2, -2], [1.0])
    result = lp_dual_bound_diagnostics(model, [2.0000001, 1.0], [2.0])
    assert result["dual_lower_bound"] == pytest.approx(2.0, abs=1e-6)


def test_uncertifiable_direction_and_near_infeasible_candidate():
    model = _domain_lp([[1.0]], [2], [np.inf], [0.0])
    result = lp_dual_bound_diagnostics(model, [1e-12], [2.0])
    assert result["dual_lower_bound"] is None
    assert result["dual_bound_reason"] == "unbounded_residual_direction"
    near = _domain_lp([[1.0]], [2], [2 - 1e-6], [1.0])
    assert (
        lp_dual_bound_diagnostics(near, [1.000001], [2.0])["dual_lower_bound"] is None
    )


@pytest.mark.parametrize("seed", range(12))
def test_general_sublevel_bound_against_reference_lp(seed):
    from scipy.optimize import linprog

    rng = np.random.default_rng(seed)
    n = 5
    center = rng.normal(size=n)
    extra = rng.normal(size=(7, n))
    A = np.vstack([np.eye(n), extra])
    lower = np.r_[center - 2, extra @ center - rng.uniform(0.1, 2, 7)]
    upper = np.r_[center + 2, extra @ center + rng.uniform(0.1, 2, 7)]
    c = rng.normal(size=n) * (10.0 ** ((seed % 7) - 3))
    model = _domain_lp(A, lower, upper, c)
    reference = linprog(
        c,
        A_ub=np.vstack([A, -A]),
        b_ub=np.r_[upper, -lower],
        bounds=[(None, None)] * n,
        method="highs",
    )
    assert reference.success
    result = lp_dual_bound_diagnostics(model, rng.normal(size=len(lower)), center)
    assert result["dual_lower_bound"] is not None
    assert result["dual_bound_method"] == "sublevel_box_lagrangian"
    assert result["dual_lower_bound"] <= reference.fun + 1e-7 * (1 + abs(reference.fun))


def test_sublevel_bound_never_truncates_unbounded_residual():
    model = _lp()
    # 对偶略高于 1，直接盒界为负无穷。可行但非最优候选 3 只能用作上界，不能当下界。
    direct = lp_dual_bound_diagnostics(model, [1.0001])
    assert direct["dual_lower_bound"] is None
    assert direct["dual_bound_reason"] == "unbounded_residual_direction"
    bounded = lp_dual_bound_diagnostics(model, [1.0001], [3.0])
    assert bounded["dual_bound_method"] == "sublevel_box_lagrangian"
    assert 8.999 < bounded["dual_lower_bound"] < 9.0
    invalid = lp_dual_bound_diagnostics(model, [1.0001], [1.0])
    assert invalid["dual_lower_bound"] is None
    assert lp_dual_bound_diagnostics(model, [np.nan])["dual_lower_bound"] is None


@pytest.mark.parametrize("backend", ["highs", "mosek", "clarabel"])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_native_lp_dual_sign_scale_and_offset(backend, scale):
    model = _lp()
    model = LinearProgram(model.kind, model.domain, model.c * scale, 7.0)
    if backend == "highs":
        result = HighsBackend().solve(model, BackendOptions(collect_dual_bound=True))
    else:
        qp = QuadraticProgram(
            CanonicalKind.QP,
            model.domain,
            sp.csc_matrix((1, 1)),
            model.c,
            objective_offset=model.objective_offset,
        )
        selected = MosekBackend() if backend == "mosek" else ClarabelBackend()
        result = selected.solve(qp, BackendOptions(collect_dual_bound=True))
    if result.reason is not None and "unavailable" in result.reason.value:
        pytest.skip("后端或 license 不可用")
    assert result.status.has_solution
    lower = result.diagnostics["dual_lower_bound"]
    assert lower is not None
    assert lower <= 7 + 2 * scale + 1e-6
    assert lower == pytest.approx(7 + 2 * scale, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("backend", ["auto", "mosek", "clarabel"])
def test_diagnostic_conflict_conclusions_consistent(sample_lp_problem, backend):
    problem = sample_lp_problem.with_constraints(turnover=TurnoverLimit(0)).with_data(
        initial_weight=np.array([1.0, 0.0, 0.0, 0.0])
    )
    report = PortfolioOptimizer().diagnose(problem, backend=backend)
    if any(
        a.reason is not None and "unavailable" in a.reason.value
        for a in report.attempts
    ):
        pytest.skip("后端或 license 不可用")
    assert report.linear_feasible is False
    assert report.turnover_linear_lower_bound == pytest.approx(0.9, abs=1e-5)
    assert report.native_evidence["phase_one_dual_lower_bound"] == pytest.approx(
        0.8, abs=1e-5
    )
    assert all(
        a.metadata["dual_bound_status"] == "numerical_estimate" for a in report.attempts
    )


@pytest.mark.parametrize("backend_type", [HighsBackend, MosekBackend, ClarabelBackend])
def test_regular_solve_skips_dual_bound(backend_type, monkeypatch):
    import importlib

    module = importlib.import_module(backend_type.__module__)

    def forbidden(*args, **kwargs):
        raise AssertionError("普通求解不得计算诊断下界")

    monkeypatch.setattr(module, "lp_dual_bound_diagnostics", forbidden)
    model = _lp()
    if backend_type is not HighsBackend:
        model = QuadraticProgram(
            CanonicalKind.QP, model.domain, sp.csc_matrix((1, 1)), model.c
        )
    result = backend_type().solve(model, BackendOptions())
    assert "dual_lower_bound" not in result.diagnostics

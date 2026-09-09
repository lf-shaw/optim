"""研究原型的代数回归，不注册生产后端；缺少可选 CVXOPT 时跳过。"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

cvxopt = pytest.importorskip("cvxopt")
path = Path(__file__).resolve().parents[1] / "benchmarks/structured_kkt_prototype.py"
spec = importlib.util.spec_from_file_location("structured_kkt_prototype", path)
prototype = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prototype
spec.loader.exec_module(prototype)


def small_cone():
    """三个资产，每资产一个 L1 辅助变量，再加一个全局线性行。"""
    local = np.kron(np.eye(3), np.array([[1.0, -1.0], [-1.0, -1.0]]))
    linear = np.vstack([local, np.eye(6), -np.eye(6), np.ones(6)])
    rng = np.random.default_rng(54)
    factor = rng.normal(size=(2, 6))
    specific = np.diag(np.arange(1, 7) * 0.1)
    R = np.vstack([factor, specific])
    G = np.vstack([linear, np.zeros(6), -R])
    return prototype.ConeProblem(
        G=sp.csc_matrix(G),
        h=np.ones(len(G)),
        A=np.ones((1, 6)),
        b=np.ones(1),
        c=np.zeros(6),
        R=sp.csc_matrix(R),
        nf=2,
        nl=len(linear),
        free=np.ones(6, dtype=bool),
        full_values=np.zeros(6),
        model=None,
    )


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
@pytest.mark.parametrize(
    "method", ["schur", "compressed_augmented", "compressed_augmented_mmd"]
)
def test_structured_matches_dense_scaled_kkt(scale, method):
    """验证 SOC 缩放、右端消元符号及 CVXOPT 要求的 z 返回尺度。"""
    p = small_cone()
    rng = np.random.default_rng(42)
    vr = rng.normal(size=p.R.shape[0]) * scale
    v = np.r_[np.sqrt(1 + vr @ vr), vr]
    d = np.exp(rng.normal(size=p.nl))
    beta = scale
    W = dict(di=cvxopt.matrix(1 / d), beta=[beta], v=[cvxopt.matrix(v)])
    solver = prototype.StructuredKKT(p, method=method)
    solve = solver(W)
    J = np.diag(np.r_[1.0, -np.ones(len(v) - 1)])
    fullW = la.block_diag(np.diag(d), beta * (2 * np.outer(v, v) - J))
    G, A = p.G.toarray(), p.A
    n, ne, m = len(p.c), len(A), len(G)
    K = np.block(
        [
            [np.zeros((n, n)), A.T, G.T @ la.inv(fullW)],
            [A, np.zeros((ne, ne)), np.zeros((ne, m))],
            [G, np.zeros((m, ne)), -fullW],
        ]
    )
    rhs = rng.normal(size=n + ne + m)
    x, y, z = [cvxopt.matrix(a) for a in np.split(rhs, [n, n + ne])]
    solve(x, y, z)
    actual = np.concatenate([np.asarray(a).ravel() for a in (x, y, z)])
    assert np.max(abs(K @ actual - rhs)) / (1 + np.max(abs(rhs))) < 1e-7
    np.testing.assert_allclose(actual, la.solve(K, rhs), rtol=1e-6, atol=1e-7)
    snapshots = prototype.snapshot_comparison(solver)
    assert snapshots[0]["structured_residual"] < 1e-7


def test_time_budget():
    solver = prototype.StructuredKKT(small_cone(), seconds=-1)
    with pytest.raises(TimeoutError):
        solver({})


def test_compressed_hessian_matches_original():
    """共享低秩基底不得改变当前缩放下的 Hessian（浮点误差范围内）。"""
    p = small_cone()
    v = np.r_[np.sqrt(1.0 + 0.04 * p.R.shape[0]), np.full(p.R.shape[0], 0.2)]
    W = dict(di=cvxopt.matrix(np.ones(p.nl)), beta=[0.8], v=[cvxopt.matrix(v)])
    solver = prototype.StructuredKKT(p, method="compressed")
    solver(W)
    B, U, _ = solver.snapshots[0]
    J = np.diag(np.r_[1.0, -np.ones(len(v) - 1)])
    fullW = la.block_diag(np.eye(p.nl), 0.8 * (2 * np.outer(v, v) - J))
    scaled = la.solve(fullW, p.G.toarray())
    expected = scaled.T @ scaled
    np.testing.assert_allclose(
        solver.local.sparse(B).toarray() + U @ U.T, expected, rtol=1e-11, atol=1e-11
    )


def test_risk_cone_matches_canonical(sample_lp_problem):
    """检查真实 compiler 的风险锥方向和基准平移，不凭候选目标相近判断等价。"""
    from optim import TrackingErrorLimit
    from optim.model.compiler import compile_problem

    model = compile_problem(
        sample_lp_problem.with_constraints(tracking_error=TrackingErrorLimit(0.06))
    ).model
    p = prototype.compile_cone(model)
    reduced = np.random.default_rng(31).normal(size=len(p.c))
    full = p.full_values.copy()
    full[p.free] = reduced
    a = full[model.domain.weight_indices] - model.risk_operator.benchmark
    f = model.risk_operator.exposure.T @ a
    risk = f @ model.risk_operator.covariance @ f + np.sum(
        (model.risk_operator.specific_volatility * a) ** 2
    )
    cone = p.h[p.nl :] - p.G[p.nl :] @ reduced
    assert cone[0] == pytest.approx(model.risk_limit)
    assert cone[1:] @ cone[1:] == pytest.approx(risk, rel=1e-12, abs=1e-12)

"""实验表达的数学回归：与完整 epigraph 独立对照，尚未改变生产默认值。"""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from optim import (
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioConstraints,
    PortfolioOptimizer,
    PortfolioProblem,
    RiskAdjustedAlpha,
    SolverPolicy,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)
from optim._impl.compiler import _DomainBuilder
from optim._impl import solver_adapter, compiler as compiler_module

# benchmarks 不属于发布包，按源文件加载，避免与环境中同名包冲突。
_spec = importlib.util.spec_from_file_location(
    "optim_sparse_active_experiment",
    Path(__file__).resolve().parents[1] / "benchmarks/sparse_active_prototype.py",
)
assert _spec is not None and _spec.loader is not None
_prototype = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prototype)
sparse_active_domain = _prototype.sparse_active_domain
split_active_domain = _prototype.split_active_domain


def _build(problem, *, simplify=True):
    """只构造线性域，不调用求解器。"""
    # 保留原型的完整布局参照；生产默认已直接生成紧凑布局，不再对其二次裁剪。
    builder = _DomainBuilder(problem, include_factor_variables=False, simplify=False)
    builder.simplify = simplify
    builder.add_common_constraints()
    return builder, builder.finish()


def _lp(domain, alpha):
    """用独立 SciPy/HiGHS 接口检验投影可行域的线性支撑值。"""
    equal = np.isfinite(domain.lower) & (domain.lower == domain.upper)
    hi = np.isfinite(domain.upper) & ~equal
    lo = np.isfinite(domain.lower) & ~equal
    import scipy.sparse as sp

    c = np.zeros(domain.n_variables)
    c[domain.weight_indices] = -alpha
    return linprog(
        c,
        A_ub=sp.vstack([domain.A[hi], -domain.A[lo]]),
        b_ub=np.r_[domain.upper[hi], -domain.lower[lo]],
        A_eq=domain.A[equal],
        b_eq=domain.upper[equal],
        bounds=list(zip(domain.variable_lower, domain.variable_upper)),
        method="highs",
    )


def test_all_asset_intervals_can_straddle_benchmark(sample_data):
    """每只资产都能高配或低配，与全组合预算固定为 1 并不矛盾。"""
    problem = PortfolioProblem(
        replace(sample_data, initial_weight=None),
        MaximizeAlpha(),
        PortfolioConstraints(
            budget=1.0,
            asset_weight=WeightBounds(0.2, 0.3),
            total_active=0.2,
            freeze_nontradable=False,
        ),
    )
    builder, original = _build(problem)
    sparse = sparse_active_domain(builder, original)
    assert sparse.n_variables == 8  # 四只股票都需要辅助变量，不能按符号消除。
    for i in range(4):
        direction = np.eye(4)[i]
        largest = _lp(sparse, direction)
        smallest = _lp(sparse, -direction)
        assert largest.success and smallest.success
        assert largest.x[i] == pytest.approx(0.3)
        assert smallest.x[i] == pytest.approx(0.2)
        assert largest.x[sparse.weight_indices].sum() == pytest.approx(1.0)
        assert smallest.x[sparse.weight_indices].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("budget", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("long_only", [True, False])
def test_random_support_values(sample_data, budget, long_only):
    """不同预算/卖空和正负固定边界下，比较可行/不可行及多个随机目标的最优值。"""
    rng = np.random.default_rng(20260909)
    for i in range(20):
        benchmark = rng.dirichlet(np.ones(2))
        benchmark = np.r_[benchmark, 0.0, 0.0]
        lower = np.zeros(4) if long_only else np.full(4, -0.6)
        upper = np.full(4, 1.2)
        if i % 3 == 0:
            # 固定资产可能在基准两侧，亦可与基准恰好相等。
            lower[0] = upper[0] = benchmark[0] * (i % 4) / 2
        alpha = rng.normal(size=4)
        data = replace(
            sample_data, benchmark=benchmark, initial_weight=None, alpha=alpha
        )
        constraints = PortfolioConstraints(
            budget=budget,
            long_only=long_only,
            asset_weight=WeightBounds(lower, upper),
            total_active=float(rng.uniform(0.01, 2.0)),
            freeze_nontradable=False,
        )
        p = PortfolioProblem(data, MaximizeAlpha(), constraints)
        builder, original = _build(p)
        ref = _lp(original, alpha)
        for reform in (sparse_active_domain, split_active_domain):
            domain = reform(builder, original)
            actual = _lp(domain, alpha)
            assert actual.status == ref.status
            if ref.success:
                assert actual.fun == pytest.approx(ref.fun, abs=1e-8)
                w = actual.x[domain.weight_indices]
                assert np.abs(w - benchmark).sum() <= constraints.total_active + 1e-8


@pytest.mark.parametrize("kind", ["above", "below", "equal"])
def test_no_auxiliary_and_infeasible_limit(sample_data, kind):
    """空辅助支撑集仍保留正确常数和不可能的主动权重上限。"""
    b = np.full(4, 0.25)
    w = b + {"above": 0.05, "below": -0.05, "equal": 0.0}[kind]
    for limit in (0.0, 0.1, 0.3):
        p = PortfolioProblem(
            replace(sample_data, benchmark=b, initial_weight=None),
            MaximizeAlpha(),
            PortfolioConstraints(
                budget=float(w.sum()),
                asset_weight=WeightBounds(w, w),
                total_active=limit,
                freeze_nontradable=False,
            ),
        )
        builder, original = _build(p)
        sparse = sparse_active_domain(builder, original)
        assert sparse.n_variables == len(b)
        assert (
            _lp(sparse, sample_data.alpha).status
            == _lp(original, sample_data.alpha).status
        )


def test_raw_budget_and_benchmark_offset(sample_data):
    """直接检查编译层恒等式，不把 benchmark.sum 硬编码为 1。

    公共 API 仍要求基准归一化；这里故意绕开公共校验验证数学转换本身。
    """
    b = np.array([0.4, 0.3, 0.0, 0.0])
    w = np.array([0.2, 0.2, 0.3, 0.2])
    p = PortfolioProblem(
        replace(sample_data, benchmark=b, initial_weight=None),
        MaximizeAlpha(),
        PortfolioConstraints(budget=0.9, total_active=0.71, freeze_nontradable=False),
    )
    builder, original = _build(p)
    domain = sparse_active_domain(builder, original)
    z = np.zeros(domain.n_variables)
    z[domain.weight_indices] = w
    for var in domain.variables:
        if var.group == "active_aux":
            i = list(sample_data.assets).index(var.key)
            z[var.index] = max(b[i] - w[i], 0.0)
    row = next(r for r in domain.constraints if r.group == "total_active")
    evaluated = float((domain.A @ z)[row.index] + row.metadata["expression_offset"])
    assert evaluated == pytest.approx(np.abs(w - b).sum(), abs=1e-14)


def test_diagnostic_domain_remains_unsimplified(sample_data):
    p = PortfolioProblem(
        sample_data, MaximizeAlpha(), PortfolioConstraints(total_active=0.2)
    )
    builder, original = _build(p, simplify=False)
    assert sparse_active_domain(builder, original) is original
    assert split_active_domain(builder, original) is original


@pytest.mark.parametrize("scenario", ["lp", "qp", "socp"])
def test_turnover_active_factorial(sample_data, monkeypatch, scenario):
    """四组表达保留相同问题、有效权重边界和业务指纹，且投影最优值一致。"""
    data = replace(
        sample_data,
        benchmark=np.array([0.5, 0.5, 0.0, 0.0]),
        initial_weight=np.array([0.6, 0.0, 0.4, 0.0]),
    )
    problem = PortfolioProblem(
        data,
        RiskAdjustedAlpha() if scenario == "qp" else MaximizeAlpha(),
        PortfolioConstraints(
            total_active=0.6,
            turnover=TurnoverLimit(1.2),
            # 负的显式下界用于检验临时视图不会意外丢失 long_only=True。
            asset_weight=WeightBounds(-0.2, 1.0),
            freeze_nontradable=False,
            tracking_error=TrackingErrorLimit(0.12) if scenario == "socp" else None,
        ),
    )
    original_builder = _DomainBuilder(problem, include_factor_variables=False)
    original_builder.add_common_constraints()
    original_domain = original_builder.finish()
    full_builder = _prototype.FullTurnoverBuilder(
        problem, include_factor_variables=False
    )
    assert full_builder.problem is problem
    assert full_builder.constraints_config is problem.constraints
    assert not full_builder.layout.sparse_turnover
    assert original_builder.layout.sparse_turnover
    assert set(original_builder.optimizations) - set(full_builder.optimizations) == {
        "exact_sparse_turnover"
    }
    full_builder.add_common_constraints()
    full_domain = full_builder.finish()
    for field in ("variable_lower", "variable_upper"):
        np.testing.assert_array_equal(
            getattr(original_domain, field)[original_domain.weight_indices],
            getattr(full_domain, field)[full_domain.weight_indices],
        )
    finish = _DomainBuilder.finish
    results = []
    for builder_class in (_DomainBuilder, _prototype.FullTurnoverBuilder):
        for sparse_active in (False, True):

            def finish_domain(builder):
                domain = finish(builder)
                return (
                    sparse_active_domain(builder, domain) if sparse_active else domain
                )

            with monkeypatch.context() as patcher:
                patcher.setattr(compiler_module, "_DomainBuilder", builder_class)
                patcher.setattr(_DomainBuilder, "finish", finish_domain)
                patcher.setattr(
                    solver_adapter, "lift_weights", _prototype.lift_active_weights
                )
                result = PortfolioOptimizer().solve(problem)
            assert result.status.has_solution
            w = result.require_weights().to_numpy()
            assert w.min() >= -1e-7
            assert np.abs(w - data.benchmark).sum() <= 0.6 + 1e-6
            assert np.abs(w - data.initial_weight).sum() <= 1.2 + 1e-6
            results.append(result)
    assert len({r.fingerprint.semantic_hash for r in results}) == 1
    for result in results[1:]:
        assert result.objective_value == pytest.approx(
            results[0].objective_value, abs=1e-6
        )


@pytest.mark.parametrize(
    "objective", [MaximizeAlpha(), MinimizeTrackingError(), RiskAdjustedAlpha()]
)
@pytest.mark.parametrize("reform", [sparse_active_domain, split_active_domain])
def test_public_lp_qp_and_conic_acceptance(sample_data, monkeypatch, objective, reform):
    """紧 total_active 的 LP、直接 PIQP QP 和 Clarabel 风险预算路径都验收原始业务约束。"""
    constraints = PortfolioConstraints(
        total_active=0.1,
        asset_weight=WeightBounds(0.0, 1.0),
        freeze_nontradable=False,
        tracking_error=TrackingErrorLimit(0.2)
        if isinstance(objective, MaximizeAlpha)
        else None,
    )
    p = PortfolioProblem(sample_data, objective, constraints)
    ref = PortfolioOptimizer().solve(p)
    original = _DomainBuilder.finish

    def finish(builder):
        return reform(builder, original(builder))

    monkeypatch.setattr(_DomainBuilder, "finish", finish)
    monkeypatch.setattr(solver_adapter, "lift_weights", _prototype.lift_active_weights)
    result = PortfolioOptimizer().solve(p)
    assert ref.status.has_solution and result.status.has_solution
    assert result.objective_value == pytest.approx(ref.objective_value, abs=1e-5)
    assert result.metrics.total_active_l1 <= 0.1 + 1e-6
    if isinstance(objective, MaximizeAlpha):
        lp = PortfolioOptimizer(SolverPolicy(backend="highs")).solve(
            p.with_constraints(tracking_error=None)
        )
        assert lp.status.has_solution
        assert lp.metrics.total_active_l1 == pytest.approx(0.1, abs=1e-7)

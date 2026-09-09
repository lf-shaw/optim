"""生产紧凑 L1 编译的等价性、边界条件、原始总量验收及诊断回归。"""

from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

from optim import (
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioConstraints,
    PortfolioOptimizer,
    PortfolioProblem,
    RiskAdjustedAlpha,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)
from optim._impl import compiler
from optim._impl.compiler import _DomainBuilder, _diagnostic_domain
from optim._impl.solution import evaluate_solution, lift_weights
from optim.model import compile_problem


def _domain(problem, simplify=True):
    """独立构造线性域；完整对照不启用任何基于原约束的简化。"""
    builder = _DomainBuilder(problem, include_factor_variables=False, simplify=simplify)
    builder.add_common_constraints()
    return builder, builder.finish()


def _support(domain, alpha):
    """用 SciPy HiGHS 独立比较持仓投影可行域的线性支撑函数。"""
    eq = np.isfinite(domain.lower) & (domain.lower == domain.upper)
    hi, lo = np.isfinite(domain.upper) & ~eq, np.isfinite(domain.lower) & ~eq
    c = np.zeros(domain.n_variables)
    c[domain.weight_indices] = -alpha
    return linprog(
        c,
        A_ub=sp.vstack([domain.A[hi], -domain.A[lo]]),
        b_ub=np.r_[domain.upper[hi], -domain.lower[lo]],
        A_eq=domain.A[eq],
        b_eq=domain.lower[eq],
        bounds=list(zip(domain.variable_lower, domain.variable_upper)),
        method="highs",
    )


@pytest.mark.parametrize("budget", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("long_only", [False, True])
def test_random_compact_domain_matches_full(sample_data, budget, long_only):
    """非单位预算、卖空、零基准和固定资产下，可行性与多个目标的最优值一致。"""
    rng = np.random.default_rng(20260910)
    for i in range(25):
        b = np.r_[rng.dirichlet(np.ones(2)), 0.0, 0.0]
        initial = np.r_[budget * rng.dirichlet(np.ones(2)), 0.0, 0.0]
        lower = np.zeros(4) if long_only else np.full(4, -0.3)
        upper = np.full(4, 1.2)
        if i % 3 == 0:
            lower[2] = upper[2] = 0.05
        p = PortfolioProblem(
            replace(sample_data, benchmark=b, initial_weight=initial),
            MaximizeAlpha(),
            PortfolioConstraints(
                budget=budget,
                long_only=long_only,
                asset_weight=WeightBounds(lower, upper),
                total_active=float(rng.uniform(0.05, 2.5)),
                turnover=TurnoverLimit(float(rng.uniform(0.05, 2.5))),
                freeze_nontradable=False,
            ),
        )
        _, compact = _domain(p)
        _, full = _domain(p, False)
        alpha = rng.normal(size=4)
        left, right = _support(compact, alpha), _support(full, alpha)
        assert left.status == right.status
        if left.success:
            assert left.fun == pytest.approx(right.fun, abs=1e-8)
            w = left.x[compact.weight_indices]
            assert np.abs(w - b).sum() <= p.constraints.total_active + 1e-8
            assert np.abs(w - initial).sum() <= p.constraints.turnover.l1_limit + 1e-8


def test_sparse_layout_uses_resolved_signs_and_preserves_offset(sample_data):
    """只有跨基准资产分配辅助列，确定低配的常数没有丢失。"""
    p = PortfolioProblem(
        replace(sample_data, benchmark=np.array([0.4, 0.3, 0.2, 0.1])),
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(
                np.array([0.0, 0.31, 0.0, 0.1]), np.array([0.3, 0.6, 0.4, 0.1])
            ),
            total_active=0.5,
            freeze_nontradable=False,
        ),
    )
    builder, domain = _domain(p)
    assert domain.n_variables == 5
    np.testing.assert_array_equal(builder.layout.active_support, [2])
    np.testing.assert_array_equal(builder.layout.active_negative, [0])
    assert next(v for v in domain.variables if v.group == "active_aux").key == "c"
    row = next(r for r in domain.constraints if r.group == "total_active")
    assert row.metadata["expression_offset"] == pytest.approx(0.8)


@pytest.mark.parametrize("limit,omitted", [(2.0, True), (2.0 - 1e-13, False)])
def test_redundancy_does_not_use_acceptance_tolerance(sample_data, limit, omitted):
    """严格比较冗余上界，不把略低于可证上界的用户限制当作冗余。"""
    p = PortfolioProblem(
        replace(sample_data, initial_weight=np.array([1.0, 0.0, 0.0, 0.0])),
        MaximizeAlpha(),
        PortfolioConstraints(
            turnover=TurnoverLimit(limit), total_active=limit, freeze_nontradable=False
        ),
    )
    builder, _ = _domain(p)
    assert ("omit_redundant_turnover" in builder.optimizations) == omitted
    assert ("omit_redundant_total_active" in builder.optimizations) == omitted
    diagnostic = _diagnostic_domain(p)
    assert {"turnover", "total_active"} <= {r.group for r in diagnostic.constraints}
    assert diagnostic.n_variables == 12


def test_short_retains_full_active_expression(sample_data):
    p = PortfolioProblem(
        sample_data,
        MaximizeAlpha(),
        PortfolioConstraints(
            long_only=False,
            asset_weight=WeightBounds(-0.5, 1.0),
            total_active=0.4,
            turnover=TurnoverLimit(0.5),
            freeze_nontradable=False,
        ),
    )
    builder, domain = _domain(p)
    assert not builder.layout.sparse_active
    assert not builder.layout.sparse_turnover
    assert domain.n_variables == 12


def test_no_auxiliary_variables_does_not_drop_infeasible_active_bound(sample_data):
    """所有主动符号确定但总量不可行时，零辅助列不等于可以删除约束。"""
    fixed = np.array([0.0, 0.0, 0.5, 0.5])
    p = PortfolioProblem(
        replace(sample_data, benchmark=np.array([0.5, 0.5, 0.0, 0.0])),
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(fixed, fixed),
            total_active=1.9,
            freeze_nontradable=False,
        ),
    )
    builder, domain = _domain(p)
    assert builder.layout.sparse_active
    assert len(builder.layout.active_aux) == 0
    assert any(r.group == "total_active" for r in domain.constraints)
    assert _support(domain, sample_data.alpha).status == 2


def test_final_frozen_negative_bound_disables_nonnegative_shortcuts(sample_data):
    """不能仅根据 long_only 标志推断硬冻结后的资产都非负。"""
    p = PortfolioProblem(
        replace(
            sample_data,
            initial_weight=np.array([-0.1, 0.4, 0.4, 0.3]),
            tradable=np.array([False, True, True, True]),
        ),
        MaximizeAlpha(),
        PortfolioConstraints(total_active=0.8, turnover=TurnoverLimit(0.3)),
    )
    builder, _ = _domain(p)
    assert builder.resolved_bounds.lower[0] == pytest.approx(-0.1)
    assert not builder.layout.sparse_active
    assert not builder.layout.sparse_turnover


def test_redundancy_uses_actual_budget_and_initial_mass(sample_data):
    p = PortfolioProblem(
        sample_data,
        MaximizeAlpha(),
        PortfolioConstraints(
            budget=0.8, turnover=TurnoverLimit(1.8), freeze_nontradable=False
        ),
    )
    builder, _ = _domain(p)
    assert "omit_redundant_turnover" in builder.optimizations
    builder, _ = _domain(p.with_constraints(turnover=TurnoverLimit(1.7)))
    assert "omit_redundant_turnover" not in builder.optimizations


def test_original_turnover_checked_even_when_rows_were_omitted(sample_data):
    """删除冗余行后，略微违反预算的数值候选仍需复算原始换手总量。"""
    p = PortfolioProblem(
        replace(sample_data, initial_weight=np.array([1.0, 0.0, 0.0, 0.0])),
        MaximizeAlpha(),
        PortfolioConstraints(turnover=TurnoverLimit(2.0), freeze_nontradable=False),
    )
    compiled = compile_problem(p)
    assert "omit_redundant_turnover" in compiled.compiler_optimizations
    w = np.array([0.0, 0.5, 0.3, 0.2 + 2e-5])
    _, violations, _ = evaluate_solution(p, compiled, lift_weights(p, compiled, w))
    assert any(
        v.constraint_id == "turnover:l1" and v.amount > 1.9e-5 for v in violations
    )


@pytest.mark.parametrize(
    "objective,risk",
    [
        (MaximizeAlpha(), None),
        (RiskAdjustedAlpha(), None),
        (MinimizeTrackingError(), None),
        (MaximizeAlpha(), TrackingErrorLimit(0.12)),
    ],
)
def test_public_routes_equal_full_model(sample_data, monkeypatch, objective, risk):
    """真实 LP/QP/SOCP 后端对照，包含权重清理后的辅助变量重建。"""
    p = PortfolioProblem(
        replace(sample_data, benchmark=np.array([0.5, 0.5, 0.0, 0.0])),
        objective,
        PortfolioConstraints(
            total_active=0.6,
            turnover=TurnoverLimit(2.0),
            tracking_error=risk,
            freeze_nontradable=False,
        ),
    )
    compact = PortfolioOptimizer().solve(p)
    original = _DomainBuilder

    class FullBuilder(original):
        def __init__(self, problem, *, include_factor_variables, simplify=True):
            super().__init__(
                problem,
                include_factor_variables=include_factor_variables,
                simplify=False,
            )

    with monkeypatch.context() as patcher:
        patcher.setattr(compiler, "_DomainBuilder", FullBuilder)
        full = PortfolioOptimizer().solve(p)
    assert compact.status.has_solution and full.status.has_solution
    assert compact.fingerprint.semantic_hash == full.fingerprint.semantic_hash
    assert compact.objective_value == pytest.approx(full.objective_value, abs=1e-5)
    compiled = compile_problem(p)
    w = compact.require_weights().to_numpy()
    _, _, violation = evaluate_solution(p, compiled, lift_weights(p, compiled, w))
    assert violation <= 1e-5


def test_aggregate_validation_catches_accumulated_epigraph_error(sample_data):
    """每个辅助行只有 4e-6 误差，但真实主动 L1 累计超过 1e-5 时不能通过。"""
    p = PortfolioProblem(
        sample_data,
        MaximizeAlpha(),
        PortfolioConstraints(total_active=0.1, freeze_nontradable=False),
    )
    compiled = compile_problem(p)
    w = np.array([0.225 - 4e-6, 0.225 - 4e-6, 0.275 + 4e-6, 0.275 + 4e-6])
    z = lift_weights(p, compiled, w)
    for record in compiled.model.domain.variables:
        if record.group == "active_aux" and record.key in ("a", "b"):
            z[record.index] -= 4e-6
    metrics, violations, maximum = evaluate_solution(p, compiled, z)
    assert metrics.total_active_l1 == pytest.approx(0.100016)
    assert maximum >= 1.5e-5
    assert any(
        v.constraint_id == "total_active:l1" and v.amount >= 1.5e-5 for v in violations
    )


def test_total_active_must_be_strictly_positive(sample_data):
    p = PortfolioProblem(
        sample_data, MaximizeAlpha(), PortfolioConstraints(total_active=0.0)
    )
    with pytest.raises(ValueError, match="strictly positive"):
        compile_problem(p)


@pytest.mark.parametrize(
    "limit,requested,expected",
    [
        (None, 1e-8, 1e-8),
        (0.1, 1e-8, 1e-10),
        (0.1, 1e-12, 1e-12),
    ],
)
def test_mosek_primal_request_preserves_stricter_user_tolerance(
    sample_data,
    limit,
    requested,
    expected,
):
    """两个 MOSEK 原生接口共享精度处理，不降低用户更严格的精度请求。"""
    from optim._core.backends.base import BackendOptions
    from optim._core.backends.mosek import _primal_feasibility_tolerance

    p = PortfolioProblem(
        sample_data, MaximizeAlpha(), PortfolioConstraints(total_active=limit)
    )
    domain = compile_problem(p).model.domain
    assert (
        _primal_feasibility_tolerance(domain, BackendOptions(eps_abs=requested))
        == expected
    )

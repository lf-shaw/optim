from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optim import (
    AssetTradeConstraints,
    MaximizeAlpha,
    PortfolioOptimizer,
    PortfolioProblem,
    PortfolioValidationError,
    RiskAdjustedAlpha,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
)
from optim.model import compile_problem


def test_single_period_blacklist_is_hard_and_does_not_leak(
    sample_data, sample_constraints
):
    optimizer = PortfolioOptimizer()
    constraints = replace(sample_constraints, turnover=TurnoverLimit(0.50))

    excluded = optimizer.optimize(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=constraints,
        blacklist=["a"],
    )
    assert excluded.require_weights()["a"] == pytest.approx(0.0, abs=1e-10)
    assert excluded.metrics.turnover_l1 == pytest.approx(0.50, abs=1e-8)

    # The same optimizer has no mutable blacklist state.
    ordinary = optimizer.optimize(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=constraints,
    )
    assert ordinary.require_weights()["a"] > 0.25


def test_frozen_and_one_sided_lists_use_initial_weight(sample_data, sample_constraints):
    optimizer = PortfolioOptimizer()
    constraints = replace(sample_constraints, turnover=None)

    frozen = optimizer.optimize(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=constraints,
        frozen=["a"],
    )
    assert frozen.require_weights()["a"] == pytest.approx(0.25, abs=1e-9)

    one_sided = optimizer.optimize(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=constraints,
        not_buyable=["a"],
        not_sellable=["c"],
    )
    weight = one_sided.require_weights()
    assert weight["a"] <= 0.25 + 1e-9
    assert weight["c"] >= 0.25 - 1e-9


def test_weight_override_accepts_exact_and_interval(sample_data, sample_constraints):
    result = PortfolioOptimizer().optimize(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=replace(sample_constraints, turnover=None),
        weight_overrides={"a": 0.20, "b": (0.29, 0.31)},
    )
    weight = result.require_weights()
    assert weight["a"] == pytest.approx(0.20, abs=1e-9)
    assert 0.29 - 1e-9 <= weight["b"] <= 0.31 + 1e-9


def test_operational_list_locally_exempts_active_bound_and_is_auditable(
    sample_data, sample_constraints
):
    constraints = replace(
        sample_constraints,
        active_weight=SymmetricBound(0.10),
        turnover=None,
        asset_trade=AssetTradeConstraints(blacklist=("a",)),
    )
    problem = PortfolioProblem(sample_data, MaximizeAlpha(), constraints)
    compiled = compile_problem(problem)
    record = next(
        item
        for item in compiled.model.domain.constraints
        if item.constraint_id == "asset_bound:a"
    )
    assert compiled.model.domain.variable_lower[0] == 0.0
    assert compiled.model.domain.variable_upper[0] == 0.0
    assert record.metadata["operational_instruction"] == "blacklist"
    assert record.metadata["ordinary_lower"] == pytest.approx(0.15)

    result = PortfolioOptimizer().solve(problem)
    assert result.require_weights()["a"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics.max_active_weight > constraints.active_weight.absolute


def test_common_asset_bound_metadata_is_interned_without_losing_audit_fields(
    sample_data, sample_constraints
):
    ordinary = compile_problem(
        PortfolioProblem(sample_data, MaximizeAlpha(), sample_constraints)
    )
    ordinary_records = tuple(
        item
        for item in ordinary.model.domain.constraints
        if item.group == "asset_bound"
    )
    assert len({id(item.metadata) for item in ordinary_records}) == 1
    assert ordinary_records[0].metadata == {
        "lower_sources": ("asset_weight",),
        "upper_sources": ("asset_weight", "active_weight"),
    }

    constrained = compile_problem(
        PortfolioProblem(
            sample_data,
            MaximizeAlpha(),
            replace(
                sample_constraints,
                asset_trade=AssetTradeConstraints(blacklist=("a",)),
            ),
        )
    )
    records = tuple(
        item
        for item in constrained.model.domain.constraints
        if item.group == "asset_bound"
    )
    # The exceptional operational record owns its audit metadata.  The
    # remaining ordinary assets still share one immutable flyweight.
    assert records[0].metadata["operational_instruction"] == "blacklist"
    assert len({id(item.metadata) for item in records[1:]}) == 1
    assert id(records[0].metadata) != id(records[1].metadata)


def test_asset_trade_changes_both_problem_fingerprints(sample_lp_problem):
    ordinary = compile_problem(sample_lp_problem)
    constrained = compile_problem(
        replace(
            sample_lp_problem,
            constraints=replace(
                sample_lp_problem.constraints,
                asset_trade=AssetTradeConstraints(blacklist=("a",)),
            ),
        )
    )
    assert ordinary.fingerprint.semantic_hash != constrained.fingerprint.semantic_hash
    assert ordinary.fingerprint.canonical_hash != constrained.fingerprint.canonical_hash


def test_same_asset_trade_domain_is_shared_by_lp_qp_and_factor_qcqp(
    sample_data, sample_constraints
):
    asset_trade = AssetTradeConstraints(frozen=("b",))
    variants = (
        (MaximizeAlpha(), replace(sample_constraints, asset_trade=asset_trade)),
        (
            RiskAdjustedAlpha(),
            replace(sample_constraints, asset_trade=asset_trade),
        ),
        (
            MaximizeAlpha(),
            replace(
                sample_constraints,
                tracking_error=TrackingErrorLimit(0.20),
                asset_trade=asset_trade,
            ),
        ),
    )
    for objective, constraints in variants:
        compiled = compile_problem(PortfolioProblem(sample_data, objective, constraints))
        assert compiled.model.domain.variable_lower[1] == pytest.approx(0.25)
        assert compiled.model.domain.variable_upper[1] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "instructions, expected_code",
    [
        (AssetTradeConstraints(blacklist=("missing",)), "unknown_asset"),
        (
            AssetTradeConstraints(blacklist=("a",), frozen=("a",)),
            "overlapping_instructions",
        ),
    ],
)
def test_asset_trade_errors_are_reported_before_backend(
    sample_lp_problem, instructions, expected_code
):
    problem = replace(
        sample_lp_problem,
        constraints=replace(sample_lp_problem.constraints, asset_trade=instructions),
    )
    report = PortfolioOptimizer().validate(problem)
    assert expected_code in {issue.code for issue in report.errors}
    with pytest.raises(PortfolioValidationError):
        PortfolioOptimizer().solve(problem)


def test_nontradable_held_asset_cannot_also_be_blacklisted(
    sample_data, sample_constraints
):
    data = replace(sample_data, tradable=np.array([False, True, True, True]))
    problem = PortfolioProblem(
        data,
        MaximizeAlpha(),
        replace(
            sample_constraints,
            asset_trade=AssetTradeConstraints(blacklist=("a",)),
        ),
    )
    report = PortfolioOptimizer().validate(problem)
    assert any(issue.code == "combined_bounds_infeasible" for issue in report.errors)


def test_range_rejects_static_one_off_asset_lists(sample_data, sample_constraints):
    constraints = replace(
        sample_constraints,
        asset_trade=AssetTradeConstraints(frozen=("a",)),
    )
    optimizer = PortfolioOptimizer()
    with pytest.raises(ValueError, match="dated PortfolioProblem"):
        optimizer.optimize_range(
            data_source=None,  # rejected before data-source access
            schedule=None,
            objective=MaximizeAlpha(),
            constraints=constraints,
            alpha_spec=sample_data.alpha_spec,
            initial_weight=None,
        )


def test_single_period_rejects_asset_lists_provided_twice(sample_data, sample_constraints):
    constraints = replace(
        sample_constraints,
        asset_trade=AssetTradeConstraints(frozen=("a",)),
    )
    with pytest.raises(ValueError, match="do not provide it twice"):
        PortfolioOptimizer().optimize(
            data=sample_data,
            objective=MaximizeAlpha(),
            constraints=constraints,
            blacklist=[],
        )

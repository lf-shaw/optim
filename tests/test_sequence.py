from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optim import (
    MaximizeAlpha,
    AlphaSpec,
    InfeasibilityReport,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    SequenceDataError,
    SequencePolicy,
    SolveStatus,
    TrackingErrorLimit,
    TurnoverLimit,
    TurnoverRecoveryPolicy,
    WeightBounds,
)
from optim.model import compile_problem


def _next_day(problem, date="2026-01-05"):
    risk_model = replace(problem.data.risk_model, asof=pd.Timestamp(date))
    data = replace(
        problem.data,
        date=pd.Timestamp(date),
        risk_model=risk_model,
        # Shape-correct placeholder; chained engine replaces it after drift.
        initial_weight=problem.data.benchmark.copy(),
    )
    return replace(problem, data=data)


def test_chained_sequence_marks_previous_target_to_market(sample_lp_problem):
    second = _next_day(sample_lp_problem)
    returns = pd.Series(
        [0.10, 0.0, 0.0, 0.0],
        index=sample_lp_problem.data.assets,
        name="close_to_close_return",
    )
    result = PortfolioOptimizer().solve_sequence(
        [sample_lp_problem, second],
        holding_period_returns={second.data.date: returns},
        sequence_policy=SequencePolicy(output_weights="sparse"),
    )
    assert result.stopped_date is None
    assert len(result.steps) == 2
    first_target = (
        result.steps[0]
        .result.require_weights()
        .reindex(sample_lp_problem.data.assets, fill_value=0.0)
    )
    assert np.isclose(result.steps[0].result.require_weights().sum(), 1.0)
    expected = first_target * (1.0 + returns)
    expected /= expected.sum()
    observed = result.steps[1].pretrade_weight.reindex(
        sample_lp_problem.data.assets, fill_value=0.0
    )
    assert np.allclose(observed.to_numpy(), expected.to_numpy())
    assert (
        np.allclose(
            observed.to_numpy(),
            second.data.initial_weight,
        )
        is False
    )


def test_sequence_prechecks_all_required_return_dates_before_solving(
    sample_lp_problem, monkeypatch
):
    second = _next_day(sample_lp_problem)
    optimizer = PortfolioOptimizer()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("solver should not run before sequence return precheck")

    monkeypatch.setattr(optimizer, "solve", fail_if_called)
    with pytest.raises(SequenceDataError, match="missing close-to-close"):
        optimizer.solve_sequence([sample_lp_problem, second])
    assert called is False


def test_sequence_does_not_repeat_static_validation_inside_daily_loop(
    sample_lp_problem, monkeypatch
):
    """全区间预检后，每日热路径不得再次扫描完整静态输入。"""

    second = _next_day(sample_lp_problem)
    optimizer = PortfolioOptimizer()
    original_validate = optimizer.validate
    validated_dates = []

    def counted_validate(problem):
        validated_dates.append(problem.data.date)
        return original_validate(problem)

    monkeypatch.setattr(optimizer, "validate", counted_validate)
    result = optimizer.solve_sequence(
        [sample_lp_problem, second],
        holding_period_returns={
            second.data.date: pd.Series(0.0, index=sample_lp_problem.data.assets)
        },
    )
    assert result.stopped_date is None
    assert validated_dates == [sample_lp_problem.data.date, second.data.date]


def test_factor_sequence_uses_clarabel_without_search_state(sample_lp_problem):
    first = replace(
        sample_lp_problem,
        constraints=replace(
            sample_lp_problem.constraints,
            tracking_error=TrackingErrorLimit(0.03),
        ),
    )
    second = _next_day(first)
    zero_returns = pd.Series(0.0, index=first.data.assets)
    result = PortfolioOptimizer().solve_sequence(
        [first, second],
        holding_period_returns={second.data.date: zero_returns},
        sequence_policy=SequencePolicy(output_weights="none"),
    )
    assert all(step.result.backend == "clarabel_qdldl" for step in result.steps)
    assert all(not hasattr(step, "theta_seed") for step in result.steps)
    assert result.steps[1].result.status.has_solution
    assert all(step.result.weights is None for step in result.steps)
    assert all(step.pretrade_weight is None for step in result.steps)


def test_explicit_turnover_recovery_uses_exact_linear_minimum():
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=pd.Index(["old", "new"], name="sid"),
        alpha=np.array([0.0, 1.0]),
        benchmark=np.array([0.0, 1.0]),
        initial_weight=np.array([1.0, 0.0]),
        tradable=np.ones(2, dtype=bool),
        alpha_spec=AlphaSpec(),
    )
    problem = PortfolioProblem(
        data,
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(
                lower=np.array([0.0, 1.0]),
                upper=np.array([0.0, 1.0]),
            ),
            turnover=TurnoverLimit(0.10),
        ),
    )
    sequence = PortfolioOptimizer().solve_sequence(
        [problem],
        sequence_policy=SequencePolicy(
            ignore_first_turnover=False,
            turnover_recovery=TurnoverRecoveryPolicy(max_turnover=2.0),
            output_weights="none",
        ),
    )
    step = sequence.steps[0]
    assert step.result.status.has_solution
    assert step.recovered_turnover is True
    assert step.configured_turnover_limit == pytest.approx(0.10)
    assert step.minimum_feasible_turnover == pytest.approx(2.0)
    assert step.effective_turnover_limit == pytest.approx(2.0)
    assert step.derived_from is not None
    assert step.derived_from != step.result.fingerprint


def test_factor_turnover_recovery_searches_full_convex_feasibility_boundary(
    sample_lp_problem,
):
    problem = replace(
        sample_lp_problem,
        constraints=replace(
            sample_lp_problem.constraints,
            turnover=TurnoverLimit(0.50),
            tracking_error=TrackingErrorLimit(0.03),
        ),
    )
    real_optimizer = PortfolioOptimizer()
    solved_template = real_optimizer.solve(problem)
    assert solved_template.status.has_solution

    class FakeOptimizer:
        policy = real_optimizer.policy

        def __init__(self):
            self.limits = []

        def validate(self, candidate):
            return real_optimizer.validate(candidate)

        def solve(self, candidate):
            limit = candidate.constraints.turnover.l1_limit
            self.limits.append(limit)
            fingerprint = compile_problem(candidate).fingerprint
            if limit < 0.65:
                return replace(
                    solved_template,
                    status=SolveStatus.INFEASIBLE,
                    weights=None,
                    backend=None,
                    fingerprint=fingerprint,
                )
            return replace(solved_template, fingerprint=fingerprint)

        def _solve_prevalidated(self, candidate):
            return self.solve(candidate)

        def diagnose(self, candidate, *, prior_result, level):
            return InfeasibilityReport(
                stage="deep",
                linear_feasible=False,
                summary_text="linear turnover lower bound",
                turnover_linear_lower_bound=0.55,
                turnover_limit=0.50,
            )

    fake = FakeOptimizer()
    sequence = PortfolioOptimizer.solve_sequence(
        fake,
        [problem],
        sequence_policy=SequencePolicy(
            ignore_first_turnover=False,
            turnover_recovery=TurnoverRecoveryPolicy(
                max_turnover=0.80,
                buffer=1e-5,
            ),
            output_weights="none",
        ),
    )
    step = sequence.steps[0]
    assert step.recovered_turnover is True
    assert step.minimum_feasible_turnover == pytest.approx(0.65, abs=2e-5)
    assert step.effective_turnover_limit == pytest.approx(0.65, abs=2e-5)
    assert max(fake.limits) == pytest.approx(0.80)
    assert len(fake.limits) > 3


def test_turnover_recovery_v1_always_resets_next_period():
    with pytest.raises(ValueError, match="always resets"):
        TurnoverRecoveryPolicy(max_turnover=0.20, reset_next_period=False)


def test_first_turnover_default_exclusion_and_second_restoration(sample_lp_problem):
    first = replace(sample_lp_problem, constraints=replace(sample_lp_problem.constraints, turnover=TurnoverLimit(0.0)))
    second = _next_day(first)
    result = PortfolioOptimizer().solve_sequence(
        [first, second], holding_period_returns={second.data.date: np.zeros(4)},
    )
    assert all(step.result.status.has_solution for step in result.steps)
    assert result.steps[0].turnover_excluded
    assert result.steps[0].result.metrics.turnover_l1 is None
    assert not result.steps[1].turnover_excluded
    assert result.steps[1].result.metrics.turnover_l1 == pytest.approx(0.0, abs=1e-7)
    assert not np.allclose(result.steps[0].result.require_weights().reindex(first.data.assets, fill_value=0.0), first.data.initial_weight)
    assert first.constraints.turnover.limit == 0.0
    constrained = PortfolioOptimizer().solve_sequence(
        [first], sequence_policy=SequencePolicy(ignore_first_turnover=False),
    )
    assert constrained.steps[0].result.metrics.turnover_l1 == pytest.approx(0.0, abs=1e-7)


def test_missing_initial_uses_benchmark_and_requires_no_operational_constraints(sample_lp_problem):
    from optim import AssetTradeConstraints

    problem = replace(sample_lp_problem, data=replace(sample_lp_problem.data, initial_weight=None))
    result = PortfolioOptimizer().solve_sequence([problem])
    np.testing.assert_array_equal(result.steps[0].pretrade_weight, problem.data.benchmark)
    assert result.steps[0].result.status.has_solution
    assert problem.data.initial_weight is None
    with pytest.raises(SequenceDataError, match="initial_weight"):
        PortfolioOptimizer().solve_sequence([problem], sequence_policy=SequencePolicy(ignore_first_turnover=False))
    with pytest.raises(SequenceDataError, match="asset_trade"):
        PortfolioOptimizer().solve_sequence([replace(problem, constraints=replace(problem.constraints, asset_trade=AssetTradeConstraints()))])
    blocked = replace(problem, data=replace(problem.data, tradable=np.array([0, 1, 1, 1])))
    initialized = PortfolioOptimizer().solve_sequence([blocked])
    np.testing.assert_allclose(initialized.steps[0].pretrade_weight.reindex(problem.data.assets, fill_value=0), [0, 1/3, 1/3, 1/3])
    assert initialized.steps[0].result.require_weights().get("a", 0.0) == 0.0
    np.testing.assert_array_equal(blocked.data.benchmark, np.full(4, 0.25))
    with pytest.raises(SequenceDataError, match="no positive tradable benchmark"):
        PortfolioOptimizer().solve_sequence([replace(blocked, data=replace(blocked.data, tradable=np.zeros(4, dtype=bool)))])
    unfrozen = replace(blocked, constraints=replace(blocked.constraints, freeze_nontradable=False))
    assert PortfolioOptimizer().solve_sequence([unfrozen]).steps[0].result.status.has_solution


def test_failed_synthetic_initial_cannot_hold_fictitious_portfolio(sample_lp_problem, monkeypatch):
    problem = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, initial_weight=None),
    )
    optimizer = PortfolioOptimizer()
    solved = optimizer.solve(sample_lp_problem)
    monkeypatch.setattr(optimizer, "_solve_prevalidated", lambda candidate: replace(solved, status=SolveStatus.INFEASIBLE, weights=None, problem=candidate))
    result = optimizer.solve_sequence([problem], sequence_policy=SequencePolicy(on_failure="hold"))
    assert result.stopped_date == problem.data.date
    assert result.final_weight is None
    assert result.stopped_problem.constraints.turnover is None


def test_independent_mode_does_not_exclude_first_turnover(sample_lp_problem):
    result = PortfolioOptimizer().solve_sequence([sample_lp_problem], sequence_policy=SequencePolicy(mode="independent"))
    assert not result.steps[0].turnover_excluded
    assert result.steps[0].result.metrics.turnover_l1 is not None


def test_explicit_initial_keeps_nontradable_holdings(sample_lp_problem):
    problem = replace(sample_lp_problem, data=replace(sample_lp_problem.data, tradable=np.array([0, 1, 1, 1])))
    result = PortfolioOptimizer().solve_sequence([problem])
    step = result.steps[0]
    assert step.result.status.has_solution
    np.testing.assert_array_equal(step.pretrade_weight, problem.data.initial_weight)
    assert step.result.require_weights()["a"] == pytest.approx(0.25)
    assert step.result.metrics.turnover_l1 is None

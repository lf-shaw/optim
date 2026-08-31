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
    first_target = result.steps[0].result.require_weights().reindex(
        sample_lp_problem.data.assets, fill_value=0.0
    )
    assert np.isclose(result.steps[0].result.require_weights().sum(), 1.0)
    expected = first_target * (1.0 + returns)
    expected /= expected.sum()
    observed = result.steps[1].pretrade_weight.reindex(
        sample_lp_problem.data.assets, fill_value=0.0
    )
    assert np.allclose(observed.to_numpy(), expected.to_numpy())
    assert np.allclose(
        observed.to_numpy(),
        second.data.initial_weight,
    ) is False


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


def test_factor_sequence_propagates_previous_theta(sample_lp_problem):
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
        sequence_policy=SequencePolicy(theta_seed="auto", output_weights="none"),
    )
    first_theta = result.steps[0].result.route[-1].metadata["theta"]
    assert result.steps[0].theta_seed == PortfolioOptimizer().policy.tuning.theta_initial
    assert result.steps[1].theta_seed == first_theta
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

        def solve(self, candidate, *, theta_seed=None):
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

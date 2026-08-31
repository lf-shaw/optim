"""Deterministic close-to-close multi-period state machine.

The sequence layer is intentionally outside solver backends.  It advances real
portfolio state, chooses a theta seed, applies an explicitly authorized failure
policy and calls the same stateless single-period optimizer on every date.
Returns are used only to mark the previous target portfolio to the next close;
there is no alternative next-open execution mode in the current contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .diagnostics import InfeasibilityReport
from .portfolio_types import (
    OptimizationResult,
    PortfolioProblem,
    ProblemFingerprint,
    SequencePolicy,
    SolveStatus,
    TurnoverLimit,
)


@dataclass(frozen=True)
class SequenceStep:
    """Auditable input/output state for one attempted rebalance date.

    ``pretrade_weight`` is the naturally drifted holding used by turnover, not
    necessarily the prior target.  Recovery fields are populated only when the
    caller explicitly enabled turnover relaxation.
    """

    date: pd.Timestamp
    result: OptimizationResult
    pretrade_weight: pd.Series | None
    theta_seed: float | None
    recovered_turnover: bool = False
    recovery_report: InfeasibilityReport | None = None
    configured_turnover_limit: float | None = None
    minimum_feasible_turnover: float | None = None
    effective_turnover_limit: float | None = None
    derived_from: ProblemFingerprint | None = None


@dataclass(frozen=True)
class PortfolioSequenceResult:
    """Ordered sequence attempts and the final actual portfolio state."""

    steps: tuple[SequenceStep, ...]
    stopped_date: pd.Timestamp | None
    final_weight: pd.Series | None
    policy: SequencePolicy
    schedule_prepare_s: float = 0.0

    def result_for_date(self, date: Any) -> OptimizationResult:
        target = pd.Timestamp(date)
        for step in self.steps:
            if step.date == target:
                return step.result
        raise KeyError(target)


class SequenceDataError(ValueError):
    pass


def solve_sequence(
    optimizer: Any,
    problems: Iterable[PortfolioProblem],
    *,
    holding_period_returns: Mapping[Any, pd.Series | np.ndarray] | None = None,
    sequence_policy: SequencePolicy | None = None,
) -> PortfolioSequenceResult:
    """Solve prevalidated dates while advancing holdings deterministically.

    Static problems and all required holding-period return dates are checked
    before the first expensive solve.  In chained mode each day's placeholder
    initial weight is replaced with the actual marked-to-market portfolio.  A
    successful solve becomes the next target; ``on_failure='hold'`` carries the
    current pretrade portfolio, while the default ``stop`` terminates the run.
    """

    policy = SequencePolicy() if sequence_policy is None else sequence_policy
    from .data.memory import PreparedPortfolioRun

    schedule_prepare_s = 0.0
    if isinstance(problems, PreparedPortfolioRun):
        problems.validation.raise_for_errors()
        dates = [pd.Timestamp(date) for date in problems.dates]
        schedule_prepare_s = problems.prepare_s

        def problem_at(position: int) -> PortfolioProblem:
            return problems.problem_at(dates[position])

        prevalidated = True
    else:
        items = tuple(problems)
        dates = [pd.Timestamp(item.data.date) for item in items]

        def problem_at(position: int) -> PortfolioProblem:
            return items[position]

        prevalidated = False
    if not dates:
        raise SequenceDataError("portfolio sequence must contain at least one problem")
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise SequenceDataError("sequence problem dates must be unique and strictly increasing")

    # Validate every static day and all return dates before the first expensive
    # solve. Later chained initial weights are dynamic, but callers still
    # provide a shape-correct placeholder so all other checks can be completed.
    if not prevalidated:
        for position in range(len(dates)):
            report = optimizer.validate(problem_at(position))
            report.raise_for_errors()
    first_template = problem_at(0)
    if policy.mode == "chained" and first_template.data.initial_weight is None:
        raise SequenceDataError("chained sequence requires first-day initial weights")
    normalized_returns: dict[pd.Timestamp, pd.Series | np.ndarray] = {}
    if holding_period_returns is not None:
        normalized_returns = {
            pd.Timestamp(date): value for date, value in holding_period_returns.items()
        }
    if policy.mode == "chained":
        missing_dates = [date for date in dates[1:] if date not in normalized_returns]
        if missing_dates:
            raise SequenceDataError(
                "missing close-to-close holding-period returns for "
                + ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:10])
            )

    steps: list[SequenceStep] = []
    actual_weight: pd.Series | None = None
    previous_theta: float | None = None
    stopped_date: pd.Timestamp | None = None

    for position in range(len(dates)):
        template = first_template if position == 0 else problem_at(position)
        date = dates[position]
        if policy.mode == "chained":
            if position == 0:
                initial = np.asarray(template.data.initial_weight, dtype=float)
                pretrade = pd.Series(
                    initial,
                    index=pd.Index(template.data.assets, name="sid"),
                    name="weight",
                )
            else:
                assert actual_weight is not None
                pretrade = _mark_to_market(
                    actual_weight,
                    normalized_returns[date],
                    pd.Index(template.data.assets, name="sid"),
                    policy,
                )
            problem = replace(
                template,
                data=replace(template.data, initial_weight=pretrade.to_numpy(copy=False)),
            )
        else:
            problem = template
            pretrade = (
                None
                if template.data.initial_weight is None
                else pd.Series(
                    np.asarray(template.data.initial_weight, dtype=float),
                    index=pd.Index(template.data.assets, name="sid"),
                    name="weight",
                )
            )

        theta_seed = _theta_seed(policy, previous_theta, optimizer.policy.tuning.theta_initial)
        result = optimizer.solve(problem, theta_seed=theta_seed)
        recovered = False
        recovery_report = None
        configured_turnover = None
        minimum_feasible_turnover = None
        effective_turnover = None
        derived_from = None
        if (
            result.status is SolveStatus.INFEASIBLE
            and policy.turnover_recovery is not None
            and problem.constraints.turnover is not None
        ):
            derived_from = result.fingerprint
            (
                result,
                recovered,
                recovery_report,
                configured_turnover,
                minimum_feasible_turnover,
                effective_turnover,
            ) = _recover_turnover(
                optimizer,
                problem,
                result,
                theta_seed,
                policy,
            )

        if result.status.has_solution:
            solved_weight = result.require_weights()
            actual_weight = solved_weight.copy()
            previous_theta = _result_theta(result)
        elif policy.mode == "chained" and policy.on_failure == "hold":
            actual_weight = pretrade.copy() if pretrade is not None else actual_weight
            previous_theta = None
        else:
            stopped_date = date

        stored_result = _apply_output_policy(result, policy)
        steps.append(
            SequenceStep(
                date=date,
                result=stored_result,
                pretrade_weight=_store_weight(pretrade, policy),
                theta_seed=theta_seed,
                recovered_turnover=recovered,
                recovery_report=recovery_report,
                configured_turnover_limit=configured_turnover,
                minimum_feasible_turnover=minimum_feasible_turnover,
                effective_turnover_limit=effective_turnover,
                derived_from=derived_from,
            )
        )
        if stopped_date is not None:
            break

    return PortfolioSequenceResult(
        steps=tuple(steps),
        stopped_date=stopped_date,
        final_weight=None if actual_weight is None else actual_weight.copy(),
        policy=policy,
        schedule_prepare_s=schedule_prepare_s,
    )


def _recover_turnover(
    optimizer: Any,
    problem: PortfolioProblem,
    original_result: OptimizationResult,
    theta_seed: float | None,
    sequence_policy: SequencePolicy,
) -> tuple[
    OptimizationResult,
    bool,
    InfeasibilityReport,
    float,
    float | None,
    float | None,
]:
    """Recover only within an explicitly authorized turnover interval.

    Deep diagnosis first computes a minimum-turnover value for the remaining
    linear domain.  Without a TE constraint that value is exact.  With a TE
    constraint it is only a lower bound, so the routine can try the authorized
    maximum and bisect the monotone turnover upper-bound relaxation.  Every
    returned candidate is still accepted by the ordinary independent validator;
    this path never treats benchmark turnover as the minimum portfolio turnover.
    """

    assert sequence_policy.turnover_recovery is not None
    assert problem.constraints.turnover is not None
    recovery = sequence_policy.turnover_recovery
    configured = problem.constraints.turnover.l1_limit
    report = optimizer.diagnose(problem, prior_result=original_result, level="deep")
    linear_lower = report.turnover_linear_lower_bound
    if (
        linear_lower is None
        or linear_lower > recovery.max_turnover + recovery.buffer
        or recovery.max_turnover <= configured
    ):
        return original_result, False, report, configured, linear_lower, None

    convention = problem.constraints.turnover.convention
    attempts = list(report.attempts)

    def solve_limit(limit: float) -> OptimizationResult:
        candidate = replace(
            problem,
            constraints=replace(
                problem.constraints,
                turnover=TurnoverLimit(limit, convention=convention),
            ),
        )
        candidate_result = optimizer.solve(candidate, theta_seed=theta_seed)
        attempts.extend(candidate_result.route)
        return candidate_result

    # For LP/QP this is the exact minimum of the remaining linear feasible
    # domain.  For factor-QCQP it is only a lower bound, but often already lies
    # close enough to the full convex boundary to recover in one solve.
    lower = max(configured, linear_lower)
    first_limit = min(recovery.max_turnover, lower + recovery.buffer)
    first_result = solve_limit(first_limit)
    if first_result.status.has_solution:
        convex_minimum = first_limit if problem.constraints.tracking_error is not None else linear_lower
        report = _with_convex_turnover(report, convex_minimum, attempts)
        return first_result, True, report, configured, convex_minimum, first_limit

    # Without a TE constraint the linear minimum is exact; a failed full solve
    # at Tmin+buffer is not evidence that a larger turnover relaxation is the
    # right repair.  Leave it as an unrecovered solver/model failure.
    if problem.constraints.tracking_error is None or first_limit >= recovery.max_turnover:
        return first_result, False, replace(report, attempts=tuple(attempts)), configured, linear_lower, first_limit

    maximum_result = solve_limit(recovery.max_turnover)
    if not maximum_result.status.has_solution:
        return maximum_result, False, replace(report, attempts=tuple(attempts)), configured, linear_lower, recovery.max_turnover

    # Feasibility is monotone in a single upper-bound relaxation.  Bisect only
    # on this explicitly enabled exceptional path; all ordinary dates pay no
    # extra solver calls.  Non-success midpoints are conservatively kept on the
    # lower side, while the returned high point is always independently valid.
    low = first_limit
    high = recovery.max_turnover
    high_result = maximum_result
    search_tolerance = max(recovery.buffer, 1e-6)
    for _ in range(24):
        if high - low <= search_tolerance:
            break
        midpoint = 0.5 * (low + high)
        midpoint_result = solve_limit(midpoint)
        if midpoint_result.status.has_solution:
            high = midpoint
            high_result = midpoint_result
        else:
            low = midpoint
    report = _with_convex_turnover(report, high, attempts)
    return high_result, True, report, configured, high, high


def _with_convex_turnover(
    report: InfeasibilityReport,
    minimum: float,
    attempts: list,
) -> InfeasibilityReport:
    addition = f"完整含风险约束模型的可行换手率上界为 {minimum:.4%}。"
    return replace(
        report,
        turnover_convex_minimum=minimum,
        summary_text=report.summary_text + addition,
        attempts=tuple(attempts),
    )


def _mark_to_market(
    previous_target: pd.Series,
    holding_return: pd.Series | np.ndarray,
    current_assets: pd.Index,
    policy: SequencePolicy,
) -> pd.Series:
    """Drift previous target weights by C2C returns and align current assets.

    Missing observations are measured using actual holding mass.  They are not
    silently filled unless the configured tolerance permits the missing mass;
    dropping securities from the new universe also requires explicit
    renormalization permission.
    """

    if isinstance(holding_return, pd.Series):
        if holding_return.index.has_duplicates:
            raise SequenceDataError("holding-period return Series contains duplicate assets")
        aligned_return = holding_return.reindex(previous_target.index)
        missing = aligned_return.isna().to_numpy()
        missing_mass = float(previous_target.to_numpy()[missing].sum())
        if missing_mass > policy.holding_missing_mass_tolerance:
            raise SequenceDataError(
                f"missing close-to-close returns cover {missing_mass:.6%} of holdings"
            )
        aligned_return = aligned_return.fillna(0.0).to_numpy(float)
    else:
        aligned_return = np.asarray(holding_return, dtype=float)
        if aligned_return.shape != (len(previous_target),):
            raise SequenceDataError(
                "array holding-period returns must follow the previous holding coordinate"
            )
    if not np.all(np.isfinite(aligned_return)):
        raise SequenceDataError("holding-period returns contain NaN or infinity")
    gross = 1.0 + aligned_return
    if np.any(gross < 0.0):
        raise SequenceDataError("holding-period return below -100% is invalid")
    drifted = previous_target.to_numpy(float) * gross
    total = float(drifted.sum())
    if total <= 0.0:
        raise SequenceDataError("mark-to-market holdings have no positive value")
    drifted /= total
    drifted_series = pd.Series(drifted, index=previous_target.index, name="weight")
    aligned = drifted_series.reindex(current_assets)
    missing_mass = float(drifted_series.loc[~drifted_series.index.isin(current_assets)].sum())
    if missing_mass > policy.holding_missing_mass_tolerance:
        raise SequenceDataError(
            f"current optimization assets omit {missing_mass:.6%} of drifted holdings"
        )
    aligned = aligned.fillna(0.0)
    retained = float(aligned.sum())
    if not np.isclose(retained, 1.0, rtol=0.0, atol=1e-12):
        if not policy.renormalize_missing_holdings:
            raise SequenceDataError(
                "current assets omit holdings; explicit renormalize_missing_holdings is disabled"
            )
        aligned /= retained
    return aligned.rename("weight")


def _theta_seed(
    policy: SequencePolicy,
    previous: float | None,
    fixed: float,
) -> float | None:
    """Resolve fixed/previous theta policy without carrying a workspace."""

    mode = policy.theta_seed
    if mode == "auto":
        mode = "previous" if policy.mode == "chained" else "fixed"
    if mode == "fixed":
        return fixed
    return fixed if previous is None or not np.isfinite(previous) else previous


def _result_theta(result: OptimizationResult) -> float | None:
    for attempt in reversed(result.route):
        value = attempt.metadata.get("theta")
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _apply_output_policy(
    result: OptimizationResult,
    policy: SequencePolicy,
) -> OptimizationResult:
    if result.weights is None or policy.output_weights == "all":
        return result
    if policy.output_weights == "none":
        return replace(result, weights=None)
    # Numerical cleanup belongs to the single-period optimizer, which lifts
    # and independently revalidates the modified vector.  The sequence output
    # layer only removes exact zeros and must never alter a validated solution.
    sparse = result.weights[result.weights != 0.0]
    return replace(result, weights=sparse)


def _store_weight(
    weight: pd.Series | None,
    policy: SequencePolicy,
) -> pd.Series | None:
    if weight is None or policy.output_weights == "none":
        return None
    if policy.output_weights == "all":
        return weight.copy()
    return weight[weight != 0.0].copy()

"""Explicit, potentially expensive infeasibility diagnostics.

Normal solve failures return immediately and never enter this module.  A caller
selects one failed problem and requests ``level='deep'`` to obtain Phase-I
relaxations, a minimum linear turnover bound and, when meaningful, minimum
tracking error.  Diagnostics reuse canonical registry metadata so evidence is
reported in business constraint names rather than anonymous matrix rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

from .backends import BackendOptions, HighsBackend
from .model.canonical import (
    ConstraintRecord,
    LinearDomain,
    LinearProgram,
    VariableRecord,
)
from .portfolio_types import (
    MinimizeTrackingError,
    OptimizationResult,
    PortfolioProblem,
    ProblemKind,
    SolveStatus,
    SolverAttempt,
    SolverPolicy,
)


@dataclass(frozen=True)
class RequiredRelaxation:
    """One relaxable canonical bound identified by Phase-I."""

    constraint_id: str
    group: str
    side: str
    amount: float
    configured_bound: float
    diagnostic_scale: float
    key: str | None = None


@dataclass(frozen=True)
class InfeasibilityReport:
    """Structured diagnostic evidence, not an automatically applied repair."""

    stage: str
    linear_feasible: bool | None
    summary_text: str
    turnover_linear_lower_bound: float | None = None
    turnover_convex_minimum: float | None = None
    turnover_limit: float | None = None
    minimum_tracking_error: float | None = None
    tracking_error_limit: float | None = None
    relaxations: tuple[RequiredRelaxation, ...] = ()
    native_evidence: Mapping[str, Any] | None = None
    attempts: tuple[SolverAttempt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "native_evidence",
            MappingProxyType({} if self.native_evidence is None else dict(self.native_evidence)),
        )


def diagnose_problem(
    problem: PortfolioProblem,
    compiled: Any,
    policy: SolverPolicy,
    *,
    prior_result: OptimizationResult | None = None,
    level: str = "deep",
) -> InfeasibilityReport:
    """Run all current deep-diagnostic phases for exactly one fingerprint.

    The Phase-I LP keeps compiler-marked hard constraints fixed and minimizes
    scaled slack on relaxable rows/bounds.  Minimum turnover removes only the
    aggregate turnover upper bound; with a nonlinear TE constraint its result is
    a lower bound for the full problem.  Minimum TE is solved only after the
    linear domain is shown feasible.
    """

    if level != "deep":
        raise ValueError("only explicit level='deep' diagnostics are supported")
    if prior_result is not None and prior_result.fingerprint != compiled.fingerprint:
        raise ValueError("prior result fingerprint does not match the diagnosed problem")

    phase_result, relaxations = _solve_phase_one(compiled.model.domain, policy)
    linear_feasible = bool(
        phase_result.status.has_solution
        and sum(item.amount for item in relaxations)
        <= policy.tuning.feasibility_tolerance
    )
    attempts = [_attempt(phase_result, "diagnostic_phase_one")]
    turnover_lower = None
    if problem.constraints.turnover is not None:
        turnover_result, turnover_lower = _minimum_linear_turnover(
            problem,
            compiled,
            policy,
        )
        attempts.append(_attempt(turnover_result, "diagnostic_minimum_turnover"))

    minimum_te = None
    if linear_feasible and problem.data.risk_model is not None and problem.data.benchmark is not None:
        from .api import PortfolioOptimizer

        min_risk_problem = replace(
            problem,
            objective=MinimizeTrackingError(),
            constraints=replace(problem.constraints, tracking_error=None),
        )
        min_risk_result = PortfolioOptimizer(policy).solve(min_risk_problem)
        attempts.extend(min_risk_result.route)
        if min_risk_result.status.has_solution:
            minimum_te = min_risk_result.metrics.tracking_error

    turnover_limit = (
        None
        if problem.constraints.turnover is None
        else problem.constraints.turnover.l1_limit
    )
    tracking_limit = (
        None
        if problem.constraints.tracking_error is None
        else problem.constraints.tracking_error.annualized
    )
    summary = _summary(
        linear_feasible,
        relaxations,
        turnover_lower,
        turnover_limit,
        minimum_te,
        tracking_limit,
    )
    return InfeasibilityReport(
        stage="deep",
        linear_feasible=linear_feasible,
        summary_text=summary,
        turnover_linear_lower_bound=turnover_lower,
        turnover_convex_minimum=None,
        turnover_limit=turnover_limit,
        minimum_tracking_error=minimum_te,
        tracking_error_limit=tracking_limit,
        relaxations=relaxations,
        native_evidence={
            "phase_one_status": phase_result.native_status,
            "phase_one_objective": phase_result.objective_value,
        },
        attempts=tuple(attempts),
    )


def _solve_phase_one(
    domain: LinearDomain,
    policy: SolverPolicy,
) -> tuple[Any, tuple[RequiredRelaxation, ...]]:
    """Construct and solve a weighted-slack LP over relaxable constraints."""

    row_records = {
        item.index: item for item in domain.constraints if item.location == "row"
    }
    variable_records = {
        item.index: item for item in domain.constraints if item.location == "variable"
    }
    row_specs: list[tuple[sp.csc_matrix, float, float, ConstraintRecord | None, str, float]] = []
    for index in range(domain.n_constraints):
        record = row_records.get(index)
        row = domain.A[index].tocsc()
        if record is None or not record.relaxable:
            row_specs.append((row, domain.lower[index], domain.upper[index], record, "hard", 0.0))
            continue
        if np.isfinite(domain.lower[index]):
            row_specs.append((row, domain.lower[index], np.inf, record, "lower", 1.0))
        if np.isfinite(domain.upper[index]):
            row_specs.append((row, -np.inf, domain.upper[index], record, "upper", -1.0))

    variable_lower = domain.variable_lower.copy()
    variable_upper = domain.variable_upper.copy()
    for index in range(domain.n_variables):
        record = variable_records.get(index)
        if record is None or not record.relaxable:
            continue
        unit_row = sp.csc_matrix(
            ([1.0], ([0], [index])), shape=(1, domain.n_variables)
        )
        if np.isfinite(domain.variable_lower[index]):
            row_specs.append((unit_row, domain.variable_lower[index], np.inf, record, "lower", 1.0))
            variable_lower[index] = -np.inf
        if np.isfinite(domain.variable_upper[index]):
            row_specs.append((unit_row, -np.inf, domain.variable_upper[index], record, "upper", -1.0))
            variable_upper[index] = np.inf

    slack_specs = [item for item in row_specs if item[4] != "hard"]
    n_slack = len(slack_specs)
    n_variables = domain.n_variables + n_slack
    base_matrix = sp.vstack([item[0] for item in row_specs], format="csc")
    slack_rows: list[int] = []
    slack_columns: list[int] = []
    slack_values: list[float] = []
    slack_meta: list[tuple[ConstraintRecord, str, float, float]] = []
    slack_index = 0
    for row_index, (_, row_lower, row_upper, record, side, sign) in enumerate(row_specs):
        if side == "hard":
            continue
        assert record is not None
        slack_rows.append(row_index)
        slack_columns.append(slack_index)
        slack_values.append(sign)
        bound = row_lower if side == "lower" else row_upper
        scale = max(1e-8, float(record.diagnostic_scale))
        slack_meta.append((record, side, float(bound), scale))
        slack_index += 1
    slack_matrix = sp.csc_matrix(
        (slack_values, (slack_rows, slack_columns)),
        shape=(len(row_specs), n_slack),
    )
    matrix = sp.hstack([base_matrix, slack_matrix], format="csc")
    phase_lower = np.asarray([item[1] for item in row_specs], dtype=float)
    phase_upper = np.asarray([item[2] for item in row_specs], dtype=float)
    variables = list(domain.variables)
    for index in range(n_slack):
        variables.append(
            VariableRecord(
                index=domain.n_variables + index,
                variable_id=f"diagnostic_slack:{index}",
                group="diagnostic_slack",
                unit="scaled_relaxation",
            )
        )
    phase_domain = LinearDomain(
        A=matrix,
        lower=phase_lower,
        upper=phase_upper,
        variable_lower=np.concatenate([variable_lower, np.zeros(n_slack)]),
        variable_upper=np.concatenate([variable_upper, np.full(n_slack, np.inf)]),
        variables=tuple(variables),
        constraints=(),
        weight_indices=domain.weight_indices,
        assets=domain.assets,
    )
    c = np.zeros(n_variables, dtype=float)
    c[domain.n_variables :] = np.asarray([1.0 / item[3] for item in slack_meta])
    phase_model = LinearProgram(ProblemKind.LP, phase_domain, c)
    result = HighsBackend().solve(phase_model, _backend_options(policy))
    relaxations: list[RequiredRelaxation] = []
    if result.status.has_solution and result.primal is not None:
        values = result.primal[domain.n_variables :]
        for value, (record, side, bound, scale) in zip(values, slack_meta):
            if value > policy.tuning.feasibility_tolerance:
                relaxations.append(
                    RequiredRelaxation(
                        constraint_id=record.constraint_id,
                        group=record.group,
                        side=side,
                        amount=float(value),
                        configured_bound=bound,
                        diagnostic_scale=scale,
                        key=record.key,
                    )
                )
    relaxations.sort(key=lambda item: item.amount / item.diagnostic_scale, reverse=True)
    return result, tuple(relaxations)


def _minimum_linear_turnover(
    problem: PortfolioProblem,
    compiled: Any,
    policy: SolverPolicy,
) -> tuple[Any, float | None]:
    """Minimize L1 turnover after removing only its configured upper bound."""

    domain = compiled.model.domain
    turnover_rows = {
        item.index
        for item in domain.constraints
        if item.location == "row" and item.group == "turnover"
    }
    keep = np.asarray(
        [index not in turnover_rows for index in range(domain.n_constraints)],
        dtype=bool,
    )
    reduced = LinearDomain(
        A=domain.A[keep].tocsc(),
        lower=domain.lower[keep],
        upper=domain.upper[keep],
        variable_lower=domain.variable_lower,
        variable_upper=domain.variable_upper,
        variables=domain.variables,
        constraints=(),
        weight_indices=domain.weight_indices,
        assets=domain.assets,
    )
    c = np.zeros(domain.n_variables, dtype=float)
    turnover_aux = [item for item in domain.variables if item.group == "turnover_aux"]
    if "exact_sparse_turnover" in compiled.compiler_optimizations:
        for item in turnover_aux:
            c[item.index] = 2.0
        assert problem.data.initial_weight is not None
        new_assets = np.asarray(problem.data.initial_weight) <= 0.0
        c[domain.weight_indices[new_assets]] = 2.0
    else:
        for item in turnover_aux:
            c[item.index] = 1.0
    result = HighsBackend().solve(
        LinearProgram(ProblemKind.LP, reduced, c),
        _backend_options(policy),
    )
    value = result.objective_value if result.status.has_solution else None
    return result, value


def _backend_options(policy: SolverPolicy) -> BackendOptions:
    tuning = policy.tuning
    return BackendOptions(
        max_iter=tuning.piqp_max_iter,
        eps_abs=tuning.final_eps,
        eps_rel=tuning.final_eps,
        inequality_form=tuning.piqp_inequality_form,
    )


def _attempt(result: Any, phase: str) -> SolverAttempt:
    return SolverAttempt(
        backend=result.backend,
        status=result.status,
        reason=result.reason,
        native_status=result.native_status,
        message=result.message,
        solve_s=result.solve_s,
        metadata={"diagnostic_phase": phase},
    )


def _summary(
    linear_feasible: bool,
    relaxations: tuple[RequiredRelaxation, ...],
    turnover_lower: float | None,
    turnover_limit: float | None,
    minimum_te: float | None,
    tracking_limit: float | None,
) -> str:
    parts = ["线性约束可行。" if linear_feasible else "问题的线性部分不可行。"]
    if turnover_lower is not None and turnover_limit is not None and turnover_lower > turnover_limit:
        parts.append(
            f"保持其他线性约束时，最小双边换手率为 {turnover_lower:.4%}，"
            f"高于约束 {turnover_limit:.4%}，至少需增加 {turnover_lower-turnover_limit:.4%}。"
        )
    if minimum_te is not None and tracking_limit is not None and minimum_te > tracking_limit:
        parts.append(
            f"线性可行域内最小年化跟踪误差为 {minimum_te:.4%}，"
            f"高于预算 {tracking_limit:.4%}，至少需增加 {minimum_te-tracking_limit:.4%}。"
        )
    if relaxations:
        leading = relaxations[0]
        parts.append(
            f"最大 Phase-I 松弛来自 {leading.constraint_id}，所需松弛 {leading.amount:.6g}。"
        )
    return "".join(parts)

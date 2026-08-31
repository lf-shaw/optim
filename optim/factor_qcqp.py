"""Structure-aware strategy for one factor-model tracking-risk constraint.

PIQP is a QP solver, not a generic SOCP solver.  This module solves the convex
QCQP by exploiting the portfolio risk structure: a low-dimensional factor block
plus diagonal specific risk and exactly one quadratic risk budget.  It builds a
parameterized QP whose matrix/domain stay fixed while theta changes only the
linear objective, then searches the one-dimensional risk/alpha frontier.

An optional HiGHS prescreen is a strict certificate, not a heuristic: the global
alpha optimum over the enclosing linear domain is also the QCQP optimum whenever
that same point satisfies the risk budget.  All public risk values remain in
annualized decimal units; internal positive scaling only preserves the audited
theta/conditioning regime.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse as sp

from .backends.base import BackendOptions, BackendResult
from .backends.highs import HighsBackend
from .backends.piqp import _constraint_data, resolve_piqp_inequality_form
from .model.canonical import (
    ConstraintRecord,
    FactorQCQP,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)
from .portfolio_types import AlphaSpec, FailureReason, ProblemKind, SolveStatus, SolverPolicy


# Public risk inputs use annualized decimal units. The production theta profile
# was calibrated with percentage-point risk, whose variance is 10,000 times
# decimal variance. Scaling the complete parametric risk objective by this
# positive constant preserves the QCQP frontier while retaining the audited
# theta magnitude and PIQP conditioning. Certificate arithmetic explicitly
# includes the same factor and remains in raw alpha units.
_FRONTIER_RISK_OBJECTIVE_SCALE = 10_000.0


@dataclass(frozen=True)
class _FrontierPoint:
    theta: float
    vector: np.ndarray
    variance: float
    alpha_value: float
    qp_gap: float
    recovered: bool


class _WorkspaceFailure(RuntimeError):
    def __init__(self, message: str, *, first_solve: bool, native_status: str):
        super().__init__(message)
        self.first_solve = first_solve
        self.native_status = native_status


class _ParametricPIQPWorkspace:
    """One-day PIQP workspace with a narrowly defined recovery lifecycle.

    ``P``, constraints and bounds are installed once.  Subsequent theta points
    update only ``q`` and warm-start through PIQP.  A first cold solve is never
    repeated with identical inputs.  After at least one successful solve, a
    failed update path may rebuild the current QP once when policy permits; the
    caller otherwise receives a normalized failure and invokes its fallback.
    Workspaces are deliberately not shared across dates.
    """

    def __init__(self, model: QuadraticProgram, policy: SolverPolicy):
        import piqp

        self.piqp = piqp
        self.model = model
        self.policy = policy
        self.form = resolve_piqp_inequality_form(policy.tuning.piqp_inequality_form)
        self.equality, self.equality_rhs, self.inequality, self.inequality_lower, self.inequality_upper = (
            _constraint_data(model, self.form)
        )
        self.solver: Any | None = None
        self.current_q: np.ndarray | None = None
        self.successful_solves = 0
        self.setup_s = 0.0
        self.update_s = 0.0
        self.solve_s = 0.0
        self.qp_solves = 0
        self.qp_iterations = 0
        self.rebuilds = 0
        self.trace: list[dict[str, Any]] = []

    def _new_solver(self, q: np.ndarray, tolerance: float) -> Any:
        solver = self.piqp.SparseSolver()
        solver.settings.verbose = False
        solver.settings.eps_abs = tolerance
        solver.settings.eps_rel = tolerance
        solver.settings.eps_duality_gap_abs = tolerance
        solver.settings.eps_duality_gap_rel = tolerance
        solver.settings.max_iter = self.policy.tuning.piqp_max_iter
        solver.settings.compute_timings = True
        started = time.perf_counter()
        solver.setup(
            self.model.P,
            q,
            self.equality,
            self.equality_rhs,
            self.inequality,
            self.inequality_lower,
            self.inequality_upper,
            self.model.domain.variable_lower,
            self.model.domain.variable_upper,
        )
        self.setup_s += time.perf_counter() - started
        self.current_q = q.copy()
        return solver

    def solve(self, q: np.ndarray, tolerance: float) -> tuple[np.ndarray, float, bool]:
        """Solve/update one theta point and return vector, QP gap and recovery flag."""

        is_first = self.solver is None
        used_update = False
        if self.solver is None:
            self.solver = self._new_solver(q, tolerance)
        else:
            self.solver.settings.eps_abs = tolerance
            self.solver.settings.eps_rel = tolerance
            self.solver.settings.eps_duality_gap_abs = tolerance
            self.solver.settings.eps_duality_gap_rel = tolerance
            if self.current_q is None or not np.array_equal(q, self.current_q):
                update_started = time.perf_counter()
                try:
                    self.solver.update(c=q)
                except Exception as exc:
                    self.update_s += time.perf_counter() - update_started
                    return self._rebuild_and_solve(
                        q,
                        tolerance,
                        trigger=f"update exception: {type(exc).__name__}: {exc}",
                    )
                self.update_s += time.perf_counter() - update_started
                self.current_q = q.copy()
                used_update = True

        try:
            vector, qp_gap, native = self._solve_current()
        except _WorkspaceFailure as failure:
            # The audited policy retries only a failed update lifecycle. A
            # first cold solve is never repeated with identical inputs.
            if used_update and self.policy.rebuild_after_update_failure:
                return self._rebuild_and_solve(
                    q,
                    tolerance,
                    trigger=f"updated solve failed: {failure.native_status}",
                )
            raise
        self.trace.append(
            {
                "phase": "solve",
                "first_solve": is_first,
                "updated": used_update,
                "recovered": False,
                "native_status": native,
                "qp_gap": qp_gap,
                "tolerance": tolerance,
            }
        )
        return vector, qp_gap, False

    def _rebuild_and_solve(
        self,
        q: np.ndarray,
        tolerance: float,
        *,
        trigger: str,
    ) -> tuple[np.ndarray, float, bool]:
        self.rebuilds += 1
        self.solver = self._new_solver(q, tolerance)
        try:
            vector, qp_gap, native = self._solve_current()
        except _WorkspaceFailure:
            self.trace.append(
                {
                    "phase": "cold_rebuild",
                    "trigger": trigger,
                    "recovered": False,
                    "tolerance": tolerance,
                }
            )
            raise
        self.trace.append(
            {
                "phase": "cold_rebuild",
                "trigger": trigger,
                "recovered": True,
                "native_status": native,
                "qp_gap": qp_gap,
                "tolerance": tolerance,
            }
        )
        return vector, qp_gap, True

    def _solve_current(self) -> tuple[np.ndarray, float, str]:
        assert self.solver is not None
        started = time.perf_counter()
        try:
            status = self.solver.solve()
        except Exception as exc:
            self.solve_s += time.perf_counter() - started
            self.qp_solves += 1
            raise _WorkspaceFailure(
                f"PIQP solve exception: {type(exc).__name__}: {exc}",
                first_solve=self.successful_solves == 0,
                native_status="solve_exception",
            ) from exc
        self.solve_s += time.perf_counter() - started
        self.qp_solves += 1
        result = self.solver.result
        info = result.info
        self.qp_iterations += int(getattr(info, "iter", 0) or 0)
        native = str(getattr(info, "status", status))
        if status != self.piqp.Status.PIQP_SOLVED or result.x is None:
            raise _WorkspaceFailure(
                f"PIQP ended with {native}",
                first_solve=self.successful_solves == 0,
                native_status=native,
            )
        self.successful_solves += 1
        qp_gap = _first_number(info, "duality_gap", "dual_gap")
        if qp_gap is None or not np.isfinite(qp_gap):
            qp_gap = math.inf
        return np.asarray(result.x, dtype=float).reshape(-1).copy(), abs(qp_gap), native


def _first_number(value: Any, *names: str) -> float | None:
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            try:
                return float(item)
            except (TypeError, ValueError):
                pass
    return None


def _extend_as_parametric_qp(
    model: FactorQCQP,
) -> tuple[QuadraticProgram, np.ndarray, float, float]:
    r"""Lift a factor-QCQP into a fixed-matrix parameterized QP.

    The base QCQP domain contains portfolio and linear auxiliary variables.  We
    append factor active exposure $f$ and the equality
    $f=E^{\mathsf T}(x-b)$. Dense style/industry rows already present in the
    base domain are replaced by direct one-column bounds on $f$ to avoid
    duplicating $E^{\mathsf T}$ in the KKT system.

    The returned QP encodes half the positively scaled risk objective.  The
    separate ``alpha_solver`` vector is centered and scaled; a theta point uses
    $q(\theta)=q_{\mathrm{risk}}-\theta\widetilde\alpha$. Centering is exact
    under the budget equality and affects conditioning, not the optimizer.
    """

    base = model.domain
    risk = model.risk_operator
    n_base = base.n_variables
    n_factors = risk.exposure.shape[1]
    factor_indices = np.arange(n_base, n_base + n_factors, dtype=np.int32)
    n_variables = n_base + n_factors

    original_rows = sp.hstack(
        [base.A, sp.csc_matrix((base.n_constraints, n_factors))], format="csc"
    )
    extended_lower = base.lower.copy()
    extended_upper = base.upper.copy()
    factor_lookup = {
        name.lower(): index for index, name in enumerate(risk.factor_names)
    }
    active_bound_records = tuple(
        record
        for record in base.constraints
        if record.location == "row"
        and record.group in {"style", "industry"}
        and record.key is not None
    )
    if active_bound_records:
        bound_rows = np.fromiter(
            (record.index for record in active_bound_records), dtype=np.int32
        )
        bound_factors = np.fromiter(
            (factor_lookup[str(record.key).lower()] for record in active_bound_records),
            dtype=np.int32,
        )
        # These canonical rows are $E_j^{\mathsf T}x$ with benchmark-shifted
        # bounds. Once $f=E^{\mathsf T}(x-b)$ is introduced, replace each dense
        # row by the corresponding one-column bound on $f_j$.
        keep = np.ones(base.n_constraints, dtype=float)
        keep[bound_rows] = 0.0
        original_rows = sp.diags(keep, format="csc") @ original_rows
        direct_factor_bounds = sp.csc_matrix(
            (
                np.ones(len(bound_rows)),
                (bound_rows, factor_indices[bound_factors]),
            ),
            shape=(base.n_constraints, n_variables),
        )
        original_rows = (original_rows + direct_factor_bounds).tocsc()
        benchmark_shift = (
            risk.exposure[:, bound_factors].T @ risk.benchmark
        )
        extended_lower[bound_rows] -= benchmark_shift
        extended_upper[bound_rows] -= benchmark_shift
    factor_rows = np.arange(n_factors, dtype=np.int32)
    factor_identity = sp.csc_matrix(
        (-np.ones(n_factors), (factor_rows, factor_indices)),
        shape=(n_factors, n_variables),
    )
    exposure_rows = sp.hstack(
        [
            sp.csc_matrix(risk.exposure.T),
            sp.csc_matrix((n_factors, n_base - len(risk.weight_indices))),
            sp.csc_matrix((n_factors, n_factors)),
        ],
        format="csc",
    )
    factor_definition = exposure_rows + factor_identity
    factor_rhs = risk.exposure.T @ risk.benchmark
    A = sp.vstack([original_rows, factor_definition], format="csc")
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    variables = list(base.variables)
    constraints = list(base.constraints)
    for local_index, (name, index) in enumerate(
        zip(risk.factor_names, factor_indices)
    ):
        variables.append(
            VariableRecord(
                index=int(index),
                variable_id=f"factor_active:{name}",
                group="factor_active",
                key=str(name),
                unit="exposure",
            )
        )
        constraints.append(
            ConstraintRecord(
                constraint_id=f"factor_definition:{name}",
                group="factor_definition",
                location="row",
                index=base.n_constraints + local_index,
                key=str(name),
                unit="exposure",
                source="factor_qcqp_strategy",
                relaxable=False,
            )
        )
    extended_domain = LinearDomain(
        A=A,
        lower=np.concatenate([extended_lower, factor_rhs]),
        upper=np.concatenate([extended_upper, factor_rhs]),
        variable_lower=np.concatenate([base.variable_lower, np.full(n_factors, -np.inf)]),
        variable_upper=np.concatenate([base.variable_upper, np.full(n_factors, np.inf)]),
        variables=tuple(variables),
        constraints=tuple(constraints),
        weight_indices=base.weight_indices,
        assets=base.assets,
    )

    specific_variance = np.square(risk.specific_volatility)
    factor_matrix = risk.covariance
    factor_row, factor_column = np.nonzero(factor_matrix)
    P = sp.csc_matrix(
        (
            np.concatenate([specific_variance, factor_matrix[factor_row, factor_column]]),
            (
                np.concatenate([base.weight_indices, factor_indices[factor_row]]),
                np.concatenate([base.weight_indices, factor_indices[factor_column]]),
            ),
        ),
        shape=(n_variables, n_variables),
    )
    P = P * _FRONTIER_RISK_OBJECTIVE_SCALE
    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    q_base = np.zeros(n_variables, dtype=float)
    q_base[base.weight_indices] = (
        -_FRONTIER_RISK_OBJECTIVE_SCALE * specific_variance * risk.benchmark
    )
    offset = (
        0.5
        * _FRONTIER_RISK_OBJECTIVE_SCALE
        * float(np.sum(specific_variance * np.square(risk.benchmark)))
    )

    alpha = np.asarray(model.alpha, dtype=float)
    centered_alpha = alpha - float(np.mean(alpha))
    max_abs = float(np.max(np.abs(centered_alpha))) if centered_alpha.size else 0.0
    alpha_scale = 1.0 if max_abs <= np.finfo(float).eps else 0.2 / max_abs
    alpha_solver = np.zeros(n_variables, dtype=float)
    alpha_solver[base.weight_indices] = centered_alpha * alpha_scale
    qp = QuadraticProgram(
        kind=ProblemKind.QP,
        domain=extended_domain,
        P=P,
        q=q_base,
        objective_offset=offset,
        risk_operator=risk,
    )
    return qp, alpha_solver, alpha_scale, max_abs


def solve_factor_qcqp(
    model: FactorQCQP,
    policy: SolverPolicy,
    *,
    theta_seed: float | None = None,
    objective_tolerance: float | None = None,
) -> BackendResult:
    """Solve a factor QCQP, optionally certifying it with the enclosing LP.

    If the globally optimal point of the identical linear domain also satisfies
    the tracking-error constraint, it is necessarily globally optimal for the
    QCQP.  This fast path is strictly opt-in through ``policy.lp_prescreen``;
    the default route remains the PIQP frontier.  A failed/unsafe screen is not
    treated as QCQP failure: its time and status are recorded before continuing
    through the same frontier route used when screening is disabled.
    """

    if not policy.lp_prescreen:
        return _solve_factor_qcqp_frontier(
            model,
            policy,
            theta_seed=theta_seed,
            objective_tolerance=objective_tolerance,
        )

    screen, screen_te, certified = _solve_lp_prescreen(model, policy)
    screen_metadata = {
        "lp_prescreen_enabled": True,
        "lp_prescreen_status": screen.status.value,
        "lp_prescreen_native_status": screen.native_status,
        "lp_prescreen_tracking_error": screen_te,
        "lp_prescreen_certified": certified,
    }
    if certified:
        assert screen.primal is not None
        alpha_value = float(
            model.alpha @ screen.primal[model.domain.weight_indices]
        )
        return replace(
            screen,
            backend="factor_qcqp_lp_prescreen_highs",
            objective_value=alpha_value,
            diagnostics={
                **screen.diagnostics,
                **screen_metadata,
                "qp_solves": 0,
                "workspace_rebuilds": 0,
            },
        )

    frontier = _solve_factor_qcqp_frontier(
        model,
        policy,
        theta_seed=theta_seed,
        objective_tolerance=objective_tolerance,
    )
    return replace(
        frontier,
        setup_s=screen.setup_s + frontier.setup_s,
        solve_s=screen.solve_s + frontier.solve_s,
        diagnostics={**frontier.diagnostics, **screen_metadata},
    )


def _solve_lp_prescreen(
    model: FactorQCQP,
    policy: SolverPolicy,
) -> tuple[BackendResult, float | None, bool]:
    """Solve the exact enclosing LP and test a margin-safe TE certificate."""

    c = np.zeros(model.domain.n_variables, dtype=float)
    c[model.domain.weight_indices] = -np.asarray(model.alpha, dtype=float)
    lp = LinearProgram(
        kind=ProblemKind.LP,
        domain=model.domain,
        c=c,
    )
    tuning = policy.tuning
    result = HighsBackend().solve(
        lp,
        BackendOptions(
            max_iter=tuning.piqp_max_iter,
            eps_abs=tuning.final_eps,
            eps_rel=tuning.final_eps,
        ),
    )
    if result.status is not SolveStatus.OPTIMAL or result.primal is None:
        return result, None, False
    variance, _, _ = model.risk_operator.components(result.primal)
    tracking_error = math.sqrt(max(0.0, variance))
    certified = bool(
        np.isfinite(tracking_error)
        and tracking_error <= max(0.0, model.risk_limit - tuning.risk_margin)
    )
    return result, tracking_error, certified


def _solve_factor_qcqp_frontier(
    model: FactorQCQP,
    policy: SolverPolicy,
    *,
    theta_seed: float | None = None,
    objective_tolerance: float | None = None,
) -> BackendResult:
    r"""Solve a factor QCQP through a safeguarded one-dimensional search.

    Search first establishes a risk-feasible/risk-infeasible theta bracket by
    geometric expansion or contraction.  It then applies bounded interpolation
    (falling back to bisection on a flat/non-monotone numerical span), performs a
    high-accuracy final solve and, when necessary, approaches the risk boundary
    again from the safe side. Success requires both risk feasibility and

    $$
    g_{\mathrm{frontier}}+g_{\mathrm{QP}}
    \le\varepsilon_{\mathrm{objective}}.
    $$

    Merely reaching a TE value close to the budget is not a stopping certificate.
    """

    tuning = policy.tuning
    started = time.perf_counter()
    try:
        qp, alpha_solver, alpha_scale, alpha_dispersion = _extend_as_parametric_qp(model)
    except Exception as exc:
        return BackendResult(
            backend="factor_qcqp_piqp",
            status=SolveStatus.SOLVER_ERROR,
            primal=None,
            objective_value=None,
            native_status="compile_error",
            reason=FailureReason.INVALID_NUMERICS,
            message=f"{type(exc).__name__}: {exc}",
        )
    matrix_build_s = time.perf_counter() - started
    workspace = _ParametricPIQPWorkspace(qp, policy)
    theta_start = tuning.theta_initial if theta_seed is None else float(theta_seed)
    if not np.isfinite(theta_start) or theta_start <= 0.0:
        theta_start = tuning.theta_initial
    theta_start = min(theta_start, tuning.theta_max)
    risk_budget_variance = model.risk_limit * model.risk_limit
    target_risk = max(0.0, model.risk_limit - tuning.risk_margin)
    target_variance = target_risk * target_risk
    if objective_tolerance is None:
        objective_tolerance = policy.objective_tolerance.raw_limit(_UNIT_ALPHA_SPEC)
    deadline = None
    # A public total time limit will be added to SolverPolicy rather than
    # overloading PIQP's per-QP iteration controls.

    points = 0

    def solve_theta(theta: float, *, final: bool = False) -> _FrontierPoint:
        nonlocal points
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("factor-QCQP total time limit reached")
        tolerance = tuning.final_eps if final else tuning.intermediate_eps
        if final:
            tolerance = max(tolerance, 1e-7)
        q = qp.q - theta * alpha_solver
        vector, qp_gap, recovered = workspace.solve(q, tolerance)
        base_vector = vector[: model.domain.n_variables].copy()
        variance, _, _ = model.risk_operator.components(base_vector)
        alpha_value = float(model.alpha @ base_vector[model.domain.weight_indices])
        points += 1
        if workspace.trace:
            workspace.trace[-1].update(
                {"theta": theta, "variance": variance, "alpha_value": alpha_value}
            )
        return _FrontierPoint(theta, base_vector, variance, alpha_value, qp_gap, recovered)

    def frontier_gap(point: _FrontierPoint) -> float:
        if point.theta <= 0.0 or alpha_scale <= 0.0:
            return math.inf
        return max(
            0.0,
            _FRONTIER_RISK_OBJECTIVE_SCALE
            * (risk_budget_variance - point.variance)
            / (2.0 * point.theta * alpha_scale),
        )

    try:
        if alpha_dispersion <= np.finfo(float).eps:
            vector, qp_gap, recovered = workspace.solve(qp.q, tuning.final_eps)
            base_vector = vector[: model.domain.n_variables].copy()
            variance, _, _ = model.risk_operator.components(base_vector)
            if variance > (model.risk_limit + tuning.feasibility_tolerance) ** 2:
                return _failure_result(
                    workspace,
                    matrix_build_s,
                    SolveStatus.INFEASIBLE,
                    FailureReason.INFEASIBLE_REPORTED,
                    "minimum-risk QP exceeds the tracking-error budget",
                )
            return _success_result(
                model,
                workspace,
                matrix_build_s,
                _FrontierPoint(0.0, base_vector, variance, float(model.alpha @ base_vector[model.domain.weight_indices]), qp_gap, recovered),
                theta_start,
                0.0,
                qp_gap,
                objective_tolerance,
                points + 1,
            )

        point = solve_theta(theta_start)
        low: _FrontierPoint | None = None
        high: _FrontierPoint | None = None
        if point.variance <= target_variance:
            low = point
            theta = theta_start
            while points < tuning.max_outer_iters and frontier_gap(low) > objective_tolerance:
                theta *= tuning.theta_growth
                if theta > tuning.theta_max:
                    break
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                else:
                    high = point
                    break
        else:
            high = point
            theta = theta_start
            while points < tuning.max_outer_iters:
                theta /= tuning.theta_growth
                if theta <= np.finfo(float).eps:
                    break
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                    break
                high = point
            if low is None:
                vector, qp_gap, recovered = workspace.solve(qp.q, tuning.final_eps)
                base_vector = vector[: model.domain.n_variables].copy()
                minimum_variance, _, _ = model.risk_operator.components(base_vector)
                if minimum_variance > (model.risk_limit + tuning.feasibility_tolerance) ** 2:
                    return _failure_result(
                        workspace,
                        matrix_build_s,
                        SolveStatus.INFEASIBLE,
                        FailureReason.INFEASIBLE_REPORTED,
                        "minimum-risk QP exceeds the tracking-error budget",
                    )
                # The minimum-risk endpoint is feasible. Use it as a theta=0
                # bracket endpoint but never compute a frontier gap at zero.
                low = _FrontierPoint(
                    0.0,
                    base_vector,
                    minimum_variance,
                    float(model.alpha @ base_vector[model.domain.weight_indices]),
                    qp_gap,
                    recovered,
                )

        assert low is not None
        if high is not None:
            while points < tuning.max_outer_iters:
                span = high.theta - low.theta
                if span <= 1e-12 * max(1.0, high.theta):
                    break
                variance_span = high.variance - low.variance
                if variance_span <= 0.0:
                    theta = low.theta + 0.5 * span
                else:
                    theta = low.theta + (
                        (target_variance - low.variance) * span / variance_span
                    )
                theta = min(high.theta - 0.1 * span, max(low.theta + 0.1 * span, theta))
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                else:
                    high = point
                if low.theta > 0.0 and frontier_gap(low) <= objective_tolerance:
                    break

        if low.theta <= 0.0:
            return _failure_result(
                workspace,
                matrix_build_s,
                SolveStatus.LIMIT_REACHED,
                FailureReason.MAX_ITER,
                "could not establish a positive-theta feasible frontier point",
            )
        final = solve_theta(low.theta, final=True)
        if final.variance > risk_budget_variance:
            risky_final = final
            theta = final.theta
            safe_final: _FrontierPoint | None = None
            for _ in range(12):
                theta /= tuning.theta_growth
                if theta <= np.finfo(float).eps:
                    break
                candidate = solve_theta(theta, final=True)
                if candidate.variance <= target_variance:
                    safe_final = candidate
                    break
                risky_final = candidate
            if safe_final is None:
                return _failure_result(
                    workspace,
                    matrix_build_s,
                    SolveStatus.NUMERICAL_ERROR,
                    FailureReason.NUMERICAL_FAILURE,
                    "could not recover a high-accuracy risk-feasible frontier point",
                )
            for _ in range(15):
                theta_span = risky_final.theta - safe_final.theta
                if theta_span <= 1e-12 * max(1.0, risky_final.theta):
                    break
                variance_span = risky_final.variance - safe_final.variance
                if variance_span <= 0.0:
                    theta = safe_final.theta + 0.5 * theta_span
                else:
                    theta = safe_final.theta + (
                        (target_variance - safe_final.variance)
                        * theta_span
                        / variance_span
                    )
                theta = min(
                    risky_final.theta - 0.05 * theta_span,
                    max(safe_final.theta + 0.05 * theta_span, theta),
                )
                candidate = solve_theta(theta, final=True)
                if candidate.variance <= target_variance:
                    safe_final = candidate
                    candidate_subproblem_gap = candidate.qp_gap / (
                        candidate.theta * alpha_scale
                    )
                    if (
                        frontier_gap(candidate) + candidate_subproblem_gap
                        <= objective_tolerance
                    ):
                        break
                else:
                    risky_final = candidate
            final = safe_final
        slack_gap = frontier_gap(final)
        subproblem_gap = final.qp_gap / (final.theta * alpha_scale)
        total_gap = slack_gap + subproblem_gap
        return _success_result(
            model,
            workspace,
            matrix_build_s,
            final,
            theta_start,
            slack_gap,
            subproblem_gap,
            objective_tolerance,
            points,
            total_gap=total_gap,
        )
    except TimeoutError as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.LIMIT_REACHED,
            FailureReason.TIME_LIMIT,
            str(exc),
        )
    except _WorkspaceFailure as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.NUMERICAL_ERROR,
            FailureReason.NUMERICAL_FAILURE,
            str(exc),
            native_status=exc.native_status,
        )
    except Exception as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.SOLVER_ERROR,
            FailureReason.UNKNOWN,
            f"{type(exc).__name__}: {exc}",
        )


def _success_result(
    model: FactorQCQP,
    workspace: _ParametricPIQPWorkspace,
    matrix_build_s: float,
    point: _FrontierPoint,
    theta_seed: float,
    frontier_gap: float,
    subproblem_gap: float,
    objective_tolerance: float,
    outer_points: int,
    *,
    total_gap: float | None = None,
) -> BackendResult:
    if total_gap is None:
        total_gap = frontier_gap + subproblem_gap
    accepted = total_gap <= objective_tolerance
    return BackendResult(
        backend="factor_qcqp_piqp",
        status=SolveStatus.OPTIMAL if accepted else SolveStatus.LIMIT_REACHED,
        primal=point.vector if accepted else None,
        objective_value=point.alpha_value if accepted else None,
        native_status="PIQP_SOLVED" if accepted else "certificate_limit",
        reason=None if accepted else FailureReason.MAX_ITER,
        message=(
            "parametric factor-QP frontier"
            if accepted
            else "frontier point is feasible but objective certificate exceeds tolerance"
        ),
        iterations=workspace.qp_iterations,
        setup_s=matrix_build_s + workspace.setup_s,
        solve_s=workspace.solve_s + workspace.update_s,
        diagnostics={
            "theta": point.theta,
            "theta_seed": theta_seed,
            "variance": point.variance,
            "tracking_error": math.sqrt(max(0.0, point.variance)),
            "frontier_gap": frontier_gap,
            "subproblem_gap": subproblem_gap,
            "total_gap": total_gap,
            "objective_tolerance": objective_tolerance,
            "risk_objective_scale": _FRONTIER_RISK_OBJECTIVE_SCALE,
            "qp_solves": workspace.qp_solves,
            "outer_points": outer_points,
            "workspace_rebuilds": workspace.rebuilds,
            "inequality_form": workspace.form,
            "trace": tuple(workspace.trace),
        },
    )


def _failure_result(
    workspace: _ParametricPIQPWorkspace,
    matrix_build_s: float,
    status: SolveStatus,
    reason: FailureReason,
    message: str,
    *,
    native_status: str = "factor_qcqp_failure",
) -> BackendResult:
    return BackendResult(
        backend="factor_qcqp_piqp",
        status=status,
        primal=None,
        objective_value=None,
        native_status=native_status,
        reason=reason,
        message=message,
        iterations=workspace.qp_iterations,
        setup_s=matrix_build_s + workspace.setup_s,
        solve_s=workspace.solve_s + workspace.update_s,
        diagnostics={
            "qp_solves": workspace.qp_solves,
            "workspace_rebuilds": workspace.rebuilds,
            "inequality_form": workspace.form,
            "trace": tuple(workspace.trace),
        },
    )


_UNIT_ALPHA_SPEC = AlphaSpec(units="unspecified", scale=1.0)

r"""Specialized factor-model QCQP solver built from parametric sparse QPs.

The original problem maximizes alpha over a polyhedral portfolio feasible set
with one factor-model variance constraint.  For a parameter $\theta\ge0$, solve

$$
\min_x\quad \frac12 R(x-b)-\theta\alpha^{\mathsf T}x
$$

over the same polyhedron.  The QP matrices stay fixed while theta changes, so
the backend can reuse its workspace and warm-start every point on the frontier.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class FactorQCQPSettings:
    backend: str = "PIQP"
    alpha_target: float = 0.2
    objective_gap_abs: float = 1e-4
    risk_margin_pct: float = 1e-5
    intermediate_eps: float = 1e-5
    final_eps: float = 1e-8
    max_iter: int = 100_000
    piqp_max_iter: int = 1_000
    max_outer_iters: int = 30
    theta_initial: float = 65_536.0
    theta_growth: float = 4.0
    theta_max: float = 1e12
    feasibility_tol: float = 2e-7
    polish: bool = True
    verbose: bool = False
    time_limit_s: float | None = None
    turnover_row_scale: float = 1.0
    piqp_inequality_form: str = "auto"


@dataclass
class FactorQCQPResult:
    weight: np.ndarray
    status: str
    objective: float
    theta: float
    theta_seed: float
    dual_gap_abs: float
    risk_pct: float
    min_risk_pct: float
    risk_active: bool
    alpha_scale: float
    alpha_shift: float
    omitted_total_active: bool
    omitted_benchmark_weight: bool
    sparse_turnover: bool
    matrix_build_s: float
    workspace_setup_s: float
    workspace_setup_solver_s: float
    search_s: float
    final_s: float
    update_s: float
    solve_wall_s: float
    solver_s: float
    polish_s: float
    qp_solves: int
    qp_iters: int
    outer_iters: int
    max_linear_violation: float
    qp_backend: str
    qp_status: str
    piqp_inequality_form: str
    message: str
    qp_trace: list[dict[str, Any]]


@dataclass
class FactorPenaltyQPResult:
    weight: np.ndarray
    status: str
    objective: float
    alpha_scale: float
    alpha_shift: float
    omitted_total_active: bool
    omitted_benchmark_weight: bool
    sparse_turnover: bool
    matrix_build_s: float
    workspace_setup_s: float
    workspace_setup_solver_s: float
    solve_wall_s: float
    solver_s: float
    qp_iters: int
    max_linear_violation: float
    qp_status: str
    piqp_inequality_form: str
    qp_trace: list[dict[str, Any]]


@dataclass
class _Point:
    theta: float
    weight: np.ndarray
    variance: float
    risk_pct: float
    alpha_value: float
    qp_status: str


@dataclass
class _QPData:
    P: sp.csc_matrix
    A: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    q_base: np.ndarray
    alpha_solver: np.ndarray
    alpha_scale: float
    alpha_shift: float
    n_assets: int
    n_factors: int
    n_variables: int
    support_idx: np.ndarray
    new_idx: np.ndarray
    trade_offset: int
    active_aux_offset: int
    n_active_aux: int
    turnover_row_index: int
    omitted_total_active: bool
    omitted_benchmark_weight: bool


def _diagnostic_value(info: Any, *names: str) -> int | float | str | None:
    """Return the first JSON-safe PIQP/OSQP diagnostic exposed by a version."""

    for name in names:
        value = getattr(info, name, None)
        if value is None:
            continue
        if name == "status":
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    return None


def _solver_info_diagnostics(info: Any) -> dict[str, int | float | str | None]:
    """Normalize diagnostics across PIQP releases without requiring one ABI."""

    return {
        "iterations": _diagnostic_value(info, "iter", "iterations"),
        "solver_status": _diagnostic_value(info, "status"),
        "primal_residual": _diagnostic_value(
            info, "primal_residual", "primal_res", "primal_res_norm"
        ),
        "dual_residual": _diagnostic_value(
            info, "dual_residual", "dual_res", "dual_res_norm"
        ),
        "duality_gap": _diagnostic_value(
            info, "duality_gap", "dual_gap", "duality_gap_abs"
        ),
        "primal_objective": _diagnostic_value(
            info, "primal_objective", "primal_obj", "objective"
        ),
        "dual_objective": _diagnostic_value(info, "dual_objective", "dual_obj"),
        "setup_time_s": _diagnostic_value(info, "setup_time"),
        "update_time_s": _diagnostic_value(info, "update_time"),
        "solver_time_s": _diagnostic_value(info, "solve_time", "run_time"),
        "rho": _diagnostic_value(info, "rho"),
        "delta": _diagnostic_value(info, "delta"),
        "mu": _diagnostic_value(info, "mu"),
    }


def _zero_columns(rows: int, columns: int) -> sp.csc_matrix:
    return sp.csc_matrix((rows, columns), dtype=float)


def _pad_block(blocks: list[sp.spmatrix], n_variables: int) -> sp.csc_matrix:
    block = sp.hstack(blocks, format="csc")
    if block.shape[1] != n_variables:
        raise AssertionError(f"constraint block has {block.shape[1]} columns, expected {n_variables}")
    return block


def _build_qp_data(day: Any, config: Any, settings: FactorQCQPSettings) -> _QPData:
    n = len(day.sids)
    k = len(day.factors)
    support = np.asarray(day.initial > config.weight_zero_tol, dtype=bool)
    support_idx = np.flatnonzero(support)
    new_idx = np.flatnonzero(~support)
    n_trade = len(support_idx)

    x0_active_l1 = float(np.abs(day.initial - day.benchmark).sum())
    omitted_total_active = (
        x0_active_l1 + config.turnover_limit <= config.total_active_ub + 1e-12
    )
    benchmark_mask = np.asarray(day.benchmark > 0, dtype=bool)
    x0_benchmark_weight = float(day.initial[benchmark_mask].sum())
    omitted_benchmark_weight = (
        x0_benchmark_weight - config.turnover_limit / 2.0
        >= config.benchmark_weight_lb - 1e-12
    )

    n_active_aux = 0 if omitted_total_active else n
    x_offset = 0
    factor_offset = n
    trade_offset = factor_offset + k
    active_aux_offset = trade_offset + n_trade
    n_variables = active_aux_offset + n_active_aux

    specific_variance = np.square(np.asarray(day.spec_risk, dtype=float))
    covariance = np.asarray(day.covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    zero_aux = sp.csc_matrix((n_trade + n_active_aux, n_trade + n_active_aux))
    P = sp.block_diag(
        (sp.diags(specific_variance, format="csc"), sp.csc_matrix(covariance), zero_aux),
        format="csc",
    )
    P = sp.triu(P, format="csc")

    alpha_shift = float(np.asarray(day.alpha) @ np.asarray(day.benchmark))
    centered_alpha = np.asarray(day.alpha, dtype=float) - alpha_shift
    max_abs_alpha = float(np.max(np.abs(centered_alpha)))
    alpha_scale = settings.alpha_target / max_abs_alpha if max_abs_alpha > 0.0 else 1.0
    alpha_solver = alpha_scale * centered_alpha
    q_base = np.zeros(n_variables, dtype=float)
    q_base[x_offset : x_offset + n] = -specific_variance * np.asarray(day.benchmark)

    blocks: list[sp.csc_matrix] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []

    # Budget equality.
    budget = sp.csc_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))), shape=(1, n_variables)
    )
    blocks.append(budget)
    lowers.append(np.array([1.0]))
    uppers.append(np.array([1.0]))

    # $f=E^{\mathsf T}(x-b)$.
    factor_eq = _pad_block(
        [
            sp.csc_matrix(np.asarray(day.exposure, dtype=float).T),
            -sp.eye(k, format="csc"),
            _zero_columns(k, n_trade + n_active_aux),
        ],
        n_variables,
    )
    factor_rhs = np.asarray(day.exposure, dtype=float).T @ np.asarray(day.benchmark)
    blocks.append(factor_eq)
    lowers.append(factor_rhs)
    uppers.append(factor_rhs)

    # Combined long-only, asset, and active-weight bounds.
    asset_lower = np.maximum(0.0, np.asarray(day.benchmark) - config.active_ub)
    asset_upper = np.minimum(config.asset_ub, np.asarray(day.benchmark) + config.active_ub)
    asset_bounds = _pad_block(
        [sp.eye(n, format="csc"), _zero_columns(n, n_variables - n)], n_variables
    )
    blocks.append(asset_bounds)
    lowers.append(asset_lower)
    uppers.append(asset_upper)

    # Style and industry constraints are direct bounds on active factor exposure f.
    factor_indices = list(day.style_idx) + list(day.industry_idx)
    factor_lowers = np.concatenate(
        [
            np.full(len(day.style_idx), config.style_default_lb),
            np.full(len(day.industry_idx), config.industry_lb),
        ]
    )
    factor_uppers = np.concatenate(
        [
            np.full(len(day.style_idx), config.style_default_ub),
            np.full(len(day.industry_idx), config.industry_ub),
        ]
    )
    size_position = list(day.style_idx).index(day.factors.index("size"))
    factor_lowers[size_position] = config.size_lb
    factor_uppers[size_position] = config.size_ub
    factor_bound_rows = np.arange(len(factor_indices), dtype=int)
    factor_bounds = sp.csc_matrix(
        (
            np.ones(len(factor_indices)),
            (factor_bound_rows, factor_offset + np.asarray(factor_indices)),
        ),
        shape=(len(factor_indices), n_variables),
    )
    blocks.append(factor_bounds)
    lowers.append(factor_lowers)
    uppers.append(factor_uppers)

    # Sparse exact turnover formulation:
    # $p_i\ge x_i-x_{0,i}$, $p_i\ge0$, and
    # $\lVert x-x_0\rVert_1=2(\sum_i p_i+\sum_{j\notin S}x_j)$.
    trade_identity = sp.csc_matrix(
        (
            np.ones(n_trade),
            (np.arange(n_trade), trade_offset + np.arange(n_trade)),
        ),
        shape=(n_trade, n_variables),
    )
    blocks.append(trade_identity)
    lowers.append(np.zeros(n_trade))
    uppers.append(np.full(n_trade, np.inf))

    support_rows = np.arange(n_trade, dtype=int)
    trade_epigraph = sp.csc_matrix(
        (
            np.concatenate([np.ones(n_trade), -np.ones(n_trade)]),
            (
                np.concatenate([support_rows, support_rows]),
                np.concatenate([support_idx, trade_offset + np.arange(n_trade)]),
            ),
        ),
        shape=(n_trade, n_variables),
    )
    blocks.append(trade_epigraph)
    lowers.append(np.full(n_trade, -np.inf))
    uppers.append(np.asarray(day.initial)[support_idx])

    turnover_columns = np.concatenate([new_idx, trade_offset + np.arange(n_trade)])
    turnover_row_index = sum(block.shape[0] for block in blocks)
    turnover_scale = settings.turnover_row_scale
    turnover = sp.csc_matrix(
        (
            np.full(len(turnover_columns), 2.0 * turnover_scale),
            (np.zeros(len(turnover_columns), dtype=int), turnover_columns),
        ),
        shape=(1, n_variables),
    )
    blocks.append(turnover)
    lowers.append(np.array([-np.inf]))
    uppers.append(np.array([config.turnover_limit * turnover_scale]))

    if not omitted_total_active:
        aux_columns = active_aux_offset + np.arange(n)
        active_aux_nonnegative = sp.csc_matrix(
            (np.ones(n), (np.arange(n), aux_columns)), shape=(n, n_variables)
        )
        blocks.append(active_aux_nonnegative)
        lowers.append(np.zeros(n))
        uppers.append(np.full(n, np.inf))

        rows = np.arange(n)
        active_positive = sp.csc_matrix(
            (
                np.concatenate([np.ones(n), -np.ones(n)]),
                (np.concatenate([rows, rows]), np.concatenate([np.arange(n), aux_columns])),
            ),
            shape=(n, n_variables),
        )
        blocks.append(active_positive)
        lowers.append(np.full(n, -np.inf))
        uppers.append(np.asarray(day.benchmark))

        active_negative = sp.csc_matrix(
            (
                np.concatenate([-np.ones(n), -np.ones(n)]),
                (np.concatenate([rows, rows]), np.concatenate([np.arange(n), aux_columns])),
            ),
            shape=(n, n_variables),
        )
        blocks.append(active_negative)
        lowers.append(np.full(n, -np.inf))
        uppers.append(-np.asarray(day.benchmark))

        total_active = sp.csc_matrix(
            (np.ones(n), (np.zeros(n, dtype=int), aux_columns)), shape=(1, n_variables)
        )
        blocks.append(total_active)
        lowers.append(np.array([-np.inf]))
        uppers.append(np.array([config.total_active_ub]))

    if not omitted_benchmark_weight:
        benchmark_idx = np.flatnonzero(benchmark_mask)
        benchmark_weight = sp.csc_matrix(
            (
                np.ones(len(benchmark_idx)),
                (np.zeros(len(benchmark_idx), dtype=int), benchmark_idx),
            ),
            shape=(1, n_variables),
        )
        blocks.append(benchmark_weight)
        lowers.append(np.array([config.benchmark_weight_lb]))
        uppers.append(np.array([np.inf]))

    A = sp.vstack(blocks, format="csc")
    A.eliminate_zeros()
    return _QPData(
        P=P,
        A=A,
        lower=np.concatenate(lowers),
        upper=np.concatenate(uppers),
        q_base=q_base,
        alpha_solver=alpha_solver,
        alpha_scale=alpha_scale,
        alpha_shift=alpha_shift,
        n_assets=n,
        n_factors=k,
        n_variables=n_variables,
        support_idx=support_idx,
        new_idx=new_idx,
        trade_offset=trade_offset,
        active_aux_offset=active_aux_offset,
        n_active_aux=n_active_aux,
        turnover_row_index=turnover_row_index,
        omitted_total_active=omitted_total_active,
        omitted_benchmark_weight=omitted_benchmark_weight,
    )


def _portfolio_metrics(day: Any, weight: np.ndarray) -> tuple[float, float, float]:
    active = weight - np.asarray(day.benchmark)
    factor = np.asarray(day.exposure).T @ active
    factor_variance = float(factor @ np.asarray(day.covariance) @ factor)
    specific_variance = float(np.sum(np.square(np.asarray(day.spec_risk) * active)))
    variance = max(0.0, factor_variance + specific_variance)
    return variance, math.sqrt(variance), float(np.asarray(day.alpha) @ weight)


def _linear_violation(data: _QPData, vector: np.ndarray) -> float:
    value = np.asarray(data.A @ vector).reshape(-1)
    lower_violation = np.where(np.isfinite(data.lower), data.lower - value, -np.inf)
    upper_violation = np.where(np.isfinite(data.upper), value - data.upper, -np.inf)
    return max(0.0, float(np.max(lower_violation)), float(np.max(upper_violation)))


def _warm_start_vector(data: _QPData, day: Any, weight: np.ndarray) -> np.ndarray:
    """Lift an asset portfolio into the exact factor/turnover epigraph variables."""

    weight = np.asarray(weight, dtype=float).reshape(-1)
    if len(weight) != data.n_assets or not np.all(np.isfinite(weight)):
        raise ValueError("factor-QP warm start has invalid asset weights")
    vector = np.zeros(data.n_variables, dtype=float)
    vector[: data.n_assets] = weight
    active = weight - np.asarray(day.benchmark)
    factor_offset = data.n_assets
    vector[factor_offset : factor_offset + data.n_factors] = np.asarray(day.exposure).T @ active
    trades = np.maximum(
        weight[data.support_idx] - np.asarray(day.initial)[data.support_idx],
        0.0,
    )
    vector[data.trade_offset : data.trade_offset + len(data.support_idx)] = trades
    if data.n_active_aux:
        vector[
            data.active_aux_offset : data.active_aux_offset + data.n_active_aux
        ] = np.abs(active)
    return vector


def piqp_distribution_version() -> str | None:
    """Return the installed PIQP distribution version, when available."""

    try:
        return metadata.version("piqp")
    except metadata.PackageNotFoundError:
        return None


def resolve_piqp_inequality_form(requested: str) -> str:
    """Resolve compact by default for the required PIQP 0.6.4+ runtime.

    The upstream 0.6.4 release contains the audited dual-recovery fix.
    ``one_sided`` is retained only for explicit diagnostics and benchmarks.
    """

    form = requested.lower()
    if form not in {"auto", "one_sided", "compact"}:
        raise ValueError(f"unsupported PIQP inequality form: {requested!r}")
    if form == "auto":
        return "compact"
    return form


def _piqp_constraint_data(
    data: _QPData,
    inequality_form: str,
) -> tuple[
    sp.csc_matrix,
    np.ndarray,
    sp.csc_matrix,
    np.ndarray,
    np.ndarray,
]:
    """Return PIQP equality/inequality arrays in a selected exact form.

    ``compact`` is the production form for the required PIQP 0.6.4+ runtime.
    Expanding every finite side into an upper-only row is mathematically exact
    and remains available as a diagnostic/performance comparison. Callers
    resolve ``auto`` before entering this helper.
    """

    form = inequality_form.lower()
    if form not in {"one_sided", "compact"}:
        raise ValueError(f"unsupported PIQP inequality form: {inequality_form!r}")
    equality = (
        np.isfinite(data.lower)
        & np.isfinite(data.upper)
        & np.isclose(data.lower, data.upper, rtol=0.0, atol=1e-14)
    )
    equality_matrix = data.A[equality].tocsc()
    equality_rhs = data.lower[equality].copy()
    general_matrix = data.A[~equality].tocsc()
    general_lower = data.lower[~equality].copy()
    general_upper = data.upper[~equality].copy()
    if form == "compact":
        return (
            equality_matrix,
            equality_rhs,
            general_matrix,
            general_lower,
            general_upper,
        )

    finite_upper = np.isfinite(general_upper)
    finite_lower = np.isfinite(general_lower)
    inequality_matrix = sp.vstack(
        [general_matrix[finite_upper], -general_matrix[finite_lower]],
        format="csc",
    )
    inequality_matrix.eliminate_zeros()
    inequality_lower = np.full(
        int(finite_upper.sum() + finite_lower.sum()), -np.inf, dtype=float
    )
    inequality_upper = np.concatenate(
        [general_upper[finite_upper], -general_lower[finite_lower]]
    )
    return (
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_lower,
        inequality_upper,
    )


def solve_factor_penalty_qp(
    day: Any,
    config: Any,
    settings: FactorQCQPSettings,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> FactorPenaltyQPResult:
    r"""Solve the fixed risk-penalty QP through PIQP's sparse interface.

    This is the direct counterpart of the CVXPY model

    $$
    \max_x\quad
    \alpha^{\mathsf T}x
    -\gamma_f R_f(x-b)
    -\gamma_s R_s(x-b).
    $$

    It reuses the same exact sparse turnover and factor-exposure reformulation
    as the parametric QCQP path, but needs only one QP solve.
    """
    import piqp

    build_started = time.perf_counter()
    data = _build_qp_data(day, config, settings)
    n = data.n_assets
    k = data.n_factors
    specific_variance = np.square(np.asarray(day.spec_risk, dtype=float))
    covariance = np.asarray(day.covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    auxiliary_count = data.n_variables - n - k

    # PIQP minimizes $\frac12z^{\mathsf T}Pz+q^{\mathsf T}z$. Expanding
    # $\gamma_s\sum_i d_i^2(x_i-b_i)^2$ gives the diagonal Hessian and linear
    # term below. Factor variables already equal $E^{\mathsf T}(x-b)$, so their
    # block has no additional linear term.
    objective_scale = data.alpha_scale
    zero_aux = sp.csc_matrix((auxiliary_count, auxiliary_count), dtype=float)
    P = sp.block_diag(
        (
            sp.diags(
                2.0
                * objective_scale
                * config.specific_risk_aversion
                * specific_variance,
                format="csc",
            ),
            sp.csc_matrix(
                2.0
                * objective_scale
                * config.common_risk_aversion
                * covariance
            ),
            zero_aux,
        ),
        format="csc",
    )
    centered_alpha = data.alpha_solver / data.alpha_scale
    q = np.zeros(data.n_variables, dtype=float)
    q[:n] = objective_scale * (
        -centered_alpha
        - 2.0
        * config.specific_risk_aversion
        * specific_variance
        * np.asarray(day.benchmark, dtype=float)
    )

    piqp_inequality_form = resolve_piqp_inequality_form(
        settings.piqp_inequality_form
    )
    (
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_lower,
        inequality_upper,
    ) = _piqp_constraint_data(data, piqp_inequality_form)
    matrix_build_s = time.perf_counter() - build_started

    solver = piqp.SparseSolver()
    solver.settings.verbose = settings.verbose
    solver.settings.eps_abs = settings.final_eps
    solver.settings.eps_rel = settings.final_eps
    solver.settings.eps_duality_gap_abs = settings.final_eps
    solver.settings.eps_duality_gap_rel = settings.final_eps
    solver.settings.max_iter = settings.piqp_max_iter
    solver.settings.compute_timings = True

    setup_started = time.perf_counter()
    solver.setup(
        P,
        q,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_lower,
        inequality_upper,
    )
    workspace_setup_s = time.perf_counter() - setup_started

    trace_enabled = diagnostics is not None
    trace = diagnostics if diagnostics is not None else []
    solve_started = time.perf_counter()
    try:
        status = solver.solve()
    except Exception as exc:
        solve_wall_s = time.perf_counter() - solve_started
        if trace_enabled:
            trace.append(
                {
                    "phase": "penalty_qp",
                    "final": True,
                    "tolerance": settings.final_eps,
                    "success": False,
                    "wall_time_s": solve_wall_s,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
        raise
    solve_wall_s = time.perf_counter() - solve_started
    result = solver.result
    info = result.info
    qp_status = str(getattr(info, "status", status))
    success = "PIQP_SOLVED" in str(status) and result.x is not None
    trace_record: dict[str, Any] = {
        "phase": "penalty_qp",
        "final": True,
        "tolerance": settings.final_eps,
        "success": success,
        "wall_time_s": solve_wall_s,
        **_solver_info_diagnostics(info),
    }
    if trace_enabled:
        trace.append(trace_record)
    if not success:
        raise RuntimeError(f"direct factor-penalty PIQP ended with status {qp_status}")

    vector = np.asarray(result.x, dtype=float).reshape(-1)
    weight = vector[:n].copy()
    linear_violation = _linear_violation(data, vector)
    if trace_enabled:
        trace_record["linear_violation"] = linear_violation
    if linear_violation > settings.feasibility_tol:
        raise RuntimeError(
            f"direct factor-penalty QP linear violation {linear_violation:.3e} "
            f"exceeds tolerance {settings.feasibility_tol:.3e}"
        )

    active = weight - np.asarray(day.benchmark, dtype=float)
    factor = np.asarray(day.exposure, dtype=float).T @ active
    factor_variance = float(factor @ covariance @ factor)
    specific_active_variance = float(
        np.sum(np.square(np.asarray(day.spec_risk, dtype=float) * active))
    )
    objective = float(np.asarray(day.alpha, dtype=float) @ weight)
    objective -= config.common_risk_aversion * factor_variance
    objective -= config.specific_risk_aversion * specific_active_variance

    return FactorPenaltyQPResult(
        weight=weight,
        status="optimal",
        objective=objective,
        alpha_scale=objective_scale,
        alpha_shift=data.alpha_shift,
        omitted_total_active=data.omitted_total_active,
        omitted_benchmark_weight=data.omitted_benchmark_weight,
        sparse_turnover=True,
        matrix_build_s=matrix_build_s,
        workspace_setup_s=workspace_setup_s,
        workspace_setup_solver_s=float(getattr(info, "setup_time", 0.0) or 0.0),
        solve_wall_s=solve_wall_s,
        solver_s=float(getattr(info, "solve_time", 0.0) or 0.0),
        qp_iters=int(getattr(info, "iter", 0) or 0),
        max_linear_violation=linear_violation,
        qp_status=qp_status,
        piqp_inequality_form=piqp_inequality_form,
        qp_trace=trace,
    )


def solve_factor_qcqp(
    day: Any,
    config: Any,
    settings: FactorQCQPSettings,
    *,
    theta_seed: float | None = None,
    linear_warm_start: np.ndarray | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> FactorQCQPResult:
    """Solve one independent factor-model QCQP and return detailed timings."""

    backend = settings.backend.upper()
    if backend not in {"OSQP", "PIQP"}:
        raise ValueError(f"unsupported factor-QP backend: {settings.backend!r}")

    theta_start = settings.theta_initial if theta_seed is None else float(theta_seed)
    theta_start = min(max(theta_start, np.finfo(float).eps), settings.theta_max)
    deadline = (
        None if settings.time_limit_s is None else time.perf_counter() + settings.time_limit_s
    )

    build_started = time.perf_counter()
    data = _build_qp_data(day, config, settings)
    matrix_build_s = time.perf_counter() - build_started

    setup_started = time.perf_counter()
    piqp_inequality_form = "not_applicable"
    if backend == "OSQP":
        import osqp

        solver = osqp.OSQP()
        solver.setup(
            P=data.P,
            q=data.q_base,
            A=data.A,
            l=data.lower,
            u=data.upper,
            verbose=settings.verbose,
            warm_starting=True,
            polishing=False,
            eps_abs=settings.intermediate_eps,
            eps_rel=settings.intermediate_eps,
            max_iter=settings.max_iter,
            adaptive_rho=True,
            check_termination=25,
        )
        warm_weight = day.initial if linear_warm_start is None else linear_warm_start
        solver.warm_start(x=_warm_start_vector(data, day, warm_weight))
    else:
        import piqp

        P_full = (data.P + sp.triu(data.P, k=1).T).tocsc()
        piqp_inequality_form = resolve_piqp_inequality_form(
            settings.piqp_inequality_form
        )
        (
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_lower,
            inequality_upper,
        ) = _piqp_constraint_data(data, piqp_inequality_form)
        solver = piqp.SparseSolver()
        solver.settings.verbose = settings.verbose
        solver.settings.eps_abs = settings.intermediate_eps
        solver.settings.eps_rel = settings.intermediate_eps
        solver.settings.eps_duality_gap_abs = settings.intermediate_eps
        solver.settings.eps_duality_gap_rel = settings.intermediate_eps
        solver.settings.max_iter = settings.piqp_max_iter
        solver.settings.compute_timings = True
        # The first frontier theta is already known, so installing its cost at
        # setup avoids an unnecessary update and follows the natural
        # setup -> solve -> update -> solve continuation lifecycle.  It is an
        # efficiency/lifecycle cleanup, not the dual-index source fix.
        initial_q = data.q_base.copy()
        initial_q[: data.n_assets] -= theta_start * data.alpha_solver
        solver.setup(
            P_full,
            initial_q,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_lower,
            inequality_upper,
        )
    workspace_setup_s = time.perf_counter() - setup_started

    update_s = 0.0
    solve_wall_s = 0.0
    solver_s = 0.0
    polish_s = 0.0
    workspace_setup_solver_s = 0.0
    qp_solves = 0
    qp_iters = 0
    last_status = "not_run"
    current_theta: float | None = None
    trace_enabled = diagnostics is not None
    trace = diagnostics if diagnostics is not None else []

    def solve_theta(theta: float, *, final: bool = False) -> tuple[_Point, np.ndarray]:
        nonlocal update_s, solve_wall_s, solver_s, polish_s, workspace_setup_solver_s
        nonlocal qp_solves, qp_iters, last_status, current_theta
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                raise TimeoutError("factor-QCQP total time limit reached")
            if backend == "OSQP":
                solver.update_settings(time_limit=max(1e-3, remaining))
        tolerance = settings.final_eps if final else settings.intermediate_eps
        if backend == "PIQP" and final:
            # PIQP already returns portfolio feasibility close to machine precision
            # at 1e-7.  Requiring 1e-8 on large-theta, inactive-risk QPs can cause
            # disproportionate regularization iterations without improving the
            # user's 1bp objective guarantee.
            tolerance = max(tolerance, 1e-7)
        if backend == "OSQP":
            solver.update_settings(
                eps_abs=tolerance,
                eps_rel=tolerance,
                polishing=bool(final and settings.polish),
                max_iter=settings.max_iter,
            )
        else:
            solver.settings.eps_abs = tolerance
            solver.settings.eps_rel = tolerance
            solver.settings.eps_duality_gap_abs = tolerance
            solver.settings.eps_duality_gap_rel = tolerance
            solver.settings.max_iter = settings.piqp_max_iter
        q = data.q_base.copy()
        q[: data.n_assets] -= theta * data.alpha_solver
        # PIQP's initial cost was installed during setup.  Updates are only
        # needed after a successful first solve, when continuation begins.
        if backend == "PIQP" and qp_solves == 0:
            if theta != theta_start:
                raise AssertionError(
                    "first PIQP frontier theta differs from the setup cost"
                )
        else:
            update_started = time.perf_counter()
            if backend == "OSQP":
                solver.update(q=q)
            else:
                solver.update(c=q)
            update_s += time.perf_counter() - update_started
        current_theta = theta
        solve_started = time.perf_counter()
        if backend == "OSQP":
            try:
                result = solver.solve(raise_error=False)
            except TypeError:  # OSQP < 1.0 compatibility.
                result = solver.solve()
            info = result.info
            vector_value = result.x
            success = str(info.status).lower() in {"solved", "solved inaccurate"}
        else:
            try:
                status = solver.solve()
            except Exception as exc:
                elapsed = time.perf_counter() - solve_started
                solve_wall_s += elapsed
                qp_solves += 1
                if trace_enabled:
                    trace.append(
                        {
                            "phase": "frontier",
                            "theta": theta,
                            "final": final,
                            "tolerance": tolerance,
                            "success": False,
                            "wall_time_s": elapsed,
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                    )
                raise
            result = solver.result
            info = result.info
            vector_value = result.x
            success = "PIQP_SOLVED" in str(status)
        elapsed = time.perf_counter() - solve_started
        solve_wall_s += elapsed
        if qp_solves == 0:
            workspace_setup_solver_s = float(getattr(info, "setup_time", 0.0) or 0.0)
        qp_solves += 1
        qp_iters += int(getattr(info, "iter", 0) or 0)
        solver_s += float(
            getattr(info, "solve_time", getattr(info, "run_time", 0.0)) or 0.0
        )
        polish_s += float(getattr(info, "polish_time", 0.0) or 0.0)
        last_status = str(getattr(info, "status", status if backend == "PIQP" else "unknown"))
        trace_record: dict[str, Any] = {
            "phase": "frontier",
            "theta": theta,
            "final": final,
            "tolerance": tolerance,
            "success": bool(success and vector_value is not None),
            "wall_time_s": elapsed,
            **_solver_info_diagnostics(info),
        }
        if trace_enabled:
            trace.append(trace_record)
        if not success or vector_value is None:
            raise RuntimeError(
                f"{backend} theta={theta:g} ended with status {last_status}"
            )
        vector = np.asarray(vector_value, dtype=float).reshape(-1)
        weight = vector[: data.n_assets].copy()
        variance, risk_pct, alpha_value = _portfolio_metrics(day, weight)
        if trace_enabled:
            trace_record.update(
                {
                    "risk_pct": risk_pct,
                    "variance_pct2": variance,
                    "alpha_value": alpha_value,
                }
            )
        return (
            _Point(
                theta=theta,
                weight=weight,
                variance=variance,
                risk_pct=risk_pct,
                alpha_value=alpha_value,
                qp_status=last_status,
            ),
            vector,
        )

    target_risk = max(0.0, config.risk_budget_pct - settings.risk_margin_pct)
    target_variance = target_risk * target_risk
    risk_budget_variance = config.risk_budget_pct * config.risk_budget_pct
    search_started = time.perf_counter()

    point, _ = solve_theta(theta_start)
    minimum_risk_pct = point.risk_pct
    outer_iters = 1

    if not np.any(data.alpha_solver):
        if point.variance > target_variance:
            raise RuntimeError(
                f"minimum-risk QP returned {point.risk_pct:.9g}%, above the risk budget"
            )
        low = point
        high: _Point | None = None
        risk_active = False
    else:
        if point.variance > target_variance:
            high = point
            low: _Point | None = None
            theta = theta_start
            # Do not solve theta=0 unconditionally.  That degenerate minimum-risk
            # endpoint is often much harder for ADMM than the frontier points we
            # actually need.  Search downward only when the initial point is risky.
            while outer_iters < settings.max_outer_iters:
                theta /= settings.theta_growth
                if theta <= np.finfo(float).eps:
                    break
                point, _ = solve_theta(theta)
                outer_iters += 1
                minimum_risk_pct = min(minimum_risk_pct, point.risk_pct)
                if point.variance <= target_variance:
                    low = point
                    break
                high = point
            if low is None:
                raise RuntimeError(
                    "could not find a risk-feasible parametric QP point; "
                    "the QCQP may be infeasible or the QP scaling may need adjustment"
                )
        else:
            low = point
            high = None
            theta = theta_start
            while outer_iters < settings.max_outer_iters:
                dual_gap = max(
                    0.0,
                    (risk_budget_variance - low.variance)
                    / (2.0 * low.theta * data.alpha_scale),
                )
                if dual_gap <= settings.objective_gap_abs:
                    break
                theta *= settings.theta_growth
                if theta > settings.theta_max:
                    break
                point, _ = solve_theta(theta)
                outer_iters += 1
                minimum_risk_pct = min(minimum_risk_pct, point.risk_pct)
                if point.variance > target_variance:
                    high = point
                    break
                low = point

        if high is not None:
            # Safeguarded secant on variance.  QP warm starts make each update cheap.
            while outer_iters < settings.max_outer_iters:
                variance_span = high.variance - low.variance
                if variance_span <= 0.0:
                    theta = (low.theta + high.theta) / 2.0
                else:
                    theta = low.theta + (
                        (target_variance - low.variance)
                        * (high.theta - low.theta)
                        / variance_span
                    )
                span = high.theta - low.theta
                theta = min(high.theta - 0.1 * span, max(low.theta + 0.1 * span, theta))
                point, _ = solve_theta(theta)
                outer_iters += 1
                minimum_risk_pct = min(minimum_risk_pct, point.risk_pct)
                if point.variance <= target_variance:
                    low = point
                else:
                    high = point
                if low.theta > 0.0:
                    dual_gap = max(
                        0.0,
                        (risk_budget_variance - low.variance)
                        / (2.0 * low.theta * data.alpha_scale),
                    )
                    if dual_gap <= settings.objective_gap_abs:
                        break
                if high.theta - low.theta <= 1e-10 * max(1.0, high.theta):
                    break
            risk_active = True
        else:
            risk_active = low.risk_pct >= target_risk - 10.0 * settings.risk_margin_pct

    search_s = time.perf_counter() - search_started

    final_started = time.perf_counter()
    final_point, final_vector = solve_theta(low.theta, final=True)
    if final_point.risk_pct > config.risk_budget_pct + settings.feasibility_tol:
        # Tightening can move a coarse frontier point slightly.  Return a genuine
        # high-accuracy parametric-QP point so that the dual-gap certificate below
        # remains valid; an interpolation between QP optima would only certify
        # feasibility, not optimality at an interpolated theta.
        risky_final = final_point
        risky_final_vector = final_vector
        theta = low.theta
        safe_final: _Point | None = None
        safe_final_vector: np.ndarray | None = None
        for _ in range(12):
            theta /= settings.theta_growth
            if theta <= np.finfo(float).eps:
                break
            candidate, candidate_vector = solve_theta(theta, final=True)
            minimum_risk_pct = min(minimum_risk_pct, candidate.risk_pct)
            if candidate.variance <= target_variance:
                safe_final = candidate
                safe_final_vector = candidate_vector
                break
        if safe_final is None or safe_final_vector is None:
            raise RuntimeError("could not recover a tight risk-feasible frontier point")

        # Refine between genuine high-accuracy QP optima.  Unlike interpolating
        # weights, every accepted safe point retains the parametric-QP dual bound.
        for _ in range(15):
            variance_span = risky_final.variance - safe_final.variance
            theta_span = risky_final.theta - safe_final.theta
            if variance_span <= 0.0:
                theta = (safe_final.theta + risky_final.theta) / 2.0
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
            candidate, candidate_vector = solve_theta(theta, final=True)
            minimum_risk_pct = min(minimum_risk_pct, candidate.risk_pct)
            if candidate.variance <= target_variance:
                safe_final = candidate
                safe_final_vector = candidate_vector
                safe_gap = max(
                    0.0,
                    (risk_budget_variance - safe_final.variance)
                    / (2.0 * safe_final.theta * data.alpha_scale),
                )
                if safe_gap <= settings.objective_gap_abs:
                    break
            else:
                risky_final = candidate
                risky_final_vector = candidate_vector
            if theta_span <= 1e-10 * max(1.0, risky_final.theta):
                break
        final_point = safe_final
        final_vector = safe_final_vector
        message = "risk-boundary high-accuracy parametric factor-QP recovery"
    else:
        message = "parametric factor-QP"
    final_s = time.perf_counter() - final_started

    linear_violation = _linear_violation(data, final_vector)
    if linear_violation > settings.feasibility_tol:
        raise RuntimeError(
            f"factor-QCQP linear violation {linear_violation:.3e} exceeds "
            f"tolerance {settings.feasibility_tol:.3e}"
        )
    if final_point.risk_pct > config.risk_budget_pct + settings.feasibility_tol:
        raise RuntimeError(
            f"factor-QCQP risk {final_point.risk_pct:.9g}% exceeds budget"
        )

    if final_point.theta > 0.0:
        dual_gap_abs = max(
            0.0,
            (risk_budget_variance - final_point.variance)
            / (2.0 * final_point.theta * data.alpha_scale),
        )
    else:
        dual_gap_abs = 0.0 if not np.any(data.alpha_solver) else math.inf

    return FactorQCQPResult(
        weight=final_point.weight,
        status="optimal" if dual_gap_abs <= settings.objective_gap_abs else "optimal_inaccurate",
        objective=final_point.alpha_value,
        theta=final_point.theta,
        theta_seed=theta_start,
        dual_gap_abs=dual_gap_abs,
        risk_pct=final_point.risk_pct,
        min_risk_pct=minimum_risk_pct,
        risk_active=risk_active,
        alpha_scale=data.alpha_scale,
        alpha_shift=data.alpha_shift,
        omitted_total_active=data.omitted_total_active,
        omitted_benchmark_weight=data.omitted_benchmark_weight,
        sparse_turnover=True,
        matrix_build_s=matrix_build_s,
        workspace_setup_s=workspace_setup_s,
        workspace_setup_solver_s=workspace_setup_solver_s,
        search_s=search_s,
        final_s=final_s,
        update_s=update_s,
        solve_wall_s=solve_wall_s,
        solver_s=solver_s,
        polish_s=polish_s,
        qp_solves=qp_solves,
        qp_iters=qp_iters,
        outer_iters=outer_iters,
        max_linear_violation=linear_violation,
        qp_backend=backend,
        qp_status=last_status,
        piqp_inequality_form=piqp_inequality_form,
        message=message,
        qp_trace=trace,
    )

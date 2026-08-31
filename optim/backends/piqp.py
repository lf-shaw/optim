"""Direct sparse PIQP adapter for one-shot canonical quadratic programs.

This adapter builds a fresh workspace and performs one solve.  The separate
factor-QCQP strategy owns its same-day multi-theta update lifecycle.  Both paths
share the same compact double-sided inequality conversion defined here.
Objective scaling multiplies ``P`` and ``q`` by the same positive value, so it
changes conditioning but not the mathematical optimizer.
"""

from __future__ import annotations

import time
from importlib import metadata
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..model.canonical import QuadraticProgram
from ..portfolio_types import FailureReason, SolveStatus
from .base import BackendOptions, BackendResult


def piqp_distribution_version() -> str | None:
    try:
        return metadata.version("piqp")
    except metadata.PackageNotFoundError:
        return None


def resolve_piqp_inequality_form(requested: str) -> str:
    """Resolve the PIQP inequality representation.

    PIQP 0.6.4 is the minimum declared package dependency and contains the upstream
    dual-recovery fix required by compact double-sided inequalities.  ``auto``
    therefore selects compact unconditionally.  ``one_sided`` remains an
    explicit diagnostic/benchmark option, not an old-version compatibility
    path.
    """

    form = requested.lower()
    if form not in {"auto", "one_sided", "compact"}:
        raise ValueError(f"unsupported PIQP inequality form: {requested!r}")
    if form == "auto":
        return "compact"
    return form


def _constraint_data(
    model: QuadraticProgram,
    form: str,
) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
    """Split canonical rows into PIQP equalities and selected inequality form."""

    domain = model.domain
    equality = (
        np.isfinite(domain.lower)
        & np.isfinite(domain.upper)
        & np.isclose(domain.lower, domain.upper, rtol=0.0, atol=1e-14)
    )
    equality_matrix = domain.A[equality].tocsc()
    equality_rhs = domain.lower[equality].copy()
    general_matrix = domain.A[~equality].tocsc()
    general_lower = domain.lower[~equality].copy()
    general_upper = domain.upper[~equality].copy()
    if form == "compact":
        return equality_matrix, equality_rhs, general_matrix, general_lower, general_upper
    finite_upper = np.isfinite(general_upper)
    finite_lower = np.isfinite(general_lower)
    inequality_matrix = sp.vstack(
        [general_matrix[finite_upper], -general_matrix[finite_lower]],
        format="csc",
    )
    inequality_lower = np.full(inequality_matrix.shape[0], -np.inf, dtype=float)
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


class PIQPBackend:
    """One-shot direct PIQP backend for a compiled convex QP."""

    name = "piqp"

    def solve(self, model: QuadraticProgram, options: BackendOptions) -> BackendResult:
        import piqp

        if not isinstance(model, QuadraticProgram):
            raise TypeError("PIQPBackend accepts QuadraticProgram only")
        form = resolve_piqp_inequality_form(options.inequality_form)
        equality, equality_rhs, inequality, inequality_lower, inequality_upper = _constraint_data(
            model, form
        )
        objective_scale = _objective_scale(model, options.objective_scale_target)
        P = (model.P * objective_scale).tocsc()
        q = np.asarray(model.q * objective_scale, dtype=float)

        solver = piqp.SparseSolver()
        solver.settings.verbose = bool(options.verbose)
        solver.settings.eps_abs = float(options.eps_abs)
        solver.settings.eps_rel = float(options.eps_rel)
        solver.settings.eps_duality_gap_abs = float(options.eps_abs)
        solver.settings.eps_duality_gap_rel = float(options.eps_rel)
        solver.settings.max_iter = int(options.max_iter)
        solver.settings.compute_timings = True

        setup_started = time.perf_counter()
        try:
            solver.setup(
                P,
                q,
                equality,
                equality_rhs,
                inequality,
                inequality_lower,
                inequality_upper,
                np.asarray(model.domain.variable_lower, dtype=float),
                np.asarray(model.domain.variable_upper, dtype=float),
            )
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status="setup_exception",
                reason=FailureReason.INVALID_NUMERICS,
                message=f"{type(exc).__name__}: {exc}",
                setup_s=time.perf_counter() - setup_started,
                diagnostics={"inequality_form": form, "objective_scale": objective_scale},
            )
        setup_s = time.perf_counter() - setup_started
        solve_started = time.perf_counter()
        try:
            native_status = solver.solve()
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.NUMERICAL_ERROR,
                primal=None,
                objective_value=None,
                native_status="solve_exception",
                reason=FailureReason.NUMERICAL_FAILURE,
                message=f"{type(exc).__name__}: {exc}",
                setup_s=setup_s,
                solve_s=time.perf_counter() - solve_started,
                diagnostics={"inequality_form": form, "objective_scale": objective_scale},
            )
        solve_s = time.perf_counter() - solve_started
        result = solver.result
        status, reason = _map_status(native_status, piqp)
        primal = None
        objective = None
        if status.has_solution and result.x is not None:
            primal = np.asarray(result.x, dtype=float).reshape(-1).copy()
            objective = float(
                0.5 * primal @ (model.P @ primal)
                + model.q @ primal
                + model.objective_offset
            )
        info = result.info
        diagnostics = {
            "inequality_form": form,
            "objective_scale": objective_scale,
            "objective_scale_reference": model.objective_scale_reference,
            "primal_residual": _number(info, "primal_residual", "primal_res"),
            "dual_residual": _number(info, "dual_residual", "dual_res"),
            "duality_gap": _number(info, "duality_gap", "dual_gap"),
            "solver_setup_s": _number(info, "setup_time"),
            "solver_solve_s": _number(info, "solve_time", "run_time"),
        }
        return BackendResult(
            backend=self.name,
            status=status,
            primal=primal,
            objective_value=objective,
            native_status=str(getattr(info, "status", native_status)),
            reason=reason,
            iterations=int(getattr(info, "iter", 0) or 0),
            setup_s=setup_s,
            solve_s=solve_s,
            diagnostics=diagnostics,
        )


def _objective_scale(model: QuadraticProgram, target: float | None) -> float:
    """Compute a bounded positive scale from explicit alpha dispersion if set."""

    if target is None:
        return 1.0
    magnitude = model.objective_scale_reference
    if magnitude is None:
        magnitude = float(np.max(np.abs(model.q))) if model.q.size else 0.0
    if not np.isfinite(magnitude) or magnitude <= np.finfo(float).eps:
        return 1.0
    return float(np.clip(target / magnitude, 1e-8, 1e8))


def _number(info: object, *names: str) -> float | None:
    for name in names:
        value = getattr(info, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _map_status(native: object, piqp: Any) -> tuple[SolveStatus, FailureReason | None]:
    if native == piqp.Status.PIQP_SOLVED:
        return SolveStatus.OPTIMAL, None
    if native == piqp.Status.PIQP_MAX_ITER_REACHED:
        return SolveStatus.LIMIT_REACHED, FailureReason.MAX_ITER
    if native == piqp.Status.PIQP_PRIMAL_INFEASIBLE:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if native == piqp.Status.PIQP_DUAL_INFEASIBLE:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    if native == piqp.Status.PIQP_NUMERICS:
        return SolveStatus.NUMERICAL_ERROR, FailureReason.NUMERICAL_FAILURE
    return SolveStatus.SOLVER_ERROR, FailureReason.UNKNOWN

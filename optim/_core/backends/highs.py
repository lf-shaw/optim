"""Canonical 线性规划的 HiGHS 适配器。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..canonical import LinearProgram
from ..contracts import (
    CoreFailureReason as FailureReason,
    CoreInfeasibilityContributor,
    CoreInfeasibilityEvidence,
    CoreSolveStatus as SolveStatus,
)
from .base import (
    BackendOptions,
    BackendResult,
    capture_infeasibility,
)
from ..dual_bounds import lp_dual_bound_diagnostics


class HighsBackend:
    """直接求解 :class:`LinearProgram` 的无状态 HiGHS 后端。"""

    name = "highs"

    def solve(self, model: LinearProgram, options: BackendOptions) -> BackendResult:
        """建立并求解一个 canonical LP。

        Parameters
        ----------
        model : LinearProgram
            待求解的 canonical 线性规划。
        options : BackendOptions
            输出和时间限制等通用数值设置。

        Returns
        -------
        BackendResult
            尚未经过公共独立验收的原生状态、完整变量候选和耗时。

        Raises
        ------
        TypeError
            ``model`` 不是 :class:`LinearProgram`。
        ImportError
            当前环境未安装 ``highspy``。
        """

        import highspy

        if not isinstance(model, LinearProgram):
            raise TypeError("HighsBackend accepts LinearProgram only")
        domain = model.domain
        matrix = domain.A.tocsc(copy=False)
        matrix.sort_indices()

        lp = highspy.HighsLp()
        lp.num_col_ = domain.n_variables
        lp.num_row_ = domain.n_constraints
        lp.col_cost_ = np.asarray(model.c, dtype=float)  # type: ignore[assignment]
        lp.col_lower_ = np.asarray(domain.variable_lower, dtype=float)  # type: ignore[assignment]
        lp.col_upper_ = np.asarray(domain.variable_upper, dtype=float)  # type: ignore[assignment]
        lp.row_lower_ = np.asarray(domain.lower, dtype=float)  # type: ignore[assignment]
        lp.row_upper_ = np.asarray(domain.upper, dtype=float)  # type: ignore[assignment]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = domain.n_variables
        lp.a_matrix_.num_row_ = domain.n_constraints
        lp.a_matrix_.start_ = np.asarray(matrix.indptr, dtype=np.int32)  # type: ignore[assignment]
        lp.a_matrix_.index_ = np.asarray(matrix.indices, dtype=np.int32)  # type: ignore[assignment]
        lp.a_matrix_.value_ = np.asarray(matrix.data, dtype=float)  # type: ignore[assignment]
        lp.sense_ = highspy.ObjSense.kMinimize

        solver = highspy.Highs()
        solver.setOptionValue("output_flag", bool(options.verbose))
        if options.time_limit_s is not None:
            solver.setOptionValue("time_limit", float(options.time_limit_s))

        setup_started = time.perf_counter()
        pass_status = solver.passModel(lp)
        setup_s = time.perf_counter() - setup_started
        if pass_status != highspy.HighsStatus.kOk:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status=str(pass_status),
                reason=FailureReason.INVALID_NUMERICS,
                message="HiGHS rejected the canonical LP",
                setup_s=setup_s,
            )

        solve_started = time.perf_counter()
        run_status = solver.run()
        solve_s = time.perf_counter() - solve_started
        native = solver.getModelStatus()
        status, reason = _map_status(native, highspy)
        solution = solver.getSolution()
        info = solver.getInfo()
        primal = None
        objective = None
        if status.has_solution and bool(solution.value_valid):
            primal = np.asarray(solution.col_value, dtype=float).copy()
            objective = float(model.c @ primal + model.objective_offset)
        iterations = int(
            max(
                getattr(info, "simplex_iteration_count", 0),
                getattr(info, "ipm_iteration_count", 0),
                getattr(info, "pdlp_iteration_count", 0),
            )
        )
        diagnostics = {
            "run_status": str(run_status),
            "max_primal_infeasibility": float(
                getattr(info, "max_primal_infeasibility", np.nan)
            ),
            "max_dual_infeasibility": float(
                getattr(info, "max_dual_infeasibility", np.nan)
            ),
            "native_objective": float(
                getattr(info, "objective_function_value", np.nan)
            ),
        }
        if options.collect_dual_bound and status.has_solution and solution.dual_valid:
            diagnostics.update(
                lp_dual_bound_diagnostics(model, solution.row_dual, primal)
            )
        evidence = None
        if status is SolveStatus.INFEASIBLE:
            evidence = capture_infeasibility(
                lambda: _dual_ray(solver, model, highspy, str(native)), diagnostics
            )
        return BackendResult(
            backend=self.name,
            status=status,
            primal=primal,
            objective_value=objective,
            native_status=str(native),
            reason=reason,
            iterations=iterations,
            setup_s=setup_s,
            solve_s=solve_s,
            diagnostics=diagnostics,
            infeasibility=evidence,
        )


def _dual_ray(
    solver: Any, model: LinearProgram, highspy: Any, native: str
) -> CoreInfeasibilityEvidence | None:
    """仅获取已经存在的 ray；presolve 没有留下 ray 时不触发 HiGHS 补充求解。"""

    status, exists = solver.getDualRayExist()
    if status != highspy.HighsStatus.kOk or not exists:
        return None
    status, exists, ray = solver.getDualRay()
    if status != highspy.HighsStatus.kOk or not exists:
        return None
    y = np.asarray(ray, dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    entries = []
    for i in np.flatnonzero(y):
        side = "lower" if y[i] > 0 else "upper"
        entries.append(CoreInfeasibilityContributor("row", int(i), side, float(y[i])))
    # HiGHS ray 给出行乘子；变量边界参与抵消行系数，因此也要映射回原变量域。
    reduced = -model.domain.A.T @ y
    for i in np.flatnonzero(reduced):
        side = "lower" if reduced[i] > 0 else "upper"
        entries.append(
            CoreInfeasibilityContributor("variable", int(i), side, float(reduced[i]))
        )
    return CoreInfeasibilityEvidence(
        "highs", "dual_ray", "numerical_estimate", native, tuple(entries)
    )


def _map_status(
    native: object, highspy: Any
) -> tuple[SolveStatus, FailureReason | None]:
    model_status = highspy.HighsModelStatus
    if native in {model_status.kOptimal, model_status.kObjectiveTarget}:
        return SolveStatus.OPTIMAL, None
    if native == model_status.kInfeasible:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if native == model_status.kUnbounded:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    if native is model_status.kUnboundedOrInfeasible:
        return SolveStatus.SOLVER_ERROR, FailureReason.UNKNOWN
    if native in {
        model_status.kTimeLimit,
        model_status.kIterationLimit,
        model_status.kSolutionLimit,
        model_status.kInterrupt,
        model_status.kHighsInterrupt,
    }:
        reason = (
            FailureReason.TIME_LIMIT
            if native == model_status.kTimeLimit
            else FailureReason.MAX_ITER
        )
        return SolveStatus.LIMIT_REACHED, reason
    if native == model_status.kMemoryLimit:
        return SolveStatus.RESOURCE_ERROR, FailureReason.NUMERICAL_FAILURE
    return SolveStatus.SOLVER_ERROR, FailureReason.NUMERICAL_FAILURE

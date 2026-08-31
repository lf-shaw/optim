"""Canonical 线性规划的 HiGHS 适配器。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..canonical import LinearProgram
from ..contracts import (
    CoreFailureReason as FailureReason,
    CoreSolveStatus as SolveStatus,
)
from .base import BackendOptions, BackendResult


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
            "max_primal_infeasibility": float(getattr(info, "max_primal_infeasibility", np.nan)),
            "max_dual_infeasibility": float(getattr(info, "max_dual_infeasibility", np.nan)),
            "native_objective": float(getattr(info, "objective_function_value", np.nan)),
        }
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
        )


def _map_status(native: object, highspy: Any) -> tuple[SolveStatus, FailureReason | None]:
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

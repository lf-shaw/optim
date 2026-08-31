"""数值核心的统一求解入口。

模块只接受 core 自己的 canonical payload、扁平选项和数值标量。问题分类、参数 QP 搜索、
后端回退及最小 canonical 可行性复算均在一次 ``solve`` 调用中完成。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .backends.base import BackendOptions, BackendResult
from .backends.clarabel import ClarabelBackend
from .backends.highs import HighsBackend
from .backends.mosek import MosekBackend
from .backends.piqp import PIQPBackend
from .canonical import CanonicalModel, FactorQCQP, LinearProgram, QuadraticProgram
from .contracts import CoreFailureReason, CoreSolveStatus, CoreSolverOptions
from .factor_qcqp import solve_factor_qcqp


CORE_ABI_VERSION = 1


@dataclass(frozen=True)
class CoreProblemHandle:
    """同一个进程内短期复用的 canonical 问题句柄。

    Attributes
    ----------
    model : CanonicalModel
        已完成上层编译的只读数值模型。
    """

    model: CanonicalModel


@dataclass(frozen=True)
class CoreSolveResult:
    """一次核心调用返回的最终尝试和完整路由证据。

    Attributes
    ----------
    final : BackendResult
        最后一次数值尝试；只有 ``accepted`` 为真时其 primal 才可进入上层结果验收。
    attempts : tuple[BackendResult, ...]
        按实际执行顺序记录的主求解和回退尝试。
    accepted : bool
        最终候选是否通过核心 canonical 可行性复算。
    max_violation : float
        核心复算发现的最大绝对违约。
    """

    final: BackendResult
    attempts: tuple[BackendResult, ...]
    accepted: bool
    max_violation: float


class CoreSolver:
    """在一个稳定数值边界内路由 LP、QP 和 factor-QCQP。"""

    def __init__(self, options: CoreSolverOptions):
        self.options = options

    def prepare(self, model: CanonicalModel) -> CoreProblemHandle:
        """验证 canonical 类型并创建不复制矩阵的短期句柄。"""

        if not isinstance(model, (LinearProgram, QuadraticProgram, FactorQCQP)):
            raise TypeError(f"unsupported canonical model: {type(model).__name__}")
        return CoreProblemHandle(model)

    def solve(
        self,
        handle: CoreProblemHandle,
        *,
        theta_seed: float | None = None,
        objective_tolerance: float,
    ) -> CoreSolveResult:
        """在一次边界调用内完成主求解、独立复算和必要回退。"""

        if not isinstance(handle, CoreProblemHandle):
            raise TypeError("invalid core problem handle")
        model = handle.model
        backend_options = self._backend_options()
        if isinstance(model, LinearProgram):
            primary = HighsBackend().solve(model, backend_options)
        elif isinstance(model, QuadraticProgram):
            primary = PIQPBackend().solve(model, backend_options)
        elif isinstance(model, FactorQCQP):
            primary = solve_factor_qcqp(
                model,
                self.options,
                theta_seed=theta_seed,
                objective_tolerance=objective_tolerance,
            )
        else:
            raise TypeError(f"unsupported canonical model: {type(model).__name__}")

        primary, accepted, max_violation = self._audit(model, primary)
        attempts = [primary]
        if not accepted and isinstance(model, (QuadraticProgram, FactorQCQP)):
            licensed_terminal = False
            if self.options.licensed_fallback.lower() == "mosek":
                fallback = MosekBackend().solve(model, backend_options)
                fallback, accepted, max_violation = self._audit(model, fallback)
                attempts.append(fallback)
                licensed_terminal = fallback.status in {
                    CoreSolveStatus.INFEASIBLE,
                    CoreSolveStatus.UNBOUNDED,
                }
            if (
                not accepted
                and not licensed_terminal
                and self.options.free_fallback.lower()
                in {"clarabel", "clarabel_qdldl"}
            ):
                fallback = ClarabelBackend().solve(model, backend_options)
                fallback, accepted, max_violation = self._audit(model, fallback)
                attempts.append(fallback)
        return CoreSolveResult(
            final=attempts[-1],
            attempts=tuple(attempts),
            accepted=accepted,
            max_violation=max_violation,
        )

    def _backend_options(self) -> BackendOptions:
        return BackendOptions(
            max_iter=self.options.piqp_max_iter,
            eps_abs=self.options.final_eps,
            eps_rel=self.options.final_eps,
            objective_scale_target=self.options.alpha_target,
            inequality_form=self.options.piqp_inequality_form,
        )

    def _audit(
        self,
        model: CanonicalModel,
        result: BackendResult,
    ) -> tuple[BackendResult, bool, float]:
        """从 primal 复算 canonical 行、变量边界和适用的风险预算。"""

        if not result.status.has_solution or result.primal is None:
            return result, False, 0.0
        vector = np.asarray(result.primal, dtype=float).reshape(-1)
        domain = model.domain
        if vector.shape != (domain.n_variables,) or not np.all(np.isfinite(vector)):
            invalid = replace(
                result,
                status=CoreSolveStatus.NUMERICAL_ERROR,
                reason=CoreFailureReason.INVALID_NUMERICS,
                message="backend returned an invalid primal vector",
            )
            return invalid, False, float("inf")

        activity = np.asarray(domain.A @ vector, dtype=float).reshape(-1)
        lower_violation = np.maximum(domain.lower - activity, 0.0)
        upper_violation = np.maximum(activity - domain.upper, 0.0)
        variable_lower_violation = np.maximum(domain.variable_lower - vector, 0.0)
        variable_upper_violation = np.maximum(vector - domain.variable_upper, 0.0)
        max_violation = max(
            _finite_max(lower_violation),
            _finite_max(upper_violation),
            _finite_max(variable_lower_violation),
            _finite_max(variable_upper_violation),
        )
        risk_operator = getattr(model, "risk_operator", None)
        risk_limit = getattr(model, "risk_limit", None)
        if risk_operator is not None and risk_limit is not None:
            variance, _, _ = risk_operator.components(vector)
            tracking_error = float(np.sqrt(max(0.0, variance)))
            max_violation = max(max_violation, tracking_error - float(risk_limit))

        accepted = bool(max_violation <= self.options.feasibility_tolerance)
        if accepted:
            return replace(result, primal=vector.copy()), True, max_violation
        invalid = replace(
            result,
            status=CoreSolveStatus.NUMERICAL_ERROR,
            reason=CoreFailureReason.INVALID_NUMERICS,
            message=(
                "backend solution failed core validation: max violation "
                f"{max_violation:.3e}"
            ),
        )
        return invalid, False, max_violation


def _finite_max(values: np.ndarray) -> float:
    """返回违约数组中的最大有限正数。"""

    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else 0.0

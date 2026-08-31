"""数值核心跨模块共享的最小不可变契约。

本模块只依赖 Python 标准库，不包含业务单位、用户数据对象或公共结果类型。上层适配器负责
把公共策略转换成这里的扁平数值选项，并把内部状态映射回稳定的公共枚举。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoreSolveStatus(str, Enum):
    """数值核心使用的求解状态。"""

    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    LIMIT_REACHED = "limit_reached"
    NUMERICAL_ERROR = "numerical_error"
    RESOURCE_ERROR = "resource_error"
    SOLVER_ERROR = "solver_error"

    @property
    def has_solution(self) -> bool:
        """返回该状态是否允许携带候选 primal 向量。"""

        return self in {self.OPTIMAL, self.OPTIMAL_INACCURATE}


class CoreFailureReason(str, Enum):
    """核心求解尝试未成功时的标准化内部原因。"""

    UPDATE_FAILURE = "update_failure"
    MAX_ITER = "max_iter"
    TIME_LIMIT = "time_limit"
    INVALID_NUMERICS = "invalid_numerics"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INFEASIBLE_REPORTED = "infeasible_reported"
    UNBOUNDED_REPORTED = "unbounded_reported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CoreSolverOptions:
    """一次核心求解使用的扁平数值与路由选项。

    Attributes
    ----------
    licensed_fallback : str
        商业回退后端标识；空字符串表示禁用。
    free_fallback : str
        免费回退后端标识；空字符串表示禁用。
    lp_prescreen : bool
        是否对 factor-QCQP 启用严格 LP 预筛选。
    rebuild_after_update_failure : bool
        PIQP 更新生命周期失败后是否允许冷重建一次。
    alpha_target : float
        目标系数数值缩放的目标最大绝对量级。
    theta_initial, theta_growth, theta_max : float
        factor-QCQP 一维前沿搜索的初值、扩张倍数和硬上限。
    max_outer_iters : int
        factor-QCQP 前沿搜索最大外层迭代数。
    intermediate_eps, final_eps : float
        中间和最终参数 QP 的数值容差。
    piqp_max_iter : int
        每个 PIQP 子问题的最大迭代数。
    piqp_inequality_form : str
        PIQP 线性不等式展开方式。
    feasibility_tolerance : float
        核心独立复算 canonical 可行性使用的绝对容差。
    risk_margin : float
        factor-QCQP 风险预算保留的年化小数安全边际。
    """

    licensed_fallback: str
    free_fallback: str
    lp_prescreen: bool
    rebuild_after_update_failure: bool
    alpha_target: float
    theta_initial: float
    theta_growth: float
    theta_max: float
    max_outer_iters: int
    intermediate_eps: float
    final_eps: float
    piqp_max_iter: int
    piqp_inequality_form: str
    feasibility_tolerance: float
    risk_margin: float

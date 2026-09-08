"""数值核心跨模块共享的最小不可变契约。

本模块只依赖 Python 标准库，不包含业务单位、用户数据对象或公共结果类型。上层适配器负责
把公共策略转换成这里的扁平数值选项，并把内部状态映射回稳定的公共枚举。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


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
class CoreInfeasibilityContributor:
    """原生不可行证书中一个 canonical 坐标的非零贡献。

    本对象只描述数值坐标，不包含资产、因子或业务约束类型。上层适配器利用 canonical
    registry 将 ``location`` 与 ``index`` 翻译成稳定的公共约束标识。

    Attributes
    ----------
    location : str
        ``"row"``、``"variable"`` 或 ``"cone"``。
    index : int
        对应 canonical 行、变量列或锥块的从零开始位置。
    side : str
        证书作用的边界侧：``"lower"``、``"upper"``、``"equal"`` 或 ``"cone"``。
    multiplier : float
        带原生符号的证书乘子；不能解释为所需放宽量或业务重要性。
    identifier : str | None
        不由线性 registry 表示的锥块稳定标识，例如 ``"tracking_error"``。
    """

    location: str
    index: int
    side: str
    multiplier: float
    identifier: str | None = None


@dataclass(frozen=True)
class CoreInfeasibilityEvidence:
    """求解器已产生的结构化原生不可行证书摘要。

    后端只提取求解时已经存在的数值向量，不在普通失败路径额外执行 IIS、重新优化或日志
    解析。``contributors`` 因而是辅助定位线索；显式 Phase-I 给出依赖惩罚尺度的一个松弛
    方案，不承诺唯一修复或逐约束最小放宽量。

    Attributes
    ----------
    backend : str
        产生证书的稳定后端标识。
    kind : str
        原生证据类型，例如 ``"dual_ray"`` 或
        ``"primal_infeasibility_certificate"``。
    proof_status : str
        ``"verified"``、``"numerical_estimate"`` 或 ``"unavailable"``。
    native_status : str
        产生证书时的原生求解状态。
    contributors : tuple[CoreInfeasibilityContributor, ...]
        保留原生符号的非零 canonical 坐标，不按量级过滤；锥向量逐分量保留。
    certificate_residual : float | None
        可计算时的归一化证书平稳性残差。
    certificate_margin : float | None
        可计算时的归一化严格不可行裕量；方向由各后端适配为正值代表有效证据。
    metadata : Mapping[str, Any]
        小型只读原生元数据；公共逻辑不得依赖其中的后端专用键。
    """

    backend: str
    kind: str
    proof_status: str
    native_status: str
    contributors: tuple[CoreInfeasibilityContributor, ...] = ()
    certificate_residual: float | None = None
    certificate_margin: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class CoreSolverOptions:
    """一次核心求解使用的扁平数值与路由选项。

    Attributes
    ----------
    backend : str
        auto 自动路由；显式值只使用指定后端，不回退。
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
    backend: str = "auto"

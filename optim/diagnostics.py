"""不可行诊断的稳定公共结果类型。

诊断算法属于私有二进制核心；本模块只定义调用方可以长期依赖、序列化和展示的结果契约。
普通求解失败不会自动运行深度诊断，调用方需显式调用
``PortfolioOptimizer.diagnose(...)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .portfolio_types import SolverAttempt


@dataclass(frozen=True)
class RequiredRelaxation:
    """Phase-I 识别出的一条需要放宽的 canonical 边界。

    Attributes
    ----------
    constraint_id : str
        稳定的 canonical 约束标识。
    group : str
        业务约束组。
    side : str
        需要放宽的边界侧，``"lower"`` 或 ``"upper"``。
    amount : float
        使用该约束原始单位表示的最小松弛量。
    configured_bound : float
        原问题配置的边界值。
    diagnostic_scale : float
        Phase-I 目标中用于跨单位比较的正数尺度。
    key : str | None
        适用时的资产、因子或属性键。
    """

    constraint_id: str
    group: str
    side: str
    amount: float
    configured_bound: float
    diagnostic_scale: float
    key: str | None = None


@dataclass(frozen=True)
class InfeasibilityReport:
    """结构化不可行证据，而不是自动应用的修复。

    Attributes
    ----------
    stage : str
        已执行的诊断阶段，当前深度诊断为 ``"deep"``。
    linear_feasible : bool | None
        去除非线性风险预算后，公共线性域是否可行。
    summary_text : str
        面向使用者的中文诊断摘要。
    turnover_linear_lower_bound : float | None
        保持其余线性约束时的最小 L1 换手率。存在 TE 约束时，它只是完整问题的下界。
    turnover_convex_minimum : float | None
        恢复搜索确认的完整凸问题可行换手率边界或上界。
    turnover_limit : float | None
        原问题配置的换手率上限。
    minimum_tracking_error : float | None
        线性域内可实现的最小年化小数跟踪误差。
    tracking_error_limit : float | None
        原问题配置的年化小数跟踪误差预算。
    relaxations : tuple[RequiredRelaxation, ...]
        Phase-I 所需松弛，按尺度化严重程度降序排列。
    native_evidence : Mapping[str, Any] | None
        只读的原生诊断状态和目标值。
    attempts : tuple[SolverAttempt, ...]
        诊断过程中执行的全部后端尝试。
    """

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
        """把后端证据冻结为只读映射，避免报告在返回后被意外修改。"""

        object.__setattr__(
            self,
            "native_evidence",
            MappingProxyType(
                {} if self.native_evidence is None else dict(self.native_evidence)
            ),
        )

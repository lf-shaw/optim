"""极薄的求解器协议和标准化原生尝试结果。

后端只接收已编译模型，不了解 pandas、数据对齐、多期状态或业务恢复策略。
``BackendResult`` 只是内部证据，只有 API 的独立验收器能将其提升为公共
:class:`OptimizationResult`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from ..contracts import (
    CoreFailureReason as FailureReason,
    CoreInfeasibilityEvidence,
    CoreInfeasibilityContributor,
    CoreSolveStatus as SolveStatus,
)


@dataclass(frozen=True)
class BackendOptions:
    """传给求解器适配器的公共数值控制项。

    Attributes
    ----------
    verbose : bool
        是否允许原生求解器输出进度。
    time_limit_s : float | None
        可选 wall-clock 秒数；原生求解器可能近似执行，``None`` 表示不设置。
    max_iter : int
        后端支持时设置的最大原生迭代数。
    eps_abs, eps_rel : float
        请求的绝对和相对数值容差。
    objective_scale_target : float | None
        适配器改善条件数时使用的目标缩放量级；``None`` 禁用后端侧目标缩放。
    inequality_form : str
        PIQP 不等式表示（``"auto"``、``"compact"`` 或 ``"one_sided"``）；其他后端忽略。
    collect_dual_bound : bool
        显式诊断 LP 是否复算拉格朗日下界；默认关闭，避免普通优化增加矩阵运算。
    """

    verbose: bool = False
    time_limit_s: float | None = None
    max_iter: int = 1000
    eps_abs: float = 1e-8
    eps_rel: float = 1e-8
    objective_scale_target: float | None = 0.2
    inequality_form: str = "auto"
    collect_dual_bound: bool = False


@dataclass(frozen=True)
class BackendResult:
    """一次后端尝试返回的标准化原生证据。

    原生状态成功并不代表公共解已验收；优化器仍会使用 canonical 模型独立检查 ``primal``。

    Attributes
    ----------
    backend : str
        稳定后端标识，例如 ``"highs"`` 或 ``"piqp"``。
    status : SolveStatus
        从原生求解器状态映射得到的标准状态。
    primal : numpy.ndarray | None
        包含辅助变量的完整 canonical 候选向量；没有候选时为 ``None``。
    objective_value : float | None
        canonical 单位下的原生最小化目标值。
    native_status : str
        用于排查的原始或字符串化后端状态。
    reason : FailureReason | None
        尝试未成功时的标准化失败类别。
    message : str
        可读的原生或适配器诊断。
    iterations : int
        可用时的原生迭代数，否则为零。
    setup_s, solve_s : float
        适配器测得的建立与求解秒数。
    diagnostics : Mapping[str, Any]
        只读后端专用遥测；调用方不得将其当作独立可行性证书。
    infeasibility : CoreInfeasibilityEvidence | None
        求解时已经产生的结构化原生不可行证书；不可用时为 ``None``。该字段不会触发额外
        求解，也不替代显式 Phase-I 诊断。
    """

    backend: str
    status: SolveStatus
    primal: np.ndarray | None
    objective_value: float | None
    native_status: str
    reason: FailureReason | None = None
    message: str = ""
    iterations: int = 0
    setup_s: float = 0.0
    solve_s: float = 0.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    infeasibility: CoreInfeasibilityEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


class SolverBackend(Protocol):
    """无状态 canonical 模型后端实现的最小协议。"""

    name: str

    def solve(self, model: Any, options: BackendOptions) -> BackendResult:
        """求解一个已编译模型并返回尚未验收的原生证据。

        Parameters
        ----------
        model : Any
            适配器支持的求解器无关 canonical 模型。
        options : BackendOptions
            本次尝试的数值控制项。

        Returns
        -------
        BackendResult
            标准化原生状态、候选向量和遥测。
        """
        ...


def capture_infeasibility(
    reader: Any, diagnostics: dict[str, Any]
) -> CoreInfeasibilityEvidence | None:
    """隔离可选原生证据读取失败；不改变原求解状态，也不执行额外求解。"""

    try:
        return reader()
    except Exception as exc:
        diagnostics["infeasibility_evidence_error"] = type(exc).__name__
        return None


def dual_entries(
    location: str, indices: Any, side: str, values: Any, identifier: str | None = None
) -> list[CoreInfeasibilityContributor]:
    """按原生顺序保留非零带符号乘子；不按幅度裁剪证书，不推导冲突排名。"""

    vector = np.asarray(values, dtype=float).reshape(-1)
    positions = np.asarray(indices).reshape(-1)
    if len(vector) != len(positions) or not np.all(np.isfinite(vector)):
        raise ValueError("invalid native certificate shape or numerics")
    return [
        CoreInfeasibilityContributor(location, int(i), side, float(v), identifier)
        for i, v in zip(positions, vector)
        if v != 0.0
    ]

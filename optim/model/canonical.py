"""所有求解器后端共享的强类型不可变数值 payload。

Canonical 对象不包含业务侧 DataFrame 对齐或原生求解器实例；其中数组是后端调用、独立验收、
fingerprint 和不可行诊断的唯一事实来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
import scipy.sparse as sp

from ..portfolio_types import ProblemFingerprint, ProblemKind


@dataclass(frozen=True)
class VariableRecord:
    """一个 canonical 变量列的审计记录。

    Attributes
    ----------
    index : int
        变量在完整 canonical 向量 ``z`` 中的从零开始列位置。
    variable_id : str
        跨求解器稳定的变量标识。
    group : str
        业务或辅助变量组，例如 ``"weight"``、``"turnover_aux"``。
    key : str | None
        适用时的资产、因子或属性标签。
    unit : str
        变量数值单位，用于诊断和展示。
    """

    index: int
    variable_id: str
    group: str
    key: str | None = None
    unit: str = "weight"


@dataclass(frozen=True)
class ConstraintRecord:
    r"""一个 canonical 行约束或变量边界的审计记录。

    Attributes
    ----------
    constraint_id : str
        跨后端稳定的约束标识。
    group : str
        对应的业务约束组。
    location : str
        ``"row"`` 表示 $l\le Az\le u$ 的矩阵行，``"variable"`` 表示变量列边界。
    index : int
        在对应 row 或 variable 坐标中的位置。
    key : str | None
        适用时的资产、因子或属性标签。
    unit : str
        约束活动值和边界的数值单位。
    source : str
        约束来源，例如用户配置、编译器恒等式或交易指令。
    relaxable : bool
        Phase-I 诊断是否允许为该边界引入 slack。
    diagnostic_scale : float
        跨单位比较所需松弛时使用的正数尺度。
    metadata : Mapping[str, Any]
        只读的补充业务审计信息。
    """

    constraint_id: str
    group: str
    location: str
    index: int
    key: str | None = None
    unit: str = "weight"
    source: str = "user"
    relaxable: bool = True
    diagnostic_scale: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.location not in {"row", "variable"}:
            raise ValueError("constraint location must be 'row' or 'variable'")
        # 编译器会驻留大规模逐资产 registry 的公共只读 metadata；此处保留共享对象，避免为
        # 每个 ConstraintRecord 重复复制字典。
        frozen = (
            self.metadata
            if isinstance(self.metadata, MappingProxyType)
            else MappingProxyType(dict(self.metadata))
        )
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True)
class LinearDomain:
    r"""由 $l\le Az\le u$ 和变量列边界构成的多面体可行域。

    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        形状为 ``(n_constraints, n_variables)`` 的 canonical CSC 约束矩阵。
    lower, upper : numpy.ndarray
        每个矩阵行的下限和上限；无界侧使用无穷，shape 为 ``(n_constraints,)``。
    variable_lower, variable_upper : numpy.ndarray
        每个变量列的下限和上限，shape 为 ``(n_variables,)``。
    variables : tuple[VariableRecord, ...]
        覆盖全部 canonical 列的变量 registry。
    constraints : tuple[ConstraintRecord, ...]
        业务可审计的矩阵行及变量边界 registry。
    weight_indices : numpy.ndarray
        组合权重在完整 canonical 向量中的列位置，shape 为 ``(n_assets,)``。
    assets : tuple[Any, ...]
        与 ``weight_indices`` 位置一致的资产标签。
    """

    A: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    variables: tuple[VariableRecord, ...]
    constraints: tuple[ConstraintRecord, ...]
    weight_indices: np.ndarray
    assets: tuple[Any, ...]

    @property
    def n_variables(self) -> int:
        """返回 canonical 变量列数。"""

        return int(self.A.shape[1])  # type: ignore

    @property
    def n_constraints(self) -> int:
        """返回 canonical 矩阵行约束数。"""

        return int(self.A.shape[0])  # type: ignore


@dataclass(frozen=True)
class FactorRiskOperator:
    r"""年化“因子风险加特异风险”跟踪风险算子。

    对主动权重 $a=x-b$，总方差为

    $$
    R(a)=(E^{\mathsf T}a)^{\mathsf T}F(E^{\mathsf T}a)
    +\lVert d\odot a\rVert_2^2.
    $$

    算子保存公共年化小数输入，可以计算基础 canonical 向量或带更多辅助列的扩展向量。

    Attributes
    ----------
    exposure : numpy.ndarray
        资产乘因子暴露矩阵。
    covariance : numpy.ndarray
        年化小数因子协方差矩阵。
    specific_volatility : numpy.ndarray
        逐资产年化小数特异波动率。
    benchmark : numpy.ndarray
        与资产顺序一致的基准权重。
    factor_names : tuple[str, ...]
        与暴露列和协方差轴一致的因子名称。
    weight_indices : numpy.ndarray
        从完整 canonical 向量中提取组合权重的列位置。
    """

    exposure: np.ndarray
    covariance: np.ndarray
    specific_volatility: np.ndarray
    benchmark: np.ndarray
    factor_names: tuple[str, ...]
    weight_indices: np.ndarray

    def components(self, vector: np.ndarray) -> tuple[float, float, float]:
        """计算总方差、因子方差和特异方差。

        Parameters
        ----------
        vector : numpy.ndarray
            完整 canonical 候选向量；至少包含 ``weight_indices`` 指向的列。

        Returns
        -------
        tuple[float, float, float]
            依次为总方差、因子方差和特异方差，均使用年化小数方差单位。
        """

        weight = np.asarray(vector, dtype=float)[self.weight_indices]
        active = weight - self.benchmark
        factor = self.exposure.T @ active
        factor_variance = float(factor @ self.covariance @ factor)
        specific_variance = float(np.square(self.specific_volatility * active).sum())
        total = max(0.0, factor_variance + specific_variance)
        return total, factor_variance, specific_variance


@dataclass(frozen=True)
class LinearProgram:
    """使用线性目标和 :class:`LinearDomain` 的 canonical LP。

    Attributes
    ----------
    kind : ProblemKind
        必须为 :attr:`ProblemKind.LP`。
    domain : LinearDomain
        与目标共享变量顺序的多面体可行域。
    c : numpy.ndarray
        canonical 最小化目标系数，shape 为 ``(n_variables,)``。
    objective_offset : float
        不影响最优解、但复原目标值时必须计入的常数项。
    """

    kind: ProblemKind
    domain: LinearDomain
    c: np.ndarray
    objective_offset: float = 0.0


@dataclass(frozen=True)
class QuadraticProgram:
    r"""多面体可行域上的凸二次最小化问题。

    $$
    \min_z\quad \frac12 z^{\mathsf T}Pz+q^{\mathsf T}z+c.
    $$

    Attributes
    ----------
    kind : ProblemKind
        必须为 :attr:`ProblemKind.QP`。
    domain : LinearDomain
        与目标共享变量顺序的多面体可行域。
    P : scipy.sparse.csc_matrix
        对称半正定 Hessian，shape 为 ``(n_variables, n_variables)``。
    q : numpy.ndarray
        线性目标系数，shape 为 ``(n_variables,)``。
    objective_offset : float
        数学目标中的常数 $c$。
    objective_scale_reference : float | None
        后端数值缩放使用的 alpha 离散量级；不改变数学目标。
    risk_operator : FactorRiskOperator | None
        可用时供独立验收复算风险的算子。
    risk_limit : float | None
        适用时的年化小数风险上限。
    """

    kind: ProblemKind
    domain: LinearDomain
    P: sp.csc_matrix
    q: np.ndarray
    objective_offset: float = 0.0
    objective_scale_reference: float | None = None
    risk_operator: FactorRiskOperator | None = None
    risk_limit: float | None = None


@dataclass(frozen=True)
class FactorQCQP:
    """在线性域上最大化 alpha，并施加一个因子模型 TE 预算。

    Attributes
    ----------
    kind : ProblemKind
        必须为 :attr:`ProblemKind.FACTOR_QCQP`。
    domain : LinearDomain
        不含风险预算的公共多面体可行域。
    alpha : numpy.ndarray
        按 ``domain.assets`` 顺序排列的线性最大化系数，shape 为 ``(n_assets,)``。
    risk_operator : FactorRiskOperator
        用于风险预算和独立复算的 factor-model 算子。
    risk_limit : float
        正数年化小数跟踪误差上限。
    """

    kind: ProblemKind
    domain: LinearDomain
    alpha: np.ndarray
    risk_operator: FactorRiskOperator
    risk_limit: float


CanonicalModel: TypeAlias = LinearProgram | QuadraticProgram | FactorQCQP


@dataclass(frozen=True)
class CompiledProblem:
    """Canonical 数值 payload、审计身份和精确编译优化记录。

    Attributes
    ----------
    model : CanonicalModel
        路由到数值后端的不可变 canonical 模型。
    fingerprint : ProblemFingerprint
        同时覆盖业务语义和最终数值 payload 的问题身份。
    compiler_optimizations : tuple[str, ...]
        已证明数学等价并实际应用的结构优化标识，例如稀疏换手率展开。
    """

    model: CanonicalModel
    fingerprint: ProblemFingerprint
    compiler_optimizations: tuple[str, ...] = ()

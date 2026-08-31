"""统一组合优化器的公共数据契约。

本模块刻意不导入任何求解器；即使未安装 MOSEK、PIQP、Clarabel、tuda2 或 carry，
调用方仍应能够安全导入 ``optim`` 并构造问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
import pandas as pd


class ProblemKind(str, Enum):
    """用于后端路由的 canonical 数学问题类型。"""

    LP = "lp"
    QP = "qp"
    FACTOR_QCQP = "factor_qcqp"
    CONIC = "conic"


class SolveStatus(str, Enum):
    """与具体求解器无关的单次求解或序列步骤状态。"""

    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    LIMIT_REACHED = "limit_reached"
    NUMERICAL_ERROR = "numerical_error"
    RESOURCE_ERROR = "resource_error"
    SOLVER_ERROR = "solver_error"
    SKIPPED = "skipped"

    @property
    def has_solution(self) -> bool:
        """该状态是否允许携带经过独立验收的组合权重。"""

        return self in {self.OPTIMAL, self.OPTIMAL_INACCURATE}


class ProofStatus(str, Enum):
    """最优性证书所依据的数学证据强度。"""

    VERIFIED = "verified"
    NUMERICAL_ESTIMATE = "numerical_estimate"
    UNAVAILABLE = "unavailable"


class FailureReason(str, Enum):
    """原生求解尝试未成功时的标准化原因。"""

    UPDATE_FAILURE = "update_failure"
    MAX_ITER = "max_iter"
    TIME_LIMIT = "time_limit"
    INVALID_NUMERICS = "invalid_numerics"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INFEASIBLE_REPORTED = "infeasible_reported"
    INFEASIBLE_CONFIRMED = "infeasible_confirmed"
    UNBOUNDED_REPORTED = "unbounded_reported"
    UNBOUNDED_CONFIRMED = "unbounded_confirmed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DataProvenance:
    """输入数据集或已物化截面的来源信息。

    Attributes
    ----------
    source : str
        稳定的数据源标识，例如 ``"memory"`` 或 ``"tuda2"``。
    source_date : pandas.Timestamp | None
        实际使用的数据日期；在严格同日语义下必须等于优化日期。
    version : str | None
        数据供应商、风险模型或适配器版本。
    metadata : Mapping[str, Any]
        只读的数据源专用审计信息。
    """

    source: str = "memory"
    source_date: pd.Timestamp | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class AlphaSpec:
    """输入 alpha 的业务单位与归一化尺度。

    Attributes
    ----------
    units : str
        业务单位说明，例如 ``"standardized_score"``；优化器不会推断或转换该单位。
    scale : float
        一个归一化目标单位对应的正数原始 alpha 数量，用于将
        ``ObjectiveTolerance.normalized`` 转换为原始目标差。
    """

    units: str = "standardized_score"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("AlphaSpec.scale must be positive and finite")


@dataclass(frozen=True)
class ObjectiveTolerance:
    """允许的原始单位和/或归一化单位目标损失。

    最终原始目标差上限是两个已配置分量之和，而不是二者的较小值。

    Attributes
    ----------
    absolute : float | None
        以输入 alpha 原始单位计量的允许损失。
    normalized : float | None
        以 :attr:`AlphaSpec.scale` 的倍数计量的允许损失。
    """

    absolute: float | None = None
    normalized: float | None = 1e-4

    def __post_init__(self) -> None:
        if self.absolute is None and self.normalized is None:
            raise ValueError("at least one objective tolerance must be provided")
        for name, value in (("absolute", self.absolute), ("normalized", self.normalized)):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"ObjectiveTolerance.{name} must be finite and non-negative")

    def raw_limit(self, alpha_spec: AlphaSpec) -> float:
        """将各容差分量换算为一个原始单位目标差上限。

        Parameters
        ----------
        alpha_spec : AlphaSpec
            当前问题 alpha 向量的尺度说明。

        Returns
        -------
        float
            ``absolute + normalized * alpha_spec.scale``；未配置的分量按零处理。
        """

        absolute = 0.0 if self.absolute is None else self.absolute
        normalized = 0.0 if self.normalized is None else self.normalized * alpha_spec.scale
        return float(absolute + normalized)


@dataclass(frozen=True)
class FactorRiskModel:
    """采用年化小数风险单位的单日因子风险模型。

    ``exposure`` 的行对应 ``PortfolioData.assets``，列对应 ``factor_names``；
    ``covariance`` 的两个轴使用同一因子顺序，``specific_volatility`` 使用资产顺序。
    对象构造后，核心层不会再推断单位或隐式重排数据。

    Attributes
    ----------
    asof : pandas.Timestamp
        所有风险模型数组共同对应的准确收盘日期。
    exposure : numpy.ndarray
        资产乘因子的暴露矩阵，shape 为 ``(n_assets, n_factors)``。
    covariance : numpy.ndarray
        对称因子协方差矩阵，shape 为 ``(n_factors, n_factors)``，单位为年化小数方差。
    specific_volatility : numpy.ndarray
        个股年化小数特异波动率，shape 为 ``(n_assets,)``。
    factor_names : tuple[str, ...]
        与暴露列、协方差两个轴位置一致的因子名称。
    factor_types : tuple[str, ...]
        每个因子的语义类型，例如 ``"style"`` 或 ``"industry"``。
    provenance : DataProvenance
        该日模型的数据来源和版本信息。
    annualization : str
        显式风险单位契约；当前编译器要求为 ``"annualized_decimal"``。
    """

    asof: pd.Timestamp
    exposure: np.ndarray
    covariance: np.ndarray
    specific_volatility: np.ndarray
    factor_names: tuple[str, ...]
    factor_types: tuple[str, ...]
    provenance: DataProvenance = field(default_factory=DataProvenance)
    annualization: str = "annualized_decimal"


@dataclass(frozen=True)
class FullCovarianceRiskModel:
    """单日全资产协方差风险模型。

    该契约目前用于问题分类和校验；当前高性能 QP/factor-QCQP 路径要求
    :class:`FactorRiskModel`。

    Attributes
    ----------
    asof : pandas.Timestamp
        协方差矩阵对应的准确收盘日期。
    covariance : numpy.ndarray
        资产协方差矩阵，shape 为 ``(n_assets, n_assets)``，单位为年化小数方差。
    provenance : DataProvenance
        矩阵的数据来源和版本信息。
    annualization : str
        显式风险单位契约，通常为 ``"annualized_decimal"``。
    """

    asof: pd.Timestamp
    covariance: np.ndarray
    provenance: DataProvenance = field(default_factory=DataProvenance)
    annualization: str = "annualized_decimal"


RiskModel: TypeAlias = FactorRiskModel | FullCovarianceRiskModel


@dataclass(frozen=True)
class PortfolioData:
    """使用唯一权威资产顺序的单日数值输入。

    所有资产维数组均按 ``assets`` 的位置解释。带标签的 pandas 对齐必须在数据源适配层完成，
    不能推迟到校验器或编译器。只有不使用 alpha 的目标才允许 ``alpha=None``。

    Attributes
    ----------
    date : pandas.Timestamp
        优化信息日；输入表示该日收盘后已知、供下一次执行机会使用的信息。
    assets : pandas.Index
        所有资产维数组唯一且无重复的权威位置坐标。
    alpha : numpy.ndarray | None
        目标得分，shape 为 ``(n_assets,)``；仅不含 alpha 下限的跟踪误差最小化允许为空。
    benchmark : numpy.ndarray | None
        基准权重，shape 为 ``(n_assets,)``，通常合计为 1。
    initial_weight : numpy.ndarray | None
        交易前实际组合权重，shape 为 ``(n_assets,)``；换手率、冻结和单边交易约束需要该字段。
    tradable : numpy.ndarray
        是否可交易的布尔掩码，shape 为 ``(n_assets,)``。
    risk_model : RiskModel | None
        风险目标或约束所需的严格同日风险模型。
    alpha_spec : AlphaSpec | None
        alpha 的单位和尺度；请求归一化目标证书时必须提供。
    extra_attributes : Mapping[str, numpy.ndarray]
        命名的逐资产属性，每项 shape 均为 ``(n_assets,)``，用于自定义绝对或主动敞口约束。
    provenance : DataProvenance
        组装后单日数据的来源信息。
    """

    date: pd.Timestamp
    assets: pd.Index
    alpha: np.ndarray | None
    benchmark: np.ndarray | None
    initial_weight: np.ndarray | None
    tradable: np.ndarray
    risk_model: RiskModel | None = None
    alpha_spec: AlphaSpec | None = None
    extra_attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", pd.Timestamp(self.date))
        object.__setattr__(self, "assets", pd.Index(self.assets, copy=False))
        object.__setattr__(
            self,
            "extra_attributes",
            MappingProxyType(dict(self.extra_attributes)),
        )


@dataclass(frozen=True)
class MaximizeAlpha:
    r"""最大化线性得分 $\alpha^{\mathsf T}x$。

    该标记对象没有字段。未配置跟踪误差上限时问题属于 LP；配置后进入专用
    factor-QCQP 路径。
    """

    pass


@dataclass(frozen=True)
class RiskAdjustedAlpha:
    """最大化扣除因子和特异方差惩罚后的 alpha。

    Attributes
    ----------
    factor_aversion : float
        乘在年化因子方差上的非负风险厌恶系数。
    specific_aversion : float
        乘在年化特异方差上的非负风险厌恶系数。

    Notes
    -----
    优化器不会自动统一 alpha 与风险的量纲。因此，这两个系数只有结合声明的 alpha 尺度和
    年化小数方差口径才具有明确业务含义。
    """

    factor_aversion: float = 0.75
    specific_aversion: float = 0.75

    def __post_init__(self) -> None:
        if self.factor_aversion < 0.0 or self.specific_aversion < 0.0:
            raise ValueError("risk aversion must be non-negative")
        if self.factor_aversion == 0.0 and self.specific_aversion == 0.0:
            raise ValueError("at least one risk aversion must be positive")


@dataclass(frozen=True)
class MinimizeTrackingError:
    r"""最小化年化跟踪误差方差。

    Attributes
    ----------
    alpha_floor : float | None
        $\alpha^{\mathsf T}x$ 的可选下限，单位为原始 alpha 目标单位；``None`` 表示允许
        :attr:`PortfolioData.alpha` 缺失。
    """

    alpha_floor: float | None = None


PortfolioObjective: TypeAlias = MaximizeAlpha | RiskAdjustedAlpha | MinimizeTrackingError


AssetWeightOverride: TypeAlias = float | tuple[float, float]


@dataclass(frozen=True)
class AssetTradeConstraints:
    """只对本次优化生效的逐资产交易指令。

    指令属于 :class:`PortfolioProblem`，不会保存为优化器的可变状态，因此不会泄漏到后续
    实盘请求。

    ``blacklist`` 将目标权重固定为零；``frozen`` 固定为输入的期初权重；
    ``not_buyable`` 和 ``not_sellable`` 相对期初权重施加单边限制；
    ``weight_overrides`` 接受精确目标或 ``(lower, upper)`` 区间。

    Attributes
    ----------
    blacklist : tuple[Any, ...]
        本次优化中目标权重强制为零的资产。
    frozen : tuple[Any, ...]
        固定在输入期初权重的资产。
    not_buyable : tuple[Any, ...]
        目标权重不得高于期初权重的资产。
    not_sellable : tuple[Any, ...]
        目标权重不得低于期初权重的资产。
    weight_overrides : Mapping[Any, AssetWeightOverride]
        每只资产的精确目标（标量）或闭区间 ``(lower, upper)``；显式覆盖仍受绝对权重和
        long-only 边界限制。
    missing_asset : str
        ``"error"`` 拒绝问题资产域以外的指令，``"ignore"`` 明确忽略这些指令。
    """

    blacklist: tuple[Any, ...] = ()
    frozen: tuple[Any, ...] = ()
    not_buyable: tuple[Any, ...] = ()
    not_sellable: tuple[Any, ...] = ()
    weight_overrides: Mapping[Any, AssetWeightOverride] = field(default_factory=dict)
    missing_asset: str = "error"

    def __post_init__(self) -> None:
        for name in ("blacklist", "frozen", "not_buyable", "not_sellable"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "weight_overrides",
            MappingProxyType(dict(self.weight_overrides)),
        )
        if self.missing_asset not in {"error", "ignore"}:
            raise ValueError("missing_asset must be 'error' or 'ignore'")

    @property
    def is_empty(self) -> bool:
        """是否完全没有配置逐资产交易指令。"""

        return not (
            self.blacklist
            or self.frozen
            or self.not_buyable
            or self.not_sellable
            or self.weight_overrides
        )


@dataclass(frozen=True)
class WeightBounds:
    """施加主动权重和交易指令前的绝对目标权重边界。

    Attributes
    ----------
    lower, upper : float | numpy.ndarray
        广播到所有资产的闭区间标量边界，或 shape 为 ``(n_assets,)`` 的位置数组。
    """

    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 1.0


@dataclass(frozen=True)
class SymmetricBound:
    r"""对称绝对值边界 $-c\le y\le c$。

    Attributes
    ----------
    absolute : float
        使用被约束量自身单位表示的非负半宽 $c$。
    """

    absolute: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError("symmetric bound must be finite and non-negative")


@dataclass(frozen=True)
class TurnoverLimit:
    r"""目标组合相对期初组合的换手率上限。

    Attributes
    ----------
    limit : float
        $\lVert x-x_0\rVert_1$ 的非负上限，单位为小数权重。
    convention : str
        ``"l1"`` 是 canonical 定义；``"two_way"`` 是保留的描述性兼容别名，目前映射到
        同一数值。
    """

    limit: float
    convention: str = "l1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.limit) or self.limit < 0.0:
            raise ValueError("turnover limit must be finite and non-negative")
        if self.convention not in {"l1", "two_way"}:
            raise ValueError("turnover convention must be 'l1' or 'two_way'")

    @property
    def l1_limit(self) -> float:
        r"""返回 canonical $\lVert x-x_0\rVert_1$ 上限。"""

        # 历史 optim 和 benchmark 均将换手率定义为
        # $\lVert x-x_0\rVert_1$.
        # ``two_way`` 仅作为该口径的描述性别名保留。
        return float(self.limit)


@dataclass(frozen=True)
class LowerBound:
    """闭区间标量下限。

    Attributes
    ----------
    value : float
        使用被约束量自身单位表示的最小允许值。
    """

    value: float


@dataclass(frozen=True)
class ExposureBounds:
    """默认及逐因子的敞口上下限。

    Attributes
    ----------
    default : tuple[float, float] | None
        除显式覆盖外应用到所有匹配因子的闭区间；``None`` 表示只约束显式命名因子。
    overrides : Mapping[str, tuple[float, float]]
        不区分大小写的逐因子覆盖；对象构造时键会统一转为小写。
    """

    default: tuple[float, float] | None = None
    overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType({str(k).lower(): tuple(v) for k, v in self.overrides.items()}),
        )


@dataclass(frozen=True)
class TrackingErrorLimit:
    """年化小数跟踪误差上限。

    Attributes
    ----------
    annualized : float
        正数年化小数波动率，例如 2% 写作 ``0.02``。
    """

    annualized: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.annualized) or self.annualized <= 0.0:
            raise ValueError("tracking error limit must be positive and finite")


@dataclass(frozen=True)
class PortfolioConstraints:
    r"""LP、QP 和 QCQP 路径共享的不可变可行域配置。

    风险预算是唯一非线性约束。换手率、主动权重、因子敞口和逐资产交易指令都编译到同一
    多面体可行域，因此它们本身不会决定求解器类型。

    Attributes
    ----------
    long_only : bool
        为真时，将每只资产的下限与零取交集。
    budget : float
        目标权重合计值。
    asset_weight : WeightBounds
        逐资产绝对目标权重区间。
    active_weight : SymmetricBound | None
        每只资产 $x_i-b_i$ 的边界；配置后必须提供基准。
    total_active : float | None
        $\lVert x-b\rVert_1$ 的上限。
    turnover : TurnoverLimit | None
        目标组合相对期初组合的换手率上限。
    benchmark_member_weight : LowerBound | None
        基准权重大于零的资产在目标组合中的最小合计权重。
    style, industry : ExposureBounds | None
        对应风险模型类型的风格或行业主动敞口边界。
    tracking_error : TrackingErrorLimit | None
        非线性的年化跟踪误差预算。
    freeze_nontradable : bool
        是否将 ``tradable=False`` 的资产固定在期初权重。
    asset_trade : AssetTradeConstraints | None
        仅对本问题生效的黑名单、冻结、单边交易和权重覆盖指令。
    extra_active : Mapping[str, tuple[float, float]]
        命名 ``extra_attributes`` 相对基准的主动敞口边界。
    extra_absolute : Mapping[str, tuple[float, float]]
        命名 ``extra_attributes`` 的组合绝对敞口边界。
    """

    long_only: bool = True
    budget: float = 1.0
    asset_weight: WeightBounds = field(default_factory=WeightBounds)
    active_weight: SymmetricBound | None = None
    total_active: float | None = None
    turnover: TurnoverLimit | None = None
    benchmark_member_weight: LowerBound | None = None
    style: ExposureBounds | None = None
    industry: ExposureBounds | None = None
    tracking_error: TrackingErrorLimit | None = None
    freeze_nontradable: bool = True
    asset_trade: AssetTradeConstraints | None = None
    extra_active: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    extra_absolute: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asset_trade is not None and not isinstance(
            self.asset_trade, AssetTradeConstraints
        ):
            raise TypeError("asset_trade must be AssetTradeConstraints or None")
        object.__setattr__(self, "extra_active", MappingProxyType(dict(self.extra_active)))
        object.__setattr__(self, "extra_absolute", MappingProxyType(dict(self.extra_absolute)))


@dataclass(frozen=True)
class PortfolioProblem:
    """单次组合优化请求的完整无状态定义。

    Attributes
    ----------
    data : PortfolioData
        使用唯一资产顺序的严格同日数值输入。
    objective : PortfolioObjective
        与求解器选择无关的首要业务目标。
    constraints : PortfolioConstraints
        不可变的可行域定义。
    """

    data: PortfolioData
    objective: PortfolioObjective
    constraints: PortfolioConstraints


@dataclass(frozen=True)
class ProblemFingerprint:
    """业务语义问题和已编译数学问题的确定性身份。

    Attributes
    ----------
    semantic_hash : str
        业务输入、目标和约束语义的哈希。
    canonical_hash : str
        已编译稀疏数组、边界和 canonical 元数据的哈希。
    compiler_version : str
        确定性“业务语义到 canonical 模型”转换版本。
    """

    semantic_hash: str
    canonical_hash: str
    compiler_version: str


@dataclass(frozen=True)
class RunFingerprint:
    """附加求解策略和运行时依赖版本的问题身份。

    Attributes
    ----------
    problem : ProblemFingerprint
        不可变的数学问题身份。
    solver_policy_hash : str
        路由、回退、验收和数值设置的哈希。
    package_versions : Mapping[str, str]
        与复现有关的只读求解器及软件包版本。
    """

    problem: ProblemFingerprint
    solver_policy_hash: str
    package_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_versions", MappingProxyType(dict(self.package_versions)))


@dataclass(frozen=True)
class OptimalityCertificate:
    """使用已声明业务目标单位表示的路由专用最优性证书。

    ``VERIFIED`` 仅用于数学上有效的界；``NUMERICAL_ESTIMATE`` 依赖求解器报告的数值 gap。
    该证书不表示逐资产权重与参考解相似。

    Attributes
    ----------
    kind : str
        证书构造方式，例如 LP 原生界或 factor-frontier 界。
    proof_status : ProofStatus
        该界属于已验证、数值估计还是不可用。
    primal_value : float
        验收组合在声明业务单位下的目标值。
    dual_bound : float | None
        最大化目标的有效或估计上界。
    absolute_gap : float | None
        原始目标单位下的 ``dual_bound - primal_value``。
    normalized_gap : float | None
        绝对目标差除以已声明的 alpha 尺度。
    objective_units : str
        继承自 :class:`AlphaSpec` 的可读单位说明。
    objective_scale : float | None
        一个归一化单位对应的原始目标数量。
    components : Mapping[str, float]
        只读的路由专用分解，例如 frontier gap 和数值 gap。
    """

    kind: str
    proof_status: ProofStatus
    primal_value: float
    dual_bound: float | None
    absolute_gap: float | None
    normalized_gap: float | None
    objective_units: str
    objective_scale: float | None
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


@dataclass(frozen=True)
class ConstraintViolation:
    """一条经独立复算得到的 canonical 约束违约记录。

    Attributes
    ----------
    constraint_id : str
        稳定的 canonical 行或变量边界标识。
    group : str
        用于归集和诊断的业务约束组。
    amount : float
        经尺度感知比较后的正数违约量。
    observed : float | None
        独立复算的约束活动值或变量值。
    lower, upper : float | None
        适用的闭区间边界；不存在的一侧为 ``None``。
    label : str | None
        面向使用者的资产、因子或约束标签。
    """

    constraint_id: str
    group: str
    amount: float
    observed: float | None = None
    lower: float | None = None
    upper: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class PortfolioMetrics:
    r"""验收候选解的独立复算业务指标。

    所有权重使用小数单位，所有风险使用年化小数单位；``None`` 表示当前问题数据无法定义该指标。

    Attributes
    ----------
    budget : float | None
        目标权重合计。
    objective : float | None
        使用原始输入单位表示的首要业务目标值。
    tracking_error : float | None
        年化小数跟踪误差波动率。
    factor_variance, specific_variance : float | None
        年化小数方差分量，二者之和为跟踪误差平方。
    turnover_l1 : float | None
        $\lVert x-x_0\rVert_1$.
    total_active_l1 : float | None
        $\lVert x-b\rVert_1$.
    benchmark_member_weight : float | None
        基准权重大于零的资产在目标组合中的合计权重。
    max_weight, min_weight : float | None
        最大和最小目标权重。
    max_active_weight : float | None
        最大逐资产主动权重绝对值。
    max_style_exposure, max_industry_exposure : float | None
        对应因子组中的最大主动敞口绝对值。
    """

    budget: float | None = None
    objective: float | None = None
    tracking_error: float | None = None
    factor_variance: float | None = None
    specific_variance: float | None = None
    turnover_l1: float | None = None
    total_active_l1: float | None = None
    benchmark_member_weight: float | None = None
    max_weight: float | None = None
    min_weight: float | None = None
    max_active_weight: float | None = None
    max_style_exposure: float | None = None
    max_industry_exposure: float | None = None


@dataclass(frozen=True)
class SolveTimings:
    """一次公共求解调用的 wall-clock 耗时分解，单位为秒。

    Attributes
    ----------
    prepare_s : float
        校验和上层准备耗时。
    compile_s : float
        业务语义到 canonical 模型的编译耗时。
    backend_setup_s : float
        所有尝试的原生 workspace/模型建立耗时合计。
    backend_solve_s : float
        所有尝试或子问题的原生数值求解耗时合计。
    validation_s : float
        独立解验收耗时。
    postprocess_s : float
        权重清理和公共结果构造耗时。
    total_s : float
        公共求解调用测得的端到端耗时。
    """

    prepare_s: float = 0.0
    compile_s: float = 0.0
    backend_setup_s: float = 0.0
    backend_solve_s: float = 0.0
    validation_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0


@dataclass(frozen=True)
class SolverAttempt:
    """一次主求解、重试或回退尝试的可审计记录。

    Attributes
    ----------
    backend : str
        后端或专用求解路径标识。
    status : SolveStatus
        标准化尝试状态。
    reason : FailureReason | None
        适用时的标准化失败类别。
    native_status : str | None
        原始求解器状态字符串。
    message : str | None
        可读的失败或路由说明。
    solve_s : float
        本次尝试耗时，单位为秒。
    recovered : bool
        是否在同一路径的前一次失败后恢复成功。
    backend_payload_hash : str | None
        用于证明发送给后端的准确 canonical payload 的哈希。
    metadata : Mapping[str, Any]
        只读的路由专用遥测信息。
    """

    backend: str
    status: SolveStatus
    reason: FailureReason | None = None
    native_status: str | None = None
    message: str | None = None
    solve_s: float = 0.0
    recovered: bool = False
    backend_payload_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class AlignmentReport:
    """附在优化结果上的数据对齐审计信息。

    Attributes
    ----------
    benchmark_missing_mass : float
        任何显式允许的归一化之前，被遗漏的基准权重。
    benchmark_renormalization_factor : float
        应用于保留基准权重的归一化乘数。
    holding_missing_mass : float
        当前资产域中缺失的期初或承接持仓权重。
    source_dates : Mapping[str, pandas.Timestamp]
        每类输入实际使用的数据日期。
    """

    benchmark_missing_mass: float = 0.0
    benchmark_renormalization_factor: float = 1.0
    holding_missing_mass: float = 0.0
    source_dates: Mapping[str, pd.Timestamp] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dates", MappingProxyType(dict(self.source_dates)))


@dataclass(frozen=True)
class OptimizationResult:
    """主求解和所有回退尝试的标准化最终结果。

    普通求解失败通过 ``status``、``route`` 和 ``weights=None`` 表达，使长序列可以按策略
    继续或事后审计；输入及模型错误会在更早阶段抛出异常。对于没有可用组合就不能继续的调用方，
    可使用 ``require_weights`` 显式选择异常式访问。

    Attributes
    ----------
    status : SolveStatus
        最终标准化状态。
    weights : pandas.Series | None
        以资产为索引的目标权重；失败或未通过独立验收时为 ``None``。
    objective_value : float | None
        使用原始业务单位独立复算的目标值。
    backend : str | None
        最终产生候选解的后端；未产生候选解时为空。
    route : tuple[SolverAttempt, ...]
        按发生顺序记录的主路径、重试和回退尝试。
    metrics : PortfolioMetrics
        独立复算的组合业务指标。
    certificate : OptimalityCertificate | None
        可用时的目标最优性证书；它不表示权重与参考解接近。
    violations : tuple[ConstraintViolation, ...]
        独立验收发现的 canonical 约束违约，通常按违约量降序排列。
    alignment : AlignmentReport
        基准、持仓和数据来源日期的对齐审计。
    diagnostics : Any | None
        显式运行深度不可行诊断后附加的结构化报告；正常求解默认不生成。
    timings : SolveTimings
        本次调用的分阶段 wall-clock 耗时。
    fingerprint : ProblemFingerprint
        本结果对应的业务问题及 canonical 模型身份。
    message : str
        面向调用方的简要状态说明。
    """

    status: SolveStatus
    weights: pd.Series | None
    objective_value: float | None
    backend: str | None
    route: tuple[SolverAttempt, ...]
    metrics: PortfolioMetrics
    certificate: OptimalityCertificate | None
    violations: tuple[ConstraintViolation, ...]
    alignment: AlignmentReport
    diagnostics: Any | None
    timings: SolveTimings
    fingerprint: ProblemFingerprint
    message: str = ""

    def require_weights(self) -> pd.Series:
        """返回可用目标权重，否则抛出异常。

        Returns
        -------
        pandas.Series
            以问题资产为索引的目标权重。

        Raises
        ------
        RuntimeError
            最终状态不可用或结果未携带权重时抛出。
        """

        if not self.status.has_solution or self.weights is None:
            raise RuntimeError(f"optimization did not produce usable weights: {self.status.value}")
        return self.weights


@dataclass(frozen=True)
class SolverTuning:
    """针对当前 CPU 快速路径校准的高级数值参数。

    公共风险容差使用年化小数单位，``weight_zero_tolerance`` 使用小数权重单位。目标验收容差
    单独配置在 :class:`SolverPolicy`，因为其原始单位含义依赖 ``AlphaSpec``。

    ``polish`` 已冻结为公共契约字段，但当前 direct PIQP 适配器尚未使用。

    Attributes
    ----------
    alpha_target : float
        将 alpha 系数缩放到的目标最大绝对量级；只改善数值条件，不改变最优解。
    theta_initial : float
        factor-QCQP frontier 搜索的初始参数 QP 权重。
    theta_growth : float
        未建立风险边界 bracket 时 theta 的乘法扩张因子。
    theta_max : float
        theta 搜索允许达到的硬上限。
    max_outer_iters : int
        frontier 扩张、插值和精修合计允许的最大外层迭代数。
    intermediate_eps : float
        中间 PIQP 子问题的绝对/相对数值容差。
    final_eps : float
        最终候选 PIQP 子问题的绝对/相对数值容差。
    piqp_max_iter : int
        每个 PIQP 子问题的最大原生迭代数。
    piqp_inequality_form : str
        PIQP 线性不等式表示；官方 0.6.4+ 下 ``"auto"`` 选择 compact 形式。
    polish : bool
        预留的最终解精修开关，当前 direct PIQP 路径尚未消费。
    feasibility_tolerance : float
        独立验收 canonical 约束时允许的最大绝对违约。
    risk_margin : float
        factor-QCQP 候选相对风险预算预留的年化小数安全边际。
    weight_zero_tolerance : float
        输出及多期状态中将权重视为数值零的绝对阈值。
    """

    alpha_target: float = 0.2
    theta_initial: float = 16384.0
    theta_growth: float = 4.0
    theta_max: float = 1e12
    max_outer_iters: int = 30
    intermediate_eps: float = 1e-5
    final_eps: float = 1e-8
    piqp_max_iter: int = 1000
    piqp_inequality_form: str = "auto"
    polish: bool = True
    feasibility_tolerance: float = 1e-5
    risk_margin: float = 1e-7
    weight_zero_tolerance: float = 1e-5


@dataclass(frozen=True)
class SolverPolicy:
    """无状态优化器的路由、回退和验收策略。

    当前实现将已验证的主路径固定为 HiGHS、PIQP 和 factor frontier；字符串选择字段为后续
    可配置路由保留公共契约。即使 ``validate_solution=False``，当前实现仍始终执行独立验收。
    ``repeat_failed_cold_solve`` 同样是保留字段，当前刻意不使用。

    Attributes
    ----------
    lp : str
        LP 首选后端标识，当前支持的生产值为 ``"highs"``。
    qp : str
        凸 QP 首选后端标识，当前支持的生产值为 ``"piqp"``。
    factor_qcqp_strategy : str
        因子风险预算问题的专用算法，当前为 ``"frontier"``。
    factor_qcqp_subproblem_backend : str
        frontier 参数 QP 的后端，当前为 ``"piqp"``。
    licensed_fallback : str
        可用 license 时优先使用的锥回退后端。
    free_fallback : str
        无商业 license 时使用的锥回退后端。
    lp_prescreen : bool
        是否显式开启 factor-QCQP 的严格 LP 最优解预筛；默认关闭。
    validate_solution : bool
        预留的独立验收开关；当前独立验收仍强制执行。
    rebuild_after_update_failure : bool
        PIQP workspace 更新失败后是否销毁并重建一次。
    repeat_failed_cold_solve : bool
        预留的冷启动重复求解开关，当前未使用。
    objective_tolerance : ObjectiveTolerance
        factor frontier 证书允许的业务目标损失。
    tuning : SolverTuning
        数值容差、theta 搜索和权重清理参数。
    """

    lp: str = "highs"
    qp: str = "piqp"
    factor_qcqp_strategy: str = "frontier"
    factor_qcqp_subproblem_backend: str = "piqp"
    licensed_fallback: str = "mosek"
    free_fallback: str = "clarabel_qdldl"
    lp_prescreen: bool = False
    validate_solution: bool = True
    rebuild_after_update_failure: bool = True
    repeat_failed_cold_solve: bool = False
    objective_tolerance: ObjectiveTolerance = field(default_factory=ObjectiveTolerance)
    tuning: SolverTuning = field(default_factory=SolverTuning)


@dataclass(frozen=True)
class TurnoverRecoveryPolicy:
    """多期问题中显式授权的换手率不可行恢复策略。

    Attributes
    ----------
    max_turnover : float
        用户授权的换手率硬上限；恢复过程不得超过该值。
    buffer : float
        在诊断得到的最小可行换手率之上增加的数值缓冲。
    require_turnover_only : bool
        为真时，只有确认移除换手率后其余约束可行才允许恢复。
    reset_next_period : bool
        是否在下一调仓期恢复原换手率；第一版必须为真，避免永久放宽约束。
    """

    max_turnover: float
    buffer: float = 1e-5
    require_turnover_only: bool = True
    reset_next_period: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_turnover) or self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be positive and finite")
        if not np.isfinite(self.buffer) or self.buffer < 0.0:
            raise ValueError("turnover recovery buffer must be finite and non-negative")
        if not self.reset_next_period:
            raise ValueError(
                "v1 turnover recovery always resets the configured limit next period"
            )


@dataclass(frozen=True)
class SequencePolicy:
    """多期求解的持仓推进、失败处理和输出策略。

    Attributes
    ----------
    mode : str
        ``"chained"`` 使用上一期实际推进后的持仓；``"independent"`` 每日使用各问题自带
        的期初权重，日期之间互不依赖。
    holding_update : str
        相邻调仓日之间的持仓估值方式；当前只支持基于日度收益复合的
        ``"mark_to_market"`` close-to-close 漂移。
    on_failure : str
        ``"stop"`` 在首个失败日停止；``"hold"`` 保持实际持仓并继续后续日期。
    theta_seed : str
        ``"fixed"`` 总用固定初值，``"previous"`` 使用上一成功日 theta，``"auto"`` 在
        链式模式使用上一日、独立模式使用固定值。
    turnover_recovery : TurnoverRecoveryPolicy | None
        显式换手率恢复授权；``None`` 表示绝不自动放宽。
    holding_missing_mass_tolerance : float
        当前风险资产域允许缺失的上一期实际持仓权重上限。
    renormalize_missing_holdings : bool
        是否在缺失质量未超限时删除缺失持仓并对剩余权重归一化。
    output_weights : str
        ``"none"`` 不保存逐日权重，``"sparse"`` 仅保存清理后的非零权重，``"all"`` 保存
        完整权重。
    """

    mode: str = "chained"
    holding_update: str = "mark_to_market"
    on_failure: str = "stop"
    theta_seed: str = "auto"
    turnover_recovery: TurnoverRecoveryPolicy | None = None
    holding_missing_mass_tolerance: float = 0.0
    renormalize_missing_holdings: bool = False
    output_weights: str = "sparse"

    def __post_init__(self) -> None:
        if self.mode not in {"chained", "independent"}:
            raise ValueError("sequence mode must be 'chained' or 'independent'")
        if self.holding_update != "mark_to_market":
            raise ValueError("only close-to-close mark_to_market holding updates are supported")
        if self.on_failure not in {"stop", "hold"}:
            raise ValueError("on_failure must be 'stop' or 'hold'")
        if self.theta_seed not in {"auto", "fixed", "previous"}:
            raise ValueError("theta_seed must be 'auto', 'fixed', or 'previous'")
        if (
            not np.isfinite(self.holding_missing_mass_tolerance)
            or self.holding_missing_mass_tolerance < 0.0
        ):
            raise ValueError("holding missing-mass tolerance must be finite and non-negative")
        if self.output_weights not in {"none", "sparse", "all"}:
            raise ValueError("output_weights must be 'none', 'sparse', or 'all'")

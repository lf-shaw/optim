"""统一组合优化器的公共数据契约。

本模块刻意不导入任何求解器；即使未安装 MOSEK、PIQP、Clarabel、tuda2 或 carry，
调用方仍应能够安全导入 ``optim`` 并构造问题。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
        稳定的数据源标识，例如 ``"memory"`` 或 ``"tuda2"``；默认为 ``"memory"``。
    source_date : pandas.Timestamp | None
        实际使用的数据日期；在严格同日语义下必须等于优化日期。默认为 ``None``，表示调用方
        未声明来源日期。
    version : str | None
        数据供应商、风险模型或适配器版本；默认为 ``None``。
    metadata : Mapping[str, Any]
        只读的数据源专用审计信息；默认为空映射。
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
        业务单位说明，例如 ``"standardized_score"``；优化器不会推断或转换该单位。默认为
        ``"standardized_score"``。
    scale : float
        一个归一化目标单位对应的正数原始 alpha 数量，用于将
        ``ObjectiveTolerance.normalized`` 转换为原始目标差；默认为 ``1.0``。
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
        以输入 alpha 原始单位计量的允许损失；默认为 ``None``，即不额外增加绝对单位容差。
    normalized : float | None
        以 :attr:`AlphaSpec.scale` 的倍数计量的允许损失；默认为 ``1e-4``。
    """

    absolute: float | None = None
    normalized: float | None = 1e-4

    def __post_init__(self) -> None:
        if self.absolute is None and self.normalized is None:
            raise ValueError("at least one objective tolerance must be provided")
        for name, value in (
            ("absolute", self.absolute),
            ("normalized", self.normalized),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"ObjectiveTolerance.{name} must be finite and non-negative"
                )

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
        normalized = (
            0.0 if self.normalized is None else self.normalized * alpha_spec.scale
        )
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
        该日模型的数据来源和版本信息；默认构造内存数据来源记录。
    annualization : str
        显式风险单位契约；默认为且当前编译器要求为 ``"annualized_decimal"``。
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
        矩阵的数据来源和版本信息；默认构造内存数据来源记录。
    annualization : str
        显式风险单位契约；默认为 ``"annualized_decimal"``。
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

    手工表格输入可用 ``make_portfolio_data(universe=..., ...)``，由股票表索引确定顺序。
    在 Notebook 中使用 ``PortfolioData?`` 查看构造参数，或 ``dataclasses.fields(PortfolioData)``
    查看字段定义；类名后的点补全不一定列出无默认值的 dataclass 字段。实例为 frozen，
    如需改值请用 ``dataclasses.replace(data, alpha=...)`` 创建新对象，不原地修改数组。

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
        风险目标或约束所需的严格同日风险模型；默认为 ``None``。
    alpha_spec : AlphaSpec | None
        alpha 的单位和尺度；请求归一化目标证书时必须提供。默认为 ``None``。
    extra_attributes : Mapping[str, numpy.ndarray]
        命名的逐资产属性，每项 shape 均为 ``(n_assets,)``，用于自定义绝对或主动敞口约束。
        默认为空映射。
    provenance : DataProvenance
        组装后单日数据的来源信息；默认构造内存数据来源记录。
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
        乘在年化因子方差上的非负风险厌恶系数；默认为 ``0.75``。
    specific_aversion : float
        乘在年化特异方差上的非负风险厌恶系数；默认为 ``0.75``。

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
        :attr:`PortfolioData.alpha` 缺失，也是默认值。
    """

    alpha_floor: float | None = None


PortfolioObjective: TypeAlias = (
    MaximizeAlpha | RiskAdjustedAlpha | MinimizeTrackingError
)


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
        本次优化中目标权重强制为零的资产；默认为空元组。
    frozen : tuple[Any, ...]
        固定在输入期初权重的资产；默认为空元组。
    not_buyable : tuple[Any, ...]
        目标权重不得高于期初权重的资产；默认为空元组。
    not_sellable : tuple[Any, ...]
        目标权重不得低于期初权重的资产；默认为空元组。
    weight_overrides : Mapping[Any, AssetWeightOverride]
        每只资产的精确目标（标量）或闭区间 ``(lower, upper)``；显式覆盖仍受绝对权重和
        long-only 边界限制。默认为空映射。
    missing_asset : str
        ``"error"`` 拒绝问题资产域以外的指令，``"ignore"`` 明确忽略这些指令；默认为
        ``"error"``。
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
    lower : float | numpy.ndarray
        广播到所有资产的下限标量，或 shape 为 ``(n_assets,)`` 的位置数组；默认为
        ``0.0``。
    upper : float | numpy.ndarray
        广播到所有资产的上限标量，或 shape 为 ``(n_assets,)`` 的位置数组；默认为
        ``1.0``。
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
        同一数值。默认为 ``"l1"``。
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
        除显式覆盖外应用到所有匹配因子的闭区间；默认为 ``None``，表示只约束显式命名
        因子。
    overrides : Mapping[str, tuple[float, float]]
        不区分大小写的逐因子覆盖；对象构造时键会统一转为小写。默认为空映射。
    """

    default: tuple[float, float] | None = None
    overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(
                {str(k).lower(): tuple(v) for k, v in self.overrides.items()}
            ),
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
        为真时，将每只资产的下限与零取交集；默认为 ``True``。设为 ``False`` 只会移除
        这一额外下限，不会自动把 :attr:`asset_weight` 的默认下限改成负数。
    budget : float
        目标权重合计值；默认为 ``1.0``。该参数不自动缩放其他权重类约束。
    asset_weight : WeightBounds
        逐资产绝对目标权重区间；默认使用 :class:`WeightBounds`，即每只资产位于
        $[0,1]$。
    active_weight : SymmetricBound | None
        每只资产 $x_i-b_i$ 的边界；配置后必须提供基准。默认为 ``None``，即不施加该约束。
    total_active : float | None
        $\lVert x-b\rVert_1$ 的上限，配置时必须为有限正数；默认为 ``None``，表示不施加
        该约束。使用小数权重单位，不随预算自动缩放。
    turnover : TurnoverLimit | None
        目标组合相对期初组合的换手率上限；默认为 ``None``。上限使用绝对权重单位，不随
        :attr:`budget` 自动缩放。
    benchmark_member_weight : LowerBound | None
        基准权重大于零的资产在目标组合中的最小合计权重；默认为 ``None``。
    style, industry : ExposureBounds | None
        对应风险模型类型的风格或行业主动敞口边界；二者均默认为 ``None``。
    tracking_error : TrackingErrorLimit | None
        非线性的年化跟踪误差预算；默认为 ``None``。
    freeze_nontradable : bool
        是否将 ``tradable=False`` 的资产固定在期初权重；默认为 ``True``。
    asset_trade : AssetTradeConstraints | None
        仅对本问题生效的黑名单、冻结、单边交易和权重覆盖指令；默认为 ``None``。
    extra_active : Mapping[str, tuple[float, float]]
        命名 ``extra_attributes`` 相对基准的主动敞口边界；默认为空映射。
    extra_absolute : Mapping[str, tuple[float, float]]
        命名 ``extra_attributes`` 的组合绝对敞口边界；默认为空映射。
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
        object.__setattr__(
            self, "extra_active", MappingProxyType(dict(self.extra_active))
        )
        object.__setattr__(
            self, "extra_absolute", MappingProxyType(dict(self.extra_absolute))
        )


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

    def with_constraints(self, **changes: Any) -> PortfolioProblem:
        """派生修改一个或多个约束的新问题，复用数据与目标。

        Parameters
        ----------
        **changes : Any
            PortfolioConstraints 字段名和值；未提供的字段保持原样。嵌套对象整体替换，
            不隐式合并。None 仅适用于可选约束，表示移除。调试建议先单项变更。

        Returns
        -------
        PortfolioProblem
            原对象不变，输入数组不复制；后续 solve 正常校验并重新生成 fingerprint。

        Raises
        ------
        TypeError
            字段名不存在，或必填约束被设为 None。
        """

        required = {
            "long_only",
            "budget",
            "asset_weight",
            "freeze_nontradable",
            "extra_active",
            "extra_absolute",
        }
        if any(name in required and value is None for name, value in changes.items()):
            raise TypeError("required constraint fields cannot be None")
        return replace(self, constraints=replace(self.constraints, **changes))

    def with_objective(self, objective: PortfolioObjective) -> PortfolioProblem:
        """替换目标，返回共享数据与约束的新问题。

        Parameters
        ----------
        objective : PortfolioObjective
            新目标；求解时执行正常模型校验。

        Returns
        -------
        PortfolioProblem
            新问题，不修改原对象。
        """

        return replace(self, objective=objective)

    def with_data(self, **changes: Any) -> PortfolioProblem:
        """替换指定输入字段，返回新问题；不取数、不重排或复制未变更数组。

        Parameters
        ----------
        **changes : Any
            PortfolioData 字段名和值，例如 initial_weight。调整 assets 时调用方必须同步
            对齐相关数组，求解时正常校验。

        Returns
        -------
        PortfolioProblem
            持有新 PortfolioData 的问题，未变更字段仍共享。

        Raises
        ------
        TypeError
            字段名不存在。
        """

        return replace(self, data=replace(self.data, **changes))


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
        与复现有关的只读求解器及软件包版本；默认为空映射。
    """

    problem: ProblemFingerprint
    solver_policy_hash: str
    package_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "package_versions", MappingProxyType(dict(self.package_versions))
        )


@dataclass(frozen=True)
class OptimalityCertificate:
    """使用已声明业务目标单位表示的路由专用最优性证书。

    ``VERIFIED`` 仅用于数学上有效的界；``NUMERICAL_ESTIMATE`` 依赖求解器报告的数值 gap。
    该证书不表示逐资产权重与参考解相似。

    Attributes
    ----------
    kind : str
        证书构造方式，例如 LP 原生界或锥原生对偶界。
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
        只读的路由专用分解，例如原生数值 gap；默认为空映射。
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
        独立复算的约束活动值或变量值；默认为 ``None``。
    lower, upper : float | None
        适用的闭区间边界；不存在的一侧默认为 ``None``。
    label : str | None
        面向使用者的资产、因子或约束标签；默认为 ``None``。
    """

    constraint_id: str
    group: str
    amount: float
    observed: float | None = None
    lower: float | None = None
    upper: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class InfeasibilityContributor:
    """原生不可行证书中一条已映射到业务约束的贡献。

    ``multiplier`` 保留 Farkas/对偶证书坐标的原生符号，受约束缩放影响；它不是
    权重、风险或“至少需要放宽多少”的业务量。需要可操作放宽量时，应显式调用
    :meth:`PortfolioOptimizer.diagnose` 获取 Phase-I 结果。

    Attributes
    ----------
    constraint_id : str
        canonical registry 中的稳定约束标识；风险锥等非线性块使用专门稳定标识。
    group : str
        业务约束组，例如 ``"asset_bound"``、``"turnover"`` 或
        ``"tracking_error"``。
    location : str
        ``"row"``、``"variable"`` 或 ``"cone"``。
    side : str
        证书作用的 ``"lower"``、``"upper"``、``"equal"`` 或 ``"cone"`` 侧。
    multiplier : float
        带原生符号的证书乘子，不能解释为业务重要性。
    key : str | None
        适用时的资产、因子或属性标签。
    configured_bound : float | None
        该侧 canonical 边界；风险锥等块可能为 ``None``。
    sources : tuple[str, ...]
        构成该有效边界的用户配置或运营指令来源。
    metadata : Mapping[str, Any]
        只读补充审计信息；公共逻辑不依赖后端专用键。
    """

    constraint_id: str
    group: str
    location: str
    side: str
    multiplier: float
    key: str | None = None
    configured_bound: float | None = None
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class NativeInfeasibilityEvidence:
    """一个后端在原求解过程中已经产生的结构化不可行证书。

    该对象有意保持很小：不同求解器的 dual ray、Farkas certificate 和 conic certificate
    都只统一成贡献坐标、残差和裕量。它不统一求解器专有 IIS，也不会为了生成结果而额外
    求解，因此普通不可行返回仍是低开销路径。

    Attributes
    ----------
    backend : str
        产生证书的后端。
    kind : str
        原生证据类型。
    proof_status : ProofStatus
        证书是已验证、数值估计还是不可用。
    native_status : str
        产生证书时的原生状态。
    contributors : tuple[InfeasibilityContributor, ...]
        按原生坐标顺序保留的具名非零约束乘子；metadata 中 canonical_index 标记具体坐标。
    certificate_residual : float | None
        归一化平稳性残差；后端无法稳定计算时为 ``None``。
    certificate_margin : float | None
        归一化严格不可行裕量；正值代表后端适配方向下的有效证据。
    metadata : Mapping[str, Any]
        小型只读原生元数据。
    fingerprint : ProblemFingerprint | None
        证据所属请求的模型身份；与 ``scope`` 一起解释，不能把派生子问题当成原问题。
    scope : str
        ``original`` 表示完整原问题，``linear_relaxation`` 表示其线性松弛。
    """

    backend: str
    kind: str
    proof_status: ProofStatus
    native_status: str
    contributors: tuple[InfeasibilityContributor, ...] = ()
    certificate_residual: float | None = None
    certificate_margin: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: ProblemFingerprint | None = None
    scope: str = "original"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class PortfolioMetrics:
    r"""验收候选解的独立复算业务指标。

    所有权重使用小数单位，所有风险使用年化小数单位。所有字段均默认为 ``None``，表示尚未
    计算或当前问题数据无法定义对应指标。

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

    所有分项均默认为 ``0.0``；求解引擎只填写实际发生并可独立计量的阶段。

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
        适用时的标准化失败类别；默认为 ``None``。
    native_status : str | None
        原始求解器状态字符串；默认为 ``None``。
    message : str | None
        可读的失败或路由说明；默认为 ``None``。
    solve_s : float
        本次尝试耗时，单位为秒；默认为 ``0.0``。
    recovered : bool
        是否在同一路径的前一次失败后恢复成功；默认为 ``False``。
    backend_payload_hash : str | None
        用于证明发送给后端的准确 canonical payload 的哈希；默认为 ``None``。
    metadata : Mapping[str, Any]
        只读的路由专用遥测信息；默认为空映射。
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
        任何显式允许的归一化之前，被遗漏的基准权重；默认为 ``0.0``。
    benchmark_renormalization_factor : float
        应用于保留基准权重的归一化乘数；默认为 ``1.0``，即不缩放。
    holding_missing_mass : float
        当前资产域中缺失的期初或承接持仓权重；默认为 ``0.0``。
    source_dates : Mapping[str, pandas.Timestamp]
        每类输入实际使用的数据日期；默认为空映射。
    """

    benchmark_missing_mass: float = 0.0
    benchmark_renormalization_factor: float = 1.0
    holding_missing_mass: float = 0.0
    source_dates: Mapping[str, pd.Timestamp] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_dates", MappingProxyType(dict(self.source_dates))
        )


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
        面向调用方的简要状态说明；默认为空字符串。
    native_infeasibility : tuple[NativeInfeasibilityEvidence, ...]
        各次实际后端尝试顺带产生的原生不可行证书；默认为空，不会触发额外诊断计算。
    problem : PortfolioProblem | None
        产生本结果的准确单期问题。单期公共求解始终保留，便于随后直接调用 ``diagnose``；
        多期结果为控制内存仅在顶层保留导致 ``stop`` 的问题。
        此字段保留引用而非深复制；输入数组不应原地修改，诊断前会重新校验 fingerprint。
    solver_policy : SolverPolicy | None
        本次实际求解策略引用，用于导出可重放问题；手工构造的结果可为空。
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
    native_infeasibility: tuple[NativeInfeasibilityEvidence, ...] = ()
    problem: PortfolioProblem | None = field(default=None, repr=False, compare=False)
    solver_policy: SolverPolicy | None = field(default=None, repr=False, compare=False)

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
            raise RuntimeError(
                f"optimization did not produce usable weights: {self.status.value}"
            )
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
        将 alpha 系数缩放到的目标最大绝对量级；只改善数值条件，不改变最优解。默认为
        ``0.2``。
    final_eps : float
        原生求解的数值精度请求；默认为 ``1e-8``。后端可采用更严格的设置，最终接受与否
        仍由 :attr:`feasibility_tolerance` 控制。
    piqp_max_iter : int
        每个 PIQP 子问题的最大原生迭代数；默认为 ``1000``。
    piqp_inequality_form : str
        PIQP 线性不等式表示；默认为 ``"auto"``，在官方 0.6.4+ 下选择 compact 形式。
    polish : bool
        预留的最终解精修开关，默认为 ``True``；当前 direct PIQP 路径尚未消费。
    feasibility_tolerance : float
        最终候选约束验收允许的最大绝对违约；默认为 ``1e-5``。
    weight_zero_tolerance : float
        输出及多期状态中将权重视为数值零的绝对阈值；默认为 ``1e-5``。
    """

    alpha_target: float = 0.2
    final_eps: float = 1e-8
    piqp_max_iter: int = 1000
    piqp_inequality_form: str = "auto"
    polish: bool = True
    feasibility_tolerance: float = 1e-5
    weight_zero_tolerance: float = 1e-5


@dataclass(frozen=True)
class SolverPolicy:
    """无状态优化器的路由、回退和验收策略。

    auto 使用 HiGHS、direct PIQP 和 Clarabel 主路径；显式 backend 只运行指定后端，
    不预筛、不回退。即使 ``validate_solution=False``，当前实现仍始终执行独立验收。
    ``repeat_failed_cold_solve`` 同样是保留字段，当前刻意不使用。

    Attributes
    ----------
    backend : str
        默认 auto；mosek、clarabel 支持 LP/QP/Factor-QCQP；highs 仅支持 LP，piqp
        仅支持 QP。不支持的模型在准备阶段报错；MOSEK 缺安装或有效授权时抛 RuntimeError，普通数值失败按结果返回。
        diagnose 默认独立自动路由，可通过其 backend 参数覆盖，不继承此开关。
    lp : str
        自动路线预留字段，仅允许默认 highs；切换后端使用 backend，非默认值报错。
    qp : str
        自动路线预留字段，仅允许默认 piqp；切换后端使用 backend，非默认值报错。
    lp_prescreen : bool
        是否显式开启 factor-QCQP 的严格 LP 最优解预筛；默认关闭。
    validate_solution : bool
        预留的独立验收开关；默认为 ``True``，且当前独立验收仍强制执行。
    repeat_failed_cold_solve : bool
        预留的冷启动重复求解开关；默认为 ``False``，当前未使用。
    objective_tolerance : ObjectiveTolerance
        权重清理允许的业务目标损失；默认构造 :class:`ObjectiveTolerance`。
    tuning : SolverTuning
        数值容差和权重清理参数；默认构造 :class:`SolverTuning`。
    """

    lp: str = "highs"
    qp: str = "piqp"
    lp_prescreen: bool = False
    validate_solution: bool = True
    repeat_failed_cold_solve: bool = False
    objective_tolerance: ObjectiveTolerance = field(default_factory=ObjectiveTolerance)
    tuning: SolverTuning = field(default_factory=SolverTuning)
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "mosek", "clarabel", "highs", "piqp"}:
            raise ValueError("backend must be auto, mosek, clarabel, highs or piqp")
        # 旧预留字段只接受既定值，避免调用者误以为它们可以切换主后端。
        for name, expected in (
            ("lp", "highs"),
            ("qp", "piqp"),
        ):
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} only supports {expected!r}; use backend to select a solver"
                )


@dataclass(frozen=True)
class TurnoverRecoveryPolicy:
    """多期问题中显式授权的换手率不可行恢复策略。

    Attributes
    ----------
    max_turnover : float
        用户授权的换手率硬上限；恢复过程不得超过该值。
    buffer : float
        在诊断得到的最小可行换手率之上增加的数值缓冲；默认为 ``1e-5``。
    require_turnover_only : bool
        为真时，只有确认移除换手率后其余约束可行才允许恢复；默认为 ``True``。
    reset_next_period : bool
        是否在下一调仓期恢复原换手率；默认为且第一版要求为 ``True``，避免永久放宽约束。
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
        的期初权重，日期之间互不依赖。默认为 ``"chained"``。
    holding_update : str
        相邻调仓日之间的持仓估值方式；当前只支持基于日度收益复合的
        ``"mark_to_market"`` close-to-close 漂移，也是默认值。
    on_failure : str
        ``"stop"`` 在首个失败日停止；``"hold"`` 保持实际持仓并继续后续日期。默认为
        ``"stop"``。
    turnover_recovery : TurnoverRecoveryPolicy | None
        显式换手率恢复授权；默认为 ``None``，表示绝不自动放宽。
    holding_missing_mass_tolerance : float
        当前风险资产域允许缺失的上一期实际持仓权重上限；默认为 ``0.0``。
    renormalize_missing_holdings : bool
        是否在缺失质量未超限时删除缺失持仓并对剩余权重归一化；默认为 ``False``。
    output_weights : str
        ``"none"`` 不保存逐日权重，``"sparse"`` 仅保存清理后的非零权重，``"all"`` 保存
        完整权重；默认为 ``"sparse"``。
    """

    mode: str = "chained"
    holding_update: str = "mark_to_market"
    on_failure: str = "stop"
    turnover_recovery: TurnoverRecoveryPolicy | None = None
    holding_missing_mass_tolerance: float = 0.0
    renormalize_missing_holdings: bool = False
    output_weights: str = "sparse"

    def __post_init__(self) -> None:
        if self.mode not in {"chained", "independent"}:
            raise ValueError("sequence mode must be 'chained' or 'independent'")
        if self.holding_update != "mark_to_market":
            raise ValueError(
                "only close-to-close mark_to_market holding updates are supported"
            )
        if self.on_failure not in {"stop", "hold"}:
            raise ValueError("on_failure must be 'stop' or 'hold'")
        if (
            not np.isfinite(self.holding_missing_mass_tolerance)
            or self.holding_missing_mass_tolerance < 0.0
        ):
            raise ValueError(
                "holding missing-mass tolerance must be finite and non-negative"
            )
        if self.output_weights not in {"none", "sparse", "all"}:
            raise ValueError("output_weights must be 'none', 'sparse', or 'all'")

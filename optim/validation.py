"""组合优化问题的低成本、与求解器无关的静态校验。

校验刻意发生在 canonical 编译和后端建立之前，并聚合彼此独立的问题，使长序列可以一次性
修复数据，而不必为每个错误字段反复启动昂贵优化。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, cast

import numpy as np
import pandas as pd

from ._impl.asset_bounds import AssetBoundsError, resolve_asset_bounds
from .portfolio_types import (
    AssetTradeConstraints,
    FactorRiskModel,
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioProblem,
    RiskAdjustedAlpha,
)


class ValidationSeverity(str, Enum):
    """静态校验问题的严重程度。"""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """一条可定位、可机器处理的静态校验问题。

    Attributes
    ----------
    field : str
        出错字段的稳定路径，例如 ``"data.alpha"``。
    code : str
        适合程序分支处理的稳定错误代码。
    severity : ValidationSeverity
        错误或警告级别。
    message : str
        面向使用者的具体说明。
    date : pandas.Timestamp | None
        问题所属优化日期；不具备日期语义时为空。
    context : Mapping[str, Any]
        只读的结构化上下文，例如期望/实际 shape 或受影响资产。
    """

    field: str
    code: str
    severity: ValidationSeverity
    message: str
    date: pd.Timestamp | None = None
    context: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.date is not None:
            object.__setattr__(self, "date", pd.Timestamp(self.date))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))


@dataclass(frozen=True)
class ValidationReport:
    """一次静态校验产生的全部问题。

    Attributes
    ----------
    issues : tuple[ValidationIssue, ...]
        按发现顺序记录的错误和警告；报告为空表示未发现静态问题。
    """

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """返回所有 error 级问题。"""

        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """返回所有 warning 级问题。"""

        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        """是否不存在会阻止编译或求解的 error。"""

        return not self.errors

    def raise_for_errors(self) -> None:
        """报告含 error 时抛出聚合异常。

        Raises
        ------
        PortfolioValidationError
            ``errors`` 非空时抛出，异常对象保留完整报告。
        """

        if self.errors:
            raise PortfolioValidationError(self)


class PortfolioValidationError(ValueError):
    """在任何后端建立前发现输入、schema 或模型错误时抛出的聚合异常。

    Attributes
    ----------
    report : ValidationReport
        触发异常的完整静态校验报告，而非截断后的异常文本。
    """

    def __init__(self, report: ValidationReport):
        self.report = report
        summary = "; ".join(
            f"{issue.field} [{issue.code}]: {issue.message}"
            for issue in report.errors[:8]
        )
        if len(report.errors) > 8:
            summary += f"; ... and {len(report.errors) - 8} more error(s)"
        super().__init__(summary)


class _Collector:
    def __init__(self, date: pd.Timestamp):
        self.date = date
        self.issues: list[ValidationIssue] = []

    def error(self, field: str, code: str, message: str, **context: Any) -> None:
        self.issues.append(
            ValidationIssue(
                field=field,
                code=code,
                severity=ValidationSeverity.ERROR,
                message=message,
                date=self.date,
                context=context,
            )
        )

    def warning(self, field: str, code: str, message: str, **context: Any) -> None:
        self.issues.append(
            ValidationIssue(
                field=field,
                code=code,
                severity=ValidationSeverity.WARNING,
                message=message,
                date=self.date,
                context=context,
            )
        )


def _array(
    collector: _Collector,
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except Exception as exc:
        collector.error(name, "not_array_like", f"cannot convert to an array: {exc}")
        return None
    if array.shape != shape:
        collector.error(
            name,
            "shape_mismatch",
            f"expected shape {shape}, got {array.shape}",
            expected=shape,
            observed=array.shape,
        )
        return None
    if finite:
        try:
            all_finite = bool(np.all(np.isfinite(array)))
        except TypeError:
            all_finite = False
        if not all_finite:
            collector.error(name, "non_finite", "contains NaN, infinity, or non-numeric values")
    return array


def _bound_array(
    collector: _Collector,
    name: str,
    value: float | np.ndarray,
    n_assets: int,
) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        collector.error(name, "not_numeric", f"bound is not numeric: {exc}")
        return None
    if array.ndim == 0:
        if not np.isfinite(array.item()):
            collector.error(name, "non_finite", "scalar bound must be finite")
            return None
        return np.full(n_assets, float(array), dtype=float)
    return _array(collector, name, array, (n_assets,))


def _check_bounds(
    collector: _Collector,
    problem: PortfolioProblem,
    n_assets: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    constraints = problem.constraints
    lower = _bound_array(
        collector,
        "constraints.asset_weight.lower",
        constraints.asset_weight.lower,
        n_assets,
    )
    upper = _bound_array(
        collector,
        "constraints.asset_weight.upper",
        constraints.asset_weight.upper,
        n_assets,
    )
    if lower is None or upper is None:
        return lower, upper
    effective_lower = np.maximum(lower, 0.0) if constraints.long_only else lower
    inverted = effective_lower > upper
    if np.any(inverted):
        collector.error(
            "constraints.asset_weight",
            "lower_exceeds_upper",
            "effective lower bound exceeds upper bound",
            positions=np.flatnonzero(inverted)[:10].tolist(),
        )
    budget = constraints.budget
    if np.isfinite(budget):
        lower_sum = float(effective_lower.sum())
        upper_sum = float(upper.sum())
        if budget < lower_sum - 1e-12 or budget > upper_sum + 1e-12:
            collector.error(
                "constraints.budget",
                "outside_asset_capacity",
                "budget is outside the aggregate asset bounds",
                budget=budget,
                lower_sum=lower_sum,
                upper_sum=upper_sum,
            )
    return effective_lower, upper


def _check_pair(collector: _Collector, field_name: str, pair: Any) -> None:
    try:
        lower, upper = pair
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError):
        collector.error(field_name, "invalid_bound_pair", "expected a finite (lower, upper) pair")
        return
    if not np.isfinite(lower) or not np.isfinite(upper):
        collector.error(field_name, "non_finite", "bounds must be finite")
    elif lower > upper:
        collector.error(field_name, "lower_exceeds_upper", "lower bound exceeds upper bound")


def _check_exposure_bounds(
    collector: _Collector,
    field_name: str,
    bounds: Any,
    risk_model: FactorRiskModel | None,
    expected_type: str,
) -> None:
    if bounds is None:
        return
    if risk_model is None:
        collector.error(field_name, "risk_model_required", "factor exposure bounds require a factor risk model")
        return
    factor_lookup = {name.lower() for name in risk_model.factor_names}
    typed_names = {
        name.lower()
        for name, factor_type in zip(risk_model.factor_names, risk_model.factor_types)
        if factor_type.lower() == expected_type
    }
    if bounds.default is not None and not typed_names:
        collector.error(
            field_name,
            "factor_type_missing",
            f"no factors are labelled {expected_type!r}",
        )
    entries: Iterable[tuple[str, tuple[float, float]]] = bounds.overrides.items()
    for key, pair in entries:
        if key.lower() not in factor_lookup:
            collector.error(field_name, "unknown_factor", f"unknown factor {key!r}")
            continue
        _check_pair(collector, f"{field_name}.{key}", pair)
    if bounds.default is not None:
        _check_pair(collector, f"{field_name}.default", bounds.default)


def _check_asset_trade_constraints(
    collector: _Collector,
    instructions: AssetTradeConstraints | None,
    assets: pd.Index,
    initial: np.ndarray | None,
) -> bool:
    """不建立求解器模型，校验仅对本次优化生效的逐资产指令。"""

    if instructions is None or instructions.is_empty:
        return True
    valid = True
    groups: tuple[tuple[str, Iterable[Any]], ...] = (
        ("blacklist", instructions.blacklist),
        ("frozen", instructions.frozen),
        ("not_buyable", instructions.not_buyable),
        ("not_sellable", instructions.not_sellable),
        ("weight_overrides", instructions.weight_overrides.keys()),
    )
    ownership: dict[Any, str] = {}
    try:
        asset_set = set(assets)
    except TypeError:
        # 重复或不可哈希的资产坐标由独立检查报告。
        return False
    for group, values in groups:
        local: set[Any] = set()
        for asset in values:
            try:
                duplicate = asset in local
            except TypeError:
                collector.error(
                    f"constraints.asset_trade.{group}",
                    "unhashable_asset",
                    f"asset identifier {asset!r} is not hashable",
                )
                valid = False
                continue
            if duplicate:
                collector.error(
                    f"constraints.asset_trade.{group}",
                    "duplicate_asset",
                    f"asset {asset!r} appears more than once",
                )
                valid = False
                continue
            local.add(asset)
            previous = ownership.get(asset)
            if previous is not None:
                collector.error(
                    "constraints.asset_trade",
                    "overlapping_instructions",
                    f"asset {asset!r} appears in both {previous} and {group}",
                    asset=str(asset),
                    first=previous,
                    second=group,
                )
                valid = False
            else:
                ownership[asset] = group
            if asset not in asset_set:
                if instructions.missing_asset == "error":
                    collector.error(
                        f"constraints.asset_trade.{group}",
                        "unknown_asset",
                        f"asset {asset!r} is not in the aligned universe",
                    )
                    valid = False
                else:
                    collector.warning(
                        f"constraints.asset_trade.{group}",
                        "ignored_unknown_asset",
                        f"asset {asset!r} is not in the aligned universe and will be ignored",
                    )

    needs_initial = bool(
        instructions.frozen
        or instructions.not_buyable
        or instructions.not_sellable
    )
    if needs_initial and initial is None:
        collector.error(
            "constraints.asset_trade",
            "initial_weight_required",
            "frozen/not-buyable/not-sellable instructions require initial weights",
        )
        valid = False

    for asset, value in instructions.weight_overrides.items():
        field_name = f"constraints.asset_trade.weight_overrides.{asset}"
        if np.isscalar(value):
            try:
                target = float(cast(Any, value))
            except (TypeError, ValueError):
                collector.error(field_name, "not_numeric", "exact target must be numeric")
                valid = False
                continue
            if not np.isfinite(target):
                collector.error(field_name, "non_finite", "exact target must be finite")
                valid = False
        else:
            before = len(collector.issues)
            _check_pair(collector, field_name, value)
            valid = valid and len(collector.issues) == before
    return valid


def validate_problem(problem: PortfolioProblem) -> ValidationReport:
    """返回一个已对齐单期问题的全部低成本静态问题。

    校验会聚合独立错误，而不是在首个异常数组处停止。检查内容包括位置 shape 和有限性、
    风险模型准确日期/单位、协方差对称性与半正定性、目标依赖、基准与期初预算、因子名称、
    交易名单冲突和最终逐资产容量。该函数不证明组合线性域的全局可行性；后者属于求解器或
    显式 Phase-I 诊断。

    Parameters
    ----------
    problem : PortfolioProblem
        数据已经按 ``PortfolioData.assets`` 对齐的单期问题。

    Returns
    -------
    ValidationReport
        完整错误和警告集合；本函数不因普通校验问题抛异常。
    """

    data = problem.data
    date = pd.Timestamp(data.date)
    collector = _Collector(date)
    assets = pd.Index(data.assets, copy=False)
    n_assets = len(assets)

    if n_assets == 0:
        collector.error("data.assets", "empty", "asset coordinate must not be empty")
    if assets.has_duplicates:
        duplicates = assets[assets.duplicated()].unique().astype(str).tolist()[:10]
        collector.error("data.assets", "duplicate", "asset coordinate contains duplicates", assets=duplicates)
    if assets.hasnans:
        collector.error("data.assets", "missing", "asset coordinate contains missing identifiers")

    alpha = None if data.alpha is None else _array(collector, "data.alpha", data.alpha, (n_assets,))
    benchmark = None if data.benchmark is None else _array(
        collector, "data.benchmark", data.benchmark, (n_assets,)
    )
    initial = None if data.initial_weight is None else _array(
        collector, "data.initial_weight", data.initial_weight, (n_assets,)
    )
    tradable = _array(collector, "data.tradable", data.tradable, (n_assets,), finite=False)

    objective_needs_alpha = isinstance(problem.objective, (MaximizeAlpha, RiskAdjustedAlpha)) or (
        isinstance(problem.objective, MinimizeTrackingError)
        and problem.objective.alpha_floor is not None
    )
    if objective_needs_alpha and alpha is None:
        collector.error("data.alpha", "required", "the selected objective requires alpha")
    if objective_needs_alpha and data.alpha_spec is None:
        collector.error(
            "data.alpha_spec",
            "required",
            "alpha units and scale must be explicit for alpha-based objectives",
        )

    if benchmark is not None:
        if np.any(benchmark < 0.0):
            collector.error("data.benchmark", "negative_weight", "benchmark weights must be non-negative")
        benchmark_sum = float(np.sum(benchmark))
        if np.isfinite(benchmark_sum) and not np.isclose(benchmark_sum, 1.0, rtol=0.0, atol=1e-8):
            collector.error(
                "data.benchmark",
                "not_normalized",
                "aligned benchmark must sum to one; alignment policy must explicitly renormalize it",
                observed=benchmark_sum,
            )
    if initial is not None:
        if problem.constraints.long_only and np.any(initial < -1e-12):
            collector.error("data.initial_weight", "negative_weight", "long-only initial weights must be non-negative")
        initial_sum = float(np.sum(initial))
        if np.isfinite(initial_sum) and not np.isclose(
            initial_sum, problem.constraints.budget, rtol=0.0, atol=1e-8
        ):
            collector.error(
                "data.initial_weight",
                "budget_mismatch",
                "initial weights must match the configured budget",
                observed=initial_sum,
                expected=problem.constraints.budget,
            )
    if tradable is not None and not np.all(np.isin(tradable, (False, True, 0, 1))):
        collector.error("data.tradable", "invalid_boolean", "tradable values must be boolean/0/1")

    risk_model = data.risk_model
    factor_risk: FactorRiskModel | None = None
    if isinstance(risk_model, FactorRiskModel):
        factor_risk = risk_model
        if pd.Timestamp(risk_model.asof) != date:
            collector.error(
                "data.risk_model.asof",
                "date_mismatch",
                "risk model must have the exact optimization date",
                observed=str(risk_model.asof),
                expected=str(date),
            )
        n_factors = len(risk_model.factor_names)
        _array(
            collector,
            "data.risk_model.exposure",
            risk_model.exposure,
            (n_assets, n_factors),
        )
        covariance = _array(
            collector,
            "data.risk_model.covariance",
            risk_model.covariance,
            (n_factors, n_factors),
        )
        specific = _array(
            collector,
            "data.risk_model.specific_volatility",
            risk_model.specific_volatility,
            (n_assets,),
        )
        if len(risk_model.factor_types) != n_factors:
            collector.error(
                "data.risk_model.factor_types",
                "shape_mismatch",
                "factor_types must match factor_names",
                expected=n_factors,
                observed=len(risk_model.factor_types),
            )
        if len(set(risk_model.factor_names)) != n_factors:
            collector.error("data.risk_model.factor_names", "duplicate", "factor names must be unique")
        if specific is not None and np.any(specific < 0.0):
            collector.error(
                "data.risk_model.specific_volatility",
                "negative",
                "specific volatility must be non-negative",
            )
        if covariance is not None and covariance.size:
            symmetric = (covariance + covariance.T) * 0.5
            scale = max(1.0, float(np.max(np.abs(symmetric))))
            asymmetry = float(np.max(np.abs(covariance - covariance.T)))
            if asymmetry > 1e-10 * scale:
                collector.error(
                    "data.risk_model.covariance",
                    "not_symmetric",
                    "factor covariance must be symmetric",
                    max_asymmetry=asymmetry,
                )
            else:
                min_eigenvalue = float(np.linalg.eigvalsh(symmetric)[0])
                if min_eigenvalue < -1e-10 * scale:
                    collector.error(
                        "data.risk_model.covariance",
                        "not_psd",
                        "factor covariance must be positive semidefinite",
                        min_eigenvalue=min_eigenvalue,
                    )
        if risk_model.annualization != "annualized_decimal":
            collector.error(
                "data.risk_model.annualization",
                "unsupported_unit",
                "core solver requires annualized_decimal risk inputs",
                observed=risk_model.annualization,
            )
    elif risk_model is not None:
        if pd.Timestamp(risk_model.asof) != date:
            collector.error("data.risk_model.asof", "date_mismatch", "risk model must have the exact optimization date")
        _array(
            collector,
            "data.risk_model.covariance",
            risk_model.covariance,
            (n_assets, n_assets),
        )

    for name, values in data.extra_attributes.items():
        _array(collector, f"data.extra_attributes.{name}", values, (n_assets,))

    constraints = problem.constraints
    if not np.isfinite(constraints.budget) or constraints.budget <= 0.0:
        collector.error("constraints.budget", "invalid", "budget must be positive and finite")
    effective_lower, effective_upper = _check_bounds(collector, problem, n_assets)
    if constraints.active_weight is not None and benchmark is None:
        collector.error("constraints.active_weight", "benchmark_required", "active bounds require a benchmark")
    if constraints.total_active is not None:
        if benchmark is None:
            collector.error("constraints.total_active", "benchmark_required", "total active bound requires a benchmark")
        if not np.isfinite(constraints.total_active) or constraints.total_active < 0.0:
            collector.error("constraints.total_active", "invalid", "total active bound must be finite and non-negative")
    if constraints.turnover is not None and initial is None:
        collector.error("constraints.turnover", "initial_weight_required", "turnover requires initial weights")
    asset_trade_valid = _check_asset_trade_constraints(
        collector,
        constraints.asset_trade,
        assets,
        initial,
    )
    nontradable_requires_initial = bool(
        constraints.freeze_nontradable
        and tradable is not None
        and np.any(~np.asarray(tradable, dtype=bool))
    )
    if nontradable_requires_initial and initial is None:
        collector.error(
            "constraints.freeze_nontradable",
            "initial_weight_required",
            "freezing non-tradable assets requires initial weights",
        )
    if constraints.benchmark_member_weight is not None and benchmark is None:
        collector.error(
            "constraints.benchmark_member_weight",
            "benchmark_required",
            "benchmark-member weight requires a benchmark",
        )
    if constraints.tracking_error is not None and risk_model is None:
        collector.error("constraints.tracking_error", "risk_model_required", "tracking error requires a risk model")
    if constraints.tracking_error is not None and benchmark is None:
        collector.error("constraints.tracking_error", "benchmark_required", "tracking error requires a benchmark")
    if isinstance(problem.objective, (RiskAdjustedAlpha, MinimizeTrackingError)) and risk_model is None:
        collector.error("objective", "risk_model_required", "the selected objective requires a risk model")
    if isinstance(problem.objective, (RiskAdjustedAlpha, MinimizeTrackingError)) and benchmark is None:
        collector.error("objective", "benchmark_required", "tracking-risk objectives require a benchmark")

    _check_exposure_bounds(collector, "constraints.style", constraints.style, factor_risk, "style")
    _check_exposure_bounds(collector, "constraints.industry", constraints.industry, factor_risk, "industry")

    for field_name, mapping in (
        ("constraints.extra_active", constraints.extra_active),
        ("constraints.extra_absolute", constraints.extra_absolute),
    ):
        for key, pair in mapping.items():
            if key not in data.extra_attributes:
                collector.error(field_name, "unknown_attribute", f"unknown extra attribute {key!r}")
            _check_pair(collector, f"{field_name}.{key}", pair)

    # 仅在基础数组和交易指令均已确认可用后解析最终逐资产边界，以便在 canonical 编译和
    # 后端建立前发现预算容量及不可交易资产冲突。
    can_resolve_asset_bounds = bool(
        effective_lower is not None
        and effective_upper is not None
        and tradable is not None
        and asset_trade_valid
        and not (nontradable_requires_initial and initial is None)
        and (constraints.active_weight is None or benchmark is not None)
    )
    if can_resolve_asset_bounds:
        try:
            resolved = resolve_asset_bounds(problem)
        except AssetBoundsError as exc:
            collector.error(
                "constraints.asset_trade",
                "combined_bounds_infeasible",
                str(exc),
            )
        else:
            lower_sum = float(resolved.lower.sum())
            upper_sum = float(resolved.upper.sum())
            if (
                constraints.budget < lower_sum - 1e-12
                or constraints.budget > upper_sum + 1e-12
            ):
                collector.error(
                    "constraints.budget",
                    "outside_effective_asset_capacity",
                    "budget is outside the aggregate bounds after one-off asset instructions",
                    budget=constraints.budget,
                    lower_sum=lower_sum,
                    upper_sum=upper_sum,
                )

    return ValidationReport(tuple(collector.issues))

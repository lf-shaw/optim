"""与后端无关的向量重建、指标计算和可行性验收。

本模块刻意不读取原生求解器状态或残差，而将后端返回向量视为不可信数值数据，使用发送给
后端的同一 canonical payload 进行检查，并从 :class:`PortfolioProblem` 独立复算业务风险
和目标指标。
"""

from __future__ import annotations

import math
import numpy as np

from ..model.canonical import CompiledProblem, FactorQCQP, QuadraticProgram
from ..portfolio_types import (
    ConstraintViolation,
    FactorRiskModel,
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioMetrics,
    PortfolioProblem,
    RiskAdjustedAlpha,
)


def lift_weights(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    weight: np.ndarray,
) -> np.ndarray:
    """根据组合权重确定性重建 canonical 辅助变量。

    权重清理只改变组合权重列。重新验收前，本函数确定性重建换手率绝对值、总主动权重和因子
    敞口辅助列，使清理后的向量与原数学模型比较，而不是与过期的求解器辅助变量比较。

    Parameters
    ----------
    problem : PortfolioProblem
        辅助变量业务语义来源。
    compiled : CompiledProblem
        定义完整变量布局和编译优化的 canonical 模型。
    weight : numpy.ndarray
        按 ``problem.data.assets`` 顺序排列的目标权重，shape 为 ``(n_assets,)``。

    Returns
    -------
    numpy.ndarray
        完整 canonical 变量向量，shape 为 ``(n_variables,)``。

    Raises
    ------
    ValueError
        权重向量与 canonical 资产坐标 shape 不一致。
    """

    domain = compiled.model.domain
    weight = np.asarray(weight, dtype=float).reshape(-1)
    if weight.shape != domain.weight_indices.shape:
        raise ValueError("weight vector does not match canonical asset coordinate")
    vector = np.zeros(domain.n_variables, dtype=float)
    vector[domain.weight_indices] = weight
    asset_positions = {str(asset): index for index, asset in enumerate(domain.assets)}
    active = (
        None
        if problem.data.benchmark is None
        else weight - np.asarray(problem.data.benchmark, dtype=float)
    )
    initial = (
        None
        if problem.data.initial_weight is None
        else np.asarray(problem.data.initial_weight, dtype=float)
    )
    sparse_turnover = "exact_sparse_turnover" in compiled.compiler_optimizations
    sparse_active = "exact_sparse_total_active" in compiled.compiler_optimizations
    factor_values = None
    if isinstance(problem.data.risk_model, FactorRiskModel) and active is not None:
        factor_values = (
            np.asarray(problem.data.risk_model.exposure, dtype=float).T @ active
        )
        factor_lookup = {
            name: factor_values[index]
            for index, name in enumerate(problem.data.risk_model.factor_names)
        }
    else:
        factor_lookup = {}
    for record in domain.variables:
        if record.group == "turnover_aux":
            assert record.key is not None and initial is not None
            asset_index = asset_positions[record.key]
            difference = weight[asset_index] - initial[asset_index]
            vector[record.index] = (
                max(difference, 0.0) if sparse_turnover else abs(difference)
            )
        elif record.group == "active_aux":
            assert record.key is not None and active is not None
            difference = active[asset_positions[record.key]]
            vector[record.index] = (
                max(-difference, 0.0) if sparse_active else abs(difference)
            )
        elif record.group == "factor_active":
            assert record.key is not None
            vector[record.index] = factor_lookup[record.key]
    return vector


def evaluate_solution(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    vector: np.ndarray,
) -> tuple[PortfolioMetrics, tuple[ConstraintViolation, ...], float]:
    """一次性独立复算全部 canonical 约束和业务指标。

    违约使用各自原始单位报告，并通过 constraint registry 映射回业务名称。跟踪误差由年化因子
    协方差和特异波动率复算，绝不从求解器锥残差推断。返回的最大违约量是公共 API 对主后端和
    回退后端统一使用的验收指标。

    Parameters
    ----------
    problem : PortfolioProblem
        目标、风险和业务指标语义来源。
    compiled : CompiledProblem
        后端实际求解的 canonical 数值模型。
    vector : numpy.ndarray
        后端候选完整变量向量，shape 必须为 ``(n_variables,)`` 且全部有限。

    Returns
    -------
    tuple[PortfolioMetrics, tuple[ConstraintViolation, ...], float]
        独立指标、逐约束违约记录和最大绝对违约量。向量 shape 或数值无效时最大违约为无穷。
    """

    model = compiled.model
    domain = model.domain
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.shape != (domain.n_variables,) or not np.all(np.isfinite(vector)):
        violation = ConstraintViolation(
            constraint_id="solution_vector",
            group="numerics",
            amount=math.inf,
            label="solution vector has invalid shape or non-finite values",
        )
        return PortfolioMetrics(), (violation,), math.inf

    row_value = np.asarray(domain.A @ vector, dtype=float).reshape(-1)
    row_violation = np.maximum(
        np.where(np.isfinite(domain.lower), domain.lower - row_value, 0.0),
        np.where(np.isfinite(domain.upper), row_value - domain.upper, 0.0),
    )
    variable_violation = np.maximum(
        np.where(
            np.isfinite(domain.variable_lower),
            domain.variable_lower - vector,
            0.0,
        ),
        np.where(
            np.isfinite(domain.variable_upper),
            vector - domain.variable_upper,
            0.0,
        ),
    )
    row_records = {
        item.index: item for item in domain.constraints if item.location == "row"
    }
    variable_records = {
        item.index: item for item in domain.constraints if item.location == "variable"
    }
    violations: list[ConstraintViolation] = []
    for index in np.flatnonzero(row_violation > 0.0):
        record = row_records.get(int(index))
        offset = (
            float(record.metadata.get("expression_offset", 0.0))
            if record is not None
            else 0.0
        )
        violations.append(
            ConstraintViolation(
                constraint_id=(
                    record.constraint_id
                    if record is not None
                    else f"canonical_row:{index}"
                ),
                group=record.group if record is not None else "canonical_row",
                amount=float(row_violation[index]),
                observed=float(row_value[index]) + offset,
                lower=float(domain.lower[index]) + offset,
                upper=float(domain.upper[index]) + offset,
                label=record.key if record is not None else None,
            )
        )
    for index in np.flatnonzero(variable_violation > 0.0):
        record = variable_records.get(int(index))
        violations.append(
            ConstraintViolation(
                constraint_id=(
                    record.constraint_id
                    if record is not None
                    else f"variable_bound:{index}"
                ),
                group=record.group if record is not None else "variable_bound",
                amount=float(variable_violation[index]),
                observed=float(vector[index]),
                lower=float(domain.variable_lower[index]),
                upper=float(domain.variable_upper[index]),
                label=record.key if record is not None else None,
            )
        )

    weight = vector[domain.weight_indices]
    data = problem.data
    benchmark = (
        None if data.benchmark is None else np.asarray(data.benchmark, dtype=float)
    )
    active = None if benchmark is None else weight - benchmark
    factor_variance = specific_variance = tracking_error = None
    if isinstance(data.risk_model, FactorRiskModel) and active is not None:
        factor = np.asarray(data.risk_model.exposure, dtype=float).T @ active
        factor_variance = float(
            factor @ np.asarray(data.risk_model.covariance, dtype=float) @ factor
        )
        specific_variance = float(
            np.square(np.asarray(data.risk_model.specific_volatility) * active).sum()
        )
        tracking_error = math.sqrt(max(0.0, factor_variance + specific_variance))

    risk_limit = None
    if isinstance(model, FactorQCQP):
        risk_limit = model.risk_limit
    elif isinstance(model, QuadraticProgram):
        risk_limit = model.risk_limit
    if tracking_error is not None and risk_limit is not None:
        risk_violation = max(0.0, tracking_error - risk_limit)
        if risk_violation > 0.0:
            violations.append(
                ConstraintViolation(
                    constraint_id="tracking_error:annualized",
                    group="tracking_error",
                    amount=risk_violation,
                    observed=tracking_error,
                    lower=0.0,
                    upper=risk_limit,
                    label="annualized_decimal",
                )
            )

    alpha_value = None if data.alpha is None else float(np.asarray(data.alpha) @ weight)
    objective_value: float | None
    if isinstance(problem.objective, MaximizeAlpha):
        objective_value = alpha_value
    elif isinstance(problem.objective, RiskAdjustedAlpha):
        assert alpha_value is not None
        objective_value = alpha_value
        if factor_variance is not None:
            objective_value -= problem.objective.factor_aversion * factor_variance
        if specific_variance is not None:
            objective_value -= problem.objective.specific_aversion * specific_variance
    elif isinstance(problem.objective, MinimizeTrackingError):
        objective_value = tracking_error
    else:
        objective_value = None

    turnover = (
        None
        if data.initial_weight is None
        else float(np.abs(weight - np.asarray(data.initial_weight, dtype=float)).sum())
    )
    total_active = None if active is None else float(np.abs(active).sum())
    benchmark_member_weight = None
    if benchmark is not None:
        benchmark_member_weight = float(weight[benchmark > 0.0].sum())

    # 使用已经复算的业务总量验收：逐行 epigraph 小残差可能累计超过总量容差。
    # 即使编译器删除了冗余行，也必须覆盖其原始约束；不增加新的矩阵运算。
    aggregate_checks = (
        (
            "turnover:l1",
            "turnover",
            turnover,
            None,
            problem.constraints.turnover.l1_limit
            if problem.constraints.turnover is not None
            else None,
        ),
        (
            "total_active:l1",
            "total_active",
            total_active,
            None,
            problem.constraints.total_active,
        ),
        (
            "benchmark_member_weight:members",
            "benchmark_member_weight",
            benchmark_member_weight,
            problem.constraints.benchmark_member_weight.value
            if problem.constraints.benchmark_member_weight is not None
            else None,
            None,
        ),
    )
    for constraint_id, group, observed, lower, upper in aggregate_checks:
        if observed is None or (lower is None and upper is None):
            continue
        amount = max(
            0.0,
            lower - observed if lower is not None else 0.0,
            observed - upper if upper is not None else 0.0,
        )
        # 总量行若已有残差，保留较大值；相同时使用原始业务坐标，避免常数移位误读。
        existing = next(
            (i for i, v in enumerate(violations) if v.constraint_id == constraint_id),
            None,
        )
        if amount > 0.0 and (existing is None or amount >= violations[existing].amount):
            violation = ConstraintViolation(
                constraint_id=constraint_id,
                group=group,
                amount=amount,
                observed=observed,
                lower=lower,
                upper=upper,
                label="original_weight_total",
            )
            if existing is None:
                violations.append(violation)
            else:
                violations[existing] = violation

    max_style = max_industry = None
    if isinstance(data.risk_model, FactorRiskModel) and active is not None:
        factor_active = np.asarray(data.risk_model.exposure, dtype=float).T @ active
        style_idx = [
            index
            for index, kind in enumerate(data.risk_model.factor_types)
            if kind.lower() == "style"
        ]
        industry_idx = [
            index
            for index, kind in enumerate(data.risk_model.factor_types)
            if kind.lower() == "industry"
        ]
        if style_idx:
            max_style = float(np.max(np.abs(factor_active[style_idx])))
        if industry_idx:
            max_industry = float(np.max(np.abs(factor_active[industry_idx])))

    metrics = PortfolioMetrics(
        budget=float(weight.sum()),
        objective=objective_value,
        tracking_error=tracking_error,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        turnover_l1=turnover,
        total_active_l1=total_active,
        benchmark_member_weight=benchmark_member_weight,
        max_weight=float(np.max(weight)) if weight.size else None,
        min_weight=float(np.min(weight)) if weight.size else None,
        max_active_weight=(
            float(np.max(np.abs(active))) if active is not None else None
        ),
        max_style_exposure=max_style,
        max_industry_exposure=max_industry,
    )
    max_violation = max((item.amount for item in violations), default=0.0)
    return metrics, tuple(violations), float(max_violation)

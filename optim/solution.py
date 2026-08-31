"""Backend-independent reconstruction, metrics and feasibility validation.

This module intentionally does not inspect native solver status or residuals.
It treats a returned vector as untrusted numerical data and checks it against
the same canonical payload supplied to the backend, while separately recomputing
business risk and objective metrics from :class:`PortfolioProblem`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model.canonical import CompiledProblem, FactorQCQP, QuadraticProgram
from .portfolio_types import (
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
    """Reconstruct canonical auxiliary variables for a portfolio weight vector.

    Weight cleanup changes only portfolio columns.  Before revalidation this
    function deterministically rebuilds absolute-turnover, total-active and
    factor-exposure columns so the cleaned vector is compared with the original
    mathematical model rather than with stale solver auxiliaries.
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
    factor_values = None
    if isinstance(problem.data.risk_model, FactorRiskModel) and active is not None:
        factor_values = np.asarray(problem.data.risk_model.exposure, dtype=float).T @ active
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
            vector[record.index] = max(difference, 0.0) if sparse_turnover else abs(difference)
        elif record.group == "active_aux":
            assert record.key is not None and active is not None
            vector[record.index] = abs(active[asset_positions[record.key]])
        elif record.group == "factor_active":
            assert record.key is not None
            vector[record.index] = factor_lookup[record.key]
    return vector


def evaluate_solution(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    vector: np.ndarray,
) -> tuple[PortfolioMetrics, tuple[ConstraintViolation, ...], float]:
    """Recompute all canonical constraints and business metrics once.

    Violations are reported in their native units and linked back through the
    constraint registry.  Tracking error is recomputed from annualized factor
    covariance and specific volatility; it is never inferred from a solver's
    cone residual.  The returned maximum is the acceptance quantity used by the
    common API for primary and fallback backends alike.
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
        violations.append(
            ConstraintViolation(
                constraint_id=(
                    record.constraint_id if record is not None else f"canonical_row:{index}"
                ),
                group=record.group if record is not None else "canonical_row",
                amount=float(row_violation[index]),
                observed=float(row_value[index]),
                lower=float(domain.lower[index]),
                upper=float(domain.upper[index]),
                label=record.key if record is not None else None,
            )
        )
    for index in np.flatnonzero(variable_violation > 0.0):
        record = variable_records.get(int(index))
        violations.append(
            ConstraintViolation(
                constraint_id=(
                    record.constraint_id if record is not None else f"variable_bound:{index}"
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
    benchmark = None if data.benchmark is None else np.asarray(data.benchmark, dtype=float)
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
        max_active_weight=(float(np.max(np.abs(active))) if active is not None else None),
        max_style_exposure=max_style,
        max_industry_exposure=max_industry,
    )
    max_violation = max((item.amount for item in violations), default=0.0)
    return metrics, tuple(violations), float(max_violation)

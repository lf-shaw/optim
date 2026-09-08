"""显式且可能较昂贵的不可行诊断。

普通求解不会自动进入本模块。调用方选择一个问题并请求 ``level='deep'``，
才会计算 Phase-I 松弛、最小线性换手率下界，以及适用时的最小跟踪误差。诊断复用 canonical
registry 元数据，因此证据以业务约束名称报告，而不是匿名矩阵行。
公共入口在校验和编译前拒绝成功结果；单独传入问题时不查询历史求解状态。
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import scipy.sparse as sp

from ._core import CoreBackendResult, CoreSolver
from .diagnostics import InfeasibilityReport, RequiredRelaxation
from .model.canonical import (
    CanonicalKind,
    CompiledProblem,
    ConstraintRecord,
    LinearDomain,
    LinearProgram,
    VariableRecord,
)
from .portfolio_types import (
    FailureReason,
    MinimizeTrackingError,
    OptimizationResult,
    PortfolioProblem,
    SolveStatus,
    SolverAttempt,
    SolverPolicy,
)


def diagnose_problem(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    policy: SolverPolicy,
    *,
    prior_result: OptimizationResult | None = None,
    level: str = "deep",
) -> InfeasibilityReport:
    """对一个准确 fingerprint 执行当前全部深度诊断阶段。

    Phase-I LP 固定编译器标记的硬约束，仅最小化可放宽行/边界的尺度化 slack。最小换手率问题
    只删除汇总换手率上限；存在非线性 TE 约束时，其结果只是完整问题下界。只有确认线性域
    可行后才求最小 TE。

    Parameters
    ----------
    problem : PortfolioProblem
        需要诊断的原始业务问题。
    compiled : CompiledProblem
        与 ``problem`` 对应的 canonical 编译结果。
    policy : SolverPolicy
        诊断子问题使用的数值容差和后端设置。
    prior_result : OptimizationResult | None
        同一问题此前的失败结果；提供时必须具有完全相同的 fingerprint。
    level : str
        显式诊断深度；当前仅支持 ``"deep"``。

    Returns
    -------
    InfeasibilityReport
        线性可行性、所需松弛、换手率/TE 边界及全部诊断尝试。

    Raises
    ------
    ValueError
        level 不受支持，或先前结果与当前问题 fingerprint 不一致。
    """

    if level != "deep":
        raise ValueError("only explicit level='deep' diagnostics are supported")
    if prior_result is not None and prior_result.fingerprint != compiled.fingerprint:
        raise ValueError(
            "prior result fingerprint does not match the diagnosed problem"
        )

    from .model.compiler import _diagnostic_domain

    domain = _diagnostic_domain(problem)
    phase_result, relaxations = _solve_phase_one(domain, policy)
    phase_feasible = _linear_feasibility(phase_result, domain, policy)
    attempts = [_attempt(phase_result, "diagnostic_phase_one")]
    turnover_lower = None
    if problem.constraints.turnover is not None:
        turnover_result, turnover_lower = _minimum_linear_turnover(
            problem,
            compiled,
            policy,
            domain,
        )
        attempts.append(_attempt(turnover_result, "diagnostic_minimum_turnover"))

    reported_turnover_lower = turnover_lower
    linear_feasible, evidence_conflict = _combine_linear_evidence(
        phase_feasible,
        phase_result,
        domain,
        problem,
        turnover_lower,
        policy.tuning.feasibility_tolerance,
    )
    if evidence_conflict is not None:
        # 不把相互矛盾的下界交给自动换手率恢复；原值仅保留为待核查遥测。
        turnover_lower = None

    minimum_te = None
    if linear_feasible is True and problem.constraints.tracking_error is not None:
        from .api import PortfolioOptimizer

        min_risk_problem = replace(
            problem,
            objective=MinimizeTrackingError(),
            constraints=replace(problem.constraints, tracking_error=None),
        )
        # 原目标中的 alpha_floor 实际也是约束；改变目标时必须保留其可行域。
        if isinstance(problem.objective, MinimizeTrackingError):
            min_risk_problem = replace(min_risk_problem, objective=problem.objective)
        min_risk_result = PortfolioOptimizer(policy).solve(min_risk_problem)
        attempts.extend(min_risk_result.route)
        if min_risk_result.status.has_solution:
            minimum_te = min_risk_result.metrics.tracking_error

    turnover_limit = (
        None
        if problem.constraints.turnover is None
        else problem.constraints.turnover.l1_limit
    )
    tracking_limit = (
        None
        if problem.constraints.tracking_error is None
        else problem.constraints.tracking_error.annualized
    )
    summary = _summary(
        linear_feasible,
        relaxations,
        reported_turnover_lower,
        turnover_limit,
        minimum_te,
        tracking_limit,
        evidence_conflict=evidence_conflict,
        tolerance=policy.tuning.feasibility_tolerance,
    )
    return InfeasibilityReport(
        stage="deep",
        linear_feasible=linear_feasible,
        summary_text=summary,
        turnover_linear_lower_bound=turnover_lower,
        turnover_convex_minimum=None,
        turnover_limit=turnover_limit,
        minimum_tracking_error=minimum_te,
        tracking_error_limit=tracking_limit,
        relaxations=relaxations,
        native_evidence={
            "phase_one_status": phase_result.native_status,
            "phase_one_objective": phase_result.objective_value,
            "phase_one_dual_lower_bound": phase_result.diagnostics.get(
                "dual_lower_bound"
            ),
            "conclusion_basis": "conflicting_evidence"
            if evidence_conflict is not None
            else "numerical_estimate"
            if linear_feasible is False
            else "primal_witness"
            if linear_feasible is True
            else "unavailable",
            "phase_one_linear_feasible": phase_feasible,
            "reported_turnover_lower_bound": reported_turnover_lower,
            "evidence_conflict": evidence_conflict,
            "diagnostic_model": "full_linear_domain",
            "phase_one_turnover_l1": (
                float(
                    np.abs(
                        phase_result.primal[domain.weight_indices]
                        - problem.data.initial_weight
                    ).sum()
                )
                if phase_result.status.has_solution
                and phase_result.primal is not None
                and problem.data.initial_weight is not None
                else None
            ),
            "protected_bounds": "nonnegative_and_operational",
            "certificate_availability": _certificate_availability(prior_result),
        },
        native_certificates=(
            () if prior_result is None else prior_result.native_infeasibility
        ),
        attempts=tuple(attempts),
    )


def _certificate_availability(
    result: OptimizationResult | None,
) -> tuple[dict[str, str], ...]:
    """解释证据缺失，不为补取证书追加求解；缺少细节时明确保留未知原因。"""

    if result is None:
        return (
            {
                "availability": "no_prior_result",
                "message": "未提供原求解结果，未采集原生证据。",
            },
        )
    available = {item.backend for item in result.native_infeasibility}
    records = []
    for attempt in result.route:
        error = attempt.metadata.get("infeasibility_evidence_error")
        state = (
            "available"
            if attempt.backend in available
            else "read_error"
            if error
            else "not_retained"
            if attempt.status is SolveStatus.INFEASIBLE
            else "not_reported_infeasible"
        )
        records.append(
            {
                "backend": attempt.backend,
                "availability": state,
                "native_status": attempt.native_status or "",
                "message": str(error)
                if error
                else {
                    "available": "已保留原生数值证据。",
                    "not_retained": "该尝试未附带可读取的原生证据；未为补取证据额外求解。",
                    "not_reported_infeasible": "该尝试未报告不可行，不将末次对偶向量视作不可行证书。",
                }.get(state, ""),
            }
        )
    return tuple(records)


def _combine_linear_evidence(
    phase_feasible: bool | None,
    phase_result: CoreBackendResult,
    domain: LinearDomain,
    problem: PortfolioProblem,
    turnover_lower: float | None,
    tolerance: float,
) -> tuple[bool | None, str | None]:
    """综合数值下界与可行候选；用原始候选检查冲突，不依赖展示用松弛列表。"""

    if (
        problem.constraints.turnover is None
        or turnover_lower is None
        or not np.isfinite(turnover_lower)
    ):
        return phase_feasible, None
    limit = problem.constraints.turnover.l1_limit
    if phase_feasible is True and turnover_lower > limit + tolerance:
        return None, "原线性域已有可行候选，但最小换手率数值下界高于原上限。"
    if phase_result.status.has_solution and phase_result.primal is not None:
        x = phase_result.primal[: domain.n_variables]
        if np.all(np.isfinite(x)) and problem.data.initial_weight is not None:
            activity = domain.A @ x
            keep = np.ones(domain.n_constraints, dtype=bool)
            for record in domain.constraints:
                if record.location == "row" and record.group == "turnover":
                    keep[record.index] = False
            violations = np.concatenate(
                (
                    domain.lower[keep] - activity[keep],
                    activity[keep] - domain.upper[keep],
                    domain.variable_lower - x,
                    x - domain.variable_upper,
                )
            )
            remaining_feasible = (
                np.all(np.isfinite(activity))
                and np.max(violations, initial=0.0) <= tolerance
            )
            observed_turnover = float(
                np.abs(x[domain.weight_indices] - problem.data.initial_weight).sum()
            )
            if remaining_feasible and turnover_lower > observed_turnover + tolerance:
                return None, (
                    f"Phase-I 原始候选满足除换手率上限外的线性约束，实际双边换手率为 {observed_turnover:.4%}，"
                    f"却低于最小换手率检查报告的数值下界 {turnover_lower:.4%}。"
                )
    if turnover_lower > limit + tolerance:
        return False, None
    return phase_feasible, None


def _linear_feasibility(
    result: CoreBackendResult, domain: LinearDomain, policy: SolverPolicy
) -> bool | None:
    """以完整原域残差确认可行，以数值对偶下界支持不可行，其余情况保留未确定。"""

    tolerance = policy.tuning.feasibility_tolerance
    if result.status.has_solution and result.primal is not None:
        x = result.primal[: domain.n_variables]
        activity = domain.A @ x
        if np.all(np.isfinite(x)) and np.all(np.isfinite(activity)):
            violations = np.concatenate(
                (
                    domain.lower - activity,
                    activity - domain.upper,
                    domain.variable_lower - x,
                    x - domain.variable_upper,
                )
            )
            if np.max(violations, initial=0.0) <= tolerance:
                return True
        lower = result.diagnostics.get("dual_lower_bound")
        if lower is not None and np.isfinite(lower) and lower > tolerance:
            return False
    return None


def _solve_phase_one(
    domain: LinearDomain,
    policy: SolverPolicy,
) -> tuple[CoreBackendResult, tuple[RequiredRelaxation, ...]]:
    """在可放宽约束上构造并求解加权 slack Phase-I LP。"""

    row_records = {
        item.index: item for item in domain.constraints if item.location == "row"
    }
    variable_records = {
        item.index: item for item in domain.constraints if item.location == "variable"
    }
    row_specs: list[
        tuple[sp.csc_matrix, float, float, ConstraintRecord | None, str, float]
    ] = []
    for index in range(domain.n_constraints):
        record = row_records.get(index)
        row = domain.A[index].tocsc()
        if record is None or not record.relaxable:
            row_specs.append(
                (row, domain.lower[index], domain.upper[index], record, "hard", 0.0)
            )
            continue
        if np.isfinite(domain.lower[index]):
            row_specs.append((row, domain.lower[index], np.inf, record, "lower", 1.0))
        if np.isfinite(domain.upper[index]):
            row_specs.append((row, -np.inf, domain.upper[index], record, "upper", -1.0))

    variable_lower = domain.variable_lower.copy()
    variable_upper = domain.variable_upper.copy()
    for index in range(domain.n_variables):
        record = variable_records.get(index)
        if record is None or not record.relaxable:
            continue
        unit_row = sp.csc_matrix(([1.0], ([0], [index])), shape=(1, domain.n_variables))
        hard_lower = float(record.metadata.get("diagnostic_hard_lower", -np.inf))
        if (
            np.isfinite(domain.variable_lower[index])
            and domain.variable_lower[index] > hard_lower
        ):
            row_specs.append(
                (unit_row, domain.variable_lower[index], np.inf, record, "lower", 1.0)
            )
            variable_lower[index] = hard_lower
        if np.isfinite(domain.variable_upper[index]):
            row_specs.append(
                (unit_row, -np.inf, domain.variable_upper[index], record, "upper", -1.0)
            )
            variable_upper[index] = np.inf

    slack_specs = [item for item in row_specs if item[4] != "hard"]
    n_slack = len(slack_specs)
    n_variables = domain.n_variables + n_slack
    base_matrix = sp.vstack([item[0] for item in row_specs], format="csc")
    slack_rows: list[int] = []
    slack_columns: list[int] = []
    slack_values: list[float] = []
    slack_meta: list[tuple[ConstraintRecord, str, float, float]] = []
    slack_index = 0
    for row_index, (_, row_lower, row_upper, record, side, sign) in enumerate(
        row_specs
    ):
        if side == "hard":
            continue
        assert record is not None
        slack_rows.append(row_index)
        slack_columns.append(slack_index)
        slack_values.append(sign)
        bound = row_lower if side == "lower" else row_upper
        scale = max(1e-8, float(record.diagnostic_scale))
        slack_meta.append((record, side, float(bound), scale))
        slack_index += 1
    slack_matrix = sp.csc_matrix(
        (slack_values, (slack_rows, slack_columns)),
        shape=(len(row_specs), n_slack),
    )
    matrix = sp.hstack([base_matrix, slack_matrix], format="csc")
    phase_lower = np.asarray([item[1] for item in row_specs], dtype=float)
    phase_upper = np.asarray([item[2] for item in row_specs], dtype=float)
    variables = list(domain.variables)
    for index in range(n_slack):
        variables.append(
            VariableRecord(
                index=domain.n_variables + index,
                variable_id=f"diagnostic_slack:{index}",
                group="diagnostic_slack",
                unit="scaled_relaxation",
            )
        )
    phase_domain = LinearDomain(
        A=matrix,
        lower=phase_lower,
        upper=phase_upper,
        variable_lower=np.concatenate([variable_lower, np.zeros(n_slack)]),
        variable_upper=np.concatenate([variable_upper, np.full(n_slack, np.inf)]),
        variables=tuple(variables),
        constraints=(),
        weight_indices=domain.weight_indices,
        assets=domain.assets,
    )
    c: np.ndarray = np.zeros(n_variables, dtype=float)
    c[domain.n_variables :] = np.asarray([1.0 / item[3] for item in slack_meta])
    phase_model = LinearProgram(CanonicalKind.LP, phase_domain, c)
    result = _solve_core_lp(phase_model, policy)
    relaxations: list[RequiredRelaxation] = []
    if result.status.has_solution and result.primal is not None:
        values = result.primal[domain.n_variables :]
        for value, (record, side, bound, scale) in zip(values, slack_meta):
            if value > policy.tuning.feasibility_tolerance:
                relaxations.append(
                    RequiredRelaxation(
                        constraint_id=record.constraint_id,
                        group=record.group,
                        side=side,
                        amount=float(value),
                        configured_bound=bound,
                        diagnostic_scale=scale,
                        key=record.key,
                        sources=_record_sources(record, side),
                        metadata=record.metadata,
                    )
                )
    relaxations.sort(key=lambda item: item.amount / item.diagnostic_scale, reverse=True)
    return result, tuple(relaxations)


def _record_sources(record: ConstraintRecord, side: str) -> tuple[str, ...]:
    """返回有效边界的配置来源；普通约束退回其单一 ``source``。"""

    raw = record.metadata.get(f"{side}_sources")
    if raw is None:
        return (record.source,)
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(value) for value in raw)


def _minimum_linear_turnover(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    policy: SolverPolicy,
    diagnostic_domain: LinearDomain | None = None,
) -> tuple[CoreBackendResult, float | None]:
    """仅移除已配置换手率上限后，最小化 L1 换手率。"""

    from .model.compiler import _diagnostic_domain

    domain = (
        _diagnostic_domain(problem) if diagnostic_domain is None else diagnostic_domain
    )
    turnover_rows = {
        item.index
        for item in domain.constraints
        if item.location == "row" and item.group == "turnover"
    }
    keep = np.asarray(
        [index not in turnover_rows for index in range(domain.n_constraints)],
        dtype=bool,
    )
    reduced = LinearDomain(
        A=domain.A[keep].tocsc(),
        lower=domain.lower[keep],
        upper=domain.upper[keep],
        variable_lower=domain.variable_lower,
        variable_upper=domain.variable_upper,
        variables=domain.variables,
        constraints=(),
        weight_indices=domain.weight_indices,
        assets=domain.assets,
    )
    c: np.ndarray = np.zeros(domain.n_variables, dtype=float)
    turnover_aux = [item for item in domain.variables if item.group == "turnover_aux"]
    for item in turnover_aux:
        c[item.index] = 1.0
    result = _solve_core_lp(LinearProgram(CanonicalKind.LP, reduced, c), policy)
    # 可行候选目标是最小化问题的上界；不能作为后续恢复的不可行排除下界。
    value = (
        result.diagnostics.get("dual_lower_bound")
        if result.status.has_solution
        else None
    )
    if value is not None:
        value = max(0.0, float(value))
    return result, value


def _solve_core_lp(
    model: LinearProgram,
    policy: SolverPolicy,
) -> CoreBackendResult:
    """通过 core 根入口求解诊断 LP，不访问具体后端模块。"""

    from ._solver_adapter import core_options_from_policy

    core = CoreSolver(core_options_from_policy(policy))
    return core.solve(
        core.prepare(model),
        objective_tolerance=0.0,
        collect_dual_bound=True,
    ).final


def _attempt(result: CoreBackendResult, phase: str) -> SolverAttempt:
    return SolverAttempt(
        backend=result.backend,
        status=SolveStatus(result.status.value),
        reason=(None if result.reason is None else FailureReason(result.reason.value)),
        native_status=result.native_status,
        message=result.message,
        solve_s=result.solve_s,
        metadata={"diagnostic_phase": phase},
    )


def _summary(
    linear_feasible: bool | None,
    relaxations: tuple[RequiredRelaxation, ...],
    turnover_lower: float | None,
    turnover_limit: float | None,
    minimum_te: float | None,
    tracking_limit: float | None,
    *,
    evidence_conflict: str | None = None,
    tolerance: float = 1e-5,
) -> str:
    parts = [
        "已找到满足容差的线性可行候选。"
        if linear_feasible is True
        else "综合数值证据支持原问题的线性部分不可行。"
        if linear_feasible is False
        else "线性可行性尚未确定，诊断未得到充分证据。"
    ]
    if evidence_conflict is not None:
        parts = [
            "诊断证据存在冲突，线性可行性暂不下结论。",
            evidence_conflict,
            "该换手率下界暂不采信，不能据此给出修复幅度；需检查数值误差或模型转换。",
        ]
    if (
        evidence_conflict is None
        and turnover_lower is not None
        and turnover_limit is not None
        and turnover_lower > turnover_limit + tolerance
    ):
        parts.append(
            f"保持其他线性约束时，最小双边换手率的数值下界为 {turnover_lower:.4%}，"
            f"高于约束 {turnover_limit:.4%}。若仅放宽换手率，按此数值下界至少需增加 "
            f"{(turnover_lower - turnover_limit) * 100:.4f} 个百分点；此界为浮点数值估计"
            "，不保证放宽到下界就足够，尤其当还存在风险预算时。"
        )
    if (
        minimum_te is not None
        and tracking_limit is not None
        and minimum_te > tracking_limit
    ):
        parts.append(
            f"风险最小化候选的年化跟踪误差为 {minimum_te:.4%}，"
            f"高于预算 {tracking_limit:.4%}；候选值是最小风险的上界，不能单凭它确认预算不可行。"
        )
    if relaxations:
        turnover_slack = next(
            (
                item
                for item in relaxations
                if item.group == "turnover" and item.side == "upper"
            ),
            None,
        )
        parts.append(
            f"一个加权 Phase-I 方案列出了 {len(relaxations)} 条超过展示阈值的边界松弛。"
        )
        if turnover_slack is not None:
            item = turnover_slack
            parts.append(
                f"其中 {item.constraint_id} 的松弛 {item.amount:.6g} 使用小数权重单位，即增加 "
                f"{item.amount * 100:.4f} 个百分点，上限从 {item.configured_bound:.4%} "
                f"变为 {item.configured_bound + item.amount:.4%}。"
            )
            others = [item for item in relaxations if item is not turnover_slack]
            if others:
                names = "、".join(dict.fromkeys(item.group for item in others))
                parts.append(
                    f"该方案还同时松弛了 {names} 等约束，应结合完整 relaxations 一起查看；它与保持其他约束不变的最小换手率检查不同。"
                )
            else:
                parts.append(
                    "展示列表中只有该项；是否与最小换手率下界矛盾，以完整原始候选检查为准，不能仅凭过滤后的列表判断。"
                )
        else:
            leading = relaxations[0]
            parts.append(
                f"其中 {leading.constraint_id} 的松弛为 {leading.amount:.6g}（该约束原始单位）。"
            )
        parts.append(
            "该方案取决于松弛权重，不是唯一修复，也不代表每条边界必须放宽该数值。"
        )
    return "".join(parts)

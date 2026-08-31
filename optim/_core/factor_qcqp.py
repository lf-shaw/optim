"""针对单一因子模型跟踪风险约束的结构感知求解策略。

PIQP 是 QP 求解器，而不是通用 SOCP 求解器。本模块利用组合风险的“低维因子块加
对角特异风险”结构，以及仅有一个二次风险预算这一条件求解凸 QCQP。实现会构造矩阵和
可行域固定、仅由 theta 改变线性目标的参数化 QP，再搜索一维风险/alpha 前沿。

可选的 HiGHS 预筛选提供严格证书，而非启发式判断：如果外层线性域上的全局 alpha
最优点同时满足风险预算，它也必然是 QCQP 最优点。所有公共风险值均保持年化小数单位；
内部正比例缩放仅用于保持已经审计的 theta 量级和数值条件。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse as sp

from .backends.base import BackendOptions, BackendResult
from .backends.highs import HighsBackend
from .backends.piqp import _constraint_data, resolve_piqp_inequality_form
from .canonical import (
    CanonicalKind,
    ConstraintRecord,
    FactorQCQP,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)
from .contracts import (
    CoreFailureReason as FailureReason,
    CoreSolveStatus as SolveStatus,
    CoreSolverOptions,
)


# 公共风险输入使用年化小数单位。生产 theta 配置最初按百分数风险校准，其方差是小数
# 方差的 10,000 倍。用这一正数缩放完整参数化风险目标不会改变 QCQP 前沿，同时保留
# 已审计的 theta 量级和 PIQP 数值条件。证书计算显式包含同一缩放，结果仍为原始 alpha
# 单位。
_FRONTIER_RISK_OBJECTIVE_SCALE = 10_000.0


@dataclass(frozen=True)
class _FrontierPoint:
    """参数化 QP 搜索得到的一个内部前沿点。

    Attributes
    ----------
    theta : float
        生成该点的正风险/alpha 权衡参数。
    vector : numpy.ndarray
        基础 QCQP 坐标中的 primal 向量。
    variance : float
        独立复算的年化跟踪方差。
    alpha_value : float
        原始 alpha 单位下的线性目标值。
    qp_gap : float
        PIQP 报告的当前参数化子问题对偶间隙。
    recovered : bool
        当前点是否来自一次获准的 workspace 冷重建恢复。
    """

    theta: float
    vector: np.ndarray
    variance: float
    alpha_value: float
    qp_gap: float
    recovered: bool


class _WorkspaceFailure(RuntimeError):
    """将 PIQP workspace 生命周期故障标准化为内部异常。"""

    def __init__(self, message: str, *, first_solve: bool, native_status: str):
        super().__init__(message)
        self.first_solve = first_solve
        self.native_status = native_status


class _ParametricPIQPWorkspace:
    """恢复生命周期受到严格限制的单日 PIQP workspace。

    ``P``、约束和边界仅安装一次。后续 theta 点只更新 ``q``，并通过 PIQP 热启动。
    首次冷求解失败后不会用相同输入重复尝试。至少一次求解成功后，如果更新路径失败且
    策略允许，可以针对当前 QP 冷重建一次；否则调用方得到标准化失败并启动回退。
    workspace 有意不跨日期共享。
    """

    def __init__(self, model: QuadraticProgram, policy: CoreSolverOptions):
        import piqp

        self.piqp = piqp
        self.model = model
        self.policy = policy
        self.form = resolve_piqp_inequality_form(policy.piqp_inequality_form)
        self.equality, self.equality_rhs, self.inequality, self.inequality_lower, self.inequality_upper = (
            _constraint_data(model, self.form)
        )
        self.solver: Any | None = None
        self.current_q: np.ndarray | None = None
        self.successful_solves = 0
        self.setup_s = 0.0
        self.update_s = 0.0
        self.solve_s = 0.0
        self.qp_solves = 0
        self.qp_iterations = 0
        self.rebuilds = 0
        self.trace: list[dict[str, Any]] = []

    def _new_solver(self, q: np.ndarray, tolerance: float) -> Any:
        solver = self.piqp.SparseSolver()
        solver.settings.verbose = False
        solver.settings.eps_abs = tolerance
        solver.settings.eps_rel = tolerance
        solver.settings.eps_duality_gap_abs = tolerance
        solver.settings.eps_duality_gap_rel = tolerance
        solver.settings.max_iter = self.policy.piqp_max_iter
        solver.settings.compute_timings = True
        started = time.perf_counter()
        solver.setup(
            self.model.P,
            q,
            self.equality,
            self.equality_rhs,
            self.inequality,
            self.inequality_lower,
            self.inequality_upper,
            self.model.domain.variable_lower,
            self.model.domain.variable_upper,
        )
        self.setup_s += time.perf_counter() - started
        self.current_q = q.copy()
        return solver

    def solve(self, q: np.ndarray, tolerance: float) -> tuple[np.ndarray, float, bool]:
        """求解或更新一个 theta 点，返回向量、QP 间隙和恢复标志。"""

        is_first = self.solver is None
        used_update = False
        if self.solver is None:
            self.solver = self._new_solver(q, tolerance)
        else:
            self.solver.settings.eps_abs = tolerance
            self.solver.settings.eps_rel = tolerance
            self.solver.settings.eps_duality_gap_abs = tolerance
            self.solver.settings.eps_duality_gap_rel = tolerance
            if self.current_q is None or not np.array_equal(q, self.current_q):
                update_started = time.perf_counter()
                try:
                    self.solver.update(c=q)
                except Exception as exc:
                    self.update_s += time.perf_counter() - update_started
                    return self._rebuild_and_solve(
                        q,
                        tolerance,
                        trigger=f"update exception: {type(exc).__name__}: {exc}",
                    )
                self.update_s += time.perf_counter() - update_started
                self.current_q = q.copy()
                used_update = True

        try:
            vector, qp_gap, native = self._solve_current()
        except _WorkspaceFailure as failure:
            # 经审计策略只重试失败的更新生命周期；首次冷求解不会用相同输入再次运行。
            if used_update and self.policy.rebuild_after_update_failure:
                return self._rebuild_and_solve(
                    q,
                    tolerance,
                    trigger=f"updated solve failed: {failure.native_status}",
                )
            raise
        self.trace.append(
            {
                "phase": "solve",
                "first_solve": is_first,
                "updated": used_update,
                "recovered": False,
                "native_status": native,
                "qp_gap": qp_gap,
                "tolerance": tolerance,
            }
        )
        return vector, qp_gap, False

    def _rebuild_and_solve(
        self,
        q: np.ndarray,
        tolerance: float,
        *,
        trigger: str,
    ) -> tuple[np.ndarray, float, bool]:
        self.rebuilds += 1
        self.solver = self._new_solver(q, tolerance)
        try:
            vector, qp_gap, native = self._solve_current()
        except _WorkspaceFailure:
            self.trace.append(
                {
                    "phase": "cold_rebuild",
                    "trigger": trigger,
                    "recovered": False,
                    "tolerance": tolerance,
                }
            )
            raise
        self.trace.append(
            {
                "phase": "cold_rebuild",
                "trigger": trigger,
                "recovered": True,
                "native_status": native,
                "qp_gap": qp_gap,
                "tolerance": tolerance,
            }
        )
        return vector, qp_gap, True

    def _solve_current(self) -> tuple[np.ndarray, float, str]:
        assert self.solver is not None
        started = time.perf_counter()
        try:
            status = self.solver.solve()
        except Exception as exc:
            self.solve_s += time.perf_counter() - started
            self.qp_solves += 1
            raise _WorkspaceFailure(
                f"PIQP solve exception: {type(exc).__name__}: {exc}",
                first_solve=self.successful_solves == 0,
                native_status="solve_exception",
            ) from exc
        self.solve_s += time.perf_counter() - started
        self.qp_solves += 1
        result = self.solver.result
        info = result.info
        self.qp_iterations += int(getattr(info, "iter", 0) or 0)
        native = str(getattr(info, "status", status))
        if status != self.piqp.Status.PIQP_SOLVED or result.x is None:
            raise _WorkspaceFailure(
                f"PIQP ended with {native}",
                first_solve=self.successful_solves == 0,
                native_status=native,
            )
        self.successful_solves += 1
        qp_gap = _first_number(info, "duality_gap", "dual_gap")
        if qp_gap is None or not np.isfinite(qp_gap):
            qp_gap = math.inf
        return np.asarray(result.x, dtype=float).reshape(-1).copy(), abs(qp_gap), native


def _first_number(value: Any, *names: str) -> float | None:
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            try:
                return float(item)
            except (TypeError, ValueError):
                pass
    return None


def _extend_as_parametric_qp(
    model: FactorQCQP,
) -> tuple[QuadraticProgram, np.ndarray, float, float]:
    r"""将因子 QCQP 提升为固定矩阵的参数化 QP。

    基础 QCQP 可行域包含组合变量和线性辅助变量。实现追加因子主动暴露 $f$ 以及等式
    $f=E^{\mathsf T}(x-b)$。基础域中已经存在的稠密风格/行业行会替换为 $f$ 上的直接
    单列边界，从而避免在 KKT 系统中重复 $E^{\mathsf T}$。

    返回的 QP 编码经正比例缩放后的半风险目标。独立的 ``alpha_solver`` 向量经过中心化
    与缩放；theta 点使用
    $q(\theta)=q_{\mathrm{risk}}-\theta\widetilde\alpha$。在预算等式下，中心化是精确
    等价变换，只影响数值条件而不改变最优解。
    """

    base = model.domain
    risk = model.risk_operator
    n_base = base.n_variables
    n_factors = risk.exposure.shape[1]
    factor_indices = np.arange(n_base, n_base + n_factors, dtype=np.int32)
    n_variables = n_base + n_factors

    original_rows = sp.hstack(
        [base.A, sp.csc_matrix((base.n_constraints, n_factors))], format="csc"
    )
    extended_lower = base.lower.copy()
    extended_upper = base.upper.copy()
    factor_lookup = {
        name.lower(): index for index, name in enumerate(risk.factor_names)
    }
    active_bound_records = tuple(
        record
        for record in base.constraints
        if record.location == "row"
        and record.group in {"style", "industry"}
        and record.key is not None
    )
    if active_bound_records:
        bound_rows = np.fromiter(
            (record.index for record in active_bound_records), dtype=np.int32
        )
        bound_factors = np.fromiter(
            (factor_lookup[str(record.key).lower()] for record in active_bound_records),
            dtype=np.int32,
        )
        # 这些 canonical 行为 $E_j^{\mathsf T}x$，其边界已经按基准平移。引入
        # $f=E^{\mathsf T}(x-b)$ 后，将每个稠密行替换为 $f_j$ 上对应的单列边界。
        keep = np.ones(base.n_constraints, dtype=float)
        keep[bound_rows] = 0.0
        original_rows = sp.diags(keep, format="csc") @ original_rows
        direct_factor_bounds = sp.csc_matrix(
            (
                np.ones(len(bound_rows)),
                (bound_rows, factor_indices[bound_factors]),
            ),
            shape=(base.n_constraints, n_variables),
        )
        original_rows = (original_rows + direct_factor_bounds).tocsc()
        benchmark_shift = (
            risk.exposure[:, bound_factors].T @ risk.benchmark
        )
        extended_lower[bound_rows] -= benchmark_shift
        extended_upper[bound_rows] -= benchmark_shift
    factor_rows = np.arange(n_factors, dtype=np.int32)
    factor_identity = sp.csc_matrix(
        (-np.ones(n_factors), (factor_rows, factor_indices)),
        shape=(n_factors, n_variables),
    )
    exposure_rows = sp.hstack(
        [
            sp.csc_matrix(risk.exposure.T),
            sp.csc_matrix((n_factors, n_base - len(risk.weight_indices))),
            sp.csc_matrix((n_factors, n_factors)),
        ],
        format="csc",
    )
    factor_definition = exposure_rows + factor_identity
    factor_rhs = risk.exposure.T @ risk.benchmark
    A = sp.vstack([original_rows, factor_definition], format="csc")
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    variables = list(base.variables)
    constraints = list(base.constraints)
    for local_index, (name, index) in enumerate(
        zip(risk.factor_names, factor_indices)
    ):
        variables.append(
            VariableRecord(
                index=int(index),
                variable_id=f"factor_active:{name}",
                group="factor_active",
                key=str(name),
                unit="exposure",
            )
        )
        constraints.append(
            ConstraintRecord(
                constraint_id=f"factor_definition:{name}",
                group="factor_definition",
                location="row",
                index=base.n_constraints + local_index,
                key=str(name),
                unit="exposure",
                source="factor_qcqp_strategy",
                relaxable=False,
            )
        )
    extended_domain = LinearDomain(
        A=A,
        lower=np.concatenate([extended_lower, factor_rhs]),
        upper=np.concatenate([extended_upper, factor_rhs]),
        variable_lower=np.concatenate([base.variable_lower, np.full(n_factors, -np.inf)]),
        variable_upper=np.concatenate([base.variable_upper, np.full(n_factors, np.inf)]),
        variables=tuple(variables),
        constraints=tuple(constraints),
        weight_indices=base.weight_indices,
        assets=base.assets,
    )

    specific_variance = np.square(risk.specific_volatility)
    factor_matrix = risk.covariance
    factor_row, factor_column = np.nonzero(factor_matrix)
    P = sp.csc_matrix(
        (
            np.concatenate([specific_variance, factor_matrix[factor_row, factor_column]]),
            (
                np.concatenate([base.weight_indices, factor_indices[factor_row]]),
                np.concatenate([base.weight_indices, factor_indices[factor_column]]),
            ),
        ),
        shape=(n_variables, n_variables),
    )
    P = P * _FRONTIER_RISK_OBJECTIVE_SCALE
    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    q_base = np.zeros(n_variables, dtype=float)
    q_base[base.weight_indices] = (
        -_FRONTIER_RISK_OBJECTIVE_SCALE * specific_variance * risk.benchmark
    )
    offset = (
        0.5
        * _FRONTIER_RISK_OBJECTIVE_SCALE
        * float(np.sum(specific_variance * np.square(risk.benchmark)))
    )

    alpha = np.asarray(model.alpha, dtype=float)
    centered_alpha = alpha - float(np.mean(alpha))
    max_abs = float(np.max(np.abs(centered_alpha))) if centered_alpha.size else 0.0
    alpha_scale = 1.0 if max_abs <= np.finfo(float).eps else 0.2 / max_abs
    alpha_solver = np.zeros(n_variables, dtype=float)
    alpha_solver[base.weight_indices] = centered_alpha * alpha_scale
    qp = QuadraticProgram(
        kind=CanonicalKind.QP,
        domain=extended_domain,
        P=P,
        q=q_base,
        objective_offset=offset,
        risk_operator=risk,
    )
    return qp, alpha_solver, alpha_scale, max_abs


def solve_factor_qcqp(
    model: FactorQCQP,
    policy: CoreSolverOptions,
    *,
    theta_seed: float | None = None,
    objective_tolerance: float,
) -> BackendResult:
    """求解因子 QCQP，并可选地用外层 LP 提供最优性证书。

    如果相同线性域上的全局最优点同时满足跟踪误差约束，它必然也是 QCQP 的全局最优点。
    此快速路径只能通过 ``policy.lp_prescreen`` 显式启用；默认路径仍为 PIQP 前沿。
    预筛选失败或无法安全认证不视为 QCQP 失败：实现记录其耗时和状态后，继续执行与未
    启用预筛选时相同的前沿路径。

    Parameters
    ----------
    model : FactorQCQP
        已编译的单风险预算因子 QCQP。
    policy : CoreSolverOptions
        后端路由、前沿搜索和数值容差策略。
    theta_seed : float | None
        首个 theta 候选值；``None`` 使用策略默认值。
    objective_tolerance : float
        上层已经换算为原始 alpha 单位的绝对证书间隙。

    Returns
    -------
    BackendResult
        PIQP 前沿或获认证 LP 快速路径的标准化后端结果。
    """

    if not policy.lp_prescreen:
        return _solve_factor_qcqp_frontier(
            model,
            policy,
            theta_seed=theta_seed,
            objective_tolerance=objective_tolerance,
        )

    screen, screen_te, certified = _solve_lp_prescreen(model, policy)
    screen_metadata = {
        "lp_prescreen_enabled": True,
        "lp_prescreen_status": screen.status.value,
        "lp_prescreen_native_status": screen.native_status,
        "lp_prescreen_tracking_error": screen_te,
        "lp_prescreen_certified": certified,
    }
    if certified:
        assert screen.primal is not None
        alpha_value = float(
            model.alpha @ screen.primal[model.domain.weight_indices]
        )
        return replace(
            screen,
            backend="factor_qcqp_lp_prescreen_highs",
            objective_value=alpha_value,
            diagnostics={
                **screen.diagnostics,
                **screen_metadata,
                "qp_solves": 0,
                "workspace_rebuilds": 0,
            },
        )

    frontier = _solve_factor_qcqp_frontier(
        model,
        policy,
        theta_seed=theta_seed,
        objective_tolerance=objective_tolerance,
    )
    return replace(
        frontier,
        setup_s=screen.setup_s + frontier.setup_s,
        solve_s=screen.solve_s + frontier.solve_s,
        diagnostics={**frontier.diagnostics, **screen_metadata},
    )


def _solve_lp_prescreen(
    model: FactorQCQP,
    policy: CoreSolverOptions,
) -> tuple[BackendResult, float | None, bool]:
    """求解精确外层 LP，并检查保留风险裕量的 TE 证书。"""

    c = np.zeros(model.domain.n_variables, dtype=float)
    c[model.domain.weight_indices] = -np.asarray(model.alpha, dtype=float)
    lp = LinearProgram(
        kind=CanonicalKind.LP,
        domain=model.domain,
        c=c,
    )
    result = HighsBackend().solve(
        lp,
        BackendOptions(
            max_iter=policy.piqp_max_iter,
            eps_abs=policy.final_eps,
            eps_rel=policy.final_eps,
        ),
    )
    if result.status is not SolveStatus.OPTIMAL or result.primal is None:
        return result, None, False
    variance, _, _ = model.risk_operator.components(result.primal)
    tracking_error = math.sqrt(max(0.0, variance))
    certified = bool(
        np.isfinite(tracking_error)
        and tracking_error <= max(0.0, model.risk_limit - policy.risk_margin)
    )
    return result, tracking_error, certified


def _solve_factor_qcqp_frontier(
    model: FactorQCQP,
    policy: CoreSolverOptions,
    *,
    theta_seed: float | None = None,
    objective_tolerance: float,
) -> BackendResult:
    r"""通过带保护的一维搜索求解因子 QCQP。

    搜索首先通过几何扩张或收缩建立风险可行/不可行 theta 区间，随后执行有界插值；当
    数值区间平坦或不单调时退回二分。之后进行高精度最终求解，并在必要时从安全侧再次
    接近风险边界。成功必须同时满足风险可行性以及

    $$
    g_{\mathrm{frontier}}+g_{\mathrm{QP}}
    \le\varepsilon_{\mathrm{objective}}.
    $$

    仅仅得到接近预算的 TE 值并不构成停止证书。
    """

    tuning = policy
    started = time.perf_counter()
    try:
        qp, alpha_solver, alpha_scale, alpha_dispersion = _extend_as_parametric_qp(model)
    except Exception as exc:
        return BackendResult(
            backend="factor_qcqp_piqp",
            status=SolveStatus.SOLVER_ERROR,
            primal=None,
            objective_value=None,
            native_status="compile_error",
            reason=FailureReason.INVALID_NUMERICS,
            message=f"{type(exc).__name__}: {exc}",
        )
    matrix_build_s = time.perf_counter() - started
    workspace = _ParametricPIQPWorkspace(qp, policy)
    theta_start = tuning.theta_initial if theta_seed is None else float(theta_seed)
    if not np.isfinite(theta_start) or theta_start <= 0.0:
        theta_start = tuning.theta_initial
    theta_start = min(theta_start, tuning.theta_max)
    risk_budget_variance = model.risk_limit * model.risk_limit
    target_risk = max(0.0, model.risk_limit - tuning.risk_margin)
    target_variance = target_risk * target_risk
    deadline = None
    # 公共总时间上限将通过 SolverPolicy 单独加入，而不会复用 PIQP 的逐 QP 迭代控制。

    points = 0

    def solve_theta(theta: float, *, final: bool = False) -> _FrontierPoint:
        nonlocal points
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("factor-QCQP total time limit reached")
        tolerance = tuning.final_eps if final else tuning.intermediate_eps
        if final:
            tolerance = max(tolerance, 1e-7)
        q = qp.q - theta * alpha_solver
        vector, qp_gap, recovered = workspace.solve(q, tolerance)
        base_vector = vector[: model.domain.n_variables].copy()
        variance, _, _ = model.risk_operator.components(base_vector)
        alpha_value = float(model.alpha @ base_vector[model.domain.weight_indices])
        points += 1
        if workspace.trace:
            workspace.trace[-1].update(
                {"theta": theta, "variance": variance, "alpha_value": alpha_value}
            )
        return _FrontierPoint(theta, base_vector, variance, alpha_value, qp_gap, recovered)

    def frontier_gap(point: _FrontierPoint) -> float:
        if point.theta <= 0.0 or alpha_scale <= 0.0:
            return math.inf
        return max(
            0.0,
            _FRONTIER_RISK_OBJECTIVE_SCALE
            * (risk_budget_variance - point.variance)
            / (2.0 * point.theta * alpha_scale),
        )

    try:
        if alpha_dispersion <= np.finfo(float).eps:
            vector, qp_gap, recovered = workspace.solve(qp.q, tuning.final_eps)
            base_vector = vector[: model.domain.n_variables].copy()
            variance, _, _ = model.risk_operator.components(base_vector)
            if variance > (model.risk_limit + tuning.feasibility_tolerance) ** 2:
                return _failure_result(
                    workspace,
                    matrix_build_s,
                    SolveStatus.INFEASIBLE,
                    FailureReason.INFEASIBLE_REPORTED,
                    "minimum-risk QP exceeds the tracking-error budget",
                )
            return _success_result(
                model,
                workspace,
                matrix_build_s,
                _FrontierPoint(0.0, base_vector, variance, float(model.alpha @ base_vector[model.domain.weight_indices]), qp_gap, recovered),
                theta_start,
                0.0,
                qp_gap,
                objective_tolerance,
                points + 1,
            )

        point = solve_theta(theta_start)
        low: _FrontierPoint | None = None
        high: _FrontierPoint | None = None
        if point.variance <= target_variance:
            low = point
            theta = theta_start
            while points < tuning.max_outer_iters and frontier_gap(low) > objective_tolerance:
                theta *= tuning.theta_growth
                if theta > tuning.theta_max:
                    break
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                else:
                    high = point
                    break
        else:
            high = point
            theta = theta_start
            while points < tuning.max_outer_iters:
                theta /= tuning.theta_growth
                if theta <= np.finfo(float).eps:
                    break
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                    break
                high = point
            if low is None:
                vector, qp_gap, recovered = workspace.solve(qp.q, tuning.final_eps)
                base_vector = vector[: model.domain.n_variables].copy()
                minimum_variance, _, _ = model.risk_operator.components(base_vector)
                if minimum_variance > (model.risk_limit + tuning.feasibility_tolerance) ** 2:
                    return _failure_result(
                        workspace,
                        matrix_build_s,
                        SolveStatus.INFEASIBLE,
                        FailureReason.INFEASIBLE_REPORTED,
                        "minimum-risk QP exceeds the tracking-error budget",
                    )
                # 最小风险端点可行，因此将其作为 theta=0 的区间端点，但绝不在零点计算
                # 前沿间隙。
                low = _FrontierPoint(
                    0.0,
                    base_vector,
                    minimum_variance,
                    float(model.alpha @ base_vector[model.domain.weight_indices]),
                    qp_gap,
                    recovered,
                )

        assert low is not None
        if high is not None:
            while points < tuning.max_outer_iters:
                span = high.theta - low.theta
                if span <= 1e-12 * max(1.0, high.theta):
                    break
                variance_span = high.variance - low.variance
                if variance_span <= 0.0:
                    theta = low.theta + 0.5 * span
                else:
                    theta = low.theta + (
                        (target_variance - low.variance) * span / variance_span
                    )
                theta = min(high.theta - 0.1 * span, max(low.theta + 0.1 * span, theta))
                point = solve_theta(theta)
                if point.variance <= target_variance:
                    low = point
                else:
                    high = point
                if low.theta > 0.0 and frontier_gap(low) <= objective_tolerance:
                    break

        if low.theta <= 0.0:
            return _failure_result(
                workspace,
                matrix_build_s,
                SolveStatus.LIMIT_REACHED,
                FailureReason.MAX_ITER,
                "could not establish a positive-theta feasible frontier point",
            )
        final = solve_theta(low.theta, final=True)
        if final.variance > risk_budget_variance:
            risky_final = final
            theta = final.theta
            safe_final: _FrontierPoint | None = None
            for _ in range(12):
                theta /= tuning.theta_growth
                if theta <= np.finfo(float).eps:
                    break
                candidate = solve_theta(theta, final=True)
                if candidate.variance <= target_variance:
                    safe_final = candidate
                    break
                risky_final = candidate
            if safe_final is None:
                return _failure_result(
                    workspace,
                    matrix_build_s,
                    SolveStatus.NUMERICAL_ERROR,
                    FailureReason.NUMERICAL_FAILURE,
                    "could not recover a high-accuracy risk-feasible frontier point",
                )
            for _ in range(15):
                theta_span = risky_final.theta - safe_final.theta
                if theta_span <= 1e-12 * max(1.0, risky_final.theta):
                    break
                variance_span = risky_final.variance - safe_final.variance
                if variance_span <= 0.0:
                    theta = safe_final.theta + 0.5 * theta_span
                else:
                    theta = safe_final.theta + (
                        (target_variance - safe_final.variance)
                        * theta_span
                        / variance_span
                    )
                theta = min(
                    risky_final.theta - 0.05 * theta_span,
                    max(safe_final.theta + 0.05 * theta_span, theta),
                )
                candidate = solve_theta(theta, final=True)
                if candidate.variance <= target_variance:
                    safe_final = candidate
                    candidate_subproblem_gap = candidate.qp_gap / (
                        candidate.theta * alpha_scale
                    )
                    if (
                        frontier_gap(candidate) + candidate_subproblem_gap
                        <= objective_tolerance
                    ):
                        break
                else:
                    risky_final = candidate
            final = safe_final
        slack_gap = frontier_gap(final)
        subproblem_gap = final.qp_gap / (final.theta * alpha_scale)
        total_gap = slack_gap + subproblem_gap
        return _success_result(
            model,
            workspace,
            matrix_build_s,
            final,
            theta_start,
            slack_gap,
            subproblem_gap,
            objective_tolerance,
            points,
            total_gap=total_gap,
        )
    except TimeoutError as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.LIMIT_REACHED,
            FailureReason.TIME_LIMIT,
            str(exc),
        )
    except _WorkspaceFailure as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.NUMERICAL_ERROR,
            FailureReason.NUMERICAL_FAILURE,
            str(exc),
            native_status=exc.native_status,
        )
    except Exception as exc:
        return _failure_result(
            workspace,
            matrix_build_s,
            SolveStatus.SOLVER_ERROR,
            FailureReason.UNKNOWN,
            f"{type(exc).__name__}: {exc}",
        )


def _success_result(
    model: FactorQCQP,
    workspace: _ParametricPIQPWorkspace,
    matrix_build_s: float,
    point: _FrontierPoint,
    theta_seed: float,
    frontier_gap: float,
    subproblem_gap: float,
    objective_tolerance: float,
    outer_points: int,
    *,
    total_gap: float | None = None,
) -> BackendResult:
    if total_gap is None:
        total_gap = frontier_gap + subproblem_gap
    accepted = total_gap <= objective_tolerance
    return BackendResult(
        backend="factor_qcqp_piqp",
        status=SolveStatus.OPTIMAL if accepted else SolveStatus.LIMIT_REACHED,
        primal=point.vector if accepted else None,
        objective_value=point.alpha_value if accepted else None,
        native_status="PIQP_SOLVED" if accepted else "certificate_limit",
        reason=None if accepted else FailureReason.MAX_ITER,
        message=(
            "parametric factor-QP frontier"
            if accepted
            else "frontier point is feasible but objective certificate exceeds tolerance"
        ),
        iterations=workspace.qp_iterations,
        setup_s=matrix_build_s + workspace.setup_s,
        solve_s=workspace.solve_s + workspace.update_s,
        diagnostics={
            "theta": point.theta,
            "theta_seed": theta_seed,
            "variance": point.variance,
            "tracking_error": math.sqrt(max(0.0, point.variance)),
            "frontier_gap": frontier_gap,
            "subproblem_gap": subproblem_gap,
            "total_gap": total_gap,
            "objective_tolerance": objective_tolerance,
            "risk_objective_scale": _FRONTIER_RISK_OBJECTIVE_SCALE,
            "qp_solves": workspace.qp_solves,
            "outer_points": outer_points,
            "workspace_rebuilds": workspace.rebuilds,
            "inequality_form": workspace.form,
            "trace": tuple(workspace.trace),
        },
    )


def _failure_result(
    workspace: _ParametricPIQPWorkspace,
    matrix_build_s: float,
    status: SolveStatus,
    reason: FailureReason,
    message: str,
    *,
    native_status: str = "factor_qcqp_failure",
) -> BackendResult:
    return BackendResult(
        backend="factor_qcqp_piqp",
        status=status,
        primal=None,
        objective_value=None,
        native_status=native_status,
        reason=reason,
        message=message,
        iterations=workspace.qp_iterations,
        setup_s=matrix_build_s + workspace.setup_s,
        solve_s=workspace.solve_s + workspace.update_s,
        diagnostics={
            "qp_solves": workspace.qp_solves,
            "workspace_rebuilds": workspace.rebuilds,
            "inequality_form": workspace.form,
            "trace": tuple(workspace.trace),
        },
    )

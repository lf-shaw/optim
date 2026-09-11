"""单次 canonical 二次规划的 direct sparse PIQP 适配器。

该适配器每次建立新 workspace 并执行一次求解，使用 compact 双边不等式转换。目标缩放用同一正数
同时乘 ``P`` 和 ``q``，只改变数值条件，不改变数学最优解。
"""

from __future__ import annotations

import time
from importlib import metadata
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..canonical import QuadraticProgram
from ..contracts import (
    CoreFailureReason as FailureReason,
    CoreInfeasibilityEvidence,
    CoreSolveStatus as SolveStatus,
)
from .base import (
    BackendOptions,
    BackendResult,
    capture_infeasibility,
    dual_entries,
    thread_diagnostics,
)


def piqp_distribution_version() -> str | None:
    """返回已安装 PIQP distribution 版本，未安装时返回 ``None``。"""

    try:
        return metadata.version("piqp")
    except metadata.PackageNotFoundError:
        return None


def resolve_piqp_inequality_form(requested: str) -> str:
    """解析 PIQP 线性不等式表示。

    PIQP 0.6.4 是声明的最低依赖，并已包含 compact 双边不等式所需的上游 dual-recovery
    修复，因此 ``auto`` 无条件选择 ``compact``。``one_sided`` 仅保留为显式诊断/benchmark
    选项，不是旧版本兼容路径。

    Parameters
    ----------
    requested : str
        ``"auto"``、``"compact"`` 或 ``"one_sided"``，不区分大小写。

    Returns
    -------
    str
        明确的 ``"compact"`` 或 ``"one_sided"``。

    Raises
    ------
    ValueError
        输入值不受支持。
    """

    form = requested.lower()
    if form not in {"auto", "one_sided", "compact"}:
        raise ValueError(f"unsupported PIQP inequality form: {requested!r}")
    if form == "auto":
        return "compact"
    return form


def _constraint_data(
    model: QuadraticProgram,
    form: str,
) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
    """将 canonical 行拆分为 PIQP 等式和选定形式的不等式。"""

    domain = model.domain
    equality = (
        np.isfinite(domain.lower)
        & np.isfinite(domain.upper)
        & np.isclose(domain.lower, domain.upper, rtol=0.0, atol=1e-14)
    )
    equality_matrix = domain.A[equality].tocsc()
    equality_rhs = domain.lower[equality].copy()
    general_matrix = domain.A[~equality].tocsc()
    general_lower = domain.lower[~equality].copy()
    general_upper = domain.upper[~equality].copy()
    if form == "compact":
        return (
            equality_matrix,
            equality_rhs,
            general_matrix,
            general_lower,
            general_upper,
        )
    finite_upper = np.isfinite(general_upper)
    finite_lower = np.isfinite(general_lower)
    inequality_matrix = sp.vstack(
        [general_matrix[finite_upper], -general_matrix[finite_lower]],
        format="csc",
    )
    inequality_lower = np.full(inequality_matrix.shape[0], -np.inf, dtype=float)
    inequality_upper = np.concatenate(
        [general_upper[finite_upper], -general_lower[finite_lower]]
    )
    return (
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_lower,
        inequality_upper,
    )


class PIQPBackend:
    """已编译凸 QP 的单次 direct PIQP 后端。"""

    name = "piqp"

    def solve(self, model: QuadraticProgram, options: BackendOptions) -> BackendResult:
        """建立新 PIQP workspace 并求解一个 canonical QP。

        Parameters
        ----------
        model : QuadraticProgram
            对称半正定 Hessian 的 canonical 凸 QP。
        options : BackendOptions
            PIQP 容差、迭代数、目标缩放和不等式表示设置。

        Returns
        -------
        BackendResult
            尚未经过公共独立验收的候选向量、状态和原生残差。

        Raises
        ------
        TypeError
            ``model`` 不是 :class:`QuadraticProgram`。
        ImportError
            当前环境未安装项目要求的 PIQP。
        """

        import piqp

        if not isinstance(model, QuadraticProgram):
            raise TypeError("PIQPBackend accepts QuadraticProgram only")
        form = resolve_piqp_inequality_form(options.inequality_form)
        equality, equality_rhs, inequality, inequality_lower, inequality_upper = (
            _constraint_data(model, form)
        )
        objective_scale = _objective_scale(model, options.objective_scale_target)
        P = (model.P * objective_scale).tocsc()
        q = np.asarray(model.q * objective_scale, dtype=float)

        solver = piqp.SparseSolver()
        solver.settings.verbose = bool(options.verbose)
        solver.settings.eps_abs = float(options.eps_abs)
        solver.settings.eps_rel = float(options.eps_rel)
        solver.settings.eps_duality_gap_abs = float(options.eps_abs)
        solver.settings.eps_duality_gap_rel = float(options.eps_rel)
        solver.settings.max_iter = int(options.max_iter)
        solver.settings.compute_timings = True

        setup_started = time.perf_counter()
        try:
            solver.setup(
                P,
                q,
                equality,
                equality_rhs,
                inequality,
                inequality_lower,
                inequality_upper,
                np.asarray(model.domain.variable_lower, dtype=float),
                np.asarray(model.domain.variable_upper, dtype=float),
            )
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status="setup_exception",
                reason=FailureReason.INVALID_NUMERICS,
                message=f"{type(exc).__name__}: {exc}",
                setup_s=time.perf_counter() - setup_started,
                diagnostics={
                    **thread_diagnostics(
                        options, native_threads=1, native_thread_limit=1
                    ),
                    "inequality_form": form,
                    "objective_scale": objective_scale,
                },
            )
        setup_s = time.perf_counter() - setup_started
        solve_started = time.perf_counter()
        try:
            native_status = solver.solve()
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.NUMERICAL_ERROR,
                primal=None,
                objective_value=None,
                native_status="solve_exception",
                reason=FailureReason.NUMERICAL_FAILURE,
                message=f"{type(exc).__name__}: {exc}",
                setup_s=setup_s,
                solve_s=time.perf_counter() - solve_started,
                diagnostics={
                    **thread_diagnostics(
                        options, native_threads=1, native_thread_limit=1
                    ),
                    "inequality_form": form,
                    "objective_scale": objective_scale,
                },
            )
        solve_s = time.perf_counter() - solve_started
        result = solver.result
        status, reason = _map_status(native_status, piqp)
        primal = None
        objective = None
        if status.has_solution and result.x is not None:
            primal = np.asarray(result.x, dtype=float).reshape(-1).copy()
            objective = float(
                0.5 * primal @ (model.P @ primal)
                + model.q @ primal
                + model.objective_offset
            )
        info = result.info
        diagnostics = {
            **thread_diagnostics(options, native_threads=1, native_thread_limit=1),
            "inequality_form": form,
            "objective_scale": objective_scale,
            "objective_scale_reference": model.objective_scale_reference,
            "primal_residual": _number(info, "primal_residual", "primal_res"),
            "dual_residual": _number(info, "dual_residual", "dual_res"),
            "duality_gap": _number(info, "duality_gap", "dual_gap"),
            "solver_setup_s": _number(info, "setup_time"),
            "solver_solve_s": _number(info, "solve_time", "run_time"),
        }
        evidence = None
        if status is SolveStatus.INFEASIBLE:
            evidence = capture_infeasibility(
                lambda: _infeasibility(model, result, form, str(native_status)),
                diagnostics,
            )
        return BackendResult(
            backend=self.name,
            status=status,
            primal=primal,
            objective_value=objective,
            native_status=str(getattr(info, "status", native_status)),
            reason=reason,
            iterations=int(getattr(info, "iter", 0) or 0),
            setup_s=setup_s,
            solve_s=solve_s,
            diagnostics=diagnostics,
            infeasibility=evidence,
        )


def _infeasibility(
    model: QuadraticProgram, result: Any, form: str, native: str
) -> CoreInfeasibilityEvidence:
    """映射单次 QP 的原生对偶估计；前沿搜索的参数 QP 不通过此入口冒充原问题证书。"""

    d = model.domain
    equal = (
        np.isfinite(d.lower)
        & np.isfinite(d.upper)
        & np.isclose(d.lower, d.upper, rtol=0, atol=1e-14)
    )
    rows = np.flatnonzero(~equal)
    entries = dual_entries("row", np.flatnonzero(equal), "equal", result.y)
    if form == "compact":
        entries += dual_entries("row", rows, "lower", result.z_l)
        entries += dual_entries("row", rows, "upper", result.z_u)
    else:
        upper = rows[np.isfinite(d.upper[rows])]
        lower = rows[np.isfinite(d.lower[rows])]
        entries += dual_entries("row", upper, "upper", result.z_u[: len(upper)])
        entries += dual_entries("row", lower, "lower", result.z_u[len(upper) :])
    columns = np.arange(d.n_variables)
    entries += dual_entries("variable", columns, "lower", result.z_bl)
    entries += dual_entries("variable", columns, "upper", result.z_bu)
    return CoreInfeasibilityEvidence(
        "piqp", "native_dual_estimate", "numerical_estimate", native, tuple(entries)
    )


def _objective_scale(model: QuadraticProgram, target: float | None) -> float:
    """根据显式 alpha 离散度计算有界正缩放；未配置目标时返回一。"""

    if target is None:
        return 1.0
    magnitude = model.objective_scale_reference
    if magnitude is None:
        magnitude = float(np.max(np.abs(model.q))) if model.q.size else 0.0
    if not np.isfinite(magnitude) or magnitude <= np.finfo(float).eps:
        return 1.0
    return float(np.clip(target / magnitude, 1e-8, 1e8))


def _number(info: object, *names: str) -> float | None:
    for name in names:
        value = getattr(info, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _map_status(native: object, piqp: Any) -> tuple[SolveStatus, FailureReason | None]:
    if native == piqp.Status.PIQP_SOLVED:
        return SolveStatus.OPTIMAL, None
    if native == piqp.Status.PIQP_MAX_ITER_REACHED:
        return SolveStatus.LIMIT_REACHED, FailureReason.MAX_ITER
    if native == piqp.Status.PIQP_PRIMAL_INFEASIBLE:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if native == piqp.Status.PIQP_DUAL_INFEASIBLE:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    if native == piqp.Status.PIQP_NUMERICS:
        return SolveStatus.NUMERICAL_ERROR, FailureReason.NUMERICAL_FAILURE
    return SolveStatus.SOLVER_ERROR, FailureReason.UNKNOWN

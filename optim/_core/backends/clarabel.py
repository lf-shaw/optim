"""使用目标缩放和 QDLDL 的 Clarabel QP/SOCP 安全后端。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..canonical import FactorQCQP, LinearDomain, QuadraticProgram
from ..contracts import (
    CoreFailureReason as FailureReason,
    CoreInfeasibilityEvidence,
    CoreSolveStatus as SolveStatus,
)
from .base import BackendOptions, BackendResult, capture_infeasibility, dual_entries


@dataclass(frozen=True)
class _ConicData:
    """传给 Clarabel 的内部锥规划数值块。

    Attributes
    ----------
    P : scipy.sparse.csc_matrix
        Clarabel 最小化目标中的上三角二次矩阵。
    q : numpy.ndarray
        已缩放的线性目标向量。
    A : scipy.sparse.csc_matrix
        锥约束矩阵。
    b : numpy.ndarray
        锥约束右端项。
    cones : tuple[object, ...]
        与 ``A`` 行块顺序一致的 Clarabel 锥对象。
    output_variables : int
        返回给公共层的原始 canonical 变量数量；不包含仅供锥提升使用的内部列。
    objective_scale : float
        为改善数值条件应用于原始目标的正比例缩放。
    domain : LinearDomain
        实际锥转换使用的线性域引用，仅在失败时生成逆映射，普通求解不分配逐行证据对象。
    """

    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csc_matrix
    b: np.ndarray
    cones: tuple[object, ...]
    output_variables: int
    objective_scale: float
    domain: LinearDomain


class ClarabelBackend:
    """将 canonical QP 或 factor-QCQP 转换为 Clarabel 锥形式的安全后端。"""

    name = "clarabel_qdldl"

    def solve(
        self,
        model: QuadraticProgram | FactorQCQP,
        options: BackendOptions,
    ) -> BackendResult:
        """使用全新 Clarabel QDLDL workspace 求解一个 canonical 模型。

        Parameters
        ----------
        model : QuadraticProgram | FactorQCQP
            凸 QP 或含一个 factor-model TE 预算的问题。
        options : BackendOptions
            容差、迭代数、时间限制和目标缩放设置。

        Returns
        -------
        BackendResult
            标准化原生状态、完整 canonical 候选和耗时；仍须公共独立验收。

        Raises
        ------
        ImportError
            当前环境未安装 Clarabel。
        """

        import clarabel

        build_started = time.perf_counter()
        try:
            data = _build_conic_data(model, options, clarabel)
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status="compile_error",
                reason=FailureReason.INVALID_NUMERICS,
                message=f"{type(exc).__name__}: {exc}",
            )
        build_s = time.perf_counter() - build_started
        settings = clarabel.DefaultSettings()
        settings.verbose = bool(options.verbose)
        settings.max_iter = int(options.max_iter)
        settings.tol_gap_abs = float(options.eps_abs)
        settings.tol_gap_rel = float(options.eps_rel)
        settings.tol_feas = float(options.eps_abs)
        settings.direct_solve_method = "qdldl"
        if options.time_limit_s is not None:
            settings.time_limit = float(options.time_limit_s)

        setup_started = time.perf_counter()
        try:
            solver = clarabel.DefaultSolver(
                data.P,
                data.q,
                data.A,
                data.b,
                list(data.cones),
                settings,
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
                setup_s=build_s + time.perf_counter() - setup_started,
            )
        setup_s = build_s + time.perf_counter() - setup_started
        solve_started = time.perf_counter()
        try:
            solution = solver.solve()
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
            )
        solve_s = time.perf_counter() - solve_started
        native_status = str(solution.status)
        status, reason = _map_status(native_status)
        primal = None
        objective = None
        if status.has_solution and solution.x is not None:
            full = np.asarray(solution.x, dtype=float).reshape(-1)
            primal = full[: data.output_variables].copy()
            if isinstance(model, QuadraticProgram):
                objective = float(
                    0.5 * primal @ (model.P @ primal)
                    + model.q @ primal
                    + model.objective_offset
                )
            else:
                objective = float(model.alpha @ primal[model.domain.weight_indices])
        primal_objective = _number(solution, "obj_val")
        dual_objective = _number(solution, "obj_val_dual")
        native_gap = None
        if primal_objective is not None and dual_objective is not None:
            native_gap = abs(primal_objective - dual_objective) / data.objective_scale
        evidence_errors: dict[str, Any] = {}
        evidence = None
        if status is SolveStatus.INFEASIBLE:
            evidence = capture_infeasibility(
                lambda: _infeasibility(data, solution, self.name), evidence_errors
            )
        return BackendResult(
            backend=self.name,
            status=status,
            primal=primal,
            objective_value=objective,
            native_status=native_status,
            reason=reason,
            iterations=int(getattr(solution, "iterations", 0) or 0),
            setup_s=setup_s,
            solve_s=solve_s,
            infeasibility=evidence,
            diagnostics={
                **evidence_errors,
                "objective_scale": data.objective_scale,
                "primal_objective_scaled": primal_objective,
                "dual_objective_scaled": dual_objective,
                "native_gap_unscaled": native_gap,
                "r_prim": _number(solution, "r_prim"),
                "r_dual": _number(solution, "r_dual"),
                "linear_solver": "qdldl",
            },
        )


def _build_conic_data(
    model: QuadraticProgram | FactorQCQP,
    options: BackendOptions,
    clarabel: Any,
) -> _ConicData:
    if isinstance(model, QuadraticProgram):
        domain = model.domain
        objective_scale = _scale_target(model.q, options.objective_scale_target)
        P = sp.triu(model.P * objective_scale, format="csc")
        q = np.asarray(model.q * objective_scale, dtype=float)
        linear_A, linear_b, cones = _linear_cones(domain, clarabel)
        return _ConicData(
            P=P,
            q=q,
            A=linear_A,
            b=linear_b,
            cones=cones,
            output_variables=domain.n_variables,
            objective_scale=objective_scale,
            domain=domain,
        )

    # 复用为前沿 QP 编译的精确因子变量等式。
    from ..factor_qcqp import _extend_as_parametric_qp

    extended_qp, _, _, _ = _extend_as_parametric_qp(model)
    domain = extended_qp.domain
    n_variables = domain.n_variables
    centered_alpha = np.asarray(model.alpha, dtype=float) - float(np.mean(model.alpha))
    objective_scale = _scale_target(centered_alpha, options.objective_scale_target)
    q = np.zeros(n_variables, dtype=float)
    q[domain.weight_indices] = -objective_scale * centered_alpha
    P = sp.csc_matrix((n_variables, n_variables), dtype=float)
    linear_A, linear_b, linear_cones = _linear_cones(domain, clarabel)

    risk = model.risk_operator
    covariance = (risk.covariance + risk.covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor_root = np.sqrt(eigenvalues)[:, None] * eigenvectors.T
    n_factors = covariance.shape[0]
    factor_indices = np.arange(n_variables - n_factors, n_variables, dtype=np.int32)

    cone_rows = 1 + n_factors + len(risk.specific_volatility)
    row_parts: list[sp.csc_matrix] = [sp.csc_matrix((1, n_variables))]
    factor_block = sp.csc_matrix((n_factors, n_variables), dtype=float)
    factor_coo = sp.coo_matrix(-factor_root)
    factor_block = sp.csc_matrix(
        (
            factor_coo.data,
            (factor_coo.row, factor_indices[factor_coo.col]),
        ),
        shape=(n_factors, n_variables),
    )
    row_parts.append(factor_block)
    n_assets = len(risk.specific_volatility)
    specific_rows = np.arange(n_assets, dtype=np.int32)
    specific_block = sp.csc_matrix(
        (
            -np.asarray(risk.specific_volatility, dtype=float),
            (specific_rows, domain.weight_indices),
        ),
        shape=(n_assets, n_variables),
    )
    row_parts.append(specific_block)
    soc_A = sp.vstack(row_parts, format="csc")
    soc_b = np.concatenate(
        [
            np.array([model.risk_limit]),
            np.zeros(n_factors),
            -np.asarray(risk.specific_volatility) * risk.benchmark,
        ]
    )
    if soc_A.shape != (cone_rows, n_variables):
        raise AssertionError("invalid factor risk cone shape")
    A = sp.vstack([linear_A, soc_A], format="csc")
    b = np.concatenate([linear_b, soc_b])
    cones = linear_cones + (clarabel.SecondOrderConeT(cone_rows),)
    return _ConicData(
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        output_variables=model.domain.n_variables,
        objective_scale=objective_scale,
        domain=domain,
    )


def _infeasibility(
    data: _ConicData, solution: Any, backend: str
) -> CoreInfeasibilityEvidence:
    """按 _linear_cones 的块顺序恢复坐标，风险锥保留全部带符号分量。"""

    d = data.domain
    equal = (
        np.isfinite(d.lower)
        & np.isfinite(d.upper)
        & np.isclose(d.lower, d.upper, rtol=0, atol=1e-14)
    )
    z = np.asarray(solution.z, dtype=float)
    if z.shape != data.b.shape or not np.all(np.isfinite(z)):
        raise ValueError("invalid Clarabel certificate")
    entries = []
    offset = 0
    for location, side, mask in (
        ("row", "equal", equal),
        ("row", "upper", ~equal & np.isfinite(d.upper)),
        ("row", "lower", ~equal & np.isfinite(d.lower)),
        ("variable", "upper", np.isfinite(d.variable_upper)),
        ("variable", "lower", np.isfinite(d.variable_lower)),
    ):
        indices = np.flatnonzero(mask)
        entries += dual_entries(
            location, indices, side, z[offset : offset + len(indices)]
        )
        offset += len(indices)
    entries += dual_entries(
        "cone", np.arange(len(z) - offset), "cone", z[offset:], "tracking_error"
    )
    scale = max(1.0, float(np.max(np.abs(z), initial=0)))
    return CoreInfeasibilityEvidence(
        backend,
        "primal_infeasibility_certificate",
        "numerical_estimate",
        str(solution.status),
        tuple(entries),
        certificate_residual=float(np.max(np.abs(data.A.T @ z), initial=0)) / scale,
        certificate_margin=float(-data.b @ z) / scale,
        metadata={"coordinates": "conic_lift", "cone_component_indices": True},
    )


def _linear_cones(
    domain: LinearDomain,
    clarabel: Any,
) -> tuple[sp.csc_matrix, np.ndarray, tuple[object, ...]]:
    equality = (
        np.isfinite(domain.lower)
        & np.isfinite(domain.upper)
        & np.isclose(domain.lower, domain.upper, rtol=0.0, atol=1e-14)
    )
    blocks: list[sp.csc_matrix] = []
    rhs: list[np.ndarray] = []
    cones: list[object] = []
    if np.any(equality):
        blocks.append(domain.A[equality].tocsc())
        rhs.append(domain.lower[equality])
        cones.append(clarabel.ZeroConeT(int(equality.sum())))

    general = ~equality
    upper_rows = general & np.isfinite(domain.upper)
    lower_rows = general & np.isfinite(domain.lower)
    variable_upper = np.isfinite(domain.variable_upper)
    variable_lower = np.isfinite(domain.variable_lower)
    inequality_blocks: list[sp.csc_matrix] = []
    inequality_rhs: list[np.ndarray] = []
    if np.any(upper_rows):
        inequality_blocks.append(domain.A[upper_rows].tocsc())
        inequality_rhs.append(domain.upper[upper_rows])
    if np.any(lower_rows):
        inequality_blocks.append(-domain.A[lower_rows].tocsc())
        inequality_rhs.append(-domain.lower[lower_rows])
    if np.any(variable_upper):
        columns = np.flatnonzero(variable_upper)
        rows = np.arange(len(columns), dtype=np.int32)
        inequality_blocks.append(
            sp.csc_matrix(
                (np.ones(len(columns)), (rows, columns)),
                shape=(len(columns), domain.n_variables),
            )
        )
        inequality_rhs.append(domain.variable_upper[variable_upper])
    if np.any(variable_lower):
        columns = np.flatnonzero(variable_lower)
        rows = np.arange(len(columns), dtype=np.int32)
        inequality_blocks.append(
            sp.csc_matrix(
                (-np.ones(len(columns)), (rows, columns)),
                shape=(len(columns), domain.n_variables),
            )
        )
        inequality_rhs.append(-domain.variable_lower[variable_lower])
    if inequality_blocks:
        block = sp.vstack(inequality_blocks, format="csc")
        blocks.append(block)
        rhs.append(np.concatenate(inequality_rhs))
        cones.append(clarabel.NonnegativeConeT(block.shape[0]))
    if not blocks:
        return (
            sp.csc_matrix((0, domain.n_variables), dtype=float),
            np.empty(0, dtype=float),
            (),
        )
    A = sp.vstack(blocks, format="csc")
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    return A, np.concatenate(rhs), tuple(cones)


def _scale_target(values: np.ndarray, target: float | None) -> float:
    if target is None:
        return 1.0
    magnitude = float(np.max(np.abs(values))) if values.size else 0.0
    if magnitude <= np.finfo(float).eps or not np.isfinite(magnitude):
        return 1.0
    return float(np.clip(target / magnitude, 1e-8, 1e8))


def _number(value: object, *names: str) -> float | None:
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            try:
                return float(item)
            except (TypeError, ValueError):
                pass
    return None


def _map_status(native: str) -> tuple[SolveStatus, FailureReason | None]:
    normalized = native.lower()
    if normalized == "solved":
        return SolveStatus.OPTIMAL, None
    if normalized == "almostsolved":
        return SolveStatus.OPTIMAL_INACCURATE, None
    if "primalinfeasible" in normalized:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if "dualinfeasible" in normalized:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    if "maxiterations" in normalized or "maxtime" in normalized:
        reason = (
            FailureReason.TIME_LIMIT
            if "maxtime" in normalized
            else FailureReason.MAX_ITER
        )
        return SolveStatus.LIMIT_REACHED, reason
    if "numerical" in normalized or "insufficientprogress" in normalized:
        return SolveStatus.NUMERICAL_ERROR, FailureReason.NUMERICAL_FAILURE
    return SolveStatus.SOLVER_ERROR, FailureReason.UNKNOWN

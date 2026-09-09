"""QP 和 factor-model SOCP 问题的延迟 MOSEK 安全后端。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..canonical import FactorQCQP, QuadraticProgram
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
)
from ..dual_bounds import lp_dual_bound_diagnostics


class MosekBackend:
    """仅在路由实际到达时导入并签出 license 的 MOSEK 后端。"""

    name = "mosek"

    def solve(
        self,
        model: QuadraticProgram | FactorQCQP,
        options: BackendOptions,
    ) -> BackendResult:
        """求解一个 canonical QP 或 factor-QCQP。

        Parameters
        ----------
        model : QuadraticProgram | FactorQCQP
            待求解的 canonical 凸模型。
        options : BackendOptions
            输出、容差和时间限制设置。

        Returns
        -------
        BackendResult
            标准化原生证据。缺少安装或 license 时通过失败结果返回，而不是在导入 ``optim``
            时产生副作用。

        Raises
        ------
        TypeError
            canonical 模型类型不受支持。
        """

        # 只有路由实际到达该 fallback 时才导入 MOSEK 并签出 license；仅导入 optim 不会访问
        # MOSEK。
        if isinstance(model, QuadraticProgram):
            return self._solve_qp(model, options)
        if isinstance(model, FactorQCQP):
            return self._solve_factor_qcqp(model, options)
        raise TypeError("MosekBackend accepts QuadraticProgram or FactorQCQP")

    def _solve_qp(
        self,
        model: QuadraticProgram,
        options: BackendOptions,
    ) -> BackendResult:
        try:
            import mosek
        except Exception as exc:
            return _unavailable(exc)
        setup_started = time.perf_counter()
        task = None
        env = None
        try:
            env = mosek.Env()
            license_path = _default_license_path()
            if license_path is not None:
                env.putlicensepath(str(license_path))
            task = env.Task(0, 0)
            if not options.verbose:
                task.putintparam(mosek.iparam.log, 0)
            if options.time_limit_s is not None:
                task.putdouparam(
                    mosek.dparam.optimizer_max_time,
                    float(options.time_limit_s),
                )
            domain = model.domain
            primal_feasibility = _primal_feasibility_tolerance(domain, options)
            task.putdouparam(
                mosek.dparam.intpnt_qo_tol_pfeas
                if model.P.nnz
                else mosek.dparam.intpnt_tol_pfeas,
                primal_feasibility,
            )
            task.appendvars(domain.n_variables)
            task.appendcons(domain.n_constraints)
            task.putclist(
                np.arange(domain.n_variables, dtype=np.int32),
                np.asarray(model.q, dtype=float),
            )
            var_keys, var_lower, var_upper = _mosek_bounds(
                domain.variable_lower, domain.variable_upper, mosek
            )
            task.putvarboundlist(
                np.arange(domain.n_variables, dtype=np.int32),
                var_keys,
                var_lower,
                var_upper,
            )
            row_keys, row_lower, row_upper = _mosek_bounds(
                domain.lower, domain.upper, mosek
            )
            task.putconboundlist(
                np.arange(domain.n_constraints, dtype=np.int32),
                row_keys,
                row_lower,
                row_upper,
            )
            coo = domain.A.tocoo()
            task.putaijlist(
                np.asarray(coo.row, dtype=np.int32),
                np.asarray(coo.col, dtype=np.int32),
                np.asarray(coo.data, dtype=float),
            )
            lower = sp.tril(model.P, format="coo")
            task.putqobj(
                np.asarray(lower.row, dtype=np.int32),
                np.asarray(lower.col, dtype=np.int32),
                np.asarray(lower.data, dtype=float),
            )
            task.putobjsense(mosek.objsense.minimize)
            setup_s = time.perf_counter() - setup_started
            solve_started = time.perf_counter()
            task.optimize()
            solve_s = time.perf_counter() - solve_started
            solution_status = task.getsolsta(mosek.soltype.itr)
            problem_status = task.getprosta(mosek.soltype.itr)
            status, reason = _map_task_status(solution_status, problem_status, mosek)
            primal = None
            objective = None
            if status.has_solution:
                primal = np.asarray(task.getxx(mosek.soltype.itr), dtype=float)
                objective = float(
                    0.5 * primal @ (model.P @ primal)
                    + model.q @ primal
                    + model.objective_offset
                )
            primal_obj = _task_double(task, mosek.dinfitem.intpnt_primal_obj)
            dual_obj = _task_double(task, mosek.dinfitem.intpnt_dual_obj)
            evidence_errors: dict[str, Any] = {}
            if options.collect_dual_bound and status.has_solution:
                try:
                    evidence_errors.update(
                        lp_dual_bound_diagnostics(
                            model, task.gety(mosek.soltype.itr), primal
                        )
                    )
                except Exception as exc:
                    evidence_errors.update(
                        dual_lower_bound=None,
                        dual_bound_status="unavailable",
                        dual_bound_reason=f"native_dual_read_error:{type(exc).__name__}",
                    )
            evidence = None
            if status is SolveStatus.INFEASIBLE:
                evidence = capture_infeasibility(
                    lambda: _task_evidence(task, domain, mosek, str(solution_status)),
                    evidence_errors,
                )
            return BackendResult(
                backend=self.name,
                status=status,
                primal=primal,
                objective_value=objective,
                native_status=f"{solution_status}/{problem_status}",
                reason=reason,
                iterations=int(task.getintinf(mosek.iinfitem.intpnt_iter)),
                setup_s=setup_s,
                solve_s=solve_s,
                infeasibility=evidence,
                diagnostics={
                    **evidence_errors,
                    "primal_feasibility_tolerance": primal_feasibility,
                    "primal_objective": primal_obj,
                    "dual_objective": dual_obj,
                    "native_gap_unscaled": (
                        abs(primal_obj - dual_obj)
                        if primal_obj is not None and dual_obj is not None
                        else None
                    ),
                },
            )
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status="exception",
                reason=_exception_reason(exc),
                message=f"{type(exc).__name__}: {exc}",
                setup_s=time.perf_counter() - setup_started,
            )
        finally:
            if task is not None:
                task.__exit__(None, None, None)
            if env is not None:
                env.__exit__(None, None, None)

    def _solve_factor_qcqp(
        self,
        model: FactorQCQP,
        options: BackendOptions,
    ) -> BackendResult:
        try:
            from mosek import fusion
        except Exception as exc:
            return _unavailable(exc)
        from ..factor_conic import extend_factor_domain

        setup_started = time.perf_counter()
        fusion_model = None
        try:
            domain = extend_factor_domain(model)
            license_path = _default_license_path()
            if license_path is not None:
                fusion.Model.putlicensepath(str(license_path))
            fusion_model = fusion.Model("portfolio_factor_qcqp")
            primal_feasibility = _primal_feasibility_tolerance(domain, options)
            fusion_model.setSolverParam("intpntCoTolPfeas", primal_feasibility)
            if options.time_limit_s is not None:
                fusion_model.setSolverParam(
                    "optimizerMaxTime",
                    float(options.time_limit_s),
                )
            z = fusion_model.variable(
                "z",
                domain.n_variables,
                fusion.Domain.unbounded(),
            )
            _fusion_add_linear_domain(fusion_model, z, domain, fusion)
            risk = model.risk_operator
            covariance = (risk.covariance + risk.covariance.T) * 0.5
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            factor_root = (
                np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
            )
            n_factors = covariance.shape[0]
            factor = z.slice(domain.n_variables - n_factors, domain.n_variables)
            weight = z.slice(0, len(risk.benchmark))
            cone_expression = fusion.Expr.vstack(
                model.risk_limit,
                fusion.Expr.mul(factor_root, factor),
                fusion.Expr.mulElm(
                    np.asarray(risk.specific_volatility, dtype=float),
                    fusion.Expr.sub(weight, np.asarray(risk.benchmark, dtype=float)),
                ),
            )
            fusion_model.constraint(
                "tracking_error",
                cone_expression,
                fusion.Domain.inQCone(),
            )
            fusion_model.objective(
                fusion.ObjectiveSense.Maximize,
                fusion.Expr.dot(np.asarray(model.alpha, dtype=float), weight),
            )
            setup_s = time.perf_counter() - setup_started
            solve_started = time.perf_counter()
            fusion_model.solve()
            solve_s = time.perf_counter() - solve_started
            primal_status = fusion_model.getPrimalSolutionStatus()
            problem_status = fusion_model.getProblemStatus()
            status, reason = _map_fusion_status(primal_status, problem_status)
            primal = None
            objective = None
            if status.has_solution:
                full = np.asarray(z.level(), dtype=float)
                primal = full[: model.domain.n_variables].copy()
                objective = float(model.alpha @ primal[model.domain.weight_indices])
            primal_obj = _fusion_double(fusion_model, "intpntPrimalObj")
            dual_obj = _fusion_double(fusion_model, "intpntDualObj")
            evidence_errors: dict[str, Any] = {}
            evidence = None
            if status is SolveStatus.INFEASIBLE:
                evidence = capture_infeasibility(
                    lambda: _fusion_evidence(
                        fusion_model, domain, fusion, str(problem_status)
                    ),
                    evidence_errors,
                )
            return BackendResult(
                backend=self.name,
                status=status,
                primal=primal,
                objective_value=objective,
                native_status=f"{primal_status}/{problem_status}",
                reason=reason,
                iterations=int(_fusion_int(fusion_model, "intpntIter") or 0),
                setup_s=setup_s,
                solve_s=solve_s,
                infeasibility=evidence,
                diagnostics={
                    **evidence_errors,
                    "primal_feasibility_tolerance": primal_feasibility,
                    "primal_objective": primal_obj,
                    "dual_objective": dual_obj,
                    "native_gap_unscaled": (
                        abs(primal_obj - dual_obj)
                        if primal_obj is not None and dual_obj is not None
                        else None
                    ),
                },
            )
        except Exception as exc:
            return BackendResult(
                backend=self.name,
                status=SolveStatus.SOLVER_ERROR,
                primal=None,
                objective_value=None,
                native_status="exception",
                reason=_exception_reason(exc),
                message=f"{type(exc).__name__}: {exc}",
                setup_s=time.perf_counter() - setup_started,
            )
        finally:
            if fusion_model is not None:
                fusion_model.dispose()


def _primal_feasibility_tolerance(domain: Any, options: BackendOptions) -> float:
    """给两个原生接口提供一致的可行性精度请求。"""
    # L1 辅助行的小残差会累加到原始业务总量。大样本回归中默认 1e-8 仍有总量
    # 超限，1e-10 可通过既定 1e-5 验收。这里只收紧原生 primal 阈值，不放宽
    # 业务验收，也不将此阈值当作所有问题总量误差的数学保证。
    has_l1_bound = any(
        record.group in {"turnover", "total_active"} for record in domain.constraints
    )
    return min(options.eps_abs, 1e-10) if has_l1_bound else options.eps_abs


def _task_evidence(
    task: Any, domain: Any, mosek: Any, native: str
) -> CoreInfeasibilityEvidence:
    """读取 Task 已有证书的四类边界乘子，不开启或解析原生文本报告。"""

    entries = []
    for location, side, values, count in (
        ("row", "lower", task.getslc(mosek.soltype.itr), domain.n_constraints),
        ("row", "upper", task.getsuc(mosek.soltype.itr), domain.n_constraints),
        ("variable", "lower", task.getslx(mosek.soltype.itr), domain.n_variables),
        ("variable", "upper", task.getsux(mosek.soltype.itr), domain.n_variables),
    ):
        entries += dual_entries(location, np.arange(count), side, values)
    return CoreInfeasibilityEvidence(
        "mosek",
        "primal_infeasibility_certificate",
        "numerical_estimate",
        native,
        tuple(entries),
        metadata={"coordinates": "task_bounds"},
    )


def _fusion_evidence(
    model: Any, domain: Any, fusion: Any, native: str
) -> CoreInfeasibilityEvidence:
    """通过建模时的具名约束读取 Fusion 证书，映射回实际扩展线性域。"""

    model.acceptedSolutionStatus(fusion.AccSolutionStatus.Certificate)
    d = domain
    equal = (
        np.isfinite(d.lower)
        & np.isfinite(d.upper)
        & np.isclose(d.lower, d.upper, rtol=0, atol=1e-14)
    )
    entries = []
    for name, location, side, mask in (
        ("linear_equal", "row", "equal", equal),
        ("linear_upper", "row", "upper", ~equal & np.isfinite(d.upper)),
        ("linear_lower", "row", "lower", ~equal & np.isfinite(d.lower)),
        ("variable_upper", "variable", "upper", np.isfinite(d.variable_upper)),
        ("variable_lower", "variable", "lower", np.isfinite(d.variable_lower)),
    ):
        if np.any(mask):
            entries += dual_entries(
                location, np.flatnonzero(mask), side, model.getConstraint(name).dual()
            )
    cone = model.getConstraint("tracking_error").dual()
    entries += dual_entries(
        "cone", np.arange(len(cone)), "cone", cone, "tracking_error"
    )
    return CoreInfeasibilityEvidence(
        "mosek",
        "primal_infeasibility_certificate",
        "numerical_estimate",
        native,
        tuple(entries),
        metadata={"coordinates": "fusion_lift", "cone_component_indices": True},
    )


def _mosek_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    mosek: Any,
) -> tuple[list[object], np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    keys: list[object] = []
    safe_lower = np.where(np.isfinite(lower), lower, 0.0)
    safe_upper = np.where(np.isfinite(upper), upper, 0.0)
    for lo, hi in zip(lower, upper):
        if np.isfinite(lo) and np.isfinite(hi):
            key = mosek.boundkey.fx if abs(lo - hi) <= 1e-14 else mosek.boundkey.ra
        elif np.isfinite(lo):
            key = mosek.boundkey.lo
        elif np.isfinite(hi):
            key = mosek.boundkey.up
        else:
            key = mosek.boundkey.fr
        keys.append(key)
    return keys, safe_lower, safe_upper


def _fusion_matrix(matrix: sp.spmatrix, fusion: Any) -> object:
    coo = matrix.tocoo()
    return fusion.Matrix.sparse(
        matrix.shape[0],
        matrix.shape[1],
        np.asarray(coo.row, dtype=np.int32),
        np.asarray(coo.col, dtype=np.int32),
        np.asarray(coo.data, dtype=float),
    )


def _fusion_add_linear_domain(
    model: Any,
    variable: Any,
    domain: Any,
    fusion: Any,
) -> None:
    equality = (
        np.isfinite(domain.lower)
        & np.isfinite(domain.upper)
        & np.isclose(domain.lower, domain.upper, rtol=0.0, atol=1e-14)
    )
    upper = ~equality & np.isfinite(domain.upper)
    lower = ~equality & np.isfinite(domain.lower)
    for name, mask, bound, direction in (
        ("linear_equal", equality, domain.lower, "equal"),
        ("linear_upper", upper, domain.upper, "upper"),
        ("linear_lower", lower, domain.lower, "lower"),
    ):
        if not np.any(mask):
            continue
        expression = fusion.Expr.mul(_fusion_matrix(domain.A[mask], fusion), variable)
        values = np.asarray(bound[mask], dtype=float)
        if direction == "equal":
            target = fusion.Domain.equalsTo(values)
        elif direction == "upper":
            target = fusion.Domain.lessThan(values)
        else:
            target = fusion.Domain.greaterThan(values)
        model.constraint(name, expression, target)

    finite_upper = np.isfinite(domain.variable_upper)
    finite_lower = np.isfinite(domain.variable_lower)
    if np.any(finite_upper):
        indices = np.flatnonzero(finite_upper).astype(np.int32)
        model.constraint(
            "variable_upper",
            variable.pick(indices),
            fusion.Domain.lessThan(
                np.asarray(domain.variable_upper[finite_upper], dtype=float)
            ),
        )
    if np.any(finite_lower):
        indices = np.flatnonzero(finite_lower).astype(np.int32)
        model.constraint(
            "variable_lower",
            variable.pick(indices),
            fusion.Domain.greaterThan(
                np.asarray(domain.variable_lower[finite_lower], dtype=float)
            ),
        )


def _default_license_path() -> Path | None:
    candidate = Path.home() / ".mosek" / "mosek.lic"
    return candidate if candidate.is_file() else None


def _map_task_status(
    solution_status: object,
    problem_status: object,
    mosek: Any,
) -> tuple[SolveStatus, FailureReason | None]:
    if solution_status in {
        mosek.solsta.optimal,
        mosek.solsta.prim_and_dual_feas,
        mosek.solsta.prim_feas,
    }:
        return SolveStatus.OPTIMAL, None
    if solution_status == mosek.solsta.prim_infeas_cer:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if solution_status == mosek.solsta.dual_infeas_cer:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    return SolveStatus.SOLVER_ERROR, FailureReason.NUMERICAL_FAILURE


def _map_fusion_status(
    primal_status: object,
    problem_status: object,
) -> tuple[SolveStatus, FailureReason | None]:
    primal = str(primal_status).lower()
    problem = str(problem_status).lower()
    if "optimal" in primal or "feasible" in primal:
        return SolveStatus.OPTIMAL, None
    if "primalinfeasible" in problem:
        return SolveStatus.INFEASIBLE, FailureReason.INFEASIBLE_REPORTED
    if "dualinfeasible" in problem:
        return SolveStatus.UNBOUNDED, FailureReason.UNBOUNDED_REPORTED
    return SolveStatus.SOLVER_ERROR, FailureReason.NUMERICAL_FAILURE


def _task_double(task: Any, item: object) -> float | None:
    try:
        return float(task.getdouinf(item))
    except Exception:
        return None


def _fusion_double(model: Any, name: str) -> float | None:
    try:
        return float(model.getSolverDoubleInfo(name))
    except Exception:
        return None


def _fusion_int(model: Any, name: str) -> int | None:
    try:
        return int(model.getSolverIntInfo(name))
    except Exception:
        return None


def _exception_reason(exc: Exception) -> FailureReason:
    # Fusion 可能包装原生 Error；沿异常链读取错误码，不解析英文日志。
    import mosek

    license_codes = {
        getattr(mosek.rescode, name)
        for name in dir(mosek.rescode)
        if name.startswith("err_") and ("license" in name or "flexlm" in name)
    }
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "errno", None) in license_codes:
            return FailureReason.BACKEND_UNAVAILABLE
        current = current.__cause__ or current.__context__
    return FailureReason.NUMERICAL_FAILURE


def _unavailable(exc: Exception) -> BackendResult:
    return BackendResult(
        backend="mosek",
        status=SolveStatus.SOLVER_ERROR,
        primal=None,
        objective_value=None,
        native_status="unavailable",
        reason=FailureReason.BACKEND_UNAVAILABLE,
        message=f"{type(exc).__name__}: {exc}",
    )

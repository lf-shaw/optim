"""公共组合契约与数值核心之间的内部适配层。"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .portfolio_types import (
    AlignmentReport,
    ConstraintViolation,
    FailureReason,
    InfeasibilityContributor,
    MaximizeAlpha,
    NativeInfeasibilityEvidence,
    OptimalityCertificate,
    OptimizationResult,
    PortfolioMetrics,
    PortfolioProblem,
    ProblemFingerprint,
    ProofStatus,
    RiskAdjustedAlpha,
    SolveStatus,
    SolveTimings,
    SolverAttempt,
    SolverPolicy,
)
from ._core import (
    CoreBackendResult as BackendResult,
    CoreFailureReason,
    CoreProblemHandle,
    CoreSolveStatus,
    CoreSolver,
    CoreSolverOptions,
)
from .diagnostics import InfeasibilityReport
from .model.canonical import CompiledProblem, FactorQCQP, LinearProgram
from .solution import evaluate_solution, lift_weights


@dataclass(frozen=True)
class SolverHandle:
    """保留上层审计对象和核心短期句柄的内部准备状态。

    Attributes
    ----------
    compiled : CompiledProblem
        完整上层编译结果，用于结果验收、fingerprint 和诊断。
    core_handle : CoreProblemHandle
        不复制数值矩阵的核心问题句柄。
    """

    compiled: CompiledProblem
    core_handle: CoreProblemHandle


@dataclass(frozen=True)
class _Evaluation:
    """一次后端候选解的独立验收结果。

    Attributes
    ----------
    metrics : PortfolioMetrics
        从最终候选重新计算的业务指标。
    violations : tuple[ConstraintViolation, ...]
        超过公共验收容差的具名约束违约。
    max_violation : float
        全部约束中的最大绝对违约。
    accepted : bool
        候选是否通过公共结果验收。
    validation_s : float
        指标复算和可选小权重清理耗时。
    """

    metrics: PortfolioMetrics
    violations: tuple[ConstraintViolation, ...]
    max_violation: float
    accepted: bool
    validation_s: float


class SolverAdapter:
    """把公共策略和结果契约转换到稳定的数值核心边界。"""

    def __init__(self, policy: SolverPolicy):
        self.policy = policy
        self._core = CoreSolver(core_options_from_policy(policy))

    def prepare(self, compiled: CompiledProblem) -> SolverHandle:
        """把上层已经编译的数值问题封装为不透明句柄。"""

        return SolverHandle(compiled, self._core.prepare(compiled.model))

    def metadata(
        self,
        handle: object,
    ) -> tuple[ProblemFingerprint, tuple[str, ...]]:
        """返回允许公共准备对象暴露的稳定审计元信息。"""

        resolved = self._require_handle(handle)
        compiled = resolved.compiled
        return compiled.fingerprint, compiled.compiler_optimizations

    def solve(
        self,
        problem: PortfolioProblem,
        handle: object,
        *,
        prepare_s: float,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """路由、求解并验收同一个不透明 canonical 问题。"""

        total_started = time.perf_counter()
        resolved = self._require_handle(handle)
        compiled = resolved.compiled
        alpha_spec = problem.data.alpha_spec
        objective_tolerance = (
            0.0
            if alpha_spec is None
            else self.policy.objective_tolerance.raw_limit(alpha_spec)
        )
        core_result = self._core.solve(
            resolved.core_handle,
            theta_seed=theta_seed,
            objective_tolerance=objective_tolerance,
        )
        attempts = list(core_result.attempts)
        final, evaluation = self._audit_backend_result(
            problem, compiled, core_result.final
        )
        attempts[-1] = final
        return self._result(
            problem,
            compiled,
            attempts,
            evaluation,
            total_started,
            prepare_s,
            theta_seed=theta_seed,
        )

    def diagnose(
        self,
        problem: PortfolioProblem,
        handle: object,
        *,
        prior_result: OptimizationResult | None,
        level: str,
    ) -> InfeasibilityReport:
        """显式运行私有深度不可行诊断。"""

        from ._diagnostic_engine import diagnose_problem

        return diagnose_problem(
            problem,
            self._require_handle(handle).compiled,
            self.policy,
            prior_result=prior_result,
            level=level,
        )

    @staticmethod
    def _require_handle(handle: object) -> SolverHandle:
        if not isinstance(handle, SolverHandle):
            raise TypeError(
                "prepared problem does not contain a valid optim core handle"
            )
        return handle

    def _audit_backend_result(
        self,
        problem: PortfolioProblem,
        compiled: CompiledProblem,
        backend_result: BackendResult,
    ) -> tuple[BackendResult, _Evaluation]:
        """独立复算候选向量，并在通过时尝试安全清理小权重。"""

        validation_started = time.perf_counter()
        metrics = PortfolioMetrics()
        violations: tuple[ConstraintViolation, ...] = ()
        max_violation = 0.0
        accepted = (
            backend_result.status.has_solution and backend_result.primal is not None
        )
        if accepted:
            assert backend_result.primal is not None
            metrics, violations, max_violation = evaluate_solution(
                problem, compiled, backend_result.primal
            )
            accepted = max_violation <= self.policy.tuning.feasibility_tolerance
            if not accepted:
                backend_result = replace(
                    backend_result,
                    status=CoreSolveStatus.NUMERICAL_ERROR,
                    reason=CoreFailureReason.INVALID_NUMERICS,
                    message=(
                        "backend solution failed independent validation: max violation "
                        f"{max_violation:.3e}"
                    ),
                )
            else:
                backend_result, metrics, violations, max_violation = (
                    self._clean_solution(
                        problem,
                        compiled,
                        backend_result,
                        metrics,
                        violations,
                        max_violation,
                    )
                )
        return backend_result, _Evaluation(
            metrics=metrics,
            violations=violations,
            max_violation=max_violation,
            accepted=accepted,
            validation_s=time.perf_counter() - validation_started,
        )

    def _clean_solution(
        self,
        problem: PortfolioProblem,
        compiled: CompiledProblem,
        backend_result: BackendResult,
        raw_metrics: PortfolioMetrics,
        raw_violations: tuple[ConstraintViolation, ...],
        raw_max_violation: float,
    ) -> tuple[
        BackendResult,
        PortfolioMetrics,
        tuple[ConstraintViolation, ...],
        float,
    ]:
        """只在可行性和目标证书都保持成立时清理不足阈值的小权重。"""

        assert backend_result.primal is not None
        domain = compiled.model.domain
        raw_weight = backend_result.primal[domain.weight_indices]
        tolerance = self.policy.tuning.weight_zero_tolerance
        small = np.abs(raw_weight) < tolerance
        removed_l1 = float(np.abs(raw_weight[small]).sum())
        if not np.any(small) or removed_l1 == 0.0:
            return backend_result, raw_metrics, raw_violations, raw_max_violation
        cleaned = raw_weight.copy()
        cleaned[small] = 0.0
        cleaned_sum = float(cleaned.sum())
        if abs(cleaned_sum) <= 1e-15:
            return backend_result, raw_metrics, raw_violations, raw_max_violation
        cleaned *= problem.constraints.budget / cleaned_sum
        cleaned_vector = lift_weights(problem, compiled, cleaned)
        metrics, violations, max_violation = evaluate_solution(
            problem, compiled, cleaned_vector
        )
        diagnostics = dict(backend_result.diagnostics)
        diagnostics.update(
            {
                "weight_cleanup_threshold": tolerance,
                "weight_cleanup_removed_l1": removed_l1,
                "weight_cleanup_count": int(small.sum()),
                "weight_cleanup_applied": (
                    max_violation <= self.policy.tuning.feasibility_tolerance
                ),
            }
        )
        if max_violation > self.policy.tuning.feasibility_tolerance:
            return (
                replace(backend_result, diagnostics=diagnostics),
                raw_metrics,
                raw_violations,
                raw_max_violation,
            )
        cleanup_loss = 0.0
        if raw_metrics.objective is not None and metrics.objective is not None:
            if isinstance(problem.objective, (MaximizeAlpha, RiskAdjustedAlpha)):
                cleanup_loss = max(0.0, raw_metrics.objective - metrics.objective)
            else:
                cleanup_loss = max(0.0, metrics.objective - raw_metrics.objective)
        diagnostics["weight_cleanup_objective_loss"] = cleanup_loss
        certificate_sensitive = bool(
            "total_gap" in diagnostics
            or diagnostics.get("lp_prescreen_certified") is True
            or isinstance(compiled.model, LinearProgram)
        )
        if certificate_sensitive:
            prior_gap = float(diagnostics.get("total_gap", 0.0))
            certified_gap = prior_gap + cleanup_loss
            diagnostics["certified_objective_gap"] = certified_gap
            if "total_gap" in diagnostics:
                diagnostics["total_gap"] = certified_gap
            diagnostics["cleanup_objective_loss"] = cleanup_loss
            alpha_spec = problem.data.alpha_spec
            if alpha_spec is not None:
                accepted_gap = self.policy.objective_tolerance.raw_limit(alpha_spec)
                if certified_gap > accepted_gap:
                    diagnostics["weight_cleanup_applied"] = False
                    diagnostics["certified_objective_gap"] = prior_gap
                    if "total_gap" in diagnostics:
                        diagnostics["total_gap"] = prior_gap
                    return (
                        replace(backend_result, diagnostics=diagnostics),
                        raw_metrics,
                        raw_violations,
                        raw_max_violation,
                    )
        return (
            replace(backend_result, primal=cleaned_vector, diagnostics=diagnostics),
            metrics,
            violations,
            max_violation,
        )

    def _result(
        self,
        problem: PortfolioProblem,
        compiled: CompiledProblem,
        backend_results: list[BackendResult],
        evaluation: _Evaluation,
        total_started: float,
        prepare_s: float,
        *,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """把私有后端证据规范化为稳定的公共结果契约。"""

        backend_result = backend_results[-1]
        metrics = evaluation.metrics
        violations = evaluation.violations
        max_violation = evaluation.max_violation
        accepted = evaluation.accepted
        status = _public_status(backend_result.status)
        message = backend_result.message
        if not accepted and not message:
            message = _default_failure_message(backend_result)

        weights = None
        objective_value = None
        certificate = None
        if accepted and backend_result.primal is not None:
            weight = backend_result.primal[compiled.model.domain.weight_indices]
            weights = pd.Series(
                weight.copy(),
                index=pd.Index(compiled.model.domain.assets, name="sid"),
                name="weight",
            )
            objective_value = metrics.objective
            assert objective_value is not None
            certificate = _certificate(
                problem,
                compiled,
                backend_result,
                float(objective_value),
                max_violation,
            )

        route = tuple(
            SolverAttempt(
                backend=item.backend,
                status=_public_status(item.status),
                reason=_public_reason(item.reason),
                native_status=item.native_status,
                message=item.message,
                solve_s=item.solve_s,
                recovered=bool(item.diagnostics.get("workspace_rebuilds", 0)),
                metadata={
                    key: value
                    for key, value in item.diagnostics.items()
                    if key != "trace"
                },
            )
            for item in backend_results
        )
        source_dates = {"portfolio": problem.data.date}
        if problem.data.provenance.source_date is not None:
            source_dates["portfolio_source"] = problem.data.provenance.source_date
        if problem.data.risk_model is not None:
            source_dates["risk_model"] = problem.data.risk_model.asof
        metadata = problem.data.provenance.metadata
        for field, label in (
            ("benchmark_source_date", "benchmark"),
            ("alpha_source_date", "alpha"),
        ):
            value = metadata.get(field)
            if value is not None:
                try:
                    parsed_date = pd.Timestamp(value)
                except (TypeError, ValueError):
                    pass
                else:
                    if isinstance(parsed_date, pd.Timestamp):
                        source_dates[label] = parsed_date
        timings = SolveTimings(
            prepare_s=prepare_s,
            backend_setup_s=sum(item.setup_s for item in backend_results),
            backend_solve_s=sum(item.solve_s for item in backend_results),
            validation_s=evaluation.validation_s,
            total_s=time.perf_counter() - total_started + prepare_s,
        )
        return OptimizationResult(
            status=status,
            weights=weights,
            objective_value=objective_value,
            backend=backend_result.backend if accepted else None,
            route=route,
            metrics=metrics,
            certificate=certificate,
            violations=violations,
            alignment=AlignmentReport(
                benchmark_missing_mass=_metadata_float(
                    metadata, "benchmark_missing_mass", 0.0
                ),
                benchmark_renormalization_factor=_metadata_float(
                    metadata, "benchmark_renormalization_factor", 1.0
                ),
                holding_missing_mass=_metadata_float(
                    metadata, "holding_missing_mass", 0.0
                ),
                source_dates=source_dates,
            ),
            diagnostics=None,
            timings=timings,
            fingerprint=compiled.fingerprint,
            message=message,
            native_infeasibility=tuple(
                evidence
                for item in backend_results
                if (evidence := _native_infeasibility(item, compiled)) is not None
            ),
            problem=problem,
            solver_policy=self.policy,
            theta_seed=theta_seed,
        )


def _native_infeasibility(
    result: BackendResult,
    compiled: CompiledProblem,
) -> NativeInfeasibilityEvidence | None:
    """将核心证书坐标映射回稳定业务约束；不解释任何后端日志或专用文本。"""

    evidence = result.infeasibility
    if evidence is None:
        return None
    domain = compiled.model.domain
    records = {(record.location, record.index): record for record in domain.constraints}
    contributors: list[InfeasibilityContributor] = []
    for item in evidence.contributors:
        record = records.get((item.location, item.index))
        if record is None and item.location != "cone":
            # 编译器的硬辅助行不一定进入公共 registry。保留稳定 canonical 坐标，避免
            # 因展示层缺少标签而丢失求解器证据。
            constraint_id = f"canonical:{item.location}:{item.index}"
            group = "canonical_internal"
            key = None
            configured_bound = _configured_bound(
                domain, item.location, item.index, item.side
            )
            sources = ("compiler",)
            metadata: dict[str, Any] = {}
        elif record is not None:
            constraint_id = record.constraint_id
            group = record.group
            key = record.key
            configured_bound = _configured_bound(
                domain, item.location, item.index, item.side
            )
            source_key = f"{item.side}_sources"
            raw_sources = record.metadata.get(source_key)
            if raw_sources is None:
                sources = (record.source,)
            elif isinstance(raw_sources, str):
                sources = (raw_sources,)
            else:
                sources = tuple(str(value) for value in raw_sources)
            metadata = dict(record.metadata)
        else:
            constraint_id = item.identifier or f"canonical:cone:{item.index}"
            group = item.identifier or "conic_constraint"
            key = None
            configured_bound = (
                float(compiled.model.risk_limit)
                if item.identifier == "tracking_error"
                and isinstance(compiled.model, FactorQCQP)
                else None
            )
            sources = (
                ("tracking_error",)
                if item.identifier == "tracking_error"
                else ("compiler",)
            )
            metadata = {}
        contributors.append(
            InfeasibilityContributor(
                constraint_id=constraint_id,
                group=group,
                location=item.location,
                side=item.side,
                multiplier=float(item.multiplier),
                key=key,
                configured_bound=configured_bound,
                sources=sources,
                metadata={**metadata, "canonical_index": item.index},
            )
        )
    return NativeInfeasibilityEvidence(
        backend=evidence.backend,
        kind=evidence.kind,
        proof_status=ProofStatus(evidence.proof_status),
        native_status=evidence.native_status,
        contributors=tuple(contributors),
        certificate_residual=evidence.certificate_residual,
        certificate_margin=evidence.certificate_margin,
        metadata=evidence.metadata,
        fingerprint=compiled.fingerprint,
        scope=(
            "linear_relaxation"
            if isinstance(compiled.model, FactorQCQP) and evidence.backend == "highs"
            else "original"
        ),
    )


def _configured_bound(
    domain: Any,
    location: str,
    index: int,
    side: str,
) -> float | None:
    """读取一个 canonical 证书坐标所对应的原始边界。"""

    if location == "row":
        values = domain.lower if side in {"lower", "equal"} else domain.upper
    elif location == "variable":
        values = (
            domain.variable_lower
            if side in {"lower", "equal"}
            else domain.variable_upper
        )
    else:
        return None
    if index < 0 or index >= len(values):
        return None
    value = float(values[index])
    return value if np.isfinite(value) else None


def _certificate(
    problem: PortfolioProblem,
    compiled: CompiledProblem,
    result: BackendResult,
    objective_value: float,
    max_violation: float,
) -> OptimalityCertificate:
    """根据实际路由和可用原生证据构造最优性证书。"""

    model = compiled.model
    diagnostics = result.diagnostics
    if (
        isinstance(model, FactorQCQP)
        and diagnostics.get("lp_prescreen_certified") is True
    ):
        alpha_spec = problem.data.alpha_spec
        assert alpha_spec is not None
        gap = float(diagnostics.get("certified_objective_gap", 0.0))
        return OptimalityCertificate(
            kind="lp_global_optimum_feasible_for_factor_qcqp",
            proof_status=ProofStatus.VERIFIED,
            primal_value=objective_value,
            dual_bound=objective_value + gap,
            absolute_gap=gap,
            normalized_gap=gap / alpha_spec.scale,
            objective_units=alpha_spec.units,
            objective_scale=alpha_spec.scale,
            components={
                "weight_cleanup_objective_loss": gap,
                "max_constraint_violation": max_violation,
            },
        )
    if isinstance(model, FactorQCQP) and "total_gap" in diagnostics:
        alpha_spec = problem.data.alpha_spec
        assert alpha_spec is not None
        gap = float(diagnostics["total_gap"])
        return OptimalityCertificate(
            kind="factor_qcqp_lagrangian",
            proof_status=ProofStatus.NUMERICAL_ESTIMATE,
            primal_value=objective_value,
            dual_bound=objective_value + gap,
            absolute_gap=gap,
            normalized_gap=gap / alpha_spec.scale,
            objective_units=alpha_spec.units,
            objective_scale=alpha_spec.scale,
            components={
                "frontier_slack_gap": float(diagnostics["frontier_gap"]),
                "qp_subproblem_gap": float(diagnostics["subproblem_gap"]),
                "max_constraint_violation": max_violation,
            },
        )
    if isinstance(model, FactorQCQP):
        alpha_spec = problem.data.alpha_spec
        assert alpha_spec is not None
        native_gap = diagnostics.get("native_gap_unscaled")
        native_gap = None if native_gap is None else float(native_gap)
        return OptimalityCertificate(
            kind="conic_primal_dual",
            proof_status=(
                ProofStatus.VERIFIED
                if native_gap is not None
                else ProofStatus.UNAVAILABLE
            ),
            primal_value=objective_value,
            dual_bound=(
                objective_value + native_gap if native_gap is not None else None
            ),
            absolute_gap=native_gap,
            normalized_gap=(
                native_gap / alpha_spec.scale if native_gap is not None else None
            ),
            objective_units=alpha_spec.units,
            objective_scale=alpha_spec.scale,
            components={"max_constraint_violation": max_violation},
        )

    verified = (
        isinstance(model, LinearProgram) and result.status is CoreSolveStatus.OPTIMAL
    )
    gap = float(diagnostics.get("certified_objective_gap", 0.0))
    alpha_spec = problem.data.alpha_spec
    return OptimalityCertificate(
        kind="backend_primal_dual" if verified else "backend_kkt",
        proof_status=ProofStatus.VERIFIED
        if verified
        else ProofStatus.NUMERICAL_ESTIMATE,
        primal_value=objective_value,
        dual_bound=objective_value + gap if verified else None,
        absolute_gap=gap if verified else None,
        normalized_gap=(
            gap / alpha_spec.scale if verified and alpha_spec is not None else None
        ),
        objective_units=(
            alpha_spec.units if alpha_spec is not None else "annualized_decimal"
        ),
        objective_scale=alpha_spec.scale if alpha_spec is not None else None,
        components={"max_constraint_violation": max_violation},
    )


def _metadata_float(metadata: Any, key: str, default: float) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def core_options_from_policy(policy: SolverPolicy) -> CoreSolverOptions:
    """把公共策略一次性压平为数值核心选项。"""

    tuning = policy.tuning
    return CoreSolverOptions(
        backend=policy.backend,
        licensed_fallback=policy.licensed_fallback,
        free_fallback=policy.free_fallback,
        lp_prescreen=policy.lp_prescreen,
        rebuild_after_update_failure=policy.rebuild_after_update_failure,
        alpha_target=tuning.alpha_target,
        theta_initial=tuning.theta_initial,
        theta_growth=tuning.theta_growth,
        theta_max=tuning.theta_max,
        max_outer_iters=tuning.max_outer_iters,
        intermediate_eps=tuning.intermediate_eps,
        final_eps=tuning.final_eps,
        piqp_max_iter=tuning.piqp_max_iter,
        piqp_inequality_form=tuning.piqp_inequality_form,
        feasibility_tolerance=tuning.feasibility_tolerance,
        risk_margin=tuning.risk_margin,
    )


def _public_status(status: CoreSolveStatus) -> SolveStatus:
    """把核心状态映射为稳定的公共状态枚举。"""

    return SolveStatus(status.value)


def _public_reason(reason: CoreFailureReason | None) -> FailureReason | None:
    """把可选核心失败原因映射为稳定的公共枚举。"""

    return None if reason is None else FailureReason(reason.value)


def _default_failure_message(result: BackendResult) -> str:
    status_text = {
        CoreSolveStatus.INFEASIBLE: "报告 canonical 问题不可行",
        CoreSolveStatus.UNBOUNDED: "报告 canonical 问题无界",
        CoreSolveStatus.LIMIT_REACHED: "达到求解限制",
        CoreSolveStatus.NUMERICAL_ERROR: "发生数值错误",
        CoreSolveStatus.RESOURCE_ERROR: "发生资源错误",
        CoreSolveStatus.SOLVER_ERROR: "求解失败",
    }.get(result.status, f"返回状态 {result.status.value}")
    native = f"；原生状态：{result.native_status}" if result.native_status else ""
    return f"{result.backend} {status_text}{native}"

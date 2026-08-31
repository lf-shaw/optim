"""统一组合优化器的公共编排层。

本模块负责一个数学请求的完整生命周期：校验、编译、后端路由、独立验收返回向量、必要时
尝试回退后端，并构造 :class:`OptimizationResult`。业务模型构造保留在
``portfolio_types``/``model.compiler``，原生求解器细节保留在 ``backends``。该边界十分重要：
后端报告 ``solved`` 从来不足以直接返回组合权重。

单期和多期便捷接口都是同一不可变 :class:`PortfolioProblem` 之上的无状态 facade；实盘请求
传入的黑名单等临时指令不会保留在 :class:`PortfolioOptimizer` 中。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .backends import (
    BackendOptions,
    ClarabelBackend,
    HighsBackend,
    MosekBackend,
    PIQPBackend,
)
from .backends.base import BackendResult
from .factor_qcqp import solve_factor_qcqp
from .model import CompiledProblem, FactorQCQP, LinearProgram, QuadraticProgram, compile_problem
from .portfolio_types import (
    AlignmentReport,
    AssetTradeConstraints,
    ConstraintViolation,
    FailureReason,
    MaximizeAlpha,
    OptimalityCertificate,
    OptimizationResult,
    PortfolioMetrics,
    PortfolioConstraints,
    PortfolioData,
    PortfolioObjective,
    PortfolioProblem,
    ProofStatus,
    RiskAdjustedAlpha,
    SolveStatus,
    SolveTimings,
    SolverAttempt,
    SolverPolicy,
)
from .solution import evaluate_solution, lift_weights
from .validation import ValidationReport, validate_problem

if TYPE_CHECKING:
    from .data import InMemoryDataSource, PortfolioSchedule
    from .portfolio_types import AlphaSpec, SequencePolicy
    from .sequence import PortfolioSequenceResult


@dataclass(frozen=True)
class PreparedPortfolioProblem:
    """已校验问题及其与求解器无关的 canonical 表示。

    输入无效时仍返回对象，但 ``compiled=None``，调用方可在不建立后端的情况下查看完整
    :class:`ValidationReport`。``prepare_s`` 包含静态校验和 canonical 编译。当前为审阅与
    诊断直接暴露 ``compiled``；未来二进制 ``_core`` 分发可将其改为不透明句柄。

    Attributes
    ----------
    problem : PortfolioProblem
        原始不可变业务问题。
    validation : ValidationReport
        在任何后端建立前生成的聚合静态校验报告。
    compiled : CompiledProblem | None
        校验通过后的 canonical 模型；存在错误时为 ``None``。
    prepare_s : float
        校验和编译的合计 wall-clock 秒数。
    """

    problem: PortfolioProblem
    validation: ValidationReport
    compiled: CompiledProblem | None
    prepare_s: float


@dataclass(frozen=True)
class _Evaluation:
    """内部使用的单次后端候选解独立验收结果。

    Attributes
    ----------
    metrics : PortfolioMetrics
        从候选向量独立复算的业务指标。
    violations : tuple
        超过验收容差的 canonical 约束违约记录。
    max_violation : float
        全部行约束和变量边界中的最大绝对违约量。
    accepted : bool
        候选解是否通过独立数值验收。
    validation_s : float
        独立复算与可选权重清理耗费的 wall-clock 秒数。
    """

    metrics: PortfolioMetrics
    violations: tuple
    max_violation: float
    accepted: bool
    validation_s: float


class PortfolioOptimizer:
    """编译、路由、求解并独立验收组合优化问题。

    实例只保存不可变的求解策略，不累计 universe、基准、持仓、黑名单、workspace 或多期状态，
    因而可以安全地跨请求复用。

    Parameters
    ----------
    policy : SolverPolicy | None
        路由、回退和验收策略；``None`` 使用 :class:`SolverPolicy` 默认值。

    Attributes
    ----------
    policy : SolverPolicy
        当前实例使用的不可变求解策略。
    """

    def __init__(self, policy: SolverPolicy | None = None):
        self.policy = SolverPolicy() if policy is None else policy

    def validate(self, problem: PortfolioProblem) -> ValidationReport:
        """聚合输入及模型的低成本静态问题，不编译也不求解。

        Parameters
        ----------
        problem : PortfolioProblem
            待校验的单期问题。

        Returns
        -------
        ValidationReport
            包含全部可独立发现错误和警告的报告；该方法不会因普通校验错误而提前抛出。
        """

        return validate_problem(problem)

    def prepare(self, problem: PortfolioProblem) -> PreparedPortfolioProblem:
        """在建立后端前校验请求，并在有效时编译 canonical 模型。

        编译过程与求解器无关，并同时计算 semantic 和 canonical fingerprint。由于本方法已经
        产生同一份校验报告，调用 ``compile_problem`` 时不会重复校验。

        Parameters
        ----------
        problem : PortfolioProblem
            待准备的不可变单期问题。

        Returns
        -------
        PreparedPortfolioProblem
            校验报告、可选 canonical 模型及准备耗时。输入错误保留在报告中，不在此处抛出。
        """

        started = time.perf_counter()
        report = validate_problem(problem)
        compiled = None
        if report.is_valid:
            compiled = compile_problem(problem, validate=False)
        return PreparedPortfolioProblem(
            problem=problem,
            validation=report,
            compiled=compiled,
            prepare_s=time.perf_counter() - started,
        )

    def solve(
        self,
        problem: PortfolioProblem,
        *,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """准备并求解一个不可变单期问题。

        ``theta_seed`` 只影响专用 factor-QCQP 策略的初始搜索点，不改变数学约束，也不会成为
        优化器的持久状态。

        Parameters
        ----------
        problem : PortfolioProblem
            完整单期问题。
        theta_seed : float | None
            可选的正数 theta 初始值；仅 factor-QCQP 使用。

        Returns
        -------
        OptimizationResult
            标准化结果。普通不可行或数值失败通过状态返回，不自动运行深度诊断。

        Raises
        ------
        PortfolioValidationError
            输入、单位、shape 或静态模型校验失败。
        """

        return self.solve_prepared(self.prepare(problem), theta_seed=theta_seed)

    def optimize(
        self,
        *,
        data: PortfolioData,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints | None = None,
        asset_trade: AssetTradeConstraints | None = None,
        blacklist: Iterable[Any] | None = None,
        frozen: Iterable[Any] | None = None,
        not_buyable: Iterable[Any] | None = None,
        not_sellable: Iterable[Any] | None = None,
        weight_overrides: Mapping[Any, float | tuple[float, float]] | None = None,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """求解一次实盘单期请求，不保留临时交易名单。

        ``solve(PortfolioProblem(...))`` 仍是 canonical 低层 API。本便捷方法将常用单期路径
        显式化，并把逐资产名单转换成不可变、仅对本问题生效的约束。

        Parameters
        ----------
        data : PortfolioData
            已对齐的严格同日单期数据。
        objective : PortfolioObjective
            业务目标。
        constraints : PortfolioConstraints | None
            可行域配置；``None`` 使用默认约束。
        asset_trade : AssetTradeConstraints | None
            已构造的单期交易指令。不能与下面的便捷名单同时传入，也不能与
            ``constraints.asset_trade`` 重复。
        blacklist, frozen, not_buyable, not_sellable : Iterable[Any] | None
            单期黑名单、冻结、不可买入和不可卖出资产集合。
        weight_overrides : Mapping[Any, float | tuple[float, float]] | None
            单期逐资产精确目标或闭区间覆盖。
        theta_seed : float | None
            factor-QCQP 的可选 theta 初始值。

        Returns
        -------
        OptimizationResult
            标准化单期结果。

        Raises
        ------
        ValueError
            同一交易指令通过多个入口重复提供。
        PortfolioValidationError
            组装后的问题未通过静态校验。
        """

        config = PortfolioConstraints() if constraints is None else constraints
        convenience_used = any(
            value is not None
            for value in (
                blacklist,
                frozen,
                not_buyable,
                not_sellable,
                weight_overrides,
            )
        )
        if asset_trade is not None and convenience_used:
            raise ValueError(
                "pass either asset_trade or the individual one-off lists, not both"
            )
        if config.asset_trade is not None and (asset_trade is not None or convenience_used):
            raise ValueError(
                "asset_trade is already present in constraints; do not provide it twice"
            )
        if convenience_used:
            asset_trade = AssetTradeConstraints(
                blacklist=() if blacklist is None else tuple(blacklist),
                frozen=() if frozen is None else tuple(frozen),
                not_buyable=() if not_buyable is None else tuple(not_buyable),
                not_sellable=() if not_sellable is None else tuple(not_sellable),
                weight_overrides={} if weight_overrides is None else weight_overrides,
            )
        if asset_trade is not None:
            config = replace(config, asset_trade=asset_trade)
        return self.solve(
            PortfolioProblem(data=data, objective=objective, constraints=config),
            theta_seed=theta_seed,
        )

    def optimize_range(
        self,
        *,
        data_source: "InMemoryDataSource",
        schedule: "PortfolioSchedule",
        objective: "PortfolioObjective",
        constraints: "PortfolioConstraints",
        alpha_spec: "AlphaSpec | None",
        initial_weight: pd.Series,
        holding_period_returns=None,
        sequence_policy: "SequencePolicy | None" = None,
        independent_initial_weights=None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> "PortfolioSequenceResult":
        """严格对齐已加载的调仓计划并按多期序列求解。

        非空静态 ``asset_trade`` 会被拒绝，因为黑名单、冻结名单等通常只对单日生效。需要逐日
        指令的策略必须显式构造逐日问题，不能把同一名单静默广播到整个区间。

        Parameters
        ----------
        data_source : InMemoryDataSource
            已一次性加载的基准和风险模型数据源。
        schedule : PortfolioSchedule
            以严格 ``(dt, sid)`` MultiIndex 定义调仓日期、资产和 alpha 的计划。
        objective : PortfolioObjective
            各日期共享的业务目标类型。
        constraints : PortfolioConstraints
            各日期共享的静态约束；不得包含非空单期交易名单。
        alpha_spec : AlphaSpec | None
            alpha 单位和尺度；alpha 目标必须提供。
        initial_weight : pandas.Series
            链式序列首日的实际期初权重，以资产为索引。
        holding_period_returns : Any | None
            相邻调仓日之间的 close-to-close 复合收益；链式模式由序列引擎按标签读取。
        sequence_policy : SequencePolicy | None
            持仓漂移、失败、theta 传播和输出策略；``None`` 使用默认策略。
        independent_initial_weights : Any | None
            独立模式下按日期提供的期初权重。
        extra_attribute_columns : tuple[str, ...]
            从 schedule 物化到 ``PortfolioData.extra_attributes`` 的列名。

        Returns
        -------
        PortfolioSequenceResult
            按日期记录结果、实际持仓状态和最终持仓的序列结果。

        Raises
        ------
        ValueError
            ``constraints`` 含有不应跨日期广播的单期交易指令。
        DataAlignmentError
            任一日期缺少严格同日数据或标签无法安全对齐。
        """

        if constraints.asset_trade is not None and not constraints.asset_trade.is_empty:
            raise ValueError(
                "static asset_trade constraints are not accepted by optimize_range; "
                "construct dated PortfolioProblem objects for date-specific instructions"
            )

        prepared_run = data_source.prepare_run(
            schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=extra_attribute_columns,
        )
        return self.solve_sequence(
            prepared_run,
            holding_period_returns=holding_period_returns,
            sequence_policy=sequence_policy,
        )

    def solve_sequence(
        self,
        problems,
        *,
        holding_period_returns=None,
        sequence_policy=None,
    ):
        """求解按日期排序的 close-to-close 组合序列。

        序列引擎采用延迟导入，使单期导入路径保持轻量。持仓漂移、失败后是否继续和 theta 传播
        都属于序列策略，而不是后端行为。

        Parameters
        ----------
        problems : Iterable[PortfolioProblem] | PreparedPortfolioRun
            已按日期排序的问题集合，或可按日期惰性物化问题的准备对象。
        holding_period_returns : Any | None
            调仓区间的 close-to-close 复合收益；链式模式需要。
        sequence_policy : SequencePolicy | None
            多期状态推进策略；``None`` 使用默认值。

        Returns
        -------
        PortfolioSequenceResult
            包含逐日尝试和最终实际持仓的结果。
        """

        from .sequence import solve_sequence

        return solve_sequence(
            self,
            problems,
            holding_period_returns=holding_period_returns,
            sequence_policy=sequence_policy,
        )

    def diagnose(
        self,
        problem: PortfolioProblem,
        *,
        prior_result: OptimizationResult | None = None,
        level: str = "deep",
    ):
        """对指定单个问题显式运行高成本不可行诊断。

        普通失败路径从不自动触发诊断。提供 ``prior_result`` 时，诊断层会先验证其问题
        fingerprint，再将它作为证据使用。

        Parameters
        ----------
        problem : PortfolioProblem
            需要诊断的准确业务问题。
        prior_result : OptimizationResult | None
            同一问题此前的失败结果；用于补充状态和路由证据。
        level : str
            诊断深度；当前公共值为 ``"deep"``。

        Returns
        -------
        InfeasibilityReport
            Phase-I、最小换手率、最小 TE 及后端证据的结构化报告。

        Raises
        ------
        PortfolioValidationError
            问题自身存在静态输入错误，无法进入数学不可行诊断。
        ValueError
            ``level`` 不受支持，或 ``prior_result`` 不属于同一问题。
        """

        from .diagnostics import diagnose_problem

        prepared = self.prepare(problem)
        prepared.validation.raise_for_errors()
        assert prepared.compiled is not None
        return diagnose_problem(
            problem,
            prepared.compiled,
            self.policy,
            prior_result=prior_result,
            level=level,
        )

    def solve_prepared(
        self,
        prepared: PreparedPortfolioProblem,
        *,
        theta_seed: float | None = None,
    ) -> OptimizationResult:
        """路由一个 canonical 模型并验收每次主求解和回退尝试。

        当前路由刻意保持显式：LP 使用 HiGHS，QP 使用 PIQP，factor-QCQP 使用专用 frontier
        策略。QP/QCQP 候选未通过验收时先尝试配置的 MOSEK，再尝试 Clarabel。所有回退求解
        完全相同的 ``CompiledProblem`` 并经过同一独立验收；LP 当前没有回退路径。

        Parameters
        ----------
        prepared : PreparedPortfolioProblem
            ``prepare`` 返回的校验及 canonical 编译结果。
        theta_seed : float | None
            factor-QCQP 的可选 theta 初始值。

        Returns
        -------
        OptimizationResult
            包含完整尝试路由、独立指标、违约和证书的结果。

        Raises
        ------
        PortfolioValidationError
            ``prepared`` 含静态校验错误。
        RuntimeError
            校验有效但缺少 canonical 模型，表示准备对象内部不一致。
        TypeError
            canonical 模型类型尚无路由实现。
        """

        total_started = time.perf_counter()
        prepared.validation.raise_for_errors()
        if prepared.compiled is None:
            raise RuntimeError("valid prepared problem has no canonical model")
        compiled = prepared.compiled
        model = compiled.model
        options = self._backend_options()

        if isinstance(model, LinearProgram):
            primary = HighsBackend().solve(model, options)
        elif isinstance(model, QuadraticProgram):
            primary = PIQPBackend().solve(model, options)
        elif isinstance(model, FactorQCQP):
            assert prepared.problem.data.alpha_spec is not None
            primary = solve_factor_qcqp(
                model,
                self.policy,
                theta_seed=theta_seed,
                objective_tolerance=self.policy.objective_tolerance.raw_limit(
                    prepared.problem.data.alpha_spec
                ),
            )
        else:
            raise TypeError(f"unsupported canonical model: {type(model).__name__}")

        attempts: list[BackendResult] = []
        primary, evaluation = self._audit_backend_result(prepared, primary)
        attempts.append(primary)
        if not evaluation.accepted and isinstance(model, (QuadraticProgram, FactorQCQP)):
            if self.policy.licensed_fallback.lower() == "mosek":
                fallback = MosekBackend().solve(model, options)
                fallback, evaluation = self._audit_backend_result(prepared, fallback)
                attempts.append(fallback)
            if (
                not evaluation.accepted
                and self.policy.free_fallback.lower() in {"clarabel", "clarabel_qdldl"}
            ):
                fallback = ClarabelBackend().solve(model, options)
                fallback, evaluation = self._audit_backend_result(prepared, fallback)
                attempts.append(fallback)
        return self._result(prepared, attempts, evaluation, total_started)

    def _backend_options(self) -> BackendOptions:
        tuning = self.policy.tuning
        return BackendOptions(
            max_iter=tuning.piqp_max_iter,
            eps_abs=tuning.final_eps,
            eps_rel=tuning.final_eps,
            objective_scale_target=tuning.alpha_target,
            inequality_form=tuning.piqp_inequality_form,
        )

    def _audit_backend_result(
        self,
        prepared: PreparedPortfolioProblem,
        backend_result: BackendResult,
    ) -> tuple[BackendResult, _Evaluation]:
        """将求解器原生结果转换为经独立验收的结果。

        审计从返回向量重新计算 canonical 行、变量边界、跟踪风险和业务指标。可行的原始解
        随后可以执行单独审计的小权重清理。数值验收失败统一规范化为
        ``INVALID_NUMERICS``，使路由逻辑不依赖各后端的状态术语。
        """

        assert prepared.compiled is not None
        validation_started = time.perf_counter()
        metrics = PortfolioMetrics()
        violations: tuple[ConstraintViolation, ...] = ()
        max_violation = 0.0
        accepted = backend_result.status.has_solution and backend_result.primal is not None
        if accepted:
            assert backend_result.primal is not None
            metrics, violations, max_violation = evaluate_solution(
                prepared.problem,
                prepared.compiled,
                backend_result.primal,
            )
            accepted = max_violation <= self.policy.tuning.feasibility_tolerance
            if not accepted:
                backend_result = replace(
                    backend_result,
                    status=SolveStatus.NUMERICAL_ERROR,
                    reason=FailureReason.INVALID_NUMERICS,
                    message=(
                        f"backend solution failed independent validation: max violation "
                        f"{max_violation:.3e}"
                    ),
                )
            else:
                backend_result, metrics, violations, max_violation = self._clean_solution(
                    prepared,
                    backend_result,
                    metrics,
                    violations,
                    max_violation,
                )
        return backend_result, _Evaluation(
            metrics,
            violations,
            max_violation,
            accepted,
            time.perf_counter() - validation_started,
        )

    def _clean_solution(
        self,
        prepared: PreparedPortfolioProblem,
        backend_result: BackendResult,
        raw_metrics: PortfolioMetrics,
        raw_violations: tuple,
        raw_max_violation: float,
    ) -> tuple[BackendResult, PortfolioMetrics, tuple, float]:
        """仅在证书仍然成立时应用 0.1bp 输出阈值。

        绝对值小于 ``weight_zero_tolerance`` 的权重置零，其余权重重新缩放至配置预算。
        修改权重后先重建全部辅助变量，再复算约束和指标。如果清理后的向量不可行，或其
        目标损失使 LP/前沿证书超过调用方容差，则丢弃清理结果并保留原始解。
        """

        assert prepared.compiled is not None
        assert backend_result.primal is not None
        domain = prepared.compiled.model.domain
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
        cleaned *= prepared.problem.constraints.budget / cleaned_sum
        cleaned_vector = lift_weights(prepared.problem, prepared.compiled, cleaned)
        metrics, violations, max_violation = evaluate_solution(
            prepared.problem,
            prepared.compiled,
            cleaned_vector,
        )
        diagnostics = dict(backend_result.diagnostics)
        diagnostics.update(
            {
                "weight_cleanup_threshold": tolerance,
                "weight_cleanup_removed_l1": removed_l1,
                "weight_cleanup_count": int(small.sum()),
                "weight_cleanup_applied": max_violation
                <= self.policy.tuning.feasibility_tolerance,
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
            if isinstance(prepared.problem.objective, (MaximizeAlpha, RiskAdjustedAlpha)):
                cleanup_loss = max(0.0, raw_metrics.objective - metrics.objective)
            else:
                cleanup_loss = max(0.0, metrics.objective - raw_metrics.objective)
        diagnostics["weight_cleanup_objective_loss"] = cleanup_loss
        certificate_sensitive = bool(
            "total_gap" in diagnostics
            or diagnostics.get("lp_prescreen_certified") is True
            or isinstance(prepared.compiled.model, LinearProgram)
        )
        if certificate_sensitive:
            prior_gap = float(diagnostics.get("total_gap", 0.0))
            certified_gap = prior_gap + cleanup_loss
            diagnostics["certified_objective_gap"] = certified_gap
            if "total_gap" in diagnostics:
                diagnostics["total_gap"] = certified_gap
            diagnostics["cleanup_objective_loss"] = cleanup_loss
            alpha_spec = prepared.problem.data.alpha_spec
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
        prepared: PreparedPortfolioProblem,
        backend_results: list[BackendResult],
        evaluation: _Evaluation,
        total_started: float,
    ) -> OptimizationResult:
        """将最终求解尝试规范化为公共不可变结果契约。

        证书类型反映实际路由：LP 预筛选证明、因子前沿 Lagrangian 估计、锥规划
        primal/dual 间隙、精确 LP 目标或后端估计。高容量 theta 轨迹不写入公共路由
        元数据，但保留可审计的聚合诊断。
        """

        assert prepared.compiled is not None
        compiled = prepared.compiled
        problem = prepared.problem
        backend_result = backend_results[-1]
        metrics = evaluation.metrics
        violations = evaluation.violations
        max_violation = evaluation.max_violation
        accepted = evaluation.accepted
        status = backend_result.status
        message = backend_result.message

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
            if (
                isinstance(compiled.model, FactorQCQP)
                and backend_result.diagnostics.get("lp_prescreen_certified") is True
            ):
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                certified_gap = float(
                    backend_result.diagnostics.get("certified_objective_gap", 0.0)
                )
                certificate = OptimalityCertificate(
                    kind="lp_global_optimum_feasible_for_factor_qcqp",
                    proof_status=ProofStatus.VERIFIED,
                    primal_value=float(objective_value),
                    dual_bound=float(objective_value) + certified_gap,
                    absolute_gap=certified_gap,
                    normalized_gap=certified_gap / alpha_spec.scale,
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_spec.scale,
                    components={
                        "weight_cleanup_objective_loss": certified_gap,
                        "max_constraint_violation": max_violation,
                    },
                )
            elif isinstance(compiled.model, FactorQCQP) and "total_gap" in backend_result.diagnostics:
                total_gap = float(backend_result.diagnostics["total_gap"])
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                alpha_scale = alpha_spec.scale
                certificate = OptimalityCertificate(
                    kind="factor_qcqp_lagrangian",
                    proof_status=ProofStatus.NUMERICAL_ESTIMATE,
                    primal_value=float(objective_value),
                    dual_bound=float(objective_value) + total_gap,
                    absolute_gap=total_gap,
                    normalized_gap=total_gap / alpha_scale,
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_scale,
                    components={
                        "frontier_slack_gap": float(
                            backend_result.diagnostics["frontier_gap"]
                        ),
                        "qp_subproblem_gap": float(
                            backend_result.diagnostics["subproblem_gap"]
                        ),
                        "max_constraint_violation": max_violation,
                    },
                )
            elif isinstance(compiled.model, FactorQCQP):
                native_gap = backend_result.diagnostics.get("native_gap_unscaled")
                native_gap = None if native_gap is None else float(native_gap)
                alpha_spec = problem.data.alpha_spec
                assert alpha_spec is not None
                alpha_scale = alpha_spec.scale
                certificate = OptimalityCertificate(
                    kind="conic_primal_dual",
                    proof_status=(
                        ProofStatus.VERIFIED
                        if native_gap is not None
                        else ProofStatus.UNAVAILABLE
                    ),
                    primal_value=float(objective_value),
                    dual_bound=(
                        float(objective_value) + native_gap
                        if native_gap is not None
                        else None
                    ),
                    absolute_gap=native_gap,
                    normalized_gap=(
                        native_gap / alpha_scale if native_gap is not None else None
                    ),
                    objective_units=alpha_spec.units,
                    objective_scale=alpha_scale,
                    components={"max_constraint_violation": max_violation},
                )
            else:
                proof_status = (
                    ProofStatus.VERIFIED
                    if isinstance(compiled.model, LinearProgram)
                    and backend_result.status == SolveStatus.OPTIMAL
                    else ProofStatus.NUMERICAL_ESTIMATE
                )
                certified_gap = float(
                    backend_result.diagnostics.get("certified_objective_gap", 0.0)
                )
                certificate = OptimalityCertificate(
                    kind=(
                        "backend_primal_dual"
                        if proof_status is ProofStatus.VERIFIED
                        else "backend_kkt"
                    ),
                    proof_status=proof_status,
                    primal_value=float(objective_value),
                    dual_bound=(
                        float(objective_value) + certified_gap
                        if proof_status is ProofStatus.VERIFIED
                        else None
                    ),
                    absolute_gap=(
                        certified_gap if proof_status is ProofStatus.VERIFIED else None
                    ),
                    normalized_gap=(
                        certified_gap / problem.data.alpha_spec.scale
                        if proof_status is ProofStatus.VERIFIED
                        and problem.data.alpha_spec is not None
                        else None
                    ),
                    objective_units=(
                        problem.data.alpha_spec.units
                        if problem.data.alpha_spec is not None
                        else "annualized_decimal"
                    ),
                    objective_scale=(
                        problem.data.alpha_spec.scale
                        if problem.data.alpha_spec is not None
                        else None
                    ),
                    components={"max_constraint_violation": max_violation},
                )

        route = tuple(
            SolverAttempt(
                backend=item.backend,
                status=item.status,
                reason=item.reason,
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
                    source_dates[label] = pd.Timestamp(value)
                except (TypeError, ValueError):
                    pass
        timings = SolveTimings(
            prepare_s=prepared.prepare_s,
            backend_setup_s=sum(item.setup_s for item in backend_results),
            backend_solve_s=sum(item.solve_s for item in backend_results),
            validation_s=evaluation.validation_s,
            total_s=time.perf_counter() - total_started + prepared.prepare_s,
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
        )


def _metadata_float(metadata, key: str, default: float) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default

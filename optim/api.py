"""统一组合优化器的公共接口。

模块提供单期与多期问题的组装、前置校验、求解和诊断入口。所有请求都基于不可变的
:class:`PortfolioProblem`；实盘请求传入的黑名单等临时指令不会保留在优化器实例中。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pandas as pd

from .model import compile_problem
from ._solver_adapter import SolverAdapter
from .portfolio_types import (
    AssetTradeConstraints,
    OptimizationResult,
    PortfolioConstraints,
    PortfolioData,
    PortfolioObjective,
    PortfolioProblem,
    ProblemFingerprint,
    SequencePolicy,
    SolverPolicy,
)
from .validation import ValidationReport, validate_problem

if TYPE_CHECKING:
    from .data import (
        BenchmarkCoveragePolicy,
        InMemoryDataSource,
        PortfolioSchedule,
    )
    from .integrations.tuda2 import Tuda2DataSource
    from .portfolio_types import AlphaSpec
    from .sequence import PortfolioSequenceResult


@dataclass(frozen=True)
class PreparedPortfolioProblem:
    """已完成前置校验和数值准备的单期问题。

    输入无效时仍返回对象，调用方可以查看完整 :class:`ValidationReport`。有效问题同时提供
    fingerprint、已应用的等价结构优化标识和准备耗时。

    Attributes
    ----------
    problem : PortfolioProblem
        原始不可变业务问题。
    validation : ValidationReport
        在任何后端建立前生成的聚合静态校验报告。
    fingerprint : ProblemFingerprint | None
        校验通过后的语义/canonical 身份；存在错误时为 ``None``。
    compiler_optimizations : tuple[str, ...]
        私有编译器实际采用、且已证明数学等价的结构优化标识。
    prepare_s : float
        校验和编译的合计 wall-clock 秒数。
    _solver_handle : object | None
        优化器内部使用的短生命周期准备状态；调用方不得读取、序列化或跨进程复用。
    """

    problem: PortfolioProblem
    validation: ValidationReport
    fingerprint: ProblemFingerprint | None
    compiler_optimizations: tuple[str, ...]
    prepare_s: float
    _solver_handle: object | None = field(default=None, repr=False, compare=False)


class PortfolioOptimizer:
    """准备、求解并验收单期或多期组合优化问题。

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
        self._solver = SolverAdapter(self.policy)

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
        """校验请求，并在有效时创建可重复求解的准备对象。

        本方法同时生成问题 fingerprint 和完整校验报告；校验失败时不会执行求解准备。

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
        solver_handle = None
        fingerprint = None
        compiler_optimizations: tuple[str, ...] = ()
        if report.is_valid:
            solver_handle = self._solver.prepare(
                compile_problem(problem, validate=False)
            )
            fingerprint, compiler_optimizations = self._solver.metadata(solver_handle)
        return PreparedPortfolioProblem(
            problem=problem,
            validation=report,
            fingerprint=fingerprint,
            compiler_optimizations=compiler_optimizations,
            prepare_s=time.perf_counter() - started,
            _solver_handle=solver_handle,
        )

    def _prepare_prevalidated(
        self,
        problem: PortfolioProblem,
    ) -> PreparedPortfolioProblem:
        """编译已通过序列全区间预检的问题，不重复执行静态校验。

        该入口仅供序列状态机使用。序列层必须先验证全部模板；链式模式随后只会把模板中的
        期初权重替换为由受控持仓漂移生成、坐标和预算均已确定的向量。
        """

        started = time.perf_counter()
        solver_handle = self._solver.prepare(compile_problem(problem, validate=False))
        fingerprint, compiler_optimizations = self._solver.metadata(solver_handle)
        return PreparedPortfolioProblem(
            problem=problem,
            validation=ValidationReport(),
            fingerprint=fingerprint,
            compiler_optimizations=compiler_optimizations,
            prepare_s=time.perf_counter() - started,
            _solver_handle=solver_handle,
        )

    def _solve_prevalidated(
        self,
        problem: PortfolioProblem,
    ) -> OptimizationResult:
        """求解已由序列状态机预检的问题，避免逐日重复静态校验。"""

        return self.solve_prepared(
            self._prepare_prevalidated(problem),
        )

    def solve(
        self,
        problem: PortfolioProblem,
    ) -> OptimizationResult:
        """准备并求解一个不可变单期问题。

        Parameters
        ----------
        problem : PortfolioProblem
            完整单期问题。

        Returns
        -------
        OptimizationResult
            标准化结果。普通不可行或数值失败通过状态返回，不自动运行深度诊断。

        Raises
        ------
        PortfolioValidationError
            输入、单位、shape 或静态模型校验失败。
        RuntimeError
            显式选择 MOSEK，但未安装或缺少有效授权；不回退到其他后端。
        """

        return self.solve_prepared(self.prepare(problem))

    def optimize(
        self,
        *,
        data: PortfolioData | None = None,
        data_source: "Tuda2DataSource | None" = None,
        date: Any | None = None,
        universe: pd.DataFrame | None = None,
        benchmark: str | pd.Series | None = None,
        initial_weight: pd.Series | None = None,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints | None = None,
        alpha_spec: "AlphaSpec | None" = None,
        asset_trade: AssetTradeConstraints | None = None,
        blacklist: Iterable[Any] | None = None,
        frozen: Iterable[Any] | None = None,
        not_buyable: Iterable[Any] | None = None,
        not_sellable: Iterable[Any] | None = None,
        weight_overrides: Mapping[Any, float | tuple[float, float]] | None = None,
        benchmark_policy: "BenchmarkCoveragePolicy | None" = None,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> OptimizationResult:
        """从已对齐数据或数据源求解一次实盘单期请求。

        ``solve(PortfolioProblem(...))`` 仍是 canonical 低层 API。本便捷方法将常用单期路径
        显式化，并把逐资产名单转换成不可变、仅对本问题生效的约束。调用方必须在 ``data``
        与 ``data_source`` 中恰好选择一个：前者直接进入核心，后者由适配器一次获取并严格对齐
        指定日期的数据，再回到相同的核心路径。

        Parameters
        ----------
        data : PortfolioData | None
            已对齐的严格同日单期数据；与 ``data_source`` 互斥。
        data_source : Tuda2DataSource | None
            实现单期数据准备协议的 tuda2 数据源。
        date, universe, initial_weight
            使用数据源时必需的单期日期、样本空间和交易前实际持仓。
        benchmark : str | pandas.Series | None
            数据源模式必需：指数代码，或以 sid 为索引的单期权重 Series。
            Series 视为指定 date 的基准，跳过指数权重 I/O；传 data 时不得重复提供。
        objective : PortfolioObjective
            业务目标。
        constraints : PortfolioConstraints | None
            可行域配置；``None`` 使用默认约束。
        alpha_spec : AlphaSpec | None
            使用数据源物化 alpha 时的单位和尺度；直接传 ``data`` 时单位已经包含在数据中。
        asset_trade : AssetTradeConstraints | None
            已构造的单期交易指令。不能与下面的便捷名单同时传入，也不能与
            ``constraints.asset_trade`` 重复。
        blacklist, frozen, not_buyable, not_sellable : Iterable[Any] | None
            单期黑名单、冻结、不可买入和不可卖出资产集合。
        weight_overrides : Mapping[Any, float | tuple[float, float]] | None
            单期逐资产精确目标或闭区间覆盖。
        benchmark_policy : BenchmarkCoveragePolicy | None
            数据源处理基准覆盖缺口的显式策略。
        tradable_universe : str | None
            数据源可选的可交易域名称。
        extra_attribute_columns : tuple[str, ...]
            从数据源样本空间物化的额外逐资产数值列。

        Returns
        -------
        OptimizationResult
            标准化单期结果。

        Raises
        ------
        ValueError
            数据输入入口不唯一、数据源参数不完整，或同一交易指令通过多个入口重复提供。
        PortfolioValidationError
            组装后的问题未通过静态校验。
        """

        if (data is None) == (data_source is None):
            raise ValueError("pass exactly one of data or data_source")
        config = PortfolioConstraints() if constraints is None else constraints
        if data_source is not None:
            missing = [
                name
                for name, value in (
                    ("date", date),
                    ("universe", universe),
                    ("benchmark", benchmark),
                    ("initial_weight", initial_weight),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "data_source single-period optimization requires "
                    + ", ".join(missing)
                )
            assert date is not None
            assert universe is not None
            assert benchmark is not None
            assert initial_weight is not None
            source_problem = data_source.build_problem(
                date=date,
                universe=universe,
                benchmark=benchmark,
                initial_weight=initial_weight,
                objective=objective,
                constraints=config,
                alpha_spec=alpha_spec,
                benchmark_policy=benchmark_policy,
                tradable_universe=tradable_universe,
                extra_attribute_columns=extra_attribute_columns,
            )
            data = source_problem.data
        else:
            source_only_values = {
                "date": date,
                "universe": universe,
                "benchmark": benchmark,
                "initial_weight": initial_weight,
                "alpha_spec": alpha_spec,
                "benchmark_policy": benchmark_policy,
                "tradable_universe": tradable_universe,
            }
            unexpected = [
                name for name, value in source_only_values.items() if value is not None
            ]
            if extra_attribute_columns:
                unexpected.append("extra_attribute_columns")
            if unexpected:
                raise ValueError(
                    "source-only arguments cannot be combined with data: "
                    + ", ".join(unexpected)
                )
        assert data is not None
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
        if config.asset_trade is not None and (
            asset_trade is not None or convenience_used
        ):
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
        )

    def optimize_range(
        self,
        *,
        data_source: "InMemoryDataSource | Tuda2DataSource",
        schedule: "PortfolioSchedule",
        objective: "PortfolioObjective",
        constraints: "PortfolioConstraints",
        alpha_spec: "AlphaSpec | None",
        initial_weight: pd.Series,
        benchmark: str | pd.Series | None = None,
        holding_period_returns=None,
        sequence_policy: "SequencePolicy | None" = None,
        independent_initial_weights=None,
        benchmark_policy: "BenchmarkCoveragePolicy | None" = None,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> "PortfolioSequenceResult":
        """通过统一数据源 facade 严格对齐并求解多期序列。

        非空静态 ``asset_trade`` 会被拒绝，因为黑名单、冻结名单等依赖当日真实持仓和交易
        状态。此类操作性指令应使用单期 ``optimize``，不能静默广播到整个研究区间。

        Parameters
        ----------
        data_source : InMemoryDataSource | Tuda2DataSource
            已加载的 ``InMemoryDataSource``，或实现区间准备协议的 ``Tuda2DataSource``。
            外部数据源必须在逐日求解前一次性完成区间 I/O，不得回调求解器。
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
        benchmark : str | pandas.Series | None
            外部数据源使用的指数代码或严格 (dt, sid) 索引的逐日权重 Series。
            缺日期报错，不广播单期权重、不前向填充。内存源已绑定基准，必须保持 None。
        holding_period_returns : Any | None
            相邻调仓日之间的 close-to-close 复合收益；链式模式由序列引擎按标签读取。
        sequence_policy : SequencePolicy | None
            持仓漂移、失败和输出策略；``None`` 使用默认策略。
        independent_initial_weights : Any | None
            独立模式下按日期提供的期初权重。
        benchmark_policy : BenchmarkCoveragePolicy | None
            外部数据源处理基准覆盖缺口的显式策略；内存源已经绑定该策略。
        tradable_universe : str | None
            外部数据源可选的可交易域名称。
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
                "use single-period optimize for operational asset instructions"
            )

        from .data import InMemoryDataSource

        if not isinstance(data_source, InMemoryDataSource):
            if benchmark is None:
                raise ValueError("external data_source requires benchmark")
            resolved_policy = (
                SequencePolicy() if sequence_policy is None else sequence_policy
            )
            prepared = data_source.prepare_sequence(
                schedule=schedule,
                benchmark=benchmark,
                initial_weight=initial_weight,
                objective=objective,
                constraints=constraints,
                alpha_spec=alpha_spec,
                benchmark_policy=benchmark_policy,
                independent_initial_weights=independent_initial_weights,
                holding_period_returns=holding_period_returns,
                require_holding_returns=resolved_policy.mode == "chained",
                tradable_universe=tradable_universe,
                extra_attribute_columns=extra_attribute_columns,
            )
            return self.solve_sequence(
                prepared.run,
                holding_period_returns=prepared.holding_period_returns,
                sequence_policy=resolved_policy,
            )
        if (
            benchmark is not None
            or benchmark_policy is not None
            or tradable_universe is not None
        ):
            raise ValueError(
                "benchmark, benchmark_policy and tradable_universe are external-source "
                "arguments; InMemoryDataSource already binds these data"
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

        序列引擎采用延迟导入，使单期导入路径保持轻量。持仓漂移、失败后是否继续和失败恢复
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
        problem: PortfolioProblem | OptimizationResult,
        *,
        prior_result: OptimizationResult | None = None,
        level: str = "deep",
        backend: str = "auto",
    ):
        """对指定单个问题显式运行高成本可行性与约束诊断。

        普通失败路径从不自动触发诊断。提供 ``prior_result`` 时，诊断层会先验证其问题
        fingerprint，再将它作为证据使用。

        成功结果（包括已验收的近似最优结果）在校验和编译前立即拒绝，当前解的指标请读取
        ``result.metrics``。失败结果不限于不可行，也可以是迭代上限或数值失败。单独传入
        ``PortfolioProblem`` 时不查询求解历史，仍允许显式诊断；没有强制绕过成功检查的选项。

        Parameters
        ----------
        problem : PortfolioProblem | OptimizationResult
            需要诊断的准确业务问题，或单期求解返回的结果。传入结果时直接使用其中保留的
            原问题，并自动把该结果作为原生证据来源。
        prior_result : OptimizationResult | None
            同一问题此前的失败结果；用于补充状态和路由证据。默认为
            ``None``；传入 ``OptimizationResult`` 作为首个参数时自动使用该结果。
        level : str
            诊断深度；当前公共值为 ``"deep"``。
        backend : str
            辅助模型的后端，默认 auto，独立于原求解策略。mosek 或 clarabel 可直接对比；
            显式值不回退，不支持辅助模型的类型则报错。piqp 不支持必需的 Phase-I LP；
            highs 仅适合无需最小风险 QP 的诊断。原结果证书保持原后端来源。
            后端未提供数值对偶下界时，相关下界字段为 None，不以候选目标冒充下界。

        Returns
        -------
        InfeasibilityReport
            Phase-I、最小换手率、最小 TE 及后端证据的结构化报告。

        Raises
        ------
        PortfolioValidationError
            问题自身存在静态输入错误，无法进入数学不可行诊断。
        ValueError
            提供了成功结果、``level`` 不受支持、结果未保留原问题、同时提供冲突的 ``prior_result``，或先前
            结果不属于同一问题。
        """

        if (
            isinstance(problem, OptimizationResult) and problem.status.has_solution
        ) or (prior_result is not None and prior_result.status.has_solution):
            raise ValueError(
                "cannot diagnose a successful optimization result; "
                "inspect result.metrics for the solved portfolio"
            )

        if isinstance(problem, OptimizationResult):
            if problem.problem is None:
                raise ValueError(
                    "optimization result does not retain a problem; for a stopped "
                    "sequence, diagnose sequence_result.stopped_problem"
                )
            if prior_result is not None and prior_result is not problem:
                raise ValueError(
                    "prior_result must be omitted when diagnosing an OptimizationResult"
                )
            prior_result = problem
            problem = problem.problem

        # 诊断的编译准备不继承原后端的模型类型限制；具体辅助模型使用独立选择的后端。
        replace(self.policy, backend=backend)  # 在昂贵编译前校验选项。
        if backend == "piqp":
            raise ValueError("diagnostic backend='piqp' does not support Phase-I LP")
        preparation = (
            self
            if self.policy.backend == "auto"
            else PortfolioOptimizer(replace(self.policy, backend="auto"))
        )
        prepared = preparation.prepare(problem)
        prepared.validation.raise_for_errors()
        if prepared._solver_handle is None:
            raise RuntimeError("valid prepared problem has no solver handle")
        return self._solver.diagnose(
            problem,
            prepared._solver_handle,
            prior_result=prior_result,
            level=level,
            backend=backend,
        )

    def solve_prepared(
        self,
        prepared: PreparedPortfolioProblem,
    ) -> OptimizationResult:
        """求解一个已经完成前置校验和数值准备的问题。

        结果会统一给出求解状态、可用权重、约束违约、最优性证据、求解路线和分阶段耗时，
        调用方不需要根据问题类型选择具体求解方法。

        Parameters
        ----------
        prepared : PreparedPortfolioProblem
            ``prepare`` 返回的校验、审计元信息和内部准备状态。

        Returns
        -------
        OptimizationResult
            包含完整尝试路由、独立指标、违约和证书的结果。

        Raises
        ------
        PortfolioValidationError
            ``prepared`` 含静态校验错误。
        RuntimeError
            校验有效但缺少内部准备状态，表示准备对象不一致。
        TypeError
            句柄不是由当前核心版本创建，或内部数学结构尚无路由实现。
        """

        prepared.validation.raise_for_errors()
        if prepared._solver_handle is None:
            raise RuntimeError("valid prepared problem has no solver handle")
        return self._solver.solve(
            prepared.problem,
            prepared._solver_handle,
            prepare_s=prepared.prepare_s,
        )

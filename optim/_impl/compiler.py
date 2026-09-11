r"""将组合业务语义编译为与求解器无关的稀疏模型。

编译器是唯一允许把业务约束转换为矩阵行和辅助变量的层。所有问题共享多面体域
$\underline z\le z\le\overline z$ 和 $l\le Az\le u$；LP、QP 和 factor-QCQP 对象只增加
各自的目标与风险表示。

``z`` 的前 ``n_assets`` 列始终是组合权重；可选后续列表示换手率 epigraph、总主动权重
epigraph 和因子主动敞口。每一行/列都带审计记录，供 fingerprint、独立解验收和不可行诊断
使用。本模块从不导入或初始化数值求解器。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from .asset_bounds import AssetBoundsError, resolve_asset_bounds
from ..fingerprint import fingerprint
from ..portfolio_types import (
    FactorRiskModel,
    FullCovarianceRiskModel,
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioProblem,
    ProblemKind,
    RiskAdjustedAlpha,
)
from ..validation import validate_problem
from ..model.canonical import (
    CanonicalKind,
    CompiledProblem,
    CanonicalModel,
    ConstraintRecord,
    FactorQCQP,
    FactorRiskOperator,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)


class CanonicalCompilationError(ValueError):
    """有效业务请求无法由当前编译器表示时抛出。"""


def classify_problem(problem: PortfolioProblem) -> ProblemKind:
    """按数学结构而不是历史 API 名称分类问题。

    线性 alpha 问题即使包含换手率或敞口约束仍是 LP，因为这些约束都可线性编译。
    ``MaximizeAlpha`` 加一个因子跟踪误差预算时选择专用 factor-QCQP 表示。
    ``RiskAdjustedAlpha`` 再叠加风险预算属于通用锥问题；当前 v1 编译器会明确报告不支持，
    而不是静默改变用户目标。

    Parameters
    ----------
    problem : PortfolioProblem
        已构造的业务问题。

    Returns
    -------
    ProblemKind
        由目标和非线性约束共同决定的 canonical 问题类型。

    Raises
    ------
    CanonicalCompilationError
        目标类型不受支持。
    """

    objective = problem.objective
    has_risk_limit = problem.constraints.tracking_error is not None
    if isinstance(objective, MaximizeAlpha):
        return ProblemKind.FACTOR_QCQP if has_risk_limit else ProblemKind.LP
    if isinstance(objective, RiskAdjustedAlpha):
        return ProblemKind.CONIC if has_risk_limit else ProblemKind.QP
    if isinstance(objective, MinimizeTrackingError):
        return ProblemKind.QP
    raise CanonicalCompilationError(
        f"unsupported objective type: {type(objective).__name__}"
    )


@dataclass
class _Layout:
    """一个 canonical 模型选定的连续变量列布局。

    Attributes
    ----------
    n_variables : int
        canonical 向量的总列数。
    weight : numpy.ndarray
        组合目标权重列的位置索引。
    turnover_aux : numpy.ndarray | None
        换手率绝对值或稀疏买入辅助变量列；未配置换手率时为 ``None``。
    turnover_support : numpy.ndarray | None
        稀疏换手率表示中，初始非零持仓对应的资产位置。
    turnover_new : numpy.ndarray | None
        稀疏换手率表示中，初始零持仓对应的资产位置。
    sparse_turnover : bool
        是否使用只做多满仓条件下的精确稀疏换手率表示。
    active_aux : numpy.ndarray | None
        总主动权重辅助列；未配置/冗余时为 ``None``，稀疏表达允许空数组。
    active_support : numpy.ndarray | None
        稀疏总主动表达中有效区间跨越基准的资产位置。
    active_negative : numpy.ndarray | None
        稀疏总主动表达中确定低配的资产位置，其贡献直接进入总量行。
    sparse_active : bool
        是否使用精确的低配单边表达；允许卖空的问题暂保留完整表达。
    factor : numpy.ndarray | None
        QP 中的因子主动暴露变量列；无需显式因子变量时为 ``None``。
    """

    n_variables: int
    weight: np.ndarray
    turnover_aux: np.ndarray | None
    turnover_support: np.ndarray | None
    turnover_new: np.ndarray | None
    sparse_turnover: bool
    active_aux: np.ndarray | None
    active_support: np.ndarray | None
    active_negative: np.ndarray | None
    sparse_active: bool
    factor: np.ndarray | None


class _DomainBuilder:
    """增量构造一个可审计的稀疏线性域。

    列布局在 ``__init__`` 中一次确定，使后续约束块可直接按最终宽度组装为 CSC。构造器还会在
    ``optimizations`` 中记录数学上精确的结构简化，例如稀疏换手率和可证明冗余的总主动约束。
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        *,
        include_factor_variables: bool,
        simplify: bool = True,
    ) -> None:
        self.problem = problem
        self.simplify = simplify
        self.data = problem.data
        self.constraints_config = problem.constraints
        self.n_assets = len(self.data.assets)
        self.risk_model = self.data.risk_model
        self.optimizations: list[str] = []
        # 先解析一次最终有效边界，布局分类与实际变量边界必须使用同一个结果。
        try:
            self.resolved_bounds = resolve_asset_bounds(problem)
        except AssetBoundsError as exc:
            raise CanonicalCompilationError(str(exc)) from exc
        n_factors = (
            len(self.risk_model.factor_names)
            if include_factor_variables and isinstance(self.risk_model, FactorRiskModel)
            else 0
        )

        cursor = self.n_assets
        turnover_aux = None
        turnover_support = None
        turnover_new = None
        sparse_turnover = False
        if self.constraints_config.turnover is not None:
            initial = np.asarray(self.data.initial_weight, dtype=float)
            redundant_turnover = self._l1_redundant(
                initial, self.constraints_config.turnover.l1_limit
            )
            sparse_turnover = (
                not redundant_turnover and self._turnover_uses_sparse_form(initial)
            )
            if redundant_turnover:
                self.optimizations.append("omit_redundant_turnover")
            elif sparse_turnover:
                turnover_support = np.flatnonzero(initial > 0.0).astype(np.int32)
                turnover_new = np.flatnonzero(initial <= 0.0).astype(np.int32)
                turnover_aux = np.arange(
                    cursor,
                    cursor + len(turnover_support),
                    dtype=np.int32,
                )
                cursor += len(turnover_support)
                self.optimizations.append("exact_sparse_turnover")
            else:
                turnover_aux = np.arange(cursor, cursor + self.n_assets, dtype=np.int32)
                cursor += self.n_assets
        active_aux = None
        active_support = active_negative = None
        sparse_active = False
        if self.constraints_config.total_active is not None:
            turnover = self.constraints_config.turnover
            benchmark = self.data.benchmark
            current_weight = self.data.initial_weight
            redundant = self._l1_redundant(
                np.asarray(benchmark), self.constraints_config.total_active
            ) or bool(
                self.simplify
                and turnover is not None
                and benchmark is not None
                and current_weight is not None
                and float(
                    np.abs(np.asarray(current_weight) - np.asarray(benchmark)).sum()
                )
                + turnover.l1_limit
                <= self.constraints_config.total_active
            )
            if redundant:
                self.optimizations.append("omit_redundant_total_active")
            else:
                sparse_active = bool(
                    self.simplify
                    and self.constraints_config.long_only
                    and np.all(self.resolved_bounds.lower >= 0.0)
                )
                if sparse_active:
                    benchmark = np.asarray(benchmark, dtype=float)
                    lower, upper = (
                        self.resolved_bounds.lower,
                        self.resolved_bounds.upper,
                    )
                    active_support = np.flatnonzero(
                        (lower < benchmark) & (upper > benchmark)
                    ).astype(np.int32)
                    active_negative = np.flatnonzero(
                        (lower < benchmark) & (upper <= benchmark)
                    ).astype(np.int32)
                    active_size = len(active_support)
                    self.optimizations.append("exact_sparse_total_active")
                else:
                    active_size = self.n_assets
                active_aux = np.arange(cursor, cursor + active_size, dtype=np.int32)
                cursor += active_size
        factor = None
        if n_factors:
            factor = np.arange(cursor, cursor + n_factors, dtype=np.int32)
            cursor += n_factors
        self.layout = _Layout(
            n_variables=cursor,
            weight=np.arange(self.n_assets, dtype=np.int32),
            turnover_aux=turnover_aux,
            turnover_support=turnover_support,
            turnover_new=turnover_new,
            sparse_turnover=sparse_turnover,
            active_aux=active_aux,
            active_support=active_support,
            active_negative=active_negative,
            sparse_active=sparse_active,
            factor=factor,
        )
        self.blocks: list[sp.csc_matrix] = []
        self.lowers: list[np.ndarray] = []
        self.uppers: list[np.ndarray] = []
        self.row_records: list[ConstraintRecord] = []

        self.variable_lower = np.full(cursor, -np.inf, dtype=float)
        self.variable_upper = np.full(cursor, np.inf, dtype=float)
        self.variables: list[VariableRecord] = []
        self.variable_records: list[ConstraintRecord] = []
        self._configure_variables()

    def _l1_redundant(self, center: np.ndarray, limit: float) -> bool:
        r"""仅凭未放宽的资产边界/预算证明 $\|w-c\|_1\le L$ 恒成立。

        非负目标权重满足上界 $B+\|c\|_1$；任意符号的有限盒约束给出
        $\sum_i\max(|l_i-c_i|,|u_i-c_i|)$。不使用某个候选的松弛量或业务容差
        判定冗余。诊断域关闭此优化，以免放宽前提后丢失原约束。
        """
        if not self.simplify:
            return False
        lower, upper = self.resolved_bounds.lower, self.resolved_bounds.upper
        if np.any(lower > upper):
            return False
        if np.all(lower >= 0.0) and self.constraints_config.budget >= 0.0:
            bound = math.fsum([self.constraints_config.budget, *np.abs(center)])
            if bound <= limit:
                return True
        distances = np.maximum(np.abs(lower - center), np.abs(upper - center))
        return bool(np.all(np.isfinite(distances)) and math.fsum(distances) <= limit)

    def _turnover_uses_sparse_form(self, initial: np.ndarray) -> bool:
        """仅在原持仓非负、目标权重非负且预算相同时采用稀疏买入表达。"""
        return bool(
            self.simplify
            and self.constraints_config.long_only
            and np.all(self.resolved_bounds.lower >= 0.0)
            and np.all(initial >= 0.0)
            and np.isclose(
                initial.sum(), self.constraints_config.budget, rtol=0.0, atol=1e-12
            )
        )

    def _configure_variables(self) -> None:
        """解析逐资产最终边界，并登记每一个 canonical 变量列。

        操作指令已经由 ``resolve_asset_bounds`` 合并。普通的 5,000 资产问题通常只有
        少量不同的边界来源组合，因此此处驻留公共不可变元数据以降低分配开销。
        """

        assets = self.data.assets
        resolved = self.resolved_bounds
        lower = resolved.lower
        upper = resolved.upper
        self.variable_lower[self.layout.weight] = lower
        self.variable_upper[self.layout.weight] = upper
        # 普通大样本空间通常只有少量来源组合；驻留其不可变映射，避免为每个资产分配并
        # 复制等价字典。
        metadata_flyweights: dict[
            tuple[tuple[str, ...], tuple[str, ...]], MappingProxyType[str, Any]
        ] = {}
        for index, asset in enumerate(assets):
            key = str(asset)
            lower_sources = resolved.lower_sources[index]
            upper_sources = resolved.upper_sources[index]
            extra_metadata = resolved.metadata[index]
            if extra_metadata is None:
                metadata_key = (lower_sources, upper_sources)
                record_metadata = metadata_flyweights.get(metadata_key)
                if record_metadata is None:
                    record_metadata = MappingProxyType(
                        {
                            "lower_sources": lower_sources,
                            "upper_sources": upper_sources,
                        }
                    )
                    metadata_flyweights[metadata_key] = record_metadata
            else:
                record_metadata = MappingProxyType(
                    {
                        "lower_sources": lower_sources,
                        "upper_sources": upper_sources,
                        **extra_metadata,
                    }
                )
            self.variables.append(VariableRecord(index, f"weight:{key}", "weight", key))
            self.variable_records.append(
                ConstraintRecord(
                    constraint_id=f"asset_bound:{key}",
                    group="asset_bound",
                    location="variable",
                    index=index,
                    key=key,
                    metadata=record_metadata,
                )
            )

        for group, indices in (
            ("turnover_aux", self.layout.turnover_aux),
            ("active_aux", self.layout.active_aux),
        ):
            if indices is None:
                continue
            self.variable_lower[indices] = 0.0
            for local_index, variable_index in enumerate(indices):
                asset_position = local_index
                if group == "turnover_aux" and self.layout.sparse_turnover:
                    assert self.layout.turnover_support is not None
                    asset_position = int(self.layout.turnover_support[local_index])
                elif group == "active_aux" and self.layout.sparse_active:
                    assert self.layout.active_support is not None
                    asset_position = int(self.layout.active_support[local_index])
                self.variables.append(
                    VariableRecord(
                        int(variable_index),
                        f"{group}:{local_index}",
                        group,
                        str(self.data.assets[asset_position]),
                    )
                )
        if self.layout.factor is not None:
            assert isinstance(self.risk_model, FactorRiskModel)
            for factor_name, variable_index in zip(
                self.risk_model.factor_names, self.layout.factor
            ):
                self.variables.append(
                    VariableRecord(
                        int(variable_index),
                        f"factor_active:{factor_name}",
                        "factor_active",
                        factor_name,
                        "exposure",
                    )
                )
        self.variables.sort(key=lambda item: item.index)

    def _pad(self, matrix: sp.spmatrix) -> sp.csc_matrix:
        matrix = matrix.tocsc()
        if matrix.shape[1] == self.layout.n_variables:
            return matrix
        if matrix.shape[1] > self.layout.n_variables:
            raise AssertionError("constraint matrix is wider than the variable layout")
        return sp.hstack(
            [
                matrix,
                sp.csc_matrix(
                    (matrix.shape[0], self.layout.n_variables - matrix.shape[1])
                ),
            ],
            format="csc",
        )

    def add_block(
        self,
        matrix: sp.spmatrix,
        lower: np.ndarray | float,
        upper: np.ndarray | float,
        *,
        group: str,
        keys: Iterable[str | None] | None = None,
        unit: str = "weight",
        source: str = "user",
        relaxable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """追加一个有界行块及其业务级审计记录。"""

        block = self._pad(matrix)
        n_rows = block.shape[0]
        lower_array = np.broadcast_to(np.asarray(lower, dtype=float), (n_rows,)).copy()
        upper_array = np.broadcast_to(np.asarray(upper, dtype=float), (n_rows,)).copy()
        key_items: list[str | None] = [None] * n_rows if keys is None else list(keys)
        if len(key_items) != n_rows:
            raise AssertionError("constraint keys do not match row count")
        row_start = sum(item.shape[0] for item in self.blocks)
        self.blocks.append(block)
        self.lowers.append(lower_array)
        self.uppers.append(upper_array)
        for local_index, key in enumerate(key_items):
            suffix = str(key) if key is not None else str(local_index)
            self.row_records.append(
                ConstraintRecord(
                    constraint_id=f"{group}:{suffix}",
                    group=group,
                    location="row",
                    index=row_start + local_index,
                    key=key,
                    unit=unit,
                    source=source,
                    relaxable=relaxable,
                    metadata={} if metadata is None else metadata,
                )
            )

    def _weight_block(self, values: np.ndarray) -> sp.csc_matrix:
        values = np.asarray(values, dtype=float).reshape(1, -1)
        return self._pad(sp.csc_matrix(values))

    def add_common_constraints(self) -> None:
        """编译所有目标类型共享的线性可行域。

        其中包括预算、因子定义、换手率、总主动权重、基准成分覆盖、风格/行业边界、
        附加属性以及可选 alpha 下限。绝对权重和逐标的主动权重边界已在此前配置为变量
        列边界，而非矩阵行。

        对只做多且满仓的初始组合，稀疏换手率表示是精确的。由于总买入等于总卖出，L1
        换手率等于买入量的两倍：原持仓支撑集上的正增量，加上支撑集之外新建的权重。
        这里不假设换手率与跟踪误差存在任何关系。
        """

        n = self.n_assets
        x = self.layout.weight
        config = self.constraints_config
        benchmark = (
            None
            if self.data.benchmark is None
            else np.asarray(self.data.benchmark, dtype=float)
        )

        self.add_block(
            self._weight_block(np.ones(n)),
            config.budget,
            config.budget,
            group="budget",
            keys=("total",),
            source="system",
            relaxable=False,
        )

        if self.layout.factor is not None:
            assert isinstance(self.risk_model, FactorRiskModel)
            assert benchmark is not None
            exposure = np.asarray(self.risk_model.exposure, dtype=float)
            rows = np.arange(len(self.layout.factor), dtype=np.int32)
            factor_identity = sp.csc_matrix(
                (-np.ones(len(rows)), (rows, self.layout.factor)),
                shape=(len(rows), self.layout.n_variables),
            )
            factor_matrix = self._pad(sp.csc_matrix(exposure.T)) + factor_identity
            # 资产数通常远大于因子数。这里使用确定性的逐元素归约，避免一次规模很小的
            # GEMV 唤醒进程级 BLAS 线程池；在受 CPU quota 限制的容器中，线程池唤醒和
            # 自旋等待可能比这次数值计算本身更昂贵。
            rhs = np.einsum("ij,i->j", exposure, benchmark, optimize=False)
            self.add_block(
                factor_matrix,
                rhs,
                rhs,
                group="factor_definition",
                keys=self.risk_model.factor_names,
                unit="exposure",
                source="compiler",
                relaxable=False,
            )

        if self.layout.turnover_aux is not None:
            assert config.turnover is not None
            assert self.data.initial_weight is not None
            initial = np.asarray(self.data.initial_weight, dtype=float)
            asset_keys = tuple(str(item) for item in self.data.assets)
            aux = self.layout.turnover_aux
            if self.layout.sparse_turnover:
                assert self.layout.turnover_support is not None
                assert self.layout.turnover_new is not None
                support = self.layout.turnover_support
                rows = np.arange(len(support), dtype=np.int32)
                positive = sp.csc_matrix(
                    (
                        np.concatenate([np.ones(len(support)), -np.ones(len(support))]),
                        (
                            np.concatenate([rows, rows]),
                            np.concatenate([support, aux]),
                        ),
                    ),
                    shape=(len(support), self.layout.n_variables),
                )
                self.add_block(
                    positive,
                    -np.inf,
                    initial[support],
                    group="turnover_epigraph_positive",
                    keys=(asset_keys[int(index)] for index in support),
                    source="compiler",
                    relaxable=False,
                )
                turnover_columns = np.concatenate([aux, self.layout.turnover_new])
                turnover_row = sp.csc_matrix(
                    (
                        np.full(len(turnover_columns), 2.0),
                        (
                            np.zeros(len(turnover_columns), dtype=np.int32),
                            turnover_columns,
                        ),
                    ),
                    shape=(1, self.layout.n_variables),
                )
            else:
                rows = np.arange(n, dtype=np.int32)
                positive = sp.csc_matrix(
                    (
                        np.concatenate([np.ones(n), -np.ones(n)]),
                        (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                    ),
                    shape=(n, self.layout.n_variables),
                )
                negative = sp.csc_matrix(
                    (
                        np.concatenate([-np.ones(n), -np.ones(n)]),
                        (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                    ),
                    shape=(n, self.layout.n_variables),
                )
                self.add_block(
                    positive,
                    -np.inf,
                    initial,
                    group="turnover_epigraph_positive",
                    keys=asset_keys,
                    source="compiler",
                    relaxable=False,
                )
                self.add_block(
                    negative,
                    -np.inf,
                    -initial,
                    group="turnover_epigraph_negative",
                    keys=asset_keys,
                    source="compiler",
                    relaxable=False,
                )
                turnover_row = sp.csc_matrix(
                    (np.ones(n), (np.zeros(n, dtype=np.int32), aux)),
                    shape=(1, self.layout.n_variables),
                )
            self.add_block(
                turnover_row,
                -np.inf,
                config.turnover.l1_limit + (config.budget - float(initial.sum()))
                if self.layout.sparse_turnover
                else config.turnover.l1_limit,
                group="turnover",
                keys=("l1",),
                metadata={"expression_offset": float(initial.sum()) - config.budget}
                if self.layout.sparse_turnover
                else None,
            )

        if self.layout.active_aux is not None and self.layout.sparse_active:
            self._add_sparse_total_active()
        elif self.layout.active_aux is not None:
            assert config.total_active is not None
            assert benchmark is not None
            aux = self.layout.active_aux
            rows = np.arange(n, dtype=np.int32)
            positive = sp.csc_matrix(
                (
                    np.concatenate([np.ones(n), -np.ones(n)]),
                    (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                ),
                shape=(n, self.layout.n_variables),
            )
            negative = sp.csc_matrix(
                (
                    np.concatenate([-np.ones(n), -np.ones(n)]),
                    (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                ),
                shape=(n, self.layout.n_variables),
            )
            asset_keys = tuple(str(item) for item in self.data.assets)
            self.add_block(
                positive,
                -np.inf,
                benchmark,
                group="total_active_epigraph_positive",
                keys=asset_keys,
                source="compiler",
                relaxable=False,
            )
            self.add_block(
                negative,
                -np.inf,
                -benchmark,
                group="total_active_epigraph_negative",
                keys=asset_keys,
                source="compiler",
                relaxable=False,
            )
            total_active_row = sp.csc_matrix(
                (np.ones(n), (np.zeros(n, dtype=np.int32), aux)),
                shape=(1, self.layout.n_variables),
            )
            self.add_block(
                total_active_row,
                -np.inf,
                float(config.total_active),
                group="total_active",
                keys=("l1",),
            )

        if config.benchmark_member_weight is not None:
            assert benchmark is not None
            turnover = config.turnover
            current_weight = self.data.initial_weight
            member = benchmark > 0.0
            redundant = bool(
                self.simplify
                and turnover is not None
                and current_weight is not None
                and float(np.asarray(current_weight)[member].sum())
                - (
                    turnover.l1_limit
                    - (config.budget - float(np.asarray(current_weight).sum()))
                )
                / 2.0
                >= config.benchmark_member_weight.value
            )
            if redundant:
                self.optimizations.append("omit_redundant_benchmark_member_weight")
            else:
                self.add_block(
                    self._weight_block(member.astype(float)),
                    config.benchmark_member_weight.value,
                    np.inf,
                    group="benchmark_member_weight",
                    keys=("members",),
                )

        self._add_factor_bounds("style", config.style)
        self._add_factor_bounds("industry", config.industry)

        if benchmark is not None:
            for key, pair in config.extra_active.items():
                values = np.asarray(self.data.extra_attributes[key], dtype=float)
                shift = float(np.einsum("i,i->", values, benchmark, optimize=False))
                self.add_block(
                    self._weight_block(values),
                    pair[0] + shift,
                    pair[1] + shift,
                    group="extra_active",
                    keys=(key,),
                    unit="exposure",
                )
        for key, pair in config.extra_absolute.items():
            values = np.asarray(self.data.extra_attributes[key], dtype=float)
            self.add_block(
                self._weight_block(values),
                pair[0],
                pair[1],
                group="extra_absolute",
                keys=(key,),
                unit="exposure",
            )

        objective = self.problem.objective
        if (
            isinstance(objective, MinimizeTrackingError)
            and objective.alpha_floor is not None
        ):
            assert self.data.alpha is not None
            self.add_block(
                self._weight_block(np.asarray(self.data.alpha, dtype=float)),
                objective.alpha_floor,
                np.inf,
                group="alpha_floor",
                keys=("alpha",),
                unit=self.data.alpha_spec.units
                if self.data.alpha_spec is not None
                else "alpha",
            )

    def _add_sparse_total_active(self) -> None:
        r"""直接生成低配单边表达，不建立完整 epigraph 再裁剪。

        $\|w-b\|_1=B-\sum b+2\sum_i(b_i-w_i)_+$。区间确定高配的项为零，
        确定低配的项直接使用 $b_i-w_i$，仅跨基准项引入 $s_i\ge b_i-w_i$、
        $s_i\ge0$。即使没有辅助变量，也保留总量行及其常数，不误删不可行约束。
        """
        layout = self.layout
        assert layout.active_aux is not None
        assert layout.active_support is not None and layout.active_negative is not None
        assert self.data.benchmark is not None
        assert self.constraints_config.total_active is not None
        benchmark = np.asarray(self.data.benchmark, dtype=float)
        support, negative, aux = (
            layout.active_support,
            layout.active_negative,
            layout.active_aux,
        )
        size = len(support)
        if size:
            rows = np.arange(size, dtype=np.int32)
            self.add_block(
                sp.csc_matrix(
                    (
                        -np.ones(2 * size),
                        (np.r_[rows, rows], np.r_[layout.weight[support], aux]),
                    ),
                    shape=(size, layout.n_variables),
                ),
                -np.inf,
                -benchmark[support],
                group="total_active_epigraph_negative",
                keys=(str(self.data.assets[i]) for i in support),
                source="compiler",
                relaxable=False,
            )
        offset = float(
            self.constraints_config.budget
            - benchmark.sum()
            + 2.0 * benchmark[negative].sum()
        )
        self.add_block(
            sp.csc_matrix(
                (
                    np.r_[np.full(size, 2.0), np.full(len(negative), -2.0)],
                    (
                        np.zeros(size + len(negative), dtype=np.int32),
                        np.r_[aux, layout.weight[negative]],
                    ),
                ),
                shape=(1, layout.n_variables),
            ),
            -np.inf,
            self.constraints_config.total_active - offset,
            group="total_active",
            keys=("l1",),
            metadata={"expression_offset": offset, "expression": "sparse_deficit_l1"},
        )

    def _add_factor_bounds(self, expected_type: str, bounds: Any) -> None:
        r"""编译风格或行业主动暴露边界。

        QP 已包含 $f=E^{\mathsf T}(x-b)$ 变量，因此因子边界只是 $f_j$ 上的一个稀疏
        系数。LP/factor-QCQP 基础域尚不包含 $f$，所以使用 $E_j^{\mathsf T}x$ 与按基准
        平移的边界。factor-QCQP 的共享锥建模随后引入因子变量时，会替换这些稠密行。
        """

        if bounds is None:
            return
        assert isinstance(self.risk_model, FactorRiskModel)
        assert self.data.benchmark is not None
        names = self.risk_model.factor_names
        factor_types = self.risk_model.factor_types
        # 边界已由前置校验保证为二元组；此处只按 Python 序列读取。
        # 避免 Cython 将固定长度 tuple 转成 C 结构体，在异常转换路径生成
        # 未完全初始化的返回值；这里无需额外的二元组数值转换。
        selected: dict[int, Sequence[float]] = {}
        if bounds.default is not None:
            selected.update(
                {
                    index: bounds.default
                    for index, factor_type in enumerate(factor_types)
                    if factor_type.lower() == expected_type
                }
            )
        lookup = {name.lower(): index for index, name in enumerate(names)}
        for name, pair in bounds.overrides.items():
            selected[lookup[name.lower()]] = pair
        if not selected:
            return
        indices = np.fromiter(sorted(selected), dtype=np.int32)
        lower = np.asarray([selected[int(i)][0] for i in indices])
        upper = np.asarray([selected[int(i)][1] for i in indices])
        if self.layout.factor is not None:
            # 因子变量已经等于 $E^{\mathsf T}(x-b)$，因此其边界是稀疏单列行；若在此重复
            # $E^{\mathsf T}$，会在 PIQP 的 KKT 矩阵中复制稠密暴露块。
            rows = np.arange(len(indices), dtype=np.int32)
            matrix = sp.csc_matrix(
                (
                    np.ones(len(indices)),
                    (rows, self.layout.factor[indices]),
                ),
                shape=(len(indices), self.layout.n_variables),
            )
        else:
            exposure = np.asarray(self.risk_model.exposure, dtype=float)[:, indices]
            benchmark_shift = np.einsum(
                "ij,i->j",
                exposure,
                np.asarray(self.data.benchmark, dtype=float),
                optimize=False,
            )
            lower = lower + benchmark_shift
            upper = upper + benchmark_shift
            matrix = self._pad(sp.csc_matrix(exposure.T))
        self.add_block(
            matrix,
            lower,
            upper,
            group=expected_type,
            keys=(names[int(i)] for i in indices),
            unit="active_exposure",
        )

    def finish(self) -> LinearDomain:
        """将累计约束块冻结为已排序的 canonical CSC 数组。"""

        if self.blocks:
            matrix = sp.vstack(self.blocks, format="csc")
            lower = np.concatenate(self.lowers)
            upper = np.concatenate(self.uppers)
        else:
            matrix = sp.csc_matrix((0, self.layout.n_variables), dtype=float)
            lower = np.empty(0, dtype=float)
            upper = np.empty(0, dtype=float)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return LinearDomain(
            A=matrix,
            lower=lower,
            upper=upper,
            variable_lower=self.variable_lower,
            variable_upper=self.variable_upper,
            variables=tuple(self.variables),
            constraints=tuple(self.variable_records + self.row_records),
            weight_indices=self.layout.weight,
            assets=tuple(self.data.assets),
        )


def _risk_operator(
    problem: PortfolioProblem, domain: LinearDomain
) -> FactorRiskOperator:
    """构造年化“因子加特异”跟踪风险算子。"""

    risk_model = problem.data.risk_model
    if isinstance(risk_model, FullCovarianceRiskModel):
        raise CanonicalCompilationError(
            "full-covariance risk objectives are not implemented in v1"
        )
    if not isinstance(risk_model, FactorRiskModel):
        raise CanonicalCompilationError("a factor risk model is required")
    if problem.data.benchmark is None:
        raise CanonicalCompilationError("tracking risk requires a benchmark")
    covariance = np.asarray(risk_model.covariance, dtype=float)
    covariance = (covariance + covariance.T) * 0.5
    return FactorRiskOperator(
        exposure=np.asarray(risk_model.exposure, dtype=float),
        covariance=covariance,
        specific_volatility=np.asarray(risk_model.specific_volatility, dtype=float),
        benchmark=np.asarray(problem.data.benchmark, dtype=float),
        factor_names=risk_model.factor_names,
        weight_indices=domain.weight_indices,
    )


def _diagnostic_domain(problem: PortfolioProblem) -> LinearDomain:
    """构造可放宽的完整线性域，禁用依赖原约束的简化；不影响正常优化热路径。

    完整 L1 换手率及显式总主动/基准覆盖约束必须保留，避免放宽前提后改变数学含义。
    非负底线和具有操作指令的资产有效边界作为 Phase-I 硬约束，常规正下限仍可向零放宽。
    """

    builder = _DomainBuilder(problem, include_factor_variables=False, simplify=False)
    builder.add_common_constraints()
    domain = builder.finish()
    records = []
    for record in domain.constraints:
        if record.location == "variable" and record.group == "asset_bound":
            metadata = dict(record.metadata)
            if problem.constraints.long_only:
                metadata["diagnostic_hard_lower"] = 0.0
            operational = bool(metadata.get("operational_instruction"))
            records.append(
                replace(record, relaxable=not operational, metadata=metadata)
            )
        else:
            records.append(record)
    return replace(domain, constraints=tuple(records))


def _compile_lp(problem: PortfolioProblem) -> tuple[LinearProgram, tuple[str, ...]]:
    r"""将 $\max_x\alpha^{\mathsf T}x$ 编译为最小化向量 $c$。"""

    builder = _DomainBuilder(problem, include_factor_variables=False)
    builder.add_common_constraints()
    domain = builder.finish()
    assert problem.data.alpha is not None
    c = np.zeros(domain.n_variables, dtype=float)
    c[domain.weight_indices] = -np.asarray(problem.data.alpha, dtype=float)
    return LinearProgram(CanonicalKind.LP, domain, c), tuple(builder.optimizations)


def _compile_qp(problem: PortfolioProblem) -> tuple[QuadraticProgram, tuple[str, ...]]:
    """编译“低秩因子加对角特异风险”的凸 QP。

    因子敞口使用显式 ``f`` 变量，使二次矩阵保持为资产对角块加小型因子块。风险调整 alpha
    目标会平移 ``alpha @ benchmark``；预算等式使该平移只改变常数，同时让 PIQP 目标缩放不受
    alpha 任意公共平移影响。
    """

    if not isinstance(problem.data.risk_model, FactorRiskModel):
        raise CanonicalCompilationError("v1 QP compiler supports FactorRiskModel only")
    if problem.data.benchmark is None:
        raise CanonicalCompilationError("tracking-risk QP requires a benchmark")
    builder = _DomainBuilder(problem, include_factor_variables=True)
    builder.add_common_constraints()
    domain = builder.finish()
    assert builder.layout.factor is not None
    risk = _risk_operator(problem, domain)
    objective = problem.objective
    if isinstance(objective, RiskAdjustedAlpha):
        factor_aversion = objective.factor_aversion
        specific_aversion = objective.specific_aversion
    elif isinstance(objective, MinimizeTrackingError):
        factor_aversion = 1.0
        specific_aversion = 1.0
    else:
        raise CanonicalCompilationError(
            f"unsupported QP objective: {type(objective).__name__}"
        )

    n_variables = domain.n_variables
    specific_variance = np.square(risk.specific_volatility)
    diagonal_rows = domain.weight_indices
    diagonal_values = 2.0 * specific_aversion * specific_variance
    factor_matrix = 2.0 * factor_aversion * risk.covariance
    factor_rows, factor_columns = np.nonzero(factor_matrix)
    P = sp.csc_matrix(
        (
            np.concatenate(
                [diagonal_values, factor_matrix[factor_rows, factor_columns]]
            ),
            (
                np.concatenate([diagonal_rows, builder.layout.factor[factor_rows]]),
                np.concatenate([diagonal_rows, builder.layout.factor[factor_columns]]),
            ),
        ),
        shape=(n_variables, n_variables),
    )
    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    q = np.zeros(n_variables, dtype=float)
    q[domain.weight_indices] = (
        -2.0 * specific_aversion * specific_variance * risk.benchmark
    )
    objective_scale_reference = None
    alpha_shift = 0.0
    if isinstance(objective, RiskAdjustedAlpha):
        assert problem.data.alpha is not None
        alpha = np.asarray(problem.data.alpha, dtype=float)
        # 预算等式使 alpha 平移在数学上只改变常数。以基准为中心，使 PIQP 缩放取决于 alpha
        # 离散程度，而不是风险线性项或任意公共水平。
        alpha_shift = float(
            np.einsum("i,i->", alpha, risk.benchmark, optimize=False)
        )
        centered_alpha = alpha - alpha_shift
        q[domain.weight_indices] -= centered_alpha
        objective_scale_reference = float(np.max(np.abs(centered_alpha)))
    offset = float(
        specific_aversion * np.sum(specific_variance * np.square(risk.benchmark))
        - alpha_shift * problem.constraints.budget
    )
    risk_limit = (
        problem.constraints.tracking_error.annualized
        if problem.constraints.tracking_error is not None
        else None
    )
    return (
        QuadraticProgram(
            kind=CanonicalKind.QP,
            domain=domain,
            P=P,
            q=q,
            objective_offset=offset,
            objective_scale_reference=objective_scale_reference,
            risk_operator=risk,
            risk_limit=risk_limit,
        ),
        tuple(builder.optimizations),
    )


def _compile_factor_qcqp(
    problem: PortfolioProblem,
) -> tuple[FactorQCQP, tuple[str, ...]]:
    """为单一 TE 预算编译线性域和因子风险算子。"""

    if not isinstance(problem.data.risk_model, FactorRiskModel):
        raise CanonicalCompilationError("factor-QCQP requires FactorRiskModel")
    assert problem.constraints.tracking_error is not None
    assert problem.data.alpha is not None
    builder = _DomainBuilder(problem, include_factor_variables=False)
    builder.add_common_constraints()
    domain = builder.finish()
    model = FactorQCQP(
        kind=CanonicalKind.FACTOR_QCQP,
        domain=domain,
        alpha=np.asarray(problem.data.alpha, dtype=float),
        risk_operator=_risk_operator(problem, domain),
        risk_limit=problem.constraints.tracking_error.annualized,
    )
    return model, tuple(builder.optimizations)


def compile_problem(
    problem: PortfolioProblem, *, validate: bool = True
) -> CompiledProblem:
    """在不导入求解器后端的情况下校验并编译一个问题。

    ``validate=False`` 只供刚刚生成等价校验报告的调用方使用，例如
    :meth:`PortfolioOptimizer.prepare`。返回 fingerprint 同时覆盖业务语义输入和最终数值
    payload，使所有回退尝试可被审计为同一个数学问题。

    Parameters
    ----------
    problem : PortfolioProblem
        待编译的单期业务问题。
    validate : bool
        是否先运行完整静态校验。仅在调用方已经校验同一不可变问题时才应设为假。

    Returns
    -------
    CompiledProblem
        canonical 模型、问题 fingerprint 和已应用的精确结构优化。

    Raises
    ------
    PortfolioValidationError
        ``validate=True`` 且问题存在静态错误。
    CanonicalCompilationError
        数学结构当前无法表示，例如风险调整目标又叠加 TE 预算。
    """

    if validate:
        report = validate_problem(problem)
        report.raise_for_errors()
    kind = classify_problem(problem)
    model: CanonicalModel
    if kind is ProblemKind.LP:
        model, optimizations = _compile_lp(problem)
    elif kind is ProblemKind.QP:
        model, optimizations = _compile_qp(problem)
    elif kind is ProblemKind.FACTOR_QCQP:
        model, optimizations = _compile_factor_qcqp(problem)
    else:
        raise CanonicalCompilationError(
            "risk-adjusted objectives with an additional TE constraint require the conic fallback, "
            "which is not implemented in the first compiler milestone"
        )
    return CompiledProblem(
        model=model,
        fingerprint=fingerprint(problem, model),
        compiler_optimizations=optimizations,
    )

"""canonical 编译前使用的显式数据对齐策略。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ._reindex import _reindex_rows


class DataAlignmentError(ValueError):
    """输入表无法在不改变既定语义的前提下完成对齐时抛出。"""


class BenchmarkCoverageError(DataAlignmentError):
    """优化样本空间未完整覆盖给定基准时抛出。"""


@dataclass(frozen=True)
class BenchmarkCoveragePolicy:
    """定义如何处理优化样本空间之外的基准权重。

    默认值有意设为 ``error``。仅当调用方显式选择
    ``renormalize_within_tolerance``，且缺失权重不超过独立业务阈值时，才允许归一化。

    Attributes
    ----------
    action : str
        ``"error"`` 拒绝任何实质性基准权重缺口；
        ``"renormalize_within_tolerance"`` 仅允许在
        ``missing_mass_tolerance`` 范围内显式归一化。
    missing_mass_tolerance : float
        显式启用归一化时，允许位于优化样本空间之外的最大基准总权重，采用小数权重单位。
    weight_sum_tolerance : float
        检查原始基准权重和是否为一，以及将缺失权重视为数值零时使用的绝对容差。
    """

    action: str = "error"
    missing_mass_tolerance: float = 0.0
    weight_sum_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.action not in {"error", "renormalize_within_tolerance"}:
            raise ValueError(
                "benchmark coverage action must be 'error' or "
                "'renormalize_within_tolerance'"
            )
        for name, value in (
            ("missing_mass_tolerance", self.missing_mass_tolerance),
            ("weight_sum_tolerance", self.weight_sum_tolerance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class AlignedBenchmark:
    """已对齐至优化器权威资产坐标的基准。

    Attributes
    ----------
    values : numpy.ndarray
        严格按请求资产顺序排列的基准权重，形状为 ``(n_assets,)``；执行获准的归一化后
        权重和为一。
    missing_mass : float
        所有缺失资产的原始基准权重之和。
    missing_assets : tuple[str, ...]
        缺失基准成分股的字符串标签。
    maximum_missing_weight : float
        缺失资产中的最大原始成分权重；没有缺失时为零。
    renormalization_factor : float
        应用于保留权重的乘数；未执行归一化时为一。
    """

    values: np.ndarray
    missing_mass: float
    missing_assets: tuple[str, ...]
    maximum_missing_weight: float
    renormalization_factor: float


def align_benchmark(
    benchmark: pd.Series,
    assets: pd.Index,
    policy: BenchmarkCoveragePolicy | None = None,
) -> AlignedBenchmark:
    """对齐一个严格同日基准，并审计全部缺失权重。

    Parameters
    ----------
    benchmark : pandas.Series
        以唯一资产标识为索引的严格同日非负基准权重。在执行覆盖处理前，原始序列之和
        必须为一。
    assets : pandas.Index
        优化问题采用的权威资产顺序。
    policy : BenchmarkCoveragePolicy | None
        显式缺失处理策略；``None`` 使用严格默认策略。

    Returns
    -------
    AlignedBenchmark
        位置化基准权重及覆盖率审计信息。

    Raises
    ------
    TypeError
        ``benchmark`` 不是 :class:`pandas.Series` 时抛出。
    DataAlignmentError
        标签或权重重复、权重非有限或为负，或权重和未在策略容差内等于一时抛出。
    BenchmarkCoverageError
        禁止缺失、缺失权重超过显式阈值，或没有保留任何正基准权重时抛出。
    """

    return _align_benchmark(benchmark, assets, policy)


def _align_benchmark(benchmark, assets, policy=None, *, aligned_values=None):
    """共同执行原始基准审计；批量路径可传入已对齐的日切片，避免再次 reindex。"""
    policy = BenchmarkCoveragePolicy() if policy is None else policy
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series indexed by sid")
    if benchmark.index.has_duplicates:
        duplicate = benchmark.index[benchmark.index.duplicated()].unique().tolist()[:10]
        raise DataAlignmentError(f"benchmark contains duplicate assets: {duplicate}")
    numeric = pd.to_numeric(benchmark, errors="coerce").astype(float)
    values = numeric.to_numpy(copy=False)
    if not np.all(np.isfinite(values)):
        raise DataAlignmentError("benchmark contains NaN or infinity")
    if np.any(values < 0.0):
        raise DataAlignmentError("benchmark contains negative weights")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=policy.weight_sum_tolerance):
        raise DataAlignmentError(
            f"raw benchmark must sum to one before coverage handling; observed {total:.12g}"
        )

    inside = benchmark.index.isin(assets)
    omitted = numeric.loc[~inside]
    missing_mass = float(omitted.sum())
    missing_assets = tuple(str(asset) for asset in omitted.index)
    maximum_missing = float(omitted.max()) if len(omitted) else 0.0
    if missing_mass > policy.weight_sum_tolerance:
        if policy.action == "error":
            raise BenchmarkCoverageError(
                f"optimization universe omits {missing_mass:.6%} benchmark mass "
                f"across {len(omitted)} asset(s); explicit renormalization is disabled"
            )
        if missing_mass > policy.missing_mass_tolerance + policy.weight_sum_tolerance:
            raise BenchmarkCoverageError(
                f"optimization universe omits {missing_mass:.6%} benchmark mass, above "
                f"the allowed {policy.missing_mass_tolerance:.6%}"
            )

    aligned = (
        _reindex_rows(numeric, assets, fill_value=0.0).to_numpy(float)
        if aligned_values is None else aligned_values
    )
    retained = float(aligned.sum())
    if retained <= 0.0:
        raise BenchmarkCoverageError("optimization universe contains no benchmark weight")
    factor = 1.0
    if not np.isclose(retained, 1.0, rtol=0.0, atol=policy.weight_sum_tolerance):
        # 到达此分支表示上方已经确认调用方显式允许归一化。
        factor = 1.0 / retained
        aligned = aligned * factor
    return AlignedBenchmark(
        values=aligned,
        missing_mass=missing_mass,
        missing_assets=missing_assets,
        maximum_missing_weight=maximum_missing,
        renormalization_factor=factor,
    )

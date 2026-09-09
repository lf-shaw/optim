"""合并常规组合边界与仅对本次优化生效的逐资产交易指令。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

from ..portfolio_types import PortfolioProblem


class AssetBoundsError(ValueError):
    """已完成维度解析的逐资产边界无法安全合并时抛出。"""


@dataclass(frozen=True)
class ResolvedAssetBounds:
    """逐资产最终权重区间及其审计来源。

    所有数组与元组均严格按 ``problem.data.assets`` 的位置坐标排列。本对象在合并
    绝对权重、主动权重限制和单期交易指令后生成。

    Attributes
    ----------
    lower : numpy.ndarray
        目标权重的闭区间下界，形状为 ``(n_assets,)``。
    upper : numpy.ndarray
        目标权重的闭区间上界，形状为 ``(n_assets,)``。
    lower_sources : tuple[tuple[str, ...], ...]
        共同形成各资产最终下界的约束标识。外层元组按资产坐标排列。
    upper_sources : tuple[tuple[str, ...], ...]
        共同形成各资产最终上界的约束标识。外层元组按资产坐标排列。
    metadata : tuple[Mapping[str, Any] | None, ...]
        逐资产操作指令的审计信息。普通资产使用 ``None``，避免每次调仓分配数千个
        空字典；受操作约束的资产记录指令、原始区间以及是否发生了局部放宽。
    """

    lower: np.ndarray
    upper: np.ndarray
    lower_sources: tuple[tuple[str, ...], ...]
    upper_sources: tuple[tuple[str, ...], ...]
    # 普通资产没有逐标的元数据。使用 ``None`` 可避免每次调仓创建数千个空字典；
    # 下方仅为受到操作约束的资产创建独立映射。
    metadata: tuple[Mapping[str, Any] | None, ...]


def _bound_array(value: float | np.ndarray, n_assets: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_assets, float(array), dtype=float)
    if array.shape != (n_assets,):
        raise AssetBoundsError(
            f"asset bound expected shape {(n_assets,)}, got {array.shape}"
        )
    return array.copy()


def _override_pair(value: Any) -> tuple[float, float]:
    if np.isscalar(value):
        target = float(cast(Any, value))
        return target, target
    try:
        lower, upper = value
    except (TypeError, ValueError) as exc:
        raise AssetBoundsError(
            "weight override must be a scalar or a (lower, upper) pair"
        ) from exc
    return float(lower), float(upper)


def resolve_asset_bounds(problem: PortfolioProblem) -> ResolvedAssetBounds:
    """应用操作指令并返回逐资产最终边界。

    硬交易指令有意优先于该资产的常规主动权重区间。这样既能保留停牌、强制剔除和
    冻结持仓所需的历史行为，又不会全局放宽其他资产的主动权重约束。

    Parameters
    ----------
    problem : PortfolioProblem
        已校验的单期问题，其中所有数组共享同一资产位置坐标。

    Returns
    -------
    ResolvedAssetBounds
        最终闭区间上下界以及逐标的约束来源。

    Raises
    ------
    AssetBoundsError
        当边界形状错误、指令引用未知资产、缺少所需初始权重，或合并指令后某资产
        区间为空时抛出。
    """

    data = problem.data
    config = problem.constraints
    assets = data.assets
    n_assets = len(assets)
    raw_lower = _bound_array(config.asset_weight.lower, n_assets)
    base_lower = raw_lower.copy()
    base_upper = _bound_array(config.asset_weight.upper, n_assets)
    if config.long_only:
        base_lower = np.maximum(base_lower, 0.0)

    lower = base_lower.copy()
    upper = base_upper.copy()
    base_source = ("asset_weight",)
    # 元组不可变，因此所有资产可以安全共享公共来源元组；仅在其他约束改变最终边界时，
    # 才为少数对应资产创建新元组。
    lower_sources: list[tuple[str, ...]] = [base_source] * n_assets
    upper_sources: list[tuple[str, ...]] = [base_source] * n_assets
    metadata: list[Mapping[str, Any] | None] = [None] * n_assets
    if config.long_only:
        for index in np.flatnonzero(raw_lower < 0.0):
            position = int(index)
            lower_sources[position] += ("long_only",)

    benchmark = data.benchmark
    if config.active_weight is not None:
        if benchmark is None:
            raise AssetBoundsError("active bounds require a benchmark")
        active = float(config.active_weight.absolute)
        active_lower = np.asarray(benchmark, dtype=float) - active
        active_upper = np.asarray(benchmark, dtype=float) + active
        changed_lower = active_lower > lower
        changed_upper = active_upper < upper
        lower = np.maximum(lower, active_lower)
        upper = np.minimum(upper, active_upper)
        for index in np.flatnonzero(changed_lower):
            position = int(index)
            lower_sources[position] += ("active_weight",)
        for index in np.flatnonzero(changed_upper):
            position = int(index)
            upper_sources[position] += ("active_weight",)

    instructions = config.asset_trade

    if instructions is not None and not instructions.is_empty:
        positions = {asset: index for index, asset in enumerate(assets)}

        def locate(asset: Any) -> int | None:
            try:
                return positions.get(asset)
            except TypeError as exc:
                raise AssetBoundsError(
                    f"unhashable asset identifier {asset!r}"
                ) from exc

        for asset in instructions.blacklist:
            index = locate(asset)
            if index is None:
                if instructions.missing_asset == "ignore":
                    continue
                raise AssetBoundsError(f"unknown blacklist asset {asset!r}")
            metadata[index] = dict(
                operational_instruction="blacklist",
                ordinary_lower=float(lower[index]),
                ordinary_upper=float(upper[index]),
                ordinary_bound_relaxed=not (
                    lower[index] - 1e-14 <= 0.0 <= upper[index] + 1e-14
                ),
            )
            lower[index] = upper[index] = 0.0
            lower_sources[index] = ("asset_trade.blacklist",)
            upper_sources[index] = ("asset_trade.blacklist",)

        initial = (
            None
            if data.initial_weight is None
            else np.asarray(data.initial_weight, dtype=float)
        )
        for asset in instructions.frozen:
            index = locate(asset)
            if index is None:
                if instructions.missing_asset == "ignore":
                    continue
                raise AssetBoundsError(f"unknown frozen asset {asset!r}")
            if initial is None:
                raise AssetBoundsError("frozen assets require initial weights")
            target = float(initial[index])
            metadata[index] = dict(
                operational_instruction="frozen",
                ordinary_lower=float(lower[index]),
                ordinary_upper=float(upper[index]),
                ordinary_bound_relaxed=not (
                    lower[index] - 1e-14 <= target <= upper[index] + 1e-14
                ),
            )
            lower[index] = upper[index] = target
            lower_sources[index] = ("asset_trade.frozen",)
            upper_sources[index] = ("asset_trade.frozen",)

        for asset in instructions.not_buyable:
            index = locate(asset)
            if index is None:
                if instructions.missing_asset == "ignore":
                    continue
                raise AssetBoundsError(f"unknown not-buyable asset {asset!r}")
            if initial is None:
                raise AssetBoundsError("not-buyable assets require initial weights")
            ordinary_lower = float(lower[index])
            current = float(initial[index])
            # 主动权重下界不得迫使买入；绝对权重和只做多边界仍然有效，并可暴露真实的
            # 模型冲突。
            if lower[index] > current:
                lower[index] = base_lower[index]
            upper[index] = min(float(upper[index]), current)
            lower_sources[index] += ("asset_trade.not_buyable",)
            upper_sources[index] += ("asset_trade.not_buyable",)
            metadata[index] = dict(
                operational_instruction="not_buyable",
                ordinary_lower=ordinary_lower,
                initial_weight=current,
                ordinary_bound_relaxed=lower[index] < ordinary_lower - 1e-14,
            )

        for asset in instructions.not_sellable:
            index = locate(asset)
            if index is None:
                if instructions.missing_asset == "ignore":
                    continue
                raise AssetBoundsError(f"unknown not-sellable asset {asset!r}")
            if initial is None:
                raise AssetBoundsError("not-sellable assets require initial weights")
            ordinary_upper = float(upper[index])
            current = float(initial[index])
            lower[index] = max(float(lower[index]), current)
            # 主动权重上界不得迫使卖出；除非当前冻结持仓已经超过绝对上界，否则绝对上界
            # 仍然有效。
            if upper[index] < current:
                upper[index] = max(float(base_upper[index]), current)
            lower_sources[index] += ("asset_trade.not_sellable",)
            upper_sources[index] += ("asset_trade.not_sellable",)
            metadata[index] = dict(
                operational_instruction="not_sellable",
                ordinary_upper=ordinary_upper,
                initial_weight=current,
                ordinary_bound_relaxed=upper[index] > ordinary_upper + 1e-14,
            )

        for asset, value in instructions.weight_overrides.items():
            index = locate(asset)
            if index is None:
                if instructions.missing_asset == "ignore":
                    continue
                raise AssetBoundsError(f"unknown weight-override asset {asset!r}")
            requested_lower, requested_upper = _override_pair(value)
            metadata[index] = dict(
                operational_instruction="weight_override",
                ordinary_lower=float(lower[index]),
                ordinary_upper=float(upper[index]),
                requested_lower=requested_lower,
                requested_upper=requested_upper,
                ordinary_bound_relaxed=(
                    requested_lower < lower[index] - 1e-14
                    or requested_upper > upper[index] + 1e-14
                ),
            )
            # 显式目标区间不受逐资产主动权重边界限制，但仍须位于绝对权重和只做多区间内。
            lower[index] = max(float(base_lower[index]), requested_lower)
            upper[index] = min(float(base_upper[index]), requested_upper)
            lower_sources[index] = (
                "asset_weight",
                "asset_trade.weight_overrides",
            )
            upper_sources[index] = (
                "asset_weight",
                "asset_trade.weight_overrides",
            )

    if config.freeze_nontradable:
        tradable = np.asarray(data.tradable, dtype=bool)
        frozen_positions = np.flatnonzero(~tradable)
        if frozen_positions.size:
            if data.initial_weight is None:
                raise AssetBoundsError(
                    "freeze_nontradable requires initial weights when an asset is non-tradable"
                )
            initial = np.asarray(data.initial_weight, dtype=float)
            for raw_index in frozen_positions:
                index = int(raw_index)
                target = float(initial[index])
                existing_metadata = metadata[index]
                instruction = (
                    None
                    if existing_metadata is None
                    else existing_metadata.get("operational_instruction")
                )
                if instruction is not None and not (
                    lower[index] - 1e-14 <= target <= upper[index] + 1e-14
                ):
                    raise AssetBoundsError(
                        f"non-tradable asset {assets[index]!r} conflicts with "
                        f"asset_trade.{instruction}"
                    )
                merged_metadata = (
                    {} if existing_metadata is None else dict(existing_metadata)
                )
                merged_metadata.update(
                    operational_instruction="freeze_nontradable",
                    ordinary_lower=float(lower[index]),
                    ordinary_upper=float(upper[index]),
                    initial_weight=target,
                )
                metadata[index] = merged_metadata
                lower[index] = upper[index] = target
                lower_sources[index] = ("freeze_nontradable",)
                upper_sources[index] = ("freeze_nontradable",)

    conflict = lower > upper + 1e-14
    if np.any(conflict):
        labels = [str(assets[index]) for index in np.flatnonzero(conflict)[:10]]
        raise AssetBoundsError(
            "combined asset bounds are infeasible for assets " + ", ".join(labels)
        )
    return ResolvedAssetBounds(
        lower=lower,
        upper=upper,
        lower_sources=tuple(lower_sources),
        upper_sources=tuple(upper_sources),
        metadata=tuple(metadata),
    )

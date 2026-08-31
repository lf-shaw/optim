"""Resolve ordinary portfolio bounds and one-off asset trading instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

from .portfolio_types import PortfolioProblem


class AssetBoundsError(ValueError):
    """Raised when already-shaped asset bounds cannot be combined safely."""


@dataclass(frozen=True)
class ResolvedAssetBounds:
    lower: np.ndarray
    upper: np.ndarray
    lower_sources: tuple[tuple[str, ...], ...]
    upper_sources: tuple[tuple[str, ...], ...]
    # Ordinary assets have no per-name metadata.  ``None`` avoids allocating
    # thousands of empty dictionaries on every rebalance; operationally
    # constrained assets receive their own mapping below.
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
    """Return effective bounds with operational instructions applied.

    The resolver deliberately gives hard trading instructions priority over an
    asset's ordinary active-weight interval.  This preserves the useful legacy
    behavior for suspensions, forced exclusions, and frozen holdings without
    globally relaxing the active-weight constraint for other assets.
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
    # Tuples are immutable, so the common source tuple can safely be shared by
    # all assets and copied only for the few entries whose effective bound is
    # changed by another constraint.
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
            # An active lower bound must not force a purchase. Absolute/long-only
            # bounds remain effective and can expose a genuine model conflict.
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
            # An active upper bound must not force a sale. The absolute upper
            # bound still applies unless the current frozen position exceeds it.
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
            # Explicit target intervals are exempt from per-asset active bounds,
            # but remain inside the absolute/long-only asset domain.
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

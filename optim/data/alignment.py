"""Explicit alignment policies used before canonical compilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataAlignmentError(ValueError):
    """Input tables cannot be aligned without changing their stated meaning."""


class BenchmarkCoverageError(DataAlignmentError):
    """The optimization universe does not cover the supplied benchmark."""


@dataclass(frozen=True)
class BenchmarkCoveragePolicy:
    """How benchmark mass outside the optimization universe is handled.

    ``error`` is deliberately the default.  Renormalization is only permitted
    when the caller explicitly selects ``renormalize_within_tolerance`` and the
    missing mass does not exceed the independent business tolerance.
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
    """Align one exact-date benchmark and audit all omitted weight mass."""

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

    aligned = numeric.reindex(assets, fill_value=0.0).to_numpy(float)
    retained = float(aligned.sum())
    if retained <= 0.0:
        raise BenchmarkCoverageError("optimization universe contains no benchmark weight")
    factor = 1.0
    if not np.isclose(retained, 1.0, rtol=0.0, atol=policy.weight_sum_tolerance):
        # Reaching this branch implies explicit permission above.
        factor = 1.0 / retained
        aligned *= factor
    return AlignedBenchmark(
        values=aligned,
        missing_mass=missing_mass,
        missing_assets=missing_assets,
        maximum_missing_weight=maximum_missing,
        renormalization_factor=factor,
    )

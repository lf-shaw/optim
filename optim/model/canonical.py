"""Typed immutable numerical payload shared by all solver backends.

Canonical objects contain no business-side dataframe alignment and no native
solver instances.  Their arrays are the single source of truth for backend
calls, independent validation, fingerprints and infeasibility diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
import scipy.sparse as sp

from ..portfolio_types import ProblemFingerprint, ProblemKind


@dataclass(frozen=True)
class VariableRecord:
    index: int
    variable_id: str
    group: str
    key: str | None = None
    unit: str = "weight"


@dataclass(frozen=True)
class ConstraintRecord:
    constraint_id: str
    group: str
    location: str
    index: int
    key: str | None = None
    unit: str = "weight"
    source: str = "user"
    relaxable: bool = True
    diagnostic_scale: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.location not in {"row", "variable"}:
            raise ValueError("constraint location must be 'row' or 'variable'")
        # The compiler interns common immutable metadata mappings for large
        # per-asset registries.  Preserve those flyweights instead of copying
        # them once per ConstraintRecord.
        frozen = (
            self.metadata
            if isinstance(self.metadata, MappingProxyType)
            else MappingProxyType(dict(self.metadata))
        )
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True)
class LinearDomain:
    r"""Polyhedral domain $l\le Az\le u$ plus column bounds."""

    A: sp.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    variables: tuple[VariableRecord, ...]
    constraints: tuple[ConstraintRecord, ...]
    weight_indices: np.ndarray
    assets: tuple[Any, ...]

    @property
    def n_variables(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_constraints(self) -> int:
        return int(self.A.shape[0])


@dataclass(frozen=True)
class FactorRiskOperator:
    r"""Annualized factor-plus-specific tracking-risk operator.

    For active weight $a=x-b$, total variance is

    $$
    R(a)=(E^{\mathsf T}a)^{\mathsf T}F(E^{\mathsf T}a)
    +\lVert d\odot a\rVert_2^2.
    $$

    The operator stores public annualized-decimal inputs and can evaluate either
    a base canonical vector or a vector with additional auxiliary columns.
    """

    exposure: np.ndarray
    covariance: np.ndarray
    specific_volatility: np.ndarray
    benchmark: np.ndarray
    factor_names: tuple[str, ...]
    weight_indices: np.ndarray

    def components(self, vector: np.ndarray) -> tuple[float, float, float]:
        weight = np.asarray(vector, dtype=float)[self.weight_indices]
        active = weight - self.benchmark
        factor = self.exposure.T @ active
        factor_variance = float(factor @ self.covariance @ factor)
        specific_variance = float(np.square(self.specific_volatility * active).sum())
        total = max(0.0, factor_variance + specific_variance)
        return total, factor_variance, specific_variance


@dataclass(frozen=True)
class LinearProgram:
    kind: ProblemKind
    domain: LinearDomain
    c: np.ndarray
    objective_offset: float = 0.0


@dataclass(frozen=True)
class QuadraticProgram:
    r"""Convex minimization over a polyhedral domain.

    $$
    \min_z\quad \frac12 z^{\mathsf T}Pz+q^{\mathsf T}z+c.
    $$
    """

    kind: ProblemKind
    domain: LinearDomain
    P: sp.csc_matrix
    q: np.ndarray
    objective_offset: float = 0.0
    objective_scale_reference: float | None = None
    risk_operator: FactorRiskOperator | None = None
    risk_limit: float | None = None


@dataclass(frozen=True)
class FactorQCQP:
    """Maximize alpha over a linear domain with one factor-model TE budget."""

    kind: ProblemKind
    domain: LinearDomain
    alpha: np.ndarray
    risk_operator: FactorRiskOperator
    risk_limit: float


CanonicalModel: TypeAlias = LinearProgram | QuadraticProgram | FactorQCQP


@dataclass(frozen=True)
class CompiledProblem:
    """Canonical payload plus audit identity and exact compiler rewrites."""

    model: CanonicalModel
    fingerprint: ProblemFingerprint
    compiler_optimizations: tuple[str, ...] = ()

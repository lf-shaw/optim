"""Thin solver protocol and normalized native-attempt result.

Backends accept an already compiled model and do not know about pandas, data
alignment, sequence state or business repair policy.  ``BackendResult`` is
internal evidence; only the API's independent validator can promote it to a
public :class:`OptimizationResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from ..portfolio_types import FailureReason, SolveStatus


@dataclass(frozen=True)
class BackendOptions:
    """Small common subset of numerical controls understood by adapters."""

    verbose: bool = False
    time_limit_s: float | None = None
    max_iter: int = 1000
    eps_abs: float = 1e-8
    eps_rel: float = 1e-8
    objective_scale_target: float | None = 0.2
    inequality_form: str = "auto"


@dataclass(frozen=True)
class BackendResult:
    """Normalized status, primal candidate and telemetry for one attempt."""

    backend: str
    status: SolveStatus
    primal: np.ndarray | None
    objective_value: float | None
    native_status: str
    reason: FailureReason | None = None
    message: str = ""
    iterations: int = 0
    setup_s: float = 0.0
    solve_s: float = 0.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class SolverBackend(Protocol):
    name: str

    def solve(self, model: Any, options: BackendOptions) -> BackendResult: ...

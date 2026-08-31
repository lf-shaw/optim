"""Public contracts for the unified portfolio optimizer.

The contracts in this module intentionally contain no solver imports.  Importing
``optim`` must remain usable without MOSEK, PIQP, Clarabel, tuda2, or carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
import pandas as pd


class ProblemKind(str, Enum):
    LP = "lp"
    QP = "qp"
    FACTOR_QCQP = "factor_qcqp"
    CONIC = "conic"


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    LIMIT_REACHED = "limit_reached"
    NUMERICAL_ERROR = "numerical_error"
    RESOURCE_ERROR = "resource_error"
    SOLVER_ERROR = "solver_error"
    SKIPPED = "skipped"

    @property
    def has_solution(self) -> bool:
        return self in {self.OPTIMAL, self.OPTIMAL_INACCURATE}


class ProofStatus(str, Enum):
    VERIFIED = "verified"
    NUMERICAL_ESTIMATE = "numerical_estimate"
    UNAVAILABLE = "unavailable"


class FailureReason(str, Enum):
    UPDATE_FAILURE = "update_failure"
    MAX_ITER = "max_iter"
    TIME_LIMIT = "time_limit"
    INVALID_NUMERICS = "invalid_numerics"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INFEASIBLE_REPORTED = "infeasible_reported"
    INFEASIBLE_CONFIRMED = "infeasible_confirmed"
    UNBOUNDED_REPORTED = "unbounded_reported"
    UNBOUNDED_CONFIRMED = "unbounded_confirmed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DataProvenance:
    source: str = "memory"
    source_date: pd.Timestamp | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class AlphaSpec:
    """Business meaning of alpha values supplied to the optimizer."""

    units: str = "standardized_score"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("AlphaSpec.scale must be positive and finite")


@dataclass(frozen=True)
class ObjectiveTolerance:
    """Accepted objective loss in raw and/or AlphaSpec-normalized units."""

    absolute: float | None = None
    normalized: float | None = 1e-4

    def __post_init__(self) -> None:
        if self.absolute is None and self.normalized is None:
            raise ValueError("at least one objective tolerance must be provided")
        for name, value in (("absolute", self.absolute), ("normalized", self.normalized)):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"ObjectiveTolerance.{name} must be finite and non-negative")

    def raw_limit(self, alpha_spec: AlphaSpec) -> float:
        absolute = 0.0 if self.absolute is None else self.absolute
        normalized = 0.0 if self.normalized is None else self.normalized * alpha_spec.scale
        return float(absolute + normalized)


@dataclass(frozen=True)
class FactorRiskModel:
    """One exact-date factor model in annualized decimal risk units.

    ``exposure`` follows ``PortfolioData.assets`` by row and ``factor_names`` by
    column.  ``covariance`` uses the same factor coordinate on both axes;
    ``specific_volatility`` follows assets.  The core performs no unit inference
    or implicit reordering after this contract is constructed.
    """

    asof: pd.Timestamp
    exposure: np.ndarray
    covariance: np.ndarray
    specific_volatility: np.ndarray
    factor_names: tuple[str, ...]
    factor_types: tuple[str, ...]
    provenance: DataProvenance = field(default_factory=DataProvenance)
    annualization: str = "annualized_decimal"


@dataclass(frozen=True)
class FullCovarianceRiskModel:
    asof: pd.Timestamp
    covariance: np.ndarray
    provenance: DataProvenance = field(default_factory=DataProvenance)
    annualization: str = "annualized_decimal"


RiskModel: TypeAlias = FactorRiskModel | FullCovarianceRiskModel


@dataclass(frozen=True)
class PortfolioData:
    """Numerical inputs for one date on one authoritative asset coordinate.

    Every asset-shaped array is positional against ``assets``.  Labelled pandas
    alignment belongs to a data-source adapter and must be completed before this
    object reaches validation/compiler code.  ``alpha`` may be ``None`` only for
    objectives that do not consume alpha.
    """

    date: pd.Timestamp
    assets: pd.Index
    alpha: np.ndarray | None
    benchmark: np.ndarray | None
    initial_weight: np.ndarray | None
    tradable: np.ndarray
    risk_model: RiskModel | None = None
    alpha_spec: AlphaSpec | None = None
    extra_attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", pd.Timestamp(self.date))
        object.__setattr__(self, "assets", pd.Index(self.assets, copy=False))
        object.__setattr__(
            self,
            "extra_attributes",
            MappingProxyType(dict(self.extra_attributes)),
        )


@dataclass(frozen=True)
class MaximizeAlpha:
    pass


@dataclass(frozen=True)
class RiskAdjustedAlpha:
    factor_aversion: float = 0.75
    specific_aversion: float = 0.75

    def __post_init__(self) -> None:
        if self.factor_aversion < 0.0 or self.specific_aversion < 0.0:
            raise ValueError("risk aversion must be non-negative")
        if self.factor_aversion == 0.0 and self.specific_aversion == 0.0:
            raise ValueError("at least one risk aversion must be positive")


@dataclass(frozen=True)
class MinimizeTrackingError:
    alpha_floor: float | None = None


PortfolioObjective: TypeAlias = MaximizeAlpha | RiskAdjustedAlpha | MinimizeTrackingError


AssetWeightOverride: TypeAlias = float | tuple[float, float]


@dataclass(frozen=True)
class AssetTradeConstraints:
    """One-off asset trading instructions for a single optimization.

    These instructions belong to :class:`PortfolioProblem`; they are never
    stored as mutable optimizer state and therefore cannot leak into a later
    live-trading request.

    ``blacklist`` forces the target weight to zero. ``frozen`` fixes it at the
    supplied initial weight. ``not_buyable`` and ``not_sellable`` are one-sided
    restrictions relative to the initial weight. ``weight_overrides`` accepts
    either an exact target or a ``(lower, upper)`` interval.
    """

    blacklist: tuple[Any, ...] = ()
    frozen: tuple[Any, ...] = ()
    not_buyable: tuple[Any, ...] = ()
    not_sellable: tuple[Any, ...] = ()
    weight_overrides: Mapping[Any, AssetWeightOverride] = field(default_factory=dict)
    missing_asset: str = "error"

    def __post_init__(self) -> None:
        for name in ("blacklist", "frozen", "not_buyable", "not_sellable"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "weight_overrides",
            MappingProxyType(dict(self.weight_overrides)),
        )
        if self.missing_asset not in {"error", "ignore"}:
            raise ValueError("missing_asset must be 'error' or 'ignore'")

    @property
    def is_empty(self) -> bool:
        return not (
            self.blacklist
            or self.frozen
            or self.not_buyable
            or self.not_sellable
            or self.weight_overrides
        )


@dataclass(frozen=True)
class WeightBounds:
    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 1.0


@dataclass(frozen=True)
class SymmetricBound:
    absolute: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError("symmetric bound must be finite and non-negative")


@dataclass(frozen=True)
class TurnoverLimit:
    limit: float
    convention: str = "l1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.limit) or self.limit < 0.0:
            raise ValueError("turnover limit must be finite and non-negative")
        if self.convention not in {"l1", "two_way"}:
            raise ValueError("turnover convention must be 'l1' or 'two_way'")

    @property
    def l1_limit(self) -> float:
        # Historical optim and the benchmark define turnover as
        # $\lVert x-x_0\rVert_1$.
        # ``two_way`` is retained as a descriptive alias for that convention.
        return float(self.limit)


@dataclass(frozen=True)
class LowerBound:
    value: float


@dataclass(frozen=True)
class ExposureBounds:
    default: tuple[float, float] | None = None
    overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType({str(k).lower(): tuple(v) for k, v in self.overrides.items()}),
        )


@dataclass(frozen=True)
class TrackingErrorLimit:
    annualized: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.annualized) or self.annualized <= 0.0:
            raise ValueError("tracking error limit must be positive and finite")


@dataclass(frozen=True)
class PortfolioConstraints:
    """Immutable feasible-domain configuration shared by LP/QP/QCQP routes.

    Risk budget is the only nonlinear constraint.  Turnover, active weight,
    factor exposure and operational asset instructions all compile into the
    same polyhedral domain and therefore do not select a solver by themselves.
    """

    long_only: bool = True
    budget: float = 1.0
    asset_weight: WeightBounds = field(default_factory=WeightBounds)
    active_weight: SymmetricBound | None = None
    total_active: float | None = None
    turnover: TurnoverLimit | None = None
    benchmark_member_weight: LowerBound | None = None
    style: ExposureBounds | None = None
    industry: ExposureBounds | None = None
    tracking_error: TrackingErrorLimit | None = None
    freeze_nontradable: bool = True
    asset_trade: AssetTradeConstraints | None = None
    extra_active: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    extra_absolute: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asset_trade is not None and not isinstance(
            self.asset_trade, AssetTradeConstraints
        ):
            raise TypeError("asset_trade must be AssetTradeConstraints or None")
        object.__setattr__(self, "extra_active", MappingProxyType(dict(self.extra_active)))
        object.__setattr__(self, "extra_absolute", MappingProxyType(dict(self.extra_absolute)))


@dataclass(frozen=True)
class PortfolioProblem:
    """Complete stateless definition of one portfolio optimization request."""

    data: PortfolioData
    objective: PortfolioObjective
    constraints: PortfolioConstraints


@dataclass(frozen=True)
class ProblemFingerprint:
    semantic_hash: str
    canonical_hash: str
    compiler_version: str


@dataclass(frozen=True)
class RunFingerprint:
    problem: ProblemFingerprint
    solver_policy_hash: str
    package_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_versions", MappingProxyType(dict(self.package_versions)))


@dataclass(frozen=True)
class OptimalityCertificate:
    """Route-specific objective bound in the declared business objective units.

    ``VERIFIED`` is reserved for a mathematically valid bound, while
    ``NUMERICAL_ESTIMATE`` depends on solver-reported numerical gaps.  The
    certificate does not assert similarity of individual portfolio weights.
    """

    kind: str
    proof_status: ProofStatus
    primal_value: float
    dual_bound: float | None
    absolute_gap: float | None
    normalized_gap: float | None
    objective_units: str
    objective_scale: float | None
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


@dataclass(frozen=True)
class ConstraintViolation:
    constraint_id: str
    group: str
    amount: float
    observed: float | None = None
    lower: float | None = None
    upper: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class PortfolioMetrics:
    budget: float | None = None
    objective: float | None = None
    tracking_error: float | None = None
    factor_variance: float | None = None
    specific_variance: float | None = None
    turnover_l1: float | None = None
    total_active_l1: float | None = None
    benchmark_member_weight: float | None = None
    max_weight: float | None = None
    min_weight: float | None = None
    max_active_weight: float | None = None
    max_style_exposure: float | None = None
    max_industry_exposure: float | None = None


@dataclass(frozen=True)
class SolveTimings:
    prepare_s: float = 0.0
    compile_s: float = 0.0
    backend_setup_s: float = 0.0
    backend_solve_s: float = 0.0
    validation_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0


@dataclass(frozen=True)
class SolverAttempt:
    backend: str
    status: SolveStatus
    reason: FailureReason | None = None
    native_status: str | None = None
    message: str | None = None
    solve_s: float = 0.0
    recovered: bool = False
    backend_payload_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class AlignmentReport:
    benchmark_missing_mass: float = 0.0
    benchmark_renormalization_factor: float = 1.0
    holding_missing_mass: float = 0.0
    source_dates: Mapping[str, pd.Timestamp] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dates", MappingProxyType(dict(self.source_dates)))


@dataclass(frozen=True)
class OptimizationResult:
    """Normalized outcome of all primary and fallback attempts.

    Ordinary solver failures are represented by status/route and ``weights=None``
    so long runs can continue or be audited.  Input/model errors occur earlier as
    exceptions.  ``require_weights`` provides the opt-in exception-style access
    pattern for callers that cannot proceed without a usable portfolio.
    """

    status: SolveStatus
    weights: pd.Series | None
    objective_value: float | None
    backend: str | None
    route: tuple[SolverAttempt, ...]
    metrics: PortfolioMetrics
    certificate: OptimalityCertificate | None
    violations: tuple[ConstraintViolation, ...]
    alignment: AlignmentReport
    diagnostics: Any | None
    timings: SolveTimings
    fingerprint: ProblemFingerprint
    message: str = ""

    def require_weights(self) -> pd.Series:
        if not self.status.has_solution or self.weights is None:
            raise RuntimeError(f"optimization did not produce usable weights: {self.status.value}")
        return self.weights


@dataclass(frozen=True)
class SolverTuning:
    """Advanced numerical defaults calibrated for the current CPU fast paths.

    Public risk tolerances use annualized decimal units.  ``weight_zero_tolerance``
    is a weight, and objective acceptance is configured separately on
    :class:`SolverPolicy` because its raw meaning depends on ``AlphaSpec``.

    ``polish`` is a frozen contract field but is not yet consumed by the current
    direct PIQP adapter.
    """

    alpha_target: float = 0.2
    theta_initial: float = 16384.0
    theta_growth: float = 4.0
    theta_max: float = 1e12
    max_outer_iters: int = 30
    intermediate_eps: float = 1e-5
    final_eps: float = 1e-8
    piqp_max_iter: int = 1000
    piqp_inequality_form: str = "auto"
    polish: bool = True
    feasibility_tolerance: float = 1e-5
    risk_margin: float = 1e-7
    weight_zero_tolerance: float = 1e-5


@dataclass(frozen=True)
class SolverPolicy:
    """Routing, fallback and acceptance policy for a stateless optimizer.

    Current implementation fixes the tested primary routes to HiGHS, PIQP and
    factor frontier; the string selector fields reserve the public contract for
    later routing configurability.  Independent validation currently remains an
    unconditional correctness boundary even if ``validate_solution`` is false.
    ``repeat_failed_cold_solve`` is likewise reserved and intentionally unused.
    """

    lp: str = "highs"
    qp: str = "piqp"
    factor_qcqp_strategy: str = "frontier"
    factor_qcqp_subproblem_backend: str = "piqp"
    licensed_fallback: str = "mosek"
    free_fallback: str = "clarabel_qdldl"
    lp_prescreen: bool = False
    validate_solution: bool = True
    rebuild_after_update_failure: bool = True
    repeat_failed_cold_solve: bool = False
    objective_tolerance: ObjectiveTolerance = field(default_factory=ObjectiveTolerance)
    tuning: SolverTuning = field(default_factory=SolverTuning)


@dataclass(frozen=True)
class TurnoverRecoveryPolicy:
    max_turnover: float
    buffer: float = 1e-5
    require_turnover_only: bool = True
    reset_next_period: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_turnover) or self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be positive and finite")
        if not np.isfinite(self.buffer) or self.buffer < 0.0:
            raise ValueError("turnover recovery buffer must be finite and non-negative")
        if not self.reset_next_period:
            raise ValueError(
                "v1 turnover recovery always resets the configured limit next period"
            )


@dataclass(frozen=True)
class SequencePolicy:
    mode: str = "chained"
    holding_update: str = "mark_to_market"
    on_failure: str = "stop"
    theta_seed: str = "auto"
    turnover_recovery: TurnoverRecoveryPolicy | None = None
    holding_missing_mass_tolerance: float = 0.0
    renormalize_missing_holdings: bool = False
    output_weights: str = "sparse"

    def __post_init__(self) -> None:
        if self.mode not in {"chained", "independent"}:
            raise ValueError("sequence mode must be 'chained' or 'independent'")
        if self.holding_update != "mark_to_market":
            raise ValueError("only close-to-close mark_to_market holding updates are supported")
        if self.on_failure not in {"stop", "hold"}:
            raise ValueError("on_failure must be 'stop' or 'hold'")
        if self.theta_seed not in {"auto", "fixed", "previous"}:
            raise ValueError("theta_seed must be 'auto', 'fixed', or 'previous'")
        if (
            not np.isfinite(self.holding_missing_mass_tolerance)
            or self.holding_missing_mass_tolerance < 0.0
        ):
            raise ValueError("holding missing-mass tolerance must be finite and non-negative")
        if self.output_weights not in {"none", "sparse", "all"}:
            raise ValueError("output_weights must be 'none', 'sparse', or 'all'")

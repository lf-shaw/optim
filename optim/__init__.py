"""Unified portfolio optimization API.

Importing :mod:`optim` has no solver-license or optional data-source side
effects. The legacy ``opt``, ``linopt`` and ``solver`` modules remain lazily
available during migration.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .portfolio_types import (
    AlignmentReport,
    AlphaSpec,
    AssetTradeConstraints,
    AssetWeightOverride,
    ConstraintViolation,
    DataProvenance,
    ExposureBounds,
    FactorRiskModel,
    FailureReason,
    FullCovarianceRiskModel,
    LowerBound,
    MaximizeAlpha,
    MinimizeTrackingError,
    ObjectiveTolerance,
    OptimalityCertificate,
    OptimizationResult,
    PortfolioConstraints,
    PortfolioData,
    PortfolioMetrics,
    PortfolioProblem,
    ProblemFingerprint,
    ProblemKind,
    ProofStatus,
    RiskAdjustedAlpha,
    RunFingerprint,
    SolveStatus,
    SolveTimings,
    SolverAttempt,
    SolverPolicy,
    SolverTuning,
    SequencePolicy,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    TurnoverRecoveryPolicy,
    WeightBounds,
)
from .api import PortfolioOptimizer, PreparedPortfolioProblem
from .validation import (
    PortfolioValidationError,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from .diagnostics import InfeasibilityReport, RequiredRelaxation
from .sequence import PortfolioSequenceResult, SequenceDataError, SequenceStep
from .data import (
    BenchmarkCoverageError,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    FactorRiskFrames,
    InMemoryDataSource,
    PortfolioSchedule,
    PreparedPortfolioRun,
)

__all__ = [
    "AlignmentReport",
    "AlphaSpec",
    "AssetTradeConstraints",
    "AssetWeightOverride",
    "BenchmarkCoverageError",
    "BenchmarkCoveragePolicy",
    "ConstraintViolation",
    "DataProvenance",
    "DataAlignmentError",
    "ExposureBounds",
    "FactorRiskModel",
    "FailureReason",
    "FactorRiskFrames",
    "FullCovarianceRiskModel",
    "InfeasibilityReport",
    "InMemoryDataSource",
    "LowerBound",
    "MaximizeAlpha",
    "MinimizeTrackingError",
    "ObjectiveTolerance",
    "OptimalityCertificate",
    "OptimizationResult",
    "PortfolioConstraints",
    "PortfolioData",
    "PortfolioMetrics",
    "PortfolioProblem",
    "PortfolioSchedule",
    "PreparedPortfolioRun",
    "PortfolioSequenceResult",
    "PortfolioOptimizer",
    "PortfolioValidationError",
    "PreparedPortfolioProblem",
    "ProblemFingerprint",
    "ProblemKind",
    "ProofStatus",
    "RiskAdjustedAlpha",
    "RequiredRelaxation",
    "RunFingerprint",
    "SolveStatus",
    "SolveTimings",
    "SequenceDataError",
    "SequencePolicy",
    "SequenceStep",
    "SolverAttempt",
    "SolverPolicy",
    "SolverTuning",
    "SymmetricBound",
    "TrackingErrorLimit",
    "TurnoverLimit",
    "TurnoverRecoveryPolicy",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "WeightBounds",
]


def __getattr__(name: str) -> Any:
    """Load legacy modules only when a caller explicitly requests them."""

    if name in {"opt", "linopt", "solver"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

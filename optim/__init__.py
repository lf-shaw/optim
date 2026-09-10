"""统一组合优化公共 API。

导入 :mod:`optim` 不会触发求解器 license 或可选数据源的副作用。
旧版 ``opt``、``linopt``、``solver`` 均已移除，统一使用 PortfolioOptimizer。
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ImportError:
    try:
        __version__ = version("optim")
    except PackageNotFoundError:
        __version__ = "0+unknown"
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
    InfeasibilityContributor,
    LowerBound,
    MaximizeAlpha,
    MinimizeTrackingError,
    NativeInfeasibilityEvidence,
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
from .repro import ReproCase, export_repro, load_repro
from .sequence import PortfolioSequenceResult, SequenceDataError, SequenceStep
from .data import (
    BenchmarkCoverageError,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    FactorRiskFrames,
    InMemoryDataSource,
    PortfolioSchedule,
    PreparedPortfolioRun,
    make_portfolio_data,
    make_factor_risk_model,
)

__all__ = [
    "make_portfolio_data",
    "make_factor_risk_model",
    "__version__",
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
    "InfeasibilityContributor",
    "InfeasibilityReport",
    "InMemoryDataSource",
    "LowerBound",
    "MaximizeAlpha",
    "MinimizeTrackingError",
    "NativeInfeasibilityEvidence",
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
    "ReproCase",
    "export_repro",
    "load_repro",
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

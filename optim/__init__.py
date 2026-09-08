"""统一组合优化公共 API。

导入 :mod:`optim` 不会触发求解器 license 或可选数据源的副作用。迁移期间，旧版
``opt``、``linopt`` 和 ``solver`` 模块仍可按需延迟加载。
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

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
)

__all__ = [
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


def __getattr__(name: str) -> Any:
    """仅在调用方显式访问时延迟加载旧版模块。"""

    if name in {"opt", "linopt", "solver"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

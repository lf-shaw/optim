"""严格且与求解器无关的组合数据装配层。"""

from .alignment import (
    BenchmarkCoverageError,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    align_benchmark,
)
from .contracts import FactorRiskFrames, PortfolioSchedule
from .memory import InMemoryDataSource, PreparedPortfolioRun

__all__ = [
    "BenchmarkCoverageError",
    "BenchmarkCoveragePolicy",
    "DataAlignmentError",
    "FactorRiskFrames",
    "InMemoryDataSource",
    "PortfolioSchedule",
    "PreparedPortfolioRun",
    "align_benchmark",
]

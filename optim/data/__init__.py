"""严格且与求解器无关的组合数据装配层。"""

from .alignment import (
    BenchmarkCoverageError,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    align_benchmark,
)
from .contracts import FactorRiskFrames, PortfolioSchedule
from .memory import InMemoryDataSource, PreparedPortfolioRun
from .manual import make_factor_risk_model, make_portfolio_data

__all__ = [
    "BenchmarkCoverageError",
    "BenchmarkCoveragePolicy",
    "DataAlignmentError",
    "FactorRiskFrames",
    "InMemoryDataSource",
    "PortfolioSchedule",
    "PreparedPortfolioRun",
    "align_benchmark",
    "make_portfolio_data",
    "make_factor_risk_model",
]

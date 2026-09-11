"""组合优化私有二进制核心。

本包不是公共 API。正式 wheel 仅保留这个轻量入口和由 Cython 编译的扩展模块；不会分发
实现模块的 Python 源码、类型存根或 C/C++ 中间文件。外部调用方应始终使用
``optim.PortfolioOptimizer``。
"""

from .canonical import (
    CanonicalKind,
    CanonicalModel,
    ConstraintRecord,
    FactorQCQP,
    FactorRiskOperator,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)
from .contracts import (
    CoreFailureReason,
    CoreInfeasibilityContributor,
    CoreInfeasibilityEvidence,
    CoreSolveStatus,
    CoreSolverOptions,
)
from .backends.base import BackendResult as CoreBackendResult
from .engine import (
    CORE_ABI_VERSION,
    CoreProblemHandle,
    CoreSolveResult,
    CoreSolver,
)
from .thread_control import (
    NumericalThreadScope,
    ThreadResolution,
    numerical_thread_scope,
    resolve_thread_setting,
)

__all__ = [
    "CORE_ABI_VERSION",
    "CanonicalKind",
    "CanonicalModel",
    "ConstraintRecord",
    "CoreFailureReason",
    "CoreInfeasibilityContributor",
    "CoreInfeasibilityEvidence",
    "CoreBackendResult",
    "CoreProblemHandle",
    "CoreSolveResult",
    "CoreSolveStatus",
    "CoreSolver",
    "CoreSolverOptions",
    "FactorQCQP",
    "FactorRiskOperator",
    "LinearDomain",
    "LinearProgram",
    "NumericalThreadScope",
    "numerical_thread_scope",
    "QuadraticProgram",
    "ThreadResolution",
    "resolve_thread_setting",
    "VariableRecord",
]

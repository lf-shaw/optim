"""组合问题的 canonical 编译入口。"""

from .canonical import (
    CanonicalKind,
    CanonicalModel,
    CompiledProblem,
    ConstraintRecord,
    FactorQCQP,
    FactorRiskOperator,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)
from .compiler import CanonicalCompilationError, classify_problem, compile_problem

__all__ = [
    "CanonicalCompilationError",
    "CanonicalKind",
    "CanonicalModel",
    "CompiledProblem",
    "ConstraintRecord",
    "FactorQCQP",
    "FactorRiskOperator",
    "LinearDomain",
    "LinearProgram",
    "QuadraticProgram",
    "VariableRecord",
    "classify_problem",
    "compile_problem",
]

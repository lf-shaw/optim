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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._impl.compiler import (
        CanonicalCompilationError,
        classify_problem,
        compile_problem,
    )


def __getattr__(name: str) -> Any:
    """延迟获取编译入口，保持便捷导入并避免与编译结果契约循环依赖。"""
    if name in {"CanonicalCompilationError", "classify_problem", "compile_problem"}:
        from .._impl import compiler

        value = getattr(compiler, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

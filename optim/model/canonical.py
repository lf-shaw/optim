"""上层编译结果契约与数值 canonical 类型入口。

数值 payload 类型由求解核心定义；本模块只增加业务审计所需的 fingerprint 和编译优化记录，
避免数值核心反向依赖公共业务类型。
"""

from __future__ import annotations

from dataclasses import dataclass

from .._core import (
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
from ..portfolio_types import ProblemFingerprint


@dataclass(frozen=True)
class CompiledProblem:
    """Canonical 数值 payload、审计身份和精确编译优化记录。

    Attributes
    ----------
    model : CanonicalModel
        已编译的 canonical 数值模型。
    fingerprint : ProblemFingerprint
        同时覆盖业务语义和最终数值 payload 的问题身份。
    compiler_optimizations : tuple[str, ...]
        已证明数学等价并实际应用的结构优化标识，例如稀疏换手率展开。
    """

    model: CanonicalModel
    fingerprint: ProblemFingerprint
    compiler_optimizations: tuple[str, ...] = ()


__all__ = [
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
]

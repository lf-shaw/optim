"""确定性的业务语义及 canonical 问题 fingerprint。"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .model.canonical import (
    ConstraintRecord,
    FactorQCQP,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)
from .portfolio_types import DataProvenance, PortfolioProblem, ProblemFingerprint


COMPILER_VERSION = "portfolio-canonical-v4-compact-l1"


def _chunk(digest: Any, label: str, payload: bytes) -> None:
    encoded = label.encode("utf-8")
    # 保持原有字节协议；小型头部合并提交，数值矩阵 payload 不参与拼接，避免大块复制。
    digest.update(
        len(encoded).to_bytes(4, "little") + encoded
        + len(payload).to_bytes(8, "little")
    )
    digest.update(payload)


def _feed(digest: Any, label: str, value: Any) -> None:
    # 股票代码等普通字符串是对象数组递归的热点。必须用精确类型判断：str 的子类可能
    # 同时是 Enum，仍应走下面的枚举语义。此快路径与末尾的旧编码逐字节相同。
    if type(value) is str:
        _chunk(digest, label, f"str:{value}".encode("utf-8"))
    elif value is None:
        _chunk(digest, label, b"none")
    elif isinstance(value, Enum):
        _chunk(digest, label, f"enum:{type(value).__qualname__}:{value.value}".encode())
    elif isinstance(value, pd.Timestamp):
        _chunk(digest, label, f"timestamp:{value.isoformat()}".encode())
    elif isinstance(value, pd.Index):
        _feed(digest, label, value.to_numpy(dtype=object))
    elif sp.issparse(value):
        matrix = value.tocsc(copy=True)
        matrix.sort_indices()
        _chunk(digest, f"{label}.shape", np.asarray(matrix.shape, dtype="<i8").tobytes())
        _chunk(digest, f"{label}.indptr", np.asarray(matrix.indptr, dtype="<i8").tobytes())
        _chunk(digest, f"{label}.indices", np.asarray(matrix.indices, dtype="<i8").tobytes())
        _chunk(digest, f"{label}.data", np.asarray(matrix.data, dtype="<f8").tobytes())
    elif isinstance(value, np.ndarray):
        _chunk(digest, f"{label}.shape", np.asarray(value.shape, dtype="<i8").tobytes())
        if value.dtype.kind in "biufc":
            if value.dtype.kind == "b":
                payload = np.ascontiguousarray(value, dtype=np.uint8).tobytes()
            elif value.dtype.kind in "iu":
                payload = np.ascontiguousarray(value, dtype="<i8").tobytes()
            elif value.dtype.kind == "c":
                payload = np.ascontiguousarray(value, dtype="<c16").tobytes()
            else:
                payload = np.ascontiguousarray(value, dtype="<f8").tobytes()
            _chunk(digest, label, payload)
        else:
            flat = value.reshape(-1)
            if all(type(item) is str for item in flat):
                # 对股票代码数组批量提交原有字节流，不改变标签、顺序或哈希协议。
                # 分块限制临时内存；混合类型和字符串子类仍走完整语义编码。
                for start in range(0, len(flat), 1024):
                    chunks = []
                    for index in range(start, min(start + 1024, len(flat))):
                        encoded = f"{label}[{index}]".encode("utf-8")
                        payload = f"str:{flat[index]}".encode("utf-8")
                        chunks.append(
                            len(encoded).to_bytes(4, "little") + encoded
                            + len(payload).to_bytes(8, "little") + payload
                        )
                    digest.update(b"".join(chunks))
            else:
                for index, item in enumerate(flat):
                    _feed(digest, f"{label}[{index}]", item)
    elif dataclasses.is_dataclass(value):
        _chunk(digest, f"{label}.type", type(value).__qualname__.encode())
        for item in dataclasses.fields(value):
            _feed(digest, f"{label}.{item.name}", getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _feed(digest, f"{label}.key", str(key))
            _feed(digest, f"{label}.{key}", value[key])
    elif isinstance(value, (tuple, list)):
        _chunk(digest, f"{label}.len", len(value).to_bytes(8, "little"))
        for index, item in enumerate(value):
            _feed(digest, f"{label}[{index}]", item)
    elif isinstance(value, (np.bool_, bool)):
        _chunk(digest, label, b"true" if bool(value) else b"false")
    elif isinstance(value, (np.integer, int)):
        _chunk(digest, label, f"int:{int(value)}".encode())
    elif isinstance(value, (np.floating, float)):
        _chunk(digest, label, np.asarray([float(value)], dtype="<f8").tobytes())
    else:
        _chunk(digest, label, f"str:{value}".encode("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return {"__timestamp__": value.isoformat()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}
    raise TypeError(f"cannot fingerprint {type(value).__qualname__}")


def _feed_variable_registry(
    digest: Any,
    records: tuple[VariableRecord, ...],
) -> None:
    """将数千条扁平 registry 记录作为一个确定性 payload 哈希。

    通用递归编码器适合小型语义对象，但对 5,200 资产 registry 会生成数十万标签和 hash
    update。这些记录具有冻结的扁平 schema，因此紧凑 JSON 数组可以保留所有字段，同时避免
    逐字段 Python dispatch。
    """

    payload = [
        (record.index, record.variable_id, record.group, record.key, record.unit)
        for record in records
    ]
    _chunk(
        digest,
        "domain.variables",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def _feed_constraint_registry(
    digest: Any,
    records: tuple[ConstraintRecord, ...],
) -> None:
    payload = [
        (
            record.constraint_id,
            record.group,
            record.location,
            record.index,
            record.key,
            record.unit,
            record.source,
            record.relaxable,
            record.diagnostic_scale,
            record.metadata,
        )
        for record in records
    ]
    _chunk(
        digest,
        "domain.constraints",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8"),
    )


def semantic_hash(problem: PortfolioProblem) -> str:
    """计算不含数据来源元信息的业务语义哈希。

    Parameters
    ----------
    problem : PortfolioProblem
        待标识的完整业务问题。

    Returns
    -------
    str
        小写十六进制 SHA-256。来源不同但对齐后数值逐字节相同的问题会得到相同结果。
    """

    # provenance 属于运行/审计元信息，不是数学问题本身。独立来源但对齐后逐字节相同的输入
    # 必须保留相同 semantic hash，后端对比才有意义。
    data = problem.data
    risk_model = data.risk_model
    if risk_model is not None:
        risk_model = dataclasses.replace(risk_model, provenance=DataProvenance())
    semantic_problem = dataclasses.replace(
        problem,
        data=dataclasses.replace(
            data,
            risk_model=risk_model,
            provenance=DataProvenance(),
        ),
    )
    digest = hashlib.sha256()
    _feed(digest, "problem", semantic_problem)
    return digest.hexdigest()


def canonical_hash(model: LinearProgram | QuadraticProgram | FactorQCQP) -> str:
    """计算 canonical 稀疏数值 payload 和 registry 的哈希。

    Parameters
    ----------
    model : LinearProgram | QuadraticProgram | FactorQCQP
        已排序、不可变的 canonical 模型。

    Returns
    -------
    str
        覆盖矩阵、边界、目标、风险算子和审计 registry 的小写十六进制 SHA-256。
    """

    digest = hashlib.sha256()
    _feed(digest, "model.type", type(model).__qualname__)
    _feed(digest, "model.kind", model.kind)
    _feed(digest, "domain.A", model.domain.A)
    _feed(digest, "domain.lower", model.domain.lower)
    _feed(digest, "domain.upper", model.domain.upper)
    _feed(digest, "domain.variable_lower", model.domain.variable_lower)
    _feed(digest, "domain.variable_upper", model.domain.variable_upper)
    _feed_variable_registry(digest, model.domain.variables)
    _feed_constraint_registry(digest, model.domain.constraints)
    _feed(digest, "domain.weight_indices", model.domain.weight_indices)
    if isinstance(model, LinearProgram):
        _feed(digest, "model.c", model.c)
        _feed(digest, "model.objective_offset", model.objective_offset)
    elif isinstance(model, QuadraticProgram):
        _feed(digest, "model.P", model.P)
        _feed(digest, "model.q", model.q)
        _feed(digest, "model.objective_offset", model.objective_offset)
        _feed(
            digest,
            "model.objective_scale_reference",
            model.objective_scale_reference,
        )
        _feed(digest, "model.risk_limit", model.risk_limit)
        _feed(digest, "model.risk_operator", model.risk_operator)
    else:
        _feed(digest, "model.alpha", model.alpha)
        _feed(digest, "model.risk_limit", model.risk_limit)
        _feed(digest, "model.risk_operator", model.risk_operator)
    return digest.hexdigest()


def fingerprint(problem: PortfolioProblem, model: Any) -> ProblemFingerprint:
    """组合业务语义哈希、canonical 哈希和编译器版本。

    Parameters
    ----------
    problem : PortfolioProblem
        canonical 模型来源的业务问题。
    model : CanonicalModel
        由当前编译器为该问题生成的 canonical 模型。

    Returns
    -------
    ProblemFingerprint
        可用于证明主求解和回退求解针对同一数学问题的身份对象。
    """

    return ProblemFingerprint(
        semantic_hash=semantic_hash(problem),
        canonical_hash=canonical_hash(model),
        compiler_version=COMPILER_VERSION,
    )

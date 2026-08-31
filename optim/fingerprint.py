"""Deterministic semantic and canonical problem fingerprints."""

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


COMPILER_VERSION = "portfolio-canonical-v3-alpha-centered"


def _chunk(digest: Any, label: str, payload: bytes) -> None:
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "little"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _feed(digest: Any, label: str, value: Any) -> None:
    if value is None:
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
            for index, item in enumerate(value.reshape(-1)):
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
    """Hash thousands of flat records as one deterministic payload.

    The general recursive encoder is convenient for small semantic objects but
    created hundreds of thousands of labels and hash updates for a 5,200-asset
    registry.  These records have a frozen flat schema, so a compact JSON array
    preserves all fields while avoiding per-field Python dispatch.
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
    # Provenance is run/audit metadata, not part of the mathematical problem.
    # Two independently sourced but byte-identical aligned inputs must retain
    # the same semantic hash so backend comparisons remain meaningful.
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
    return ProblemFingerprint(
        semantic_hash=semantic_hash(problem),
        canonical_hash=canonical_hash(model),
        compiler_version=COMPILER_VERSION,
    )

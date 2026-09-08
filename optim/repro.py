"""单期问题的可移植复现包；仅使用显式类型白名单，不反序列化可执行对象。"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from dataclasses import dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from . import portfolio_types as pt
from .diagnostics import InfeasibilityReport, _certificate_export, _json_value
from .model.compiler import compile_problem

_TYPES = {
    name: getattr(pt, name)
    for name in (
        "PortfolioProblem",
        "PortfolioData",
        "PortfolioConstraints",
        "FactorRiskModel",
        "FullCovarianceRiskModel",
        "DataProvenance",
        "AlphaSpec",
        "ObjectiveTolerance",
        "MaximizeAlpha",
        "RiskAdjustedAlpha",
        "MinimizeTrackingError",
        "WeightBounds",
        "SymmetricBound",
        "TurnoverLimit",
        "LowerBound",
        "ExposureBounds",
        "TrackingErrorLimit",
        "AssetTradeConstraints",
        "SolverPolicy",
        "SolverTuning",
    )
}


def _environment() -> dict[str, str]:
    """记录版本而非环境变量，避免导出密钥和本机配置。"""
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in (
        "optim",
        "numpy",
        "pandas",
        "scipy",
        "piqp",
        "highspy",
        "clarabel",
        "Mosek",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _encode(value: Any, members: dict[str, bytes]) -> Any:
    """编码白名单输入，数组独立压缩，映射保留键类型。"""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {
                "type": "object_array",
                "shape": list(value.shape),
                "items": _encode(value.ravel().tolist(), members),
            }
        name = f"arrays/{len(members)}.npy"
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        members[name] = stream.getvalue()
        return {"type": "array", "member": name}
    if isinstance(value, np.generic):
        return _encode(value.item(), members)
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Index):
        if isinstance(value, pd.MultiIndex):
            raise TypeError("repro assets must be a single Index")
        return {
            "type": "index",
            "items": _encode(value.tolist(), members),
            "name": _encode(value.name, members),
            "dtype": str(value.dtype),
        }
    if type(value).__name__ in _TYPES and type(value) is _TYPES[type(value).__name__]:
        return {
            "type": type(value).__name__,
            "fields": {
                f.name: _encode(getattr(value, f.name), members) for f in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": [
                [_encode(k, members), _encode(v, members)] for k, v in value.items()
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(v, members) for v in value],
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": str(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported repro value: {type(value).__name__}")


def _decode(value: Any, members: dict[str, bytes]) -> Any:
    """只构造已知数据类型；数值数组只读，不使用 pickle 或动态导入。"""
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "array":
        result = np.load(io.BytesIO(members[value["member"]]), allow_pickle=False)
        result.setflags(write=False)
        return result
    if kind == "object_array":
        items = _decode(value["items"], members)
        result = np.empty(len(items), dtype=object)
        result[:] = items
        result = result.reshape(value["shape"])
        result.setflags(write=False)
        return result
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "index":
        return pd.Index(
            _decode(value["items"], members),
            name=_decode(value["name"], members),
            dtype=value["dtype"],
            tupleize_cols=False,
        )
    if kind == "mapping":
        return {_decode(k, members): _decode(v, members) for k, v in value["items"]}
    if kind in ("tuple", "list"):
        items = [_decode(v, members) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "float":
        return float(value["value"])
    if kind in _TYPES:
        return _TYPES[kind](
            **{k: _decode(v, members) for k, v in value["fields"].items()}
        )
    raise ValueError(f"unsupported repro type: {kind}")


@dataclass(frozen=True)
class ReproCase:
    """已加载的单期复现输入，不自动调用求解器。

    Attributes
    ----------
    problem : PortfolioProblem
        原始业务问题快照，包含该期真实初始持仓。
    policy : SolverPolicy
        原求解策略；可用于构建新的 PortfolioOptimizer。
    theta_seed : float | None
        原请求的前沿搜索初值。
    original_result : Mapping[str, Any]
        原结果的 JSON 审计快照，不是 OptimizationResult 实例。
    original_report : Mapping[str, Any] | None
        可选完整诊断证据快照，不自动重新诊断。
    environment : Mapping[str, str]
        导出机器的依赖版本与平台信息。
    version_differences : Mapping[str, tuple[str, str]]
        与本机不同的版本，值为原版本、本机版本；不强制阻断复跑。
    """

    problem: pt.PortfolioProblem
    policy: pt.SolverPolicy
    theta_seed: float | None
    original_result: Mapping[str, Any]
    original_report: Mapping[str, Any] | None
    environment: Mapping[str, str]
    version_differences: Mapping[str, tuple[str, str]]

    def solve(self) -> pt.OptimizationResult:
        """用保存的策略与 theta 初值单次求解，返回正常 OptimizationResult。

        不自动运行诊断或修改约束；正常校验和回退仍然生效。
        版本、license 和硬件差异可能导致实际后端、耗时及数值解不同。
        """
        from .api import PortfolioOptimizer

        return PortfolioOptimizer(self.policy).solve(
            self.problem, theta_seed=self.theta_seed
        )


def export_repro(
    path: str | Path,
    *,
    result: pt.OptimizationResult,
    report: InfeasibilityReport | None = None,
    problem: pt.PortfolioProblem | None = None,
    policy: pt.SolverPolicy | None = None,
    overwrite: bool = False,
) -> Path:
    """导出压缩的单期故障复现包，不重新求解或诊断。

    Parameters
    ----------
    path : str | Path
        ZIP 文件路径，父目录必须存在。
    result : OptimizationResult
        原始结果，默认从中读取 problem、solver_policy 和 theta_seed。
    report : InfeasibilityReport | None
        已有诊断报告，可选；包含完整列式原生证据，不隐式启动诊断。
    problem : PortfolioProblem | None
        显式原问题，例如序列的 stopped_problem；必须匹配结果 fingerprint。
    policy : SolverPolicy | None
        仅当旧结果未保存策略时补充，不能覆盖结果已有策略。
    overwrite : bool
        默认禁止覆盖已有文件。

    Returns
    -------
    Path
        输出文件路径。包含 alpha、持仓和风险模型等敏感数据，请限制传输范围。

    Raises
    ------
    ValueError
        缺少输入、策略冲突或原问题已变更。
    TypeError
        自定义数据包含白名单之外的对象。
    FileExistsError
        文件已存在且未允许覆盖。

    Notes
    -----
    不保存 license、日志、源码或跨期 workspace 内存；复现的是独立单期请求，
    不是进程生命周期。SHA256 检查传输完整性，不提供来源认证。
    """
    resolved = problem if problem is not None else result.problem
    actual_policy = result.solver_policy if result.solver_policy is not None else policy
    if resolved is None or actual_policy is None or result.fingerprint is None:
        raise ValueError("repro requires original problem, policy and fingerprint")
    if (
        policy is not None
        and result.solver_policy is not None
        and policy != result.solver_policy
    ):
        raise ValueError("policy differs from original solve policy")
    if compile_problem(resolved).fingerprint != result.fingerprint:
        raise ValueError("problem fingerprint differs from original result")
    members: dict[str, bytes] = {}
    snapshot = {
        f.name: _json_value(getattr(result, f.name))
        for f in fields(result)
        if f.name
        not in {
            "problem",
            "solver_policy",
            "diagnostics",
            "native_infeasibility",
            "weights",
        }
    }
    snapshot["weights"] = (
        None
        if result.weights is None
        else {
            "assets": _json_value(result.weights.index.tolist()),
            "values": _json_value(result.weights.to_numpy()),
        }
    )
    snapshot["native_infeasibility"] = [
        _certificate_export(v, "full") for v in result.native_infeasibility
    ]
    report_snapshot = None
    if report is not None:
        if not report.contributors_complete:
            raise ValueError(
                "repro requires a full report; contributors were omitted, pass report=None instead"
            )
        report_snapshot = {
            f.name: _json_value(getattr(report, f.name))
            for f in fields(report)
            if f.name != "native_certificates"
        }
        report_snapshot["native_certificates"] = [
            _certificate_export(v, "full") for v in report.native_certificates
        ]
    payload = {
        "format_version": 1,
        "problem": _encode(resolved, members),
        "policy": _encode(actual_policy, members),
        "theta_seed": result.theta_seed,
        "original_result": snapshot,
        "original_report": report_snapshot,
        "environment": _environment(),
    }
    members["payload.json"] = json.dumps(
        payload, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in members.items()
    }
    destination = Path(path)
    with ZipFile(
        destination, "w" if overwrite else "x", compression=ZIP_DEFLATED
    ) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest))
    return destination


def load_repro(
    path: str | Path, *, max_uncompressed_bytes: int = 1_073_741_824
) -> ReproCase:
    """校验并加载复现包，不取数、不求解、不诊断，不向文件系统解压。

    Parameters
    ----------
    path : str | Path
        export_repro 生成的 ZIP 文件。
    max_uncompressed_bytes : int
        解压成员总字节数上限，默认 1 GiB；内存充足时可显式增大。

    Returns
    -------
    ReproCase
        可直接 solve，或从 problem.with_constraints 派生对照问题的复现对象。

    Raises
    ------
    ValueError
        文件结构、哈希、版本、类型或大小限制不符合约定。

    Notes
    -----
    无 pickle 和动态代码加载，但仍应只接收可信来源的包；不承诺跨 schema 自动迁移。
    不要求旧 canonical hash 与新编译器相等，以便在修复后的版本验证原始输入。
    """
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if (
            len(names) != len(set(names))
            or sum(v.file_size for v in infos) > max_uncompressed_bytes
        ):
            raise ValueError("duplicate members or repro size limit exceeded")
        manifest = json.loads(archive.read("manifest.json"))
        if set(names) != set(manifest) | {"manifest.json"}:
            raise ValueError("repro manifest member mismatch")
        members = {name: archive.read(name) for name in manifest}
    if any(
        hashlib.sha256(data).hexdigest() != manifest[name]
        for name, data in members.items()
    ):
        raise ValueError("repro checksum mismatch")
    payload = json.loads(members["payload.json"])
    if payload["format_version"] != 1:
        raise ValueError("unsupported repro format version")
    problem = _decode(payload["problem"], members)
    policy = _decode(payload["policy"], members)
    if not isinstance(problem, pt.PortfolioProblem) or not isinstance(
        policy, pt.SolverPolicy
    ):
        raise ValueError("invalid repro root types")
    environment = payload["environment"]
    current = _environment()
    differences = {
        k: (v, current.get(k, "unknown"))
        for k, v in environment.items()
        if current.get(k) != v
    }
    return ReproCase(
        problem,
        policy,
        payload["theta_seed"],
        payload["original_result"],
        payload["original_report"],
        environment,
        differences,
    )

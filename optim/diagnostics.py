"""不可行诊断的稳定公共结果类型。

诊断算法属于私有二进制核心；本模块只定义调用方可以长期依赖、序列化和展示的结果契约。
普通求解失败不会自动运行深度诊断，调用方需显式调用
``PortfolioOptimizer.diagnose(...)``。
该入口拒绝成功结果；单独传入问题时不查询求解历史。报告类型名称不表示问题必然不可行。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import gzip
import inspect
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .portfolio_types import NativeInfeasibilityEvidence, SolverAttempt


@dataclass(frozen=True)
class RequiredRelaxation:
    """一个加权 Phase-I 松弛方案中的 canonical 边界。

    方案可能不唯一；某条边界的松弛量不是该边界必须放宽的独立下界。

    Attributes
    ----------
    constraint_id : str
        稳定的 canonical 约束标识。
    group : str
        业务约束组。
    side : str
        需要放宽的边界侧，``"lower"`` 或 ``"upper"``。
    amount : float
        此方案中使用该约束原始单位表示的非负松弛幅度，不是新边界。
        lower 从 configured_bound 减去 amount；upper 则加上 amount。
    configured_bound : float
        原问题配置的边界值。
    diagnostic_scale : float
        Phase-I 目标中用于跨单位比较的正数尺度。
    key : str | None
        适用时的资产、因子或属性键。
    sources : tuple[str, ...]
        构成该有效边界的用户配置或运营指令来源。
    metadata : Mapping[str, Any]
        原 canonical registry 的只读业务元数据。
    """

    constraint_id: str
    group: str
    side: str
    amount: float
    configured_bound: float
    diagnostic_scale: float
    key: str | None = None
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def relaxed_bound(self) -> float:
        """放宽后的边界；下界减去松弛幅度，上界加上幅度，不独立存储。"""
        if self.side not in {"lower", "upper"}:
            raise ValueError("relaxation side must be lower or upper")
        return self.configured_bound + (
            -self.amount if self.side == "lower" else self.amount
        )

    @property
    def unit(self) -> str:
        """展示单位：明确的权重约束用 weight，其余保留 original，不猜测自定义属性单位。"""
        return (
            "weight"
            if self.group
            in {
                "asset_bound",
                "turnover",
                "total_active",
                "benchmark_member_weight",
                "budget",
            }
            else "original"
        )

    @property
    def description(self) -> str:
        """中文边界变化说明；百分比仅用于已知权重单位，不改变原数值精度。"""
        label = "下界" if self.side == "lower" else "上界"
        direction = "降低" if self.side == "lower" else "提高"
        if self.unit == "weight":
            return (
                f"{self.constraint_id}：{label}从 {self.configured_bound:.4%} "
                f"{direction}至 {self.relaxed_bound:.4%}，{direction} {self.amount * 100:.4f} 个百分点"
            )
        return (
            f"{self.constraint_id}：{label}从 {self.configured_bound:.8g} "
            f"{direction}至 {self.relaxed_bound:.8g}，幅度 {self.amount:.8g}（原约束单位）"
        )


@dataclass(frozen=True)
class InfeasibilityReport:
    """结构化不可行证据，而不是自动应用的修复。

    Attributes
    ----------
    stage : str
        已执行的诊断阶段，当前深度诊断为 ``"deep"``。
    linear_feasible : bool | None
        综合 Phase-I 和最小换手率证据：True 表示已验收线性候选；False 表示数值下界支持
        不可行；None 表示未确定或证据冲突。
    summary_text : str
        面向使用者的中文诊断摘要。
    turnover_linear_lower_bound : float | None
        保持其余线性约束时的最小 L1 换手率的数值对偶下界，不使用 primal 候选冒充下界。
        与已检查候选冲突时返回 None，原值和冲突说明保留在 native_evidence 中供核查。
    turnover_convex_minimum : float | None
        恢复搜索确认的完整凸问题可行换手率边界或上界。
    turnover_limit : float | None
        原问题配置的换手率上限。
    minimum_tracking_error : float | None
        风险最小化的已验收候选 TE，为最小值的数值上界；高于预算不能独立证明不可行。
    tracking_error_limit : float | None
        原问题配置的年化小数跟踪误差预算。
    relaxations : tuple[RequiredRelaxation, ...]
        Phase-I 松弛方案，按尺度化严重程度降序排列；不是各边界必须放宽的最小幅度，
        也不保证满足风险预算。导出含 configured_bound（原边界）、amount（非负幅度）、
        relaxed_bound（新边界：lower 减、upper 加）、unit（weight 为小数权重，original
        为原约束单位）、description（仅供阅读的变化说明）。
    native_evidence : Mapping[str, Any] | None
        只读的诊断阶段状态和目标值；保留用于兼容已有展示代码。
    native_certificates : tuple[NativeInfeasibilityEvidence, ...]
        原问题各后端求解时已经产生的结构化证书，不会由诊断层伪造或解析日志。
        默认文本表示（``str`` / ``repr``）省略此字段，避免大量证据淹没摘要；可直接访问。
    attempts : tuple[SolverAttempt, ...]
        诊断过程中执行的全部后端尝试。
    """

    stage: str
    linear_feasible: bool | None
    summary_text: str
    turnover_linear_lower_bound: float | None = None
    turnover_convex_minimum: float | None = None
    turnover_limit: float | None = None
    minimum_tracking_error: float | None = None
    tracking_error_limit: float | None = None
    relaxations: tuple[RequiredRelaxation, ...] = ()
    native_evidence: Mapping[str, Any] | None = None
    native_certificates: tuple[NativeInfeasibilityEvidence, ...] = field(
        default=(), repr=False
    )
    attempts: tuple[SolverAttempt, ...] = ()

    def __post_init__(self) -> None:
        """把后端证据冻结为只读映射，避免报告在返回后被意外修改。"""

        object.__setattr__(
            self,
            "native_evidence",
            MappingProxyType(
                {} if self.native_evidence is None else dict(self.native_evidence)
            ),
        )

    def dump(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        overwrite: bool = False,
        evidence: str = "summary",
    ) -> Path:
        """导出带集中字段说明的完整 UTF-8 JSON 报告，便于传输和离线阅读。

        文件依次包含 ``format_version``、``field_descriptions`` 和 ``report``。
        字段说明直接取自本类 Attributes 文档。默认原生证据仅按组汇总，所有 Phase-I 松弛
        均保留。``evidence='full'`` 以列式数组导出全部原生坐标。文件不包含可重放求解的原模型。

        Parameters
        ----------
        path : str | pathlib.Path
            输出文件路径；以 ``.gz`` 结尾时使用 gzip 压缩。父目录需已存在。
        indent : int | None
            JSON 缩进空格数，默认 ``2``，便于阅读。``None`` 输出紧凑 JSON；与 gzip 压缩
            独立。非 None 时必须为非负整数。
        overwrite : bool
            是否覆盖已有文件；默认为 ``False``。
        evidence : str
            ``"summary"``（默认）只导出原生证据来源、质量、数量和约束组计数；
            ``"full"`` 保留每个贡献对象的全部字段，按列存储，减少重复字段名。
            乘子大小不作为冲突重要性排名；风险锥在摘要中只计为一个约束组。

        Returns
        -------
        pathlib.Path
            写入的文件路径。

        Raises
        ------
        FileExistsError
            文件已存在且未允许覆盖。
        OSError
            目录不存在、权限不足或写入失败。
        TypeError
            自定义 metadata 包含不支持的对象或非字符串映射键。
        ValueError
            ``indent`` 不是非负整数或 None，或 evidence 不受支持。

        Notes
        -----
        枚举导出其值，日期使用 ISO 格式，NumPy 数组导出为列表。非有限数值导出为字符串
        ``"NaN"``、``"Infinity"``、``"-Infinity"``，避免产生非标准 JSON。
        本方法只序列化现有报告，不重新诊断、不调用求解器。
        """

        if indent is not None and (type(indent) is not int or indent < 0):
            raise ValueError("indent must be a non-negative integer or None")
        if evidence not in {"summary", "full"}:
            raise ValueError("evidence must be 'summary' or 'full'")
        destination = Path(path)
        report = {
            item.name: _json_value(getattr(self, item.name))
            for item in fields(self)
            if item.name != "native_certificates"
        }
        report["native_certificates"] = [
            _certificate_export(item, evidence) for item in self.native_certificates
        ]
        payload = {
            "format_version": 2,
            "field_descriptions": _field_descriptions(type(self)),
            "evidence_mode": evidence,
            "report": report,
        }
        mode = "wt" if overwrite else "xt"
        stream = (
            gzip.open(destination, mode, encoding="utf-8")
            if destination.suffix.lower() == ".gz"
            else destination.open(mode, encoding="utf-8")
        )
        with stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
                separators=(",", ":") if indent is None else None,
            )
            stream.write("\n")
        return destination


def _certificate_export(
    certificate: NativeInfeasibilityEvidence, mode: str
) -> dict[str, Any]:
    """默认只扫描计数，不构造逐贡献 JSON；完整模式按字段列存储全部带符号证据。"""

    result = {
        item.name: _json_value(getattr(certificate, item.name))
        for item in fields(certificate)
        if item.name != "contributors"
    }
    groups: dict[str, int] = {}
    for item in certificate.contributors:
        groups[item.group] = groups.get(item.group, 0) + 1
    result["contributor_count"] = len(certificate.contributors)
    result["group_counts"] = groups
    if mode == "full":
        from .portfolio_types import InfeasibilityContributor

        result["contributors"] = {
            "encoding": "columns",
            "columns": {
                field.name: [
                    _json_value(getattr(item, field.name))
                    for item in certificate.contributors
                ]
                for field in fields(InfeasibilityContributor)
            },
        }
    return result


def _field_descriptions(report_type: type) -> dict[str, str]:
    """提取本项目固定 NumPy 风格 Attributes 段，避免复制维护一份字段字典。"""

    names = {item.name for item in fields(report_type)}
    descriptions: dict[str, str] = {}
    current = None
    for line in (inspect.getdoc(report_type) or "").splitlines():
        name, separator, _ = line.partition(" : ")
        if separator and name in names:
            current = name
            descriptions[current] = ""
        elif current is not None and line.startswith("    "):
            descriptions[current] += line.strip()
        elif line.strip():
            current = None
    if descriptions.keys() != names or not all(descriptions.values()):
        raise ValueError("report Attributes documentation is incomplete")
    return descriptions


def _json_value(value: Any) -> Any:
    """转换报告中的只读映射与数值类型；不依赖 repr 或深复制 dataclass。"""

    import numpy as np

    if is_dataclass(value) and not isinstance(value, type):
        result = {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
        if isinstance(value, RequiredRelaxation):
            result.update(
                relaxed_bound=_json_value(value.relaxed_bound),
                unit=value.unit,
                description=value.description,
            )
        return result
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("report mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")

"""导出并重放多期优化数据，不保存逐期优化结果或使用 pickle。

生产导出通过 ``Tuda2DataSource.prepare_sequence`` 一次完成 I/O、严格对齐和预检，
随后把优化器实际消费的内存表保存为 Parquet。开发机加载为 ``InMemoryDataSource``，
因此不需要原 fizzdb、tuda2 地址或生产凭据。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pandas as pd

from optim import (
    AlphaSpec,
    BenchmarkCoveragePolicy,
    InMemoryDataSource,
    PortfolioConstraints,
    PortfolioOptimizer,
    PortfolioSchedule,
    PortfolioSequenceResult,
    SequencePolicy,
    SolverPolicy,
)
from optim.data import FactorRiskFrames
from optim.portfolio_types import PortfolioObjective
from optim.repro import _decode, _encode, _environment


FORMAT_VERSION = 1
_PARQUET_MEMBERS = (
    "schedule.parquet",
    "exposure.parquet",
    "covariance.parquet",
    "specific_volatility.parquet",
    "benchmark.parquet",
)


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "sequence data package requires pyarrow; install pyarrow on export and replay machines"
        ) from exc


def _series_frame(value: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one value column")
        result = value.copy(deep=False)
        result.columns = [name]
        return result
    return value.rename(name).to_frame()


def _returns_frame(
    returns: Mapping[pd.Timestamp, pd.Series] | None,
) -> pd.DataFrame | None:
    if returns is None:
        return None
    pieces = {}
    for date, value in returns.items():
        if not isinstance(value, pd.Series):
            raise TypeError(
                "portable sequence packages require labelled Series returns"
            )
        if value.index.has_duplicates:
            raise ValueError(
                f"holding returns contain duplicate sid on {pd.Timestamp(date).date()}"
            )
        item = value.rename("return").copy(deep=False)
        item.index = pd.Index(item.index.astype(str), name="sid")
        pieces[pd.Timestamp(date)] = item
    if not pieces:
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["dt", "sid"]
        )
        return pd.DataFrame({"return": pd.Series(dtype=float)}, index=index)
    result = pd.concat(pieces, names=["dt"])
    result.index = result.index.set_names(["dt", "sid"])
    return result.sort_index().to_frame()


def _dated_weights_frame(
    values: Mapping[pd.Timestamp, pd.Series] | None,
) -> pd.DataFrame | None:
    if values is None:
        return None
    pieces = {}
    for date, value in values.items():
        if not isinstance(value, pd.Series) or value.index.has_duplicates:
            raise TypeError(
                "independent initial weights must be unique labelled Series"
            )
        item = value.rename("weight").copy(deep=False)
        item.index = pd.Index(item.index.astype(str), name="sid")
        pieces[pd.Timestamp(date)] = item
    if not pieces:
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["dt", "sid"]
        )
        return pd.DataFrame({"weight": pd.Series(dtype=float)}, index=index)
    result = pd.concat(pieces, names=["dt"])
    result.index = result.index.set_names(["dt", "sid"])
    return result.sort_index().to_frame()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(path: Path, value: pd.DataFrame) -> None:
    value.to_parquet(path, engine="pyarrow", compression="zstd", index=True)


def export_tuda2_sequence(
    path: str | Path,
    *,
    data_source: Any,
    schedule: PortfolioSchedule,
    benchmark: str | pd.Series,
    objective: PortfolioObjective,
    constraints: PortfolioConstraints,
    alpha_spec: AlphaSpec | None,
    solver_policy: SolverPolicy | None = None,
    initial_weight: pd.Series | None = None,
    sequence_policy: SequencePolicy | None = None,
    benchmark_policy: BenchmarkCoveragePolicy | None = None,
    holding_period_returns: Mapping[pd.Timestamp, pd.Series] | None = None,
    independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
    tradable_universe: str | None = None,
    extra_attribute_columns: tuple[str, ...] = (),
    overwrite: bool = False,
) -> Path:
    """批量取得并导出一次真实多期请求的可移植数据包。

    本函数执行 tuda2 区间取数和优化器同级预检，但不调用任何求解器。导出的是对齐后的
    schedule、风险表、基准、区间收益、初始持仓及业务配置；不包含逐日结果权重、license、
    环境变量、数据库地址或求解器 workspace。

    ``path`` 必须是新文件，除非显式 ``overwrite=True``。包中含 alpha、持仓与风险模型等
    敏感数据，应按内部数据权限传输。需要 export/replay 两端安装 pyarrow。
    """

    _require_parquet()
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    resolved_sequence = SequencePolicy() if sequence_policy is None else sequence_policy
    resolved_solver = SolverPolicy() if solver_policy is None else solver_policy
    prepared = data_source.prepare_sequence(
        schedule=schedule,
        benchmark=benchmark,
        initial_weight=initial_weight,
        objective=objective,
        constraints=constraints,
        alpha_spec=alpha_spec,
        benchmark_policy=benchmark_policy,
        independent_initial_weights=independent_initial_weights,
        holding_period_returns=holding_period_returns,
        require_holding_returns=resolved_sequence.mode == "chained",
        tradable_universe=tradable_universe,
        extra_attribute_columns=extra_attribute_columns,
        sequence_policy=resolved_sequence,
    )
    run = prepared.run
    risk = run.data_source.risk_data
    keep_columns = list(
        dict.fromkeys(
            [
                run.schedule.alpha_column,
                run.schedule.tradable_column,
                *run.extra_attribute_columns,
            ]
        )
    )
    missing = [column for column in keep_columns if column not in run.schedule.universe]
    if missing:
        raise ValueError(f"prepared schedule is missing exported columns: {missing}")
    config_members: dict[str, bytes] = {}
    config = _encode(
        {
            "objective": run.objective,
            "constraints": run.constraints,
            "alpha_spec": run.alpha_spec,
            "sequence_policy": resolved_sequence,
            "solver_policy": resolved_solver,
            "risk_provenance": risk.provenance,
            "factor_types": dict(risk.factor_types),
            "constant_exposures": dict(risk.constant_exposures),
        },
        config_members,
    )
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "environment": _environment(),
        "schedule": {
            "alpha_column": run.schedule.alpha_column,
            "tradable_column": run.schedule.tradable_column,
            "extra_attribute_columns": list(run.extra_attribute_columns),
        },
        "benchmark_policy": {
            "action": run.data_source.benchmark_policy.action,
            "missing_mass_tolerance": run.data_source.benchmark_policy.missing_mass_tolerance,
            "weight_sum_tolerance": run.data_source.benchmark_policy.weight_sum_tolerance,
        },
        "has_returns": prepared.holding_period_returns is not None,
        "has_initial_weight": run.initial_weight is not None,
        "has_independent_initial_weights": run.independent_initial_weights is not None,
        "members": {},
    }
    with TemporaryDirectory(prefix="optim-sequence-export-") as temporary:
        root = Path(temporary)
        frames = {
            "schedule.parquet": run.schedule.universe.loc[:, keep_columns],
            "exposure.parquet": risk.exposure,
            "covariance.parquet": risk.covariance,
            "specific_volatility.parquet": _series_frame(
                risk.specific_volatility, "specific_volatility"
            ),
            "benchmark.parquet": _series_frame(run.data_source.benchmark, "weight"),
        }
        returns_frame = _returns_frame(prepared.holding_period_returns)
        if returns_frame is not None:
            frames["returns.parquet"] = returns_frame
        if run.initial_weight is not None:
            frames["initial_weight.parquet"] = _series_frame(
                run.initial_weight, "weight"
            )
        independent = _dated_weights_frame(run.independent_initial_weights)
        if independent is not None:
            frames["independent_initial_weights.parquet"] = independent
        for name, frame in frames.items():
            _write_parquet(root / name, frame)
        config_members["config.json"] = json.dumps(
            config, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        for name, content in config_members.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        for item in root.rglob("*"):
            if item.is_file():
                relative = item.relative_to(root).as_posix()
                manifest["members"][relative] = {
                    "sha256": _hash_file(item),
                    "size": item.stat().st_size,
                }
        mode = "w" if overwrite else "x"
        with ZipFile(destination, mode, allowZip64=True) as archive:
            for name in sorted(manifest["members"]):
                compression = ZIP_STORED if name.endswith(".parquet") else ZIP_DEFLATED
                archive.write(root / name, name, compress_type=compression)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2),
                compress_type=ZIP_DEFLATED,
            )
    return destination


@dataclass(frozen=True)
class SequencePackage:
    """已加载的多期数据和配置；调用 ``solve`` 才执行优化。

    Attributes
    ----------
    data_source : InMemoryDataSource
        绑定已对齐风险表和基准的内存数据源。
    schedule : PortfolioSchedule
        原请求实际使用的日期、资产、alpha 和可交易性计划。
    objective : PortfolioObjective
        各期共享的业务目标。
    constraints : PortfolioConstraints
        各期共享的约束配置。
    alpha_spec : AlphaSpec | None
        alpha 的单位和尺度声明。
    initial_weight : pandas.Series | None
        链式首期显式持仓；原请求省略时仍为空。
    independent_initial_weights : Mapping | None
        独立模式下按日期保存的期初持仓。
    holding_period_returns : Mapping | None
        以当前调仓日为键的 close-to-close 区间复合收益。
    sequence_policy : SequencePolicy
        持仓推进、失败处理和输出策略。
    solver_policy : SolverPolicy
        导出请求使用的求解策略；重放时可只覆盖 backend。
    environment : Mapping[str, str]
        导出机 Python、平台和主要依赖版本，不含环境变量。
    extra_attribute_columns : tuple[str, ...]
        从 schedule 物化为逐资产附加数值的列名。
    """

    data_source: InMemoryDataSource
    schedule: PortfolioSchedule
    objective: PortfolioObjective
    constraints: PortfolioConstraints
    alpha_spec: AlphaSpec | None
    initial_weight: pd.Series | None
    independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None
    holding_period_returns: Mapping[pd.Timestamp, pd.Series] | None
    sequence_policy: SequencePolicy
    solver_policy: SolverPolicy
    environment: Mapping[str, str]
    extra_attribute_columns: tuple[str, ...]

    def solve(
        self, *, backend: str | None = None, show_progress: bool = True
    ) -> PortfolioSequenceResult:
        """使用包内配置重放；backend 非空时只覆盖求解后端选择。"""
        policy = self.solver_policy
        if backend is not None:
            policy = replace(policy, backend=backend)
        return PortfolioOptimizer(policy).optimize_range(
            data_source=self.data_source,
            schedule=self.schedule,
            objective=self.objective,
            constraints=self.constraints,
            alpha_spec=self.alpha_spec,
            initial_weight=self.initial_weight,
            holding_period_returns=self.holding_period_returns,
            sequence_policy=self.sequence_policy,
            independent_initial_weights=self.independent_initial_weights,
            extra_attribute_columns=self.extra_attribute_columns,
            show_progress=show_progress,
        )


def load_sequence_package(
    path: str | Path, *, max_uncompressed_bytes: int = 8 * 1024**3
) -> SequencePackage:
    """校验哈希并加载数据包；不取数、不求解，也不加载可执行对象。"""
    _require_parquet()
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("sequence package contains duplicate members")
        if "manifest.json" not in names:
            raise ValueError("sequence package has no manifest")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported sequence package format")
        expected = manifest.get("members", {})
        if set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("sequence package manifest member mismatch")
        if sum(item.file_size for item in infos) > max_uncompressed_bytes:
            raise ValueError("sequence package size limit exceeded")
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in expected
        ):
            raise ValueError("sequence package contains an unsafe member name")
        with TemporaryDirectory(prefix="optim-sequence-load-") as temporary:
            root = Path(temporary)
            for name, metadata in expected.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if (
                    target.stat().st_size != metadata["size"]
                    or _hash_file(target) != metadata["sha256"]
                ):
                    raise ValueError(f"sequence package checksum mismatch: {name}")
            frames = {
                name: pd.read_parquet(root / name, engine="pyarrow")
                for name in _PARQUET_MEMBERS
            }
            optional = {}
            for name in (
                "returns.parquet",
                "initial_weight.parquet",
                "independent_initial_weights.parquet",
            ):
                if name in expected:
                    optional[name] = pd.read_parquet(root / name, engine="pyarrow")
            member_bytes = {
                name: (root / name).read_bytes()
                for name in expected
                if name == "config.json" or name.startswith("arrays/")
            }
    config = _decode(json.loads(member_bytes["config.json"]), member_bytes)
    schedule_info = manifest["schedule"]
    schedule = PortfolioSchedule(
        frames["schedule.parquet"],
        alpha_column=schedule_info["alpha_column"],
        tradable_column=schedule_info["tradable_column"],
    )
    risk = FactorRiskFrames(
        exposure=frames["exposure.parquet"],
        covariance=frames["covariance.parquet"],
        specific_volatility=frames["specific_volatility.parquet"].iloc[:, 0],
        factor_types=config["factor_types"],
        provenance=config["risk_provenance"],
        constant_exposures=config["constant_exposures"],
    )
    coverage = BenchmarkCoveragePolicy(**manifest["benchmark_policy"])
    data_source = InMemoryDataSource(
        risk_data=risk,
        benchmark=frames["benchmark.parquet"].iloc[:, 0],
        benchmark_policy=coverage,
    )
    initial = optional.get("initial_weight.parquet")
    initial_weight = None if initial is None else initial.iloc[:, 0]
    independent_frame = optional.get("independent_initial_weights.parquet")
    independent = None
    if independent_frame is not None:
        independent = {
            pd.Timestamp(date): day.droplevel("dt").iloc[:, 0]
            for date, day in independent_frame.groupby(level="dt", sort=False)
        }
    returns_frame = optional.get("returns.parquet")
    returns = None
    if returns_frame is not None:
        returns = {
            pd.Timestamp(date): day.droplevel("dt").iloc[:, 0]
            for date, day in returns_frame.groupby(level="dt", sort=False)
        }
    return SequencePackage(
        data_source=data_source,
        schedule=schedule,
        objective=config["objective"],
        constraints=config["constraints"],
        alpha_spec=config["alpha_spec"],
        initial_weight=initial_weight,
        independent_initial_weights=independent,
        holding_period_returns=returns,
        sequence_policy=config["sequence_policy"],
        solver_policy=config["solver_policy"],
        environment=manifest["environment"],
        extra_attribute_columns=tuple(schedule_info["extra_attribute_columns"]),
    )


def performance_frame(result: PortfolioSequenceResult) -> pd.DataFrame:
    """将重放结果转换成不含权重的逐期性能表。"""
    rows = []
    for step in result.steps:
        item = step.result
        timings = item.timings
        rows.append(
            {
                "date": step.date,
                "status": item.status.value,
                "backend": item.backend,
                "attempts": len(item.route),
                "route": " -> ".join(
                    f"{v.backend}:{v.status.value}" for v in item.route
                ),
                "objective_value": item.objective_value,
                "tracking_error": item.metrics.tracking_error,
                "turnover_l1": item.metrics.turnover_l1,
                "max_violation": max(
                    (violation.amount for violation in item.violations), default=0.0
                ),
                "prepare_s": timings.prepare_s,
                "compile_s": timings.compile_s,
                "backend_setup_s": timings.backend_setup_s,
                "backend_solve_s": timings.backend_solve_s,
                "validation_s": timings.validation_s,
                "postprocess_s": timings.postprocess_s,
                "total_s": timings.total_s,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """从命令行加载数据包、重放多期问题并可导出无权重性能表。"""

    parser = argparse.ArgumentParser(description="重放已导出的多期数据包")
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--backend", choices=["auto", "mosek", "highs", "piqp", "clarabel"]
    )
    parser.add_argument("--performance-csv", type=Path)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--max-uncompressed-gib", type=float, default=8.0)
    args = parser.parse_args()
    case = load_sequence_package(
        args.package,
        max_uncompressed_bytes=int(args.max_uncompressed_gib * 1024**3),
    )
    started = time.perf_counter()
    result = case.solve(backend=args.backend, show_progress=args.show_progress)
    elapsed = time.perf_counter() - started
    frame = performance_frame(result)
    if args.performance_csv is not None:
        frame.to_csv(args.performance_csv, index=False)
    print(
        json.dumps(
            {
                "periods": len(result.steps),
                "stopped_date": None
                if result.stopped_date is None
                else str(result.stopped_date),
                "wall_s": elapsed,
                "periods_per_s": len(result.steps) / elapsed if elapsed else None,
                "statuses": frame["status"].value_counts().to_dict(),
                "backends": frame["backend"].value_counts(dropna=False).to_dict(),
                "performance_csv": None
                if args.performance_csv is None
                else str(args.performance_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

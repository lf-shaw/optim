"""在生产 Notebook 复用实际请求，拆分静态预检耗时并核对风险覆盖。

将此文件放在 Notebook 旁边并导入 run_probe。只加载风险、基准和所需可交易性；
重复预检复用同一批数据。结果不包含 alpha、原始风险值或逐股基准权重。
"""

from collections import Counter, defaultdict
from contextlib import ExitStack, nullcontext
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
import pandas as pd

from optim import InMemoryDataSource, SequencePolicy
from optim._core.thread_control import effective_cpu_count
from optim._progress import _Progress, _with_progress
from optim.data import memory
from optim.data.contracts import FactorRiskFrames
import optim.validation as validation_module


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _pool_info():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    fields = ("user_api", "internal_api", "num_threads", "version")
    return [{key: pool.get(key) for key in fields} for pool in threadpool_info()]


def _benchmark_prepare(source, schedule, request, mode):
    if mode == "blas1":
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return {"mode": mode, "skipped": "threadpoolctl is unavailable"}, None
        scope = threadpool_limits(limits=1, user_api="blas")
    else:
        scope = nullcontext()

    samples = defaultdict(list)

    def watch(owner, member, label):
        original = getattr(owner, member)

        @wraps(original)
        def observed(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                samples[label].append(perf_counter() - start)

        stack.enter_context(patch.object(owner, member, observed))

    @_with_progress
    def prepare(*, show_progress=False):
        return source.prepare_run(schedule, **request)

    with scope, ExitStack() as stack:
        if hasattr(InMemoryDataSource, "_risk_coverage"):
            watch(InMemoryDataSource, "_risk_coverage", "batch_risk_coverage")
        if hasattr(InMemoryDataSource, "_batch_benchmark"):
            watch(InMemoryDataSource, "_batch_benchmark", "batch_benchmark_audit")
        watch(InMemoryDataSource, "_batch_for_schedule", "batch_align")
        watch(InMemoryDataSource, "_materialize_problem", "materialize_problem")
        watch(FactorRiskFrames, "materialize", "materialize_risk")
        watch(memory, "validate_problem", "validate_problem")
        watch(validation_module, "resolve_asset_bounds", "validate_asset_bounds")
        watch(np.linalg, "eigvalsh", "validate_eigvalsh")
        watch(_Progress, "date", "progress_date")
        watch(_Progress, "advance", "progress_advance")
        watch(_Progress, "phase", "progress_phase")
        pools = _pool_info()
        started, cpu_started = perf_counter(), process_time()
        run = prepare(show_progress=mode == "progress")
        cpu_s, wall_s = process_time() - cpu_started, perf_counter() - started

    stages = {
        name: {
            "calls": len(times),
            "total_s": sum(times),
            "p50_ms": float(np.median(times)) * 1e3,
            "p95_ms": float(np.percentile(times, 95)) * 1e3,
        }
        for name, times in samples.items()
    }
    return {
        "mode": mode,
        "wall_s": wall_s,
        "process_cpu_s": cpu_s,
        "prepare_s": run.prepare_s,
        "after_batch_align_s": run.prepare_s - sum(
            stages.get(name, {}).get("total_s", 0.0)
            for name in ("batch_align", "batch_risk_coverage")
        ),
        "batch_alignment_used": run.data_source is not source,
        "errors": len(run.validation.errors),
        "warnings": len(run.validation.warnings),
        "blas_pools_in_run": pools,
        "stages": stages,
    }, run.validation


def _coverage(source, schedule, sid_aliases):
    target = schedule.universe.index
    exposure = source.risk_data.exposure.index
    specific = source.risk_data.specific_volatility.index
    missing_e = ~target.isin(exposure)
    missing_s = ~target.isin(specific)
    failed = missing_e | missing_s
    missing = target[failed].to_frame(index=False)
    missing["exposure_missing"] = missing_e[failed]
    missing["specific_missing"] = missing_s[failed]

    def count_by_date(index, positions=slice(None)):
        counts = np.bincount(index.codes[0][positions], minlength=len(index.levels[0]))
        return pd.Series(counts, index=index.levels[0]).reindex(
            schedule.dates, fill_value=0
        )

    daily = pd.DataFrame(
        {
            "universe_count": count_by_date(target),
            "risk_union_missing_count": count_by_date(target, failed),
            "exposure_missing_count": count_by_date(target, missing_e),
            "specific_missing_count": count_by_date(target, missing_s),
            "both_missing_count": count_by_date(target, missing_e & missing_s),
            "exposure_rows_in_model": count_by_date(exposure),
            "specific_rows_in_model": count_by_date(specific),
        }
    ).rename_axis("dt")

    benchmark = source.benchmark
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]
    affected = benchmark[benchmark.index.isin(target[failed])]
    affected = pd.to_numeric(affected, errors="coerce")
    daily["benchmark_mass_at_risk_if_dropped"] = (
        affected.groupby(level="dt")
        .sum(min_count=1)
        .reindex(daily.index, fill_value=0.0)
    )
    daily["positive_benchmark_members_if_dropped"] = (
        affected.gt(0).groupby(level="dt").sum().reindex(daily.index, fill_value=0)
    )
    daily["invalid_benchmark_values_on_missing_keys"] = (
        (~np.isfinite(affected))
        .groupby(level="dt")
        .sum()
        .reindex(daily.index, fill_value=0)
    )

    aliases = []
    for historical, current in (sid_aliases or {}).items():
        rows = missing.loc[missing["sid"].eq(current)].copy()
        keys = pd.MultiIndex.from_arrays(
            [rows["dt"], [historical] * len(rows)], names=target.names
        )
        rows["historical_sid"] = historical
        rows["historical_sid_in_exposure"] = keys.isin(exposure)
        rows["historical_sid_in_specific"] = keys.isin(specific)
        rows["historical_sid_in_universe"] = keys.isin(target)
        aliases.append(rows)
    alias_hits = pd.concat(aliases, ignore_index=True) if aliases else None
    return missing, daily, alias_hits


def run_probe(
    *,
    output_dir,
    data_source,
    schedule,
    objective,
    constraints,
    alpha_spec,
    benchmark=None,
    initial_weight=None,
    sequence_policy=None,
    benchmark_policy=None,
    risk_coverage_policy=None,
    independent_initial_weights=None,
    tradable_universe=None,
    extra_attribute_columns=(),
    modes=("baseline", "progress", "blas1"),
    sid_aliases=None,
):
    """使用原 optimize_range 参数调用，省略 optimizer、收益数据和 show_progress。

    output_dir 必须不存在。modes 按指定顺序运行，baseline 建议放在首轮；progress 使用
    当前 Notebook 下的 tqdm 显示；blas1 对该轮局部限制 BLAS，退出后自动恢复。
    sid_aliases 接受 {历史代码: 当前代码}，只对同日坐标进行对照，不改写索引。
    """
    modes = tuple(modes)
    if not modes or set(modes) - {"baseline", "progress", "blas1"}:
        raise ValueError("modes must contain only baseline, progress or blas1")
    if len(schedule.dates) == 0:
        raise ValueError("schedule must contain at least one date")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    loaded_s = None
    tradability_s = None
    if isinstance(data_source, InMemoryDataSource):
        if any(
            item is not None
            for item in (benchmark, benchmark_policy, tradable_universe)
        ):
            raise ValueError("an InMemoryDataSource already binds benchmark and policy")
        source = data_source
    else:
        if tradable_universe is not None:
            started = perf_counter()
            schedule = data_source._attach_tradability(schedule, tradable_universe)
            tradability_s = perf_counter() - started
        print("取数并展开风险敞口", flush=True)
        started = perf_counter()
        loaded = data_source.load(dates=schedule.dates, benchmark=benchmark)
        loaded_s = perf_counter() - started
        source = InMemoryDataSource(
            risk_data=loaded.risk_data,
            benchmark=loaded.benchmark,
            benchmark_policy=benchmark_policy,
        )

    request = dict(
        objective=objective,
        constraints=constraints,
        alpha_spec=alpha_spec,
        initial_weight=initial_weight,
        independent_initial_weights=independent_initial_weights,
        sequence_policy=SequencePolicy()
        if sequence_policy is None
        else sequence_policy,
        extra_attribute_columns=extra_attribute_columns,
    )
    if risk_coverage_policy is not None:
        request["risk_coverage_policy"] = risk_coverage_policy
    effective_cpus, cpu_source = effective_cpu_count()
    report = {
        "risk_model": getattr(data_source, "risk_model", None),
        "versions": {
            name: _version(name)
            for name in (
                "optim",
                "tuda2",
                "pandas",
                "numpy",
                "carry2",
                "tqdm",
                "threadpoolctl",
            )
        },
        "effective_cpus": effective_cpus,
        "effective_cpu_source": cpu_source,
        "dates": len(schedule.dates),
        "first_date": schedule.dates[0].isoformat(),
        "last_date": schedule.dates[-1].isoformat(),
        "universe_rows": len(schedule.universe),
        "load_and_expand_s": loaded_s,
        "attach_tradability_s": tradability_s,
        "runs": [],
        "timing_notes": (
            "after_batch_align_s subtracts batch_align and batch_risk_coverage from prepare_s. "
            "batch_benchmark_audit is included in batch_align. "
            "materialize_risk is inside materialize_problem; eigvalsh and bounds are inside validate_problem. "
            "Runs reuse source inputs; later runs may benefit from warmed index and data caches."
        ),
    }
    reference_issues = None
    for mode in modes:
        print(f"运行静态预检 {mode}", flush=True)
        result, report_for_run = _benchmark_prepare(source, schedule, request, mode)
        report["runs"].append(result)
        if report_for_run is not None:
            if reference_issues is None:
                reference_issues = report_for_run.issues
            result["issues_match_first_run"] = report_for_run.issues == reference_issues
            print(
                f"{mode} 用时 {result['wall_s']:.2f} 秒，发现 {result['errors']} 条错误",
                flush=True,
            )
        else:
            print(f"{mode} 跳过，{result['skipped']}", flush=True)

    issues = reference_issues or ()
    pd.DataFrame(
        [_issue_record(issue) for issue in issues],
        columns=["date", "severity", "field", "code", "message"],
    ).to_csv(output / "precheck_issues.csv", index=False)
    report["issue_counts"] = [
        {"severity": severity, "field": field, "code": code, "count": count}
        for (severity, field, code), count in Counter(
            (issue.severity.value, issue.field, issue.code) for issue in issues
        ).most_common()
    ]

    print("核对全区间风险坐标", flush=True)
    started = perf_counter()
    missing, daily, aliases = _coverage(source, schedule, sid_aliases)
    max_at_risk = float(daily["benchmark_mass_at_risk_if_dropped"].max())
    report["coverage"] = {
        "audit_s": perf_counter() - started,
        "dates_with_missing_risk": int(daily["risk_union_missing_count"].gt(0).sum()),
        "missing_coordinates": len(missing),
        "missing_sids": int(missing["sid"].nunique()),
        "dates_with_no_risk_rows": int(
            daily[["exposure_rows_in_model", "specific_rows_in_model"]]
            .eq(0)
            .any(axis=1)
            .sum()
        ),
        "max_benchmark_mass_at_risk_if_dropped": max_at_risk
        if np.isfinite(max_at_risk)
        else None,
    }
    missing.to_csv(output / "coverage_missing.csv.gz", index=False)
    daily.to_csv(output / "coverage_daily.csv")
    if aliases is not None:
        aliases.to_csv(output / "sid_alias_checks.csv", index=False)
        report["coverage"]["alias_coordinates_with_both_risk_inputs"] = int(
            (
                aliases["historical_sid_in_exposure"]
                & aliases["historical_sid_in_specific"]
            ).sum()
        )
    with (output / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"报告已写入 {output}", flush=True)
    return report


def _issue_record(issue):
    # ValidationIssue.context 是 MappingProxyType，不能用 dataclasses.asdict 深拷贝。
    return {
        "date": issue.date,
        "severity": issue.severity.value,
        "field": issue.field,
        "code": issue.code,
        "message": issue.message,
    }

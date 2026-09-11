"""用多期数据包检查生产运行时、CPU 调度和模型准备热点，不导出权重。"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import sys
import sysconfig
import time
import tracemalloc
from typing import Callable

import numpy as np
import optim
import optim._impl.compiler as compiler_module
import optim._impl.solution as solution_module
from optim.devtools.sequence_data_package import (
    load_sequence_package,
    performance_frame,
)
from optim.fingerprint import canonical_hash, semantic_hash


def _hook_name(value):
    if value is None:
        return None
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _read_numeric_key_values(path: str) -> dict[str, int] | None:
    """读取 cgroup ``key integer`` 状态文件；格式异常时不阻断探针。"""

    text = _read_text(path)
    if text is None:
        return None
    result = {}
    try:
        for line in text.splitlines():
            key, value = line.split()
            result[key] = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _numeric_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], int)
    }


def _threadpool_snapshot() -> list[dict] | None:
    """在已安装 threadpoolctl 时记录实际 BLAS/OpenMP 线程池。"""

    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return threadpool_info()


def _system_snapshot() -> dict:
    """收集不含通用环境变量和凭据的运行时状态。"""
    cpu = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, value = (item.strip() for item in line.split(":", 1))
            if key in {"model name", "cpu MHz"} and key not in cpu:
                cpu[key] = value
    except OSError:
        pass
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    known_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
            "KMP_BLOCKTIME",
            "OMP_WAIT_POLICY",
            "MKL_DYNAMIC",
            "PYTHONMALLOC",
            "PYTHONDEVMODE",
        )
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "optim_version": optim.__version__,
        "optim_module": optim.__file__,
        "compiler_module": compiler_module.__file__,
        "solution_module": solution_module.__file__,
        "trace_hook": _hook_name(sys.gettrace()),
        "profile_hook": _hook_name(sys.getprofile()),
        "tracemalloc": tracemalloc.is_tracing(),
        "dev_mode": sys.flags.dev_mode,
        "optimize_flag": sys.flags.optimize,
        "gc_enabled": gc.isenabled(),
        "gc_threshold": gc.get_threshold(),
        "gc_count": gc.get_count(),
        "affinity": affinity,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "known_environment": known_environment,
        "threadpools": _threadpool_snapshot(),
        "numpy_build": getattr(np.__config__, "CONFIG", None),
        "cpu": cpu,
        "cgroup": {
            "cpu.max": _read_text("/sys/fs/cgroup/cpu.max"),
            "cpu.stat": _read_numeric_key_values("/sys/fs/cgroup/cpu.stat"),
            "cpu.pressure": _read_text("/sys/fs/cgroup/cpu.pressure"),
            "cpuset.cpus.effective": _read_text("/sys/fs/cgroup/cpuset.cpus.effective"),
            "cpuset.mems.effective": _read_text("/sys/fs/cgroup/cpuset.mems.effective"),
            "cpu.cfs_quota_us": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "cpu.cfs_period_us": _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        },
        "python_build": {
            "CONFIG_ARGS": sysconfig.get_config_var("CONFIG_ARGS"),
            "CFLAGS": sysconfig.get_config_var("CFLAGS"),
        },
    }


def _usage() -> dict[str, float]:
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_s": value.ru_utime,
        "system_cpu_s": value.ru_stime,
        "minor_faults": value.ru_minflt,
        "major_faults": value.ru_majflt,
        "voluntary_context_switches": value.ru_nvcsw,
        "involuntary_context_switches": value.ru_nivcsw,
        "max_rss": value.ru_maxrss,
    }


def _measure(name: str, function: Callable[[], object], repeats: int) -> dict:
    """同时记录 wall、进程 CPU 和调度计数，返回每次样本而非只给均值。"""
    samples = []
    for _ in range(repeats):
        before = _usage()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        function()
        cpu_s = time.process_time() - cpu_started
        wall_s = time.perf_counter() - wall_started
        after = _usage()
        samples.append(
            {
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "cpu_to_wall": cpu_s / wall_s if wall_s else None,
                "minor_faults": after["minor_faults"] - before["minor_faults"],
                "major_faults": after["major_faults"] - before["major_faults"],
                "voluntary_context_switches": after["voluntary_context_switches"]
                - before["voluntary_context_switches"],
                "involuntary_context_switches": after["involuntary_context_switches"]
                - before["involuntary_context_switches"],
            }
        )
    walls = [item["wall_s"] for item in samples]
    return {
        "name": name,
        "repeats": repeats,
        "wall_mean_s": statistics.mean(walls),
        "wall_median_s": statistics.median(walls),
        "wall_min_s": min(walls),
        "samples": samples,
    }


def _sequence_measurement(case, backend: str) -> tuple[object, dict]:
    """运行一次未插桩序列，并记录进程与 cgroup 的前后差值。"""

    usage_before = _usage()
    cgroup_before = _read_numeric_key_values("/sys/fs/cgroup/cpu.stat")
    wall_started = time.perf_counter()
    result = case.solve(backend=backend, show_progress=False)
    wall_s = time.perf_counter() - wall_started
    usage_after = _usage()
    cgroup_after = _read_numeric_key_values("/sys/fs/cgroup/cpu.stat")
    process_cpu_s = (
        usage_after["user_cpu_s"]
        + usage_after["system_cpu_s"]
        - usage_before["user_cpu_s"]
        - usage_before["system_cpu_s"]
    )
    frame = performance_frame(result)
    summary = {
        "backend": backend,
        "wall_s": wall_s,
        "process_cpu_s": process_cpu_s,
        "process_cpu_to_wall": process_cpu_s / wall_s if wall_s else None,
        "periods": len(result.steps),
        "usage_delta": {
            key: usage_after[key] - usage_before[key]
            for key in usage_after
            if key != "max_rss"
        },
        "cgroup_cpu_stat_delta": _numeric_delta(cgroup_before, cgroup_after),
        "statuses": frame["status"].value_counts().to_dict(),
        "stopped_date": (
            None if result.stopped_date is None else str(result.stopped_date)
        ),
    }
    return result, summary


def main() -> None:
    """运行准备阶段微基准，并可额外生成一次完整重放的 cProfile。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--profile-backend", choices=["auto", "mosek", "highs", "clarabel"]
    )
    parser.add_argument(
        "--measure-backend", choices=["auto", "mosek", "highs", "clarabel"]
    )
    parser.add_argument("--max-uncompressed-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    case = load_sequence_package(
        args.package,
        max_uncompressed_bytes=int(args.max_uncompressed_gib * 1024**3),
    )
    run = case.data_source.prepare_run(
        case.schedule,
        objective=case.objective,
        constraints=case.constraints,
        alpha_spec=case.alpha_spec,
        initial_weight=case.initial_weight,
        independent_initial_weights=case.independent_initial_weights,
        extra_attribute_columns=case.extra_attribute_columns,
        sequence_policy=case.sequence_policy,
    )
    problem = run.problem_at(run.dates[0])
    compiled = compiler_module.compile_problem(problem)
    # 预热解释器、稀疏构造及哈希代码页，不计入正式样本。
    semantic_hash(problem)
    canonical_hash(compiled.model)
    compiler_module.compile_problem(problem)
    measurements = [
        _measure("semantic_hash", lambda: semantic_hash(problem), args.repeats),
        _measure(
            "canonical_hash", lambda: canonical_hash(compiled.model), args.repeats
        ),
        _measure(
            "compile_problem_gc_enabled",
            lambda: compiler_module.compile_problem(problem),
            args.repeats,
        ),
    ]
    was_enabled = gc.isenabled()
    gc.collect()
    try:
        gc.disable()
        measurements.append(
            _measure(
                "compile_problem_gc_disabled",
                lambda: compiler_module.compile_problem(problem),
                args.repeats,
            )
        )
    finally:
        if was_enabled:
            gc.enable()
    payload = {
        "format_version": 1,
        "package": str(args.package),
        "package_environment": dict(case.environment),
        "runtime": _system_snapshot(),
        "dates": len(case.schedule.dates),
        "first_date": str(case.schedule.dates[0]),
        "first_date_assets": len(problem.data.assets),
        "measurements": measurements,
        "usage_after_microbench": _usage(),
    }
    if args.measure_backend is not None:
        result, summary = _sequence_measurement(case, args.measure_backend)
        performance_frame(result).to_csv(
            args.output_dir / "performance_unprofiled.csv", index=False
        )
        payload["unprofiled_sequence"] = summary
    if args.profile_backend is not None:
        profiler = cProfile.Profile()
        usage_before = _usage()
        cgroup_before = _read_numeric_key_values("/sys/fs/cgroup/cpu.stat")
        wall_started = time.perf_counter()
        result = profiler.runcall(
            case.solve, backend=args.profile_backend, show_progress=False
        )
        wall_s = time.perf_counter() - wall_started
        usage_after = _usage()
        cgroup_after = _read_numeric_key_values("/sys/fs/cgroup/cpu.stat")
        profiler.dump_stats(args.output_dir / "sequence.prof")
        with (args.output_dir / "profile.txt").open("w") as stream:
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
            stats.print_stats(100)
        frame = performance_frame(result)
        frame.to_csv(args.output_dir / "performance_profiled.csv", index=False)
        payload["profiled_sequence"] = {
            "backend": args.profile_backend,
            "wall_s": wall_s,
            "periods": len(result.steps),
            "process_cpu_s": (
                usage_after["user_cpu_s"]
                + usage_after["system_cpu_s"]
                - usage_before["user_cpu_s"]
                - usage_before["system_cpu_s"]
            ),
            "usage_delta": {
                key: usage_after[key] - usage_before[key]
                for key in usage_after
                if key != "max_rss"
            },
            "cgroup_cpu_stat_delta": _numeric_delta(cgroup_before, cgroup_after),
            "statuses": frame["status"].value_counts().to_dict(),
            "stopped_date": None
            if result.stopped_date is None
            else str(result.stopped_date),
        }
    (args.output_dir / "runtime.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

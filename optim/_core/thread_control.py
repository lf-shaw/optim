"""数值求解期间的线程预算解析与动态作用域。"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ThreadResolution:
    """一次线程策略解析后的稳定结果。

    Attributes
    ----------
    policy : str
        ``"auto"``、``"max"`` 或 ``"fixed"``。
    limit : int
        optim 在本次数值作用域内请求的线程上限。
    effective_cpus : int
        同时考虑进程 CPU affinity 与 cgroup quota 后的可用 CPU 上界。
    source : str
        可用 CPU 上界的判定来源，便于生产性能诊断。
    """

    policy: str
    limit: int
    effective_cpus: int
    source: str


def resolve_thread_setting(setting: str | int) -> ThreadResolution:
    """解析公共线程策略，不修改当前进程状态。

    ``auto`` 将通用数值线程上限设为一，具体后端仍可采用已经单独校准的原生自动策略；
    ``max`` 使用容器/进程实际可用 CPU 上界；正整数原样作为调用者明确请求的线程上限，
    不静默截断。
    """

    effective, source = effective_cpu_count()
    if setting == "auto":
        return ThreadResolution("auto", 1, effective, source)
    if setting == "max":
        return ThreadResolution("max", effective, effective, source)
    if isinstance(setting, int) and not isinstance(setting, bool) and setting > 0:
        if setting > effective:
            raise ValueError(
                f"threads={setting} exceeds the effective CPU limit {effective} "
                f"determined from {source}"
            )
        return ThreadResolution("fixed", int(setting), effective, source)
    raise ValueError("threads must be 'auto', 'max' or a positive integer")


def effective_cpu_count(
    *,
    proc_root: str | Path = "/proc",
    cgroup_root: str | Path = "/sys/fs/cgroup",
) -> tuple[int, str]:
    """返回 affinity 与 cgroup CPU quota 共同允许的保守整数 CPU 数。

    cgroup quota 表示调度带宽而非独占物理核心。为避免 ``max`` 因小数 quota 产生持续
    throttling，这里向下取整并至少返回一。无法读取 cgroup 时仅采用 affinity；两者都不可用
    时回退到 :func:`os.cpu_count`。
    """

    try:
        affinity = len(os.sched_getaffinity(0))
        affinity_source = "affinity"
    except (AttributeError, OSError):
        affinity = int(os.cpu_count() or 1)
        affinity_source = "os_cpu_count"
    affinity = max(1, affinity)

    proc = Path(proc_root)
    root = Path(cgroup_root)
    quota = _cgroup_v2_quota(proc, root)
    quota_source = "cgroup_v2"
    if quota is None:
        quota = _cgroup_v1_quota(proc, root)
        quota_source = "cgroup_v1"
    if quota is None:
        return affinity, affinity_source
    quota_cpus = max(1, math.floor(quota))
    return min(affinity, quota_cpus), f"{affinity_source}+{quota_source}"


def _cgroup_v2_quota(proc_root: Path, cgroup_root: Path) -> float | None:
    """读取当前 v2 cgroup 及其祖先中最严格的有限 ``cpu.max``。"""

    try:
        lines = (proc_root / "self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    relative = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative = parts[2].lstrip("/")
            break
    if relative is None:
        return None
    return _minimum_quota(cgroup_root / relative, cgroup_root, "cpu.max")


def _cgroup_v1_quota(proc_root: Path, cgroup_root: Path) -> float | None:
    """读取常见 v1 cpu controller 布局中的最严格有限 quota。"""

    try:
        lines = (proc_root / "self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    relative = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and "cpu" in parts[1].split(","):
            relative = parts[2].lstrip("/")
            break
    if relative is None:
        return None
    for controller_root in (
        cgroup_root / "cpu",
        cgroup_root / "cpu,cpuacct",
        cgroup_root,
    ):
        start = controller_root / relative
        values: list[float] = []
        for directory in _ancestors_within(start, controller_root):
            try:
                quota = int((directory / "cpu.cfs_quota_us").read_text().strip())
                period = int((directory / "cpu.cfs_period_us").read_text().strip())
            except (OSError, ValueError):
                continue
            if quota > 0 and period > 0:
                values.append(quota / period)
        if values:
            return min(values)
    return None


def _minimum_quota(start: Path, root: Path, filename: str) -> float | None:
    values: list[float] = []
    for directory in _ancestors_within(start, root):
        try:
            fields = (directory / filename).read_text(encoding="utf-8").split()
        except OSError:
            continue
        if len(fields) != 2 or fields[0] == "max":
            continue
        try:
            quota, period = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if quota > 0 and period > 0:
            values.append(quota / period)
    return min(values) if values else None


def _ancestors_within(start: Path, root: Path) -> Iterator[Path]:
    """从当前 cgroup 向根遍历，不越过指定挂载根。"""

    current = start
    while True:
        try:
            current.relative_to(root)
        except ValueError:
            return
        yield current
        if current == root:
            return
        parent = current.parent
        if parent == current:
            return
        current = parent


class NumericalThreadScope:
    """为一个优化器实例提供可嵌套、异常安全的 BLAS 线程作用域。

    最外层调用才修改线程池；单期入口内部的 prepare/solve 嵌套以及多期逐日调用都不会
    重复扫描或反复设置动态库。退出最外层作用域时由 threadpoolctl 恢复原值。
    """

    def __init__(self, setting: str | int):
        self.resolution = resolve_thread_setting(setting)
        self._depth: ContextVar[int] = ContextVar(
            f"optim_thread_scope_{id(self)}", default=0
        )
        self._controller: Any | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        """进入当前实例的线程作用域，并在正常或异常退出时恢复原状态。"""

        depth = self._depth.get()
        token = self._depth.set(depth + 1)
        try:
            if depth:
                yield
                return
            if self._controller is None:
                from threadpoolctl import ThreadpoolController

                self._controller = ThreadpoolController()
            # HiGHS 的并行调度器跨实例缓存。只在最外层 optim 数值调用的边界重置，避免
            # 多期逐日重复初始化；退出后保持为未初始化状态，后续独立 HiGHS 调用将按自己的
            # options 重新创建，而不会继承 optim 的线程上限。
            _reset_highs_scheduler_if_loaded()
            try:
                with self._controller.limit(
                    limits=self.resolution.limit,
                    user_api="blas",
                ):
                    yield
            finally:
                _reset_highs_scheduler_if_loaded()
        finally:
            self._depth.reset(token)


def _reset_highs_scheduler_if_loaded() -> None:
    """仅在 highspy 已加载时重置其进程级调度器，不为其他模型额外导入后端。"""

    highspy = sys.modules.get("highspy")
    if highspy is None:
        return
    try:
        highspy.Highs.resetGlobalScheduler(True)
    except (AttributeError, RuntimeError):
        pass


def numerical_thread_scope(method):
    """让 PortfolioOptimizer 数值入口自动使用实例线程作用域。"""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        scope = getattr(self, "_thread_scope", None)
        if scope is None:
            return method(self, *args, **kwargs)
        with scope.activate():
            return method(self, *args, **kwargs)

    return wrapped

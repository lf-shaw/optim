"""请求级进度显示；不参与模型、策略或求解结果。

使用上下文隔离嵌套入口，每个阶段完成后保留独立的一行。关闭时不导入 tqdm，
循环仅保留一次空值判断，不构造日期文本。最外层请求负责关闭最后一行并显示总耗时。
"""

from contextvars import ContextVar
from functools import wraps
import sys
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")


class _Progress:
    """每次只更新当前阶段，已经结束的阶段保留最终计数和耗时。"""

    def __init__(self):
        self._started = perf_counter()
        self._tqdm = _load_tqdm()
        self._bar = None
        self._label = ""

    def phase(self, label: str, total: int | None = None) -> None:
        """开始新阶段；总数未知时只显示活动阶段及计时。"""
        self._close_phase()
        self._label = label
        self._bar = self._tqdm(
            total=total,
            desc=label,
            unit="期",
            mininterval=0.2,
            dynamic_ncols=True,
            leave=True,
            bar_format=(
                "{desc} [{elapsed_s:.2f}s]"
                if total is None
                else "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed_s:.2f}s<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def date(self, date) -> None:
        """在处理前显示当前日期，长时间的单期求解也能定位。"""
        self._bar.set_postfix_str(str(date.date()), refresh=True)

    def advance(self) -> None:
        """一日期处理完成后推进计数，内部刷新由 tqdm 节流。"""
        self._bar.update(1)

    def finish(self, label: str) -> None:
        """在当前阶段的名称旁追加终止状态，保留真实完成计数。"""
        if self._bar is not None:
            self._bar.set_description_str(f"{self._label}（{label}）", refresh=False)

    def _close_phase(self) -> None:
        if self._bar is not None:
            bar, self._bar = self._bar, None
            bar.close()

    def close(self) -> None:
        """结束活动阶段并显示本次请求的实际墙钟耗时。"""
        self._close_phase()
        self._tqdm.write(f"总耗时 {perf_counter() - self._started:.2f} 秒", file=sys.stderr)


_active: ContextVar[_Progress | None] = ContextVar("optim_progress", default=None)


def _current_progress() -> _Progress | None:
    """在入口处读取一次，避免在逐日循环中重复查找上下文。"""
    return _active.get()


def _load_tqdm():
    """仅用户开启时加载可选依赖。"""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "show_progress=True requires tqdm; install 'optim[progress]' or tqdm"
        ) from exc
    return tqdm


def _with_progress(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """为含 show_progress 关键字的入口管理显示生命周期及嵌套复用。"""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        enabled = kwargs.get("show_progress", False)
        if not isinstance(enabled, bool):
            raise TypeError("show_progress must be bool")
        if not enabled or _active.get() is not None:
            return function(*args, **kwargs)
        progress = _Progress()
        token = _active.set(progress)
        try:
            return function(*args, **kwargs)
        except BaseException:
            progress.finish("已中断")
            raise
        finally:
            try:
                progress.close()
            finally:
                _active.reset(token)

    return wrapped

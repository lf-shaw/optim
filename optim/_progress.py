"""请求级进度显示；不参与模型、策略或求解结果。

使用上下文隔离嵌套入口。终端保留各阶段的文字记录，Jupyter 使用一张阶段卡片。
关闭显示时不初始化控件或构造日期文本。
"""

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import sys
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _format_duration(seconds: float, *, remaining: bool = False) -> str:
    """耗时不足一分钟时保留小数，长耗时和预计剩余时间使用时钟格式。"""
    if remaining:
        whole_seconds = round(seconds)
    else:
        seconds = round(seconds, 2)
        if seconds < 60:
            return f"{seconds:.2f} s"
        whole_seconds = int(seconds)
    hours, rest = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


@dataclass
class _SolveProgress:
    """逐期求解的展示计数。

    ``total`` 是计划期数，``solved`` 和 ``failed`` 分别计入有解和无解的尝试。
    ``held`` 是无解后继续沿用持仓的期数，``recovered`` 已计入有解。
    ``first_failed`` 和 ``last_solved`` 是首个无解日和最后有解日。
    """

    total: int | None = None
    solved: int = 0
    failed: int = 0
    held: int = 0
    recovered: int = 0
    first_failed: datetime | None = None
    last_solved: datetime | None = None

    def summary(self) -> str:
        """整理已处理、可用解及失败推进数量。"""
        assert self.total is not None
        return (
            f"已处理 {self.solved + self.failed:,} / {self.total:,} 期"
            f" · 有解 {self.solved:,} 期 · 无解 {self.failed:,} 期"
            f" · 沿用旧仓 {self.held:,} 期 · 其中换手恢复 {self.recovered:,} 期"
        )


class _Progress:
    """连接运行事件和展示层，记录逐期求解结果的计数。"""

    def __init__(self):
        self._started = perf_counter()
        self._counts = _SolveProgress()
        self._notebook = _load_notebook(self._counts)
        self._tqdm = None if self._notebook is not None else _load_tqdm()
        self._bar = None
        self._label = ""
        self._date: datetime | None = None

    def plan(self, *stages: str) -> None:
        """提供主图节点及顺序，嵌套入口沿用外层计划。"""
        if self._notebook is not None:
            self._notebook.plan(stages)

    def phase(self, label: str, total: int | None = None, *, solving: bool = False) -> None:
        """开始新阶段；总数未知时只显示活动阶段及计时。"""
        if solving:
            self._counts.total = total
        self._date = None
        if self._notebook is not None:
            self._notebook.phase(label, total)
            return
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
                "{desc} [{elapsed}]"
                if total is None
                else "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def date(self, date) -> None:
        """在处理前显示当前日期，长时间的单期求解也能定位。"""
        self._date = date
        if self._notebook is not None:
            self._notebook.date(date)
        else:
            self._bar.set_postfix_str(str(date.date()), refresh=True)

    def advance(self, *, solved: bool | None = None, held: bool = False, recovered: bool = False) -> None:
        """处理完一期；逐期求解另外记录结果、旧仓推进及恢复情况。"""
        if solved is True:
            self._counts.solved += 1
            self._counts.recovered += recovered
            self._counts.last_solved = self._date
        elif solved is False:
            self._counts.failed += 1
            self._counts.held += held
            if self._counts.first_failed is None:
                self._counts.first_failed = self._date
        if self._notebook is not None:
            self._notebook.advance()
        else:
            self._bar.update(1)

    def finish(self, label: str, *, state: str = "done", detail: str = "") -> None:
        """在当前阶段的名称旁追加终止状态，保留真实完成计数。"""
        if self._notebook is not None:
            self._notebook.finish(label, state, detail)
        elif self._bar is not None:
            self._bar.set_description_str(f"{self._label}（{label}）", refresh=False)

    def _close_phase(self) -> None:
        if self._bar is not None:
            bar, self._bar = self._bar, None
            bar.close()

    def close(self) -> None:
        """结束活动阶段并显示本次请求的实际墙钟耗时。"""
        if self._notebook is not None:
            self._notebook.close(self._started)
            return
        self._close_phase()
        if self._counts.total is not None:
            self._tqdm.write(self._counts.summary(), file=sys.stderr)
        self._tqdm.write(f"总耗时 {_format_duration(perf_counter() - self._started)}", file=sys.stderr)


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


def _load_notebook(counts: _SolveProgress):
    """在支持 Jupyter 控件的内核中创建单一 HTML 控件。"""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    if getattr(get_ipython(), "kernel", None) is None:
        return None
    try:
        from IPython.display import display
        from ipywidgets import HTML
    except ImportError:
        return None

    from ._notebook_progress import _NotebookProgress

    widget = HTML(value="准备优化…")
    display(widget)
    return _NotebookProgress(widget, counts)


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
        except BaseException as exc:
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            progress.finish("已中断", state=state, detail=f"{type(exc).__name__}：{exc}")
            raise
        finally:
            try:
                progress.close()
            finally:
                _active.reset(token)

    return wrapped

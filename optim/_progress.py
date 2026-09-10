"""请求级进度显示；不参与模型、策略或求解结果。

使用上下文隔离嵌套入口，数据准备与求解共用一条进度条。关闭时不导入 tqdm，
循环仅保留一次空值判断，不构造日期文本。进度条由最外层请求负责关闭。
"""

from contextvars import ContextVar
from functools import wraps
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")


class _Progress:
    """轻量显示适配器：按阶段重置计数，不把未知长度 I/O 伪装成百分比。"""

    def __init__(self, bar):
        self.bar = bar

    def phase(self, label: str, total: int | None = None) -> None:
        """开始新阶段；总数未知时只显示活动阶段及计时。"""
        self.bar.total = total
        self.bar.set_description_str(label, refresh=False)
        self.bar.set_postfix_str("", refresh=False)
        self.bar.reset(total=total)

    def date(self, date) -> None:
        """在处理前显示当前日期，长时间的单期求解也能定位。"""
        self.bar.set_postfix_str(str(date.date()), refresh=True)

    def advance(self) -> None:
        """一日期处理完成后推进计数，内部刷新由 tqdm 节流。"""
        self.bar.update(1)

    def finish(self, label: str) -> None:
        """仅更新结束状态，提前停止时不补齐未处理日期的计数。"""
        self.bar.set_description_str(label)


_active: ContextVar[_Progress | None] = ContextVar("optim_progress", default=None)


def _current_progress() -> _Progress | None:
    """在入口处读取一次，避免在逐日循环中重复查找上下文。"""
    return _active.get()


def _create_bar():
    """仅用户开启时加载可选依赖。"""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "show_progress=True requires tqdm; install 'optim[progress]' or tqdm"
        ) from exc
    return tqdm(
        total=None, desc="准备请求", unit="期", mininterval=0.2, dynamic_ncols=True
    )


def _with_progress(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """为含 show_progress 关键字的入口管理显示生命周期及嵌套复用。"""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        enabled = kwargs.get("show_progress", False)
        if not isinstance(enabled, bool):
            raise TypeError("show_progress must be bool")
        if not enabled or _active.get() is not None:
            return function(*args, **kwargs)
        progress = _Progress(_create_bar())
        token = _active.set(progress)
        try:
            return function(*args, **kwargs)
        except BaseException:
            progress.finish("已中断")
            raise
        finally:
            try:
                progress.bar.close()
            finally:
                _active.reset(token)

    return wrapped

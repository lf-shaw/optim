"""在 Jupyter 的一个 HTML 控件中原位更新阶段和结果。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from time import perf_counter
from typing import TYPE_CHECKING

from ._progress import _format_duration

if TYPE_CHECKING:
    from ._progress import _SolveProgress


_STYLE = """
<style>
.optim-run-progress {font: 13px/1.5 var(--jp-ui-font-family, sans-serif);
  color: var(--jp-ui-font-color1, #24292f); padding: 8px 2px;}
.optim-run-progress .orp-flow {display: flex; flex-wrap: wrap; align-items: center;
  gap: 5px 8px; list-style: none; padding: 0; margin: 0 0 8px;}
.optim-run-progress .orp-step {display: inline-flex; align-items: center; gap: 6px;
  white-space: nowrap; border: 1px solid var(--jp-border-color2, #d0d7de);
  border-radius: 6px; padding: 4px 8px;}
.optim-run-progress .orp-arrow, .optim-run-progress .orp-meta {opacity: .74;}
.optim-run-progress .orp-pending {opacity: .65;}
.optim-run-progress .orp-active {border-color: #4589e4;}
.optim-run-progress .orp-done .orp-icon {color: #28834f;}
.optim-run-progress .orp-warning .orp-icon, .optim-run-progress .orp-warning .orp-title {color: #a8660d;}
.optim-run-progress .orp-error .orp-icon, .optim-run-progress .orp-error .orp-title {color: #d34a49;}
.optim-run-progress .orp-spinner {display: inline-block; width: 9px; height: 9px;
  border-radius: 50%; border: 2px solid #4589e4; border-right-color: transparent;
  animation: optim-progress-spin .85s linear infinite;}
@keyframes optim-progress-spin {to {transform: rotate(360deg);}}
@media (prefers-reduced-motion: reduce) {
  .optim-run-progress .orp-spinner {animation: none;}
}
.optim-run-progress .orp-detail {border-left: 3px solid var(--jp-border-color1, #d0d7de);
  padding: 2px 10px; overflow-wrap: anywhere;}
.optim-run-progress .orp-title {font-weight: 600;}
.optim-run-progress progress {width: min(360px, 100%); display: block;
  height: 10px; margin: 5px 0; accent-color: #4589e4;}
</style>
"""

_SYMBOLS = {"pending": "○", "done": "✓", "warning": "!", "error": "×"}
_SHORT_LABELS = {
    "数据对齐": "对齐",
    "读取持仓收益": "读取收益",
}


@dataclass
class _Stage:
    """一个节点的记录。

    ``elapsed`` 累计已经完成的阶段用时，``status`` 表示节点的运行状态。
    ``count`` 和 ``total`` 仅在阶段有已知期数时更新。
    """

    elapsed: float = 0.0
    status: str = "pending"
    count: int = 0
    total: int | None = None


class _NotebookProgress:
    """按阶段切换、按时间节流，结束时保留同一个控件。"""

    def __init__(self, widget, counts: _SolveProgress):
        self._widget = widget
        self._counts = counts
        self._stages: dict[str, _Stage] = {}
        self._active: str | None = None
        self._label = "准备请求"
        self._date: datetime | None = None
        self._total: int | None = None
        self._started = self._last_render = perf_counter()
        self._final_state = "done"
        self._final_label = "运行结束"
        self._reason = ""

    def plan(self, stages: tuple[str, ...]) -> None:
        if not self._stages:
            self._stages.update((name, _Stage()) for name in stages)

    def phase(self, label: str, total: int | None) -> None:
        now = perf_counter()
        self._end_phase(now, "done")
        # 序列核验可能在收益读取之后执行，耗时仍累计在预检节点里。
        node = "静态预检" if label == "序列预检" and "静态预检" in self._stages else label
        stage = self._stages.setdefault(node, _Stage())
        stage.status = "active"
        if total is not None:
            stage.total = total
            stage.count = 0
        self._active, self._label, self._total = node, label, total
        self._date = None
        self._started = now
        self._render(now)

    def date(self, date: datetime) -> None:
        first = self._date is None
        self._date = date
        now = perf_counter()
        if first or now - self._last_render >= 0.2:
            self._render(now)

    def advance(self) -> None:
        if self._active is not None and self._total is not None:
            self._stages[self._active].count += 1
        now = perf_counter()
        if now - self._last_render >= 0.2:
            self._render(now)

    def finish(self, label: str, state: str, detail: str) -> None:
        self._final_state = state
        self._final_label = {
            "error": "运行出错",
            "stopped": "已停止",
            "interrupted": "已中断",
        }.get(state, label)
        self._reason = " ".join(detail.split())[:360]

    def close(self, started: float) -> None:
        now = perf_counter()
        state = self._final_state
        if state == "done" and self._counts.failed:
            self._final_label = "运行结束，有未求解期"
            state = "warning"
        style = {"error": "error", "stopped": "warning", "interrupted": "warning"}.get(state, state)
        self._end_phase(now, style)
        details = [f"{self._final_label} · 总耗时 {_format_duration(now - started)}"]
        if self._counts.total is not None:
            details.append(self._counts.summary())
        dates = []
        if self._counts.first_failed is not None:
            dates.append(f"首个无解日 {self._counts.first_failed.date()}")
        if self._counts.last_solved is not None:
            dates.append(f"最后有解日 {self._counts.last_solved.date()}")
        if dates:
            details.append(" · ".join(dates))
        if state in {"error", "interrupted", "stopped"}:
            context = [self._label] if self._label else []
            if self._date is not None:
                context.append(str(self._date.date()))
            if context:
                details.append("断点 " + " · ".join(context))
            if self._reason:
                details.append(self._reason)
        self._draw(details, style=style)

    def _end_phase(self, now: float, status: str) -> None:
        if self._active is not None:
            stage = self._stages[self._active]
            stage.elapsed += now - self._started
            stage.status = status
            self._active = None

    def _render(self, now: float) -> None:
        self._last_render = now
        elapsed = now - self._started
        details = [f"{self._label} · 进行中 · 已耗时 {_format_duration(elapsed)}"]
        progress = ""
        if self._active is not None and self._total:
            count = self._stages[self._active].count
            progress = (
                f'<progress aria-label="{escape(self._label)}" value="{count}" max="{self._total}"></progress>'
            )
            metrics = [f"已处理 {count:,} / {self._total:,} 期"]
            if self._date is not None:
                metrics.append(f"当前 {self._date.date()}")
            if count and elapsed > 0:
                rate = count / elapsed
                remaining = _format_duration((self._total - count) / rate, remaining=True)
                metrics.extend([f"{rate:.2f} 期/s", f"预计剩余 {remaining}"])
            details.append(" · ".join(metrics))
        self._draw(details, style="active", progress=progress)

    def _draw(self, details: list[str], *, style: str, progress: str = "") -> None:
        nodes = []
        for label, stage in self._stages.items():
            icon = (
                '<span class="orp-spinner" aria-hidden="true"></span>'
                if stage.status == "active"
                else f'<span class="orp-icon" aria-hidden="true">{_SYMBOLS[stage.status]}</span>'
            )
            if stage.status == "pending":
                meta = "待执行"
            elif stage.status == "active":
                meta = "进行中"
            else:
                meta = _format_duration(stage.elapsed)
                if stage.total is not None:
                    meta = f"{stage.count:,}/{stage.total:,} 期 · " + meta
            nodes.append(
                f'<li class="orp-step orp-{stage.status}">{icon}'
                f'{escape(_SHORT_LABELS.get(label, label))} <span class="orp-meta">{meta}</span></li>'
            )
        flow = '<li class="orp-arrow" aria-hidden="true">→</li>'.join(nodes)
        lines = "".join(f"<div>{escape(line)}</div>" for line in details[1:])
        self._widget.value = (
            f'{_STYLE}<section class="optim-run-progress" aria-label="组合优化进度">'
            f'<ol class="orp-flow">{flow}</ol><div class="orp-detail orp-{style}">'
            f'<div class="orp-title">{escape(details[0])}</div>{progress}{lines}</div></section>'
        )

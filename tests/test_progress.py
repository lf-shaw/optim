"""可选进度显示的生命周期、阶段和结果不变性。"""

import builtins
from dataclasses import replace
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optim import PortfolioOptimizer, SolveStatus
from optim import _progress
from optim import _notebook_progress
from optim.portfolio_types import SequencePolicy
from optim.data import FactorRiskFrames, InMemoryDataSource, PortfolioSchedule


class RecordingBar:
    """记录 tqdm 稳定接口调用，不依赖终端输出格式。"""

    def __init__(self, *, total, desc, leave, bar_format, **kwargs):
        self.total = total
        self.n = 0
        self.closed = 0
        self.labels = [desc]
        self.dates = []
        self.leave = leave
        self.bar_format = bar_format

    def set_description_str(self, text, refresh=True):
        self.labels.append(text)

    def set_postfix_str(self, text, refresh=True):
        if text:
            self.dates.append(text)

    def update(self, n):
        self.n += n

    def close(self):
        self.closed += 1


@pytest.fixture
def bars(monkeypatch):
    created = []

    def create(**kwargs):
        assert all(bar.closed == 1 for bar in created)
        bar = RecordingBar(**kwargs)
        created.append(bar)
        return bar

    create.write = lambda text, *, file: print(text, file=file)
    monkeypatch.setattr(_progress, "_load_tqdm", lambda: create)
    return created


def test_optional_progress_preserves_results_and_stays_lazy(sample_lp_problem, bars):
    optimizer = PortfolioOptimizer()
    plain = optimizer.solve_sequence([sample_lp_problem])
    assert bars == []
    shown = optimizer.solve_sequence([sample_lp_problem], show_progress=True)
    assert [bar.labels[0] for bar in bars] == ["序列预检", "静态预检", "逐期求解"]
    assert all(bar.closed == 1 and bar.leave for bar in bars)
    assert (bars[1].n, bars[1].total) == (1, 1)
    bar = bars[-1]
    assert (bar.n, bar.total, bar.closed) == (1, 1, 1)
    assert bar.labels[-1] == "逐期求解（求解完成）"
    assert bar.dates[-1] == "2026-01-02"
    pd.testing.assert_series_equal(plain.final_weight, shown.final_weight)
    assert plain.steps[0].result.metrics == shown.steps[0].result.metrics
    assert _progress._current_progress() is None


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_progress_closes_on_solver_exception(
    sample_lp_problem, bars, monkeypatch, error, capsys
):
    optimizer = PortfolioOptimizer()

    def fail(*args, **kwargs):
        raise error("test interrupt")

    monkeypatch.setattr(optimizer, "_solve_prevalidated", fail)
    with pytest.raises(error):
        optimizer.solve_sequence([sample_lp_problem], show_progress=True)
    assert all(bar.closed == 1 for bar in bars)
    assert bars[-1].n == 0
    assert bars[-1].labels[-1] == "逐期求解（已中断）"
    assert capsys.readouterr().err.count("总耗时") == 1
    assert _progress._current_progress() is None


def test_early_stop_keeps_actual_count(sample_lp_problem, bars, monkeypatch):
    optimizer = PortfolioOptimizer()
    native = optimizer.solve(sample_lp_problem)
    later_date = pd.Timestamp("2026-01-05")
    second = replace(
        sample_lp_problem,
        data=replace(
            sample_lp_problem.data,
            date=later_date,
            risk_model=replace(sample_lp_problem.data.risk_model, asof=later_date),
        ),
    )
    monkeypatch.setattr(
        optimizer,
        "_solve_prevalidated",
        lambda p: replace(
            native,
            status=SolveStatus.INFEASIBLE,
            weights=None,
            problem=p,
        ),
    )
    result = optimizer.solve_sequence(
        [sample_lp_problem, second],
        show_progress=True,
        holding_period_returns={later_date: np.zeros(len(second.data.assets))},
    )
    assert result.stopped_date == sample_lp_problem.data.date
    assert result.stop_reason is not None
    assert "2026-01-02" in result.stop_reason
    assert "status=infeasible" in result.stop_reason
    assert f"backend={native.backend}" in result.stop_reason
    assert (bars[-1].n, bars[-1].total, bars[-1].closed) == (1, 2, 1)
    assert bars[-1].labels[-1] == "逐期求解（求解已停止：2026-01-02 infeasible）"


def test_range_keeps_each_progress_stage_and_prints_one_total(sample_lp_problem, bars, capsys):
    data = sample_lp_problem.data
    risk = data.risk_model
    index = pd.MultiIndex.from_product([[data.date], data.assets], names=["dt", "sid"])
    frames = FactorRiskFrames(
        exposure=pd.DataFrame(risk.exposure, index=index, columns=risk.factor_names),
        covariance=pd.DataFrame(
            risk.covariance,
            index=pd.MultiIndex.from_product(
                [[data.date], risk.factor_names], names=["dt", "factor"]
            ),
            columns=risk.factor_names,
        ),
        specific_volatility=pd.Series(risk.specific_volatility, index=index),
        factor_types=dict(zip(risk.factor_names, risk.factor_types)),
    )
    source = InMemoryDataSource(
        risk_data=frames, benchmark=pd.Series(data.benchmark, index=index)
    )
    result = PortfolioOptimizer().optimize_range(
        data_source=source,
        schedule=PortfolioSchedule(pd.DataFrame({"alpha": data.alpha}, index=index)),
        objective=sample_lp_problem.objective,
        constraints=sample_lp_problem.constraints,
        alpha_spec=data.alpha_spec,
        show_progress=True,
    )
    assert result.steps[0].result.status.has_solution
    assert [bar.labels[0] for bar in bars] == ["组装", "静态预检", "逐期求解"]
    assert all(bar.closed == 1 for bar in bars)
    assert capsys.readouterr().err.count("总耗时") == 1


def test_missing_optional_dependency_is_actionable(sample_lp_problem, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "tqdm.auto":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    optimizer = PortfolioOptimizer()
    assert (
        optimizer.solve_sequence([sample_lp_problem])
        .steps[0]
        .result.status.has_solution
    )
    with pytest.raises(ImportError, match=r"optim\[progress\]"):
        optimizer.solve_sequence([sample_lp_problem], show_progress=True)
    assert _progress._current_progress() is None


def test_real_tqdm_smoke(sample_lp_problem, capsys):
    pytest.importorskip("tqdm.auto")
    result = PortfolioOptimizer().solve_sequence(
        [sample_lp_problem], show_progress=True
    )
    assert result.steps[0].result.status.has_solution
    final_lines = [line.rsplit("\r", 1)[-1] for line in capsys.readouterr().err.split("\n")]
    for label in ("序列预检", "静态预检", "逐期求解（求解完成）"):
        line, = [line for line in final_lines if line.startswith(label)]
        assert re.search(r"\[\d{2}:\d{2}", line)
    assert final_lines[-2].startswith("总耗时 ")


def test_progress_closes_on_precheck_failure(sample_lp_problem, bars):
    from optim import PortfolioValidationError

    bad = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=np.full(4, np.nan)),
    )
    with pytest.raises(PortfolioValidationError):
        PortfolioOptimizer().solve_sequence([bad], show_progress=True)
    assert all(bar.closed == 1 for bar in bars)
    assert [bar.labels[0] for bar in bars] == ["序列预检", "静态预检"]
    assert bars[-1].labels[-1] == "静态预检（已中断）"
    assert _progress._current_progress() is None


@pytest.fixture
def notebook_card(monkeypatch):
    widget = SimpleNamespace(value="")
    created = []

    def create(counts):
        created.append(widget)
        return _notebook_progress._NotebookProgress(widget, counts)

    monkeypatch.setattr(_progress, "_load_notebook", create)
    return widget, created


def test_notebook_stage_card_shows_progress_only_for_counted_active_phase(
    monkeypatch, notebook_card
):
    widget, created = notebook_card
    clock = [0.0]
    monkeypatch.setattr(_progress, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(_notebook_progress, "perf_counter", lambda: clock[0])
    display = _progress._Progress()
    display.plan("取数", "组装", "静态预检", "逐期求解")
    display.phase("取数")
    assert '<span class="orp-spinner"' in widget.value and "<progress" not in widget.value
    assert widget.value.count('class="orp-step orp-pending"') == 3
    clock[0] = 2.22
    display.phase("组装")
    assert "2.22 s" in widget.value
    assert "<progress" not in widget.value
    clock[0] = 8.87
    display.phase("静态预检", 1)
    display.advance()
    clock[0] = 17.78
    display.phase("序列预检")
    clock[0] = 17.80
    display.phase("逐期求解", 2, solving=True)
    display.date(pd.Timestamp("2026-01-02"))
    clock[0] = 418.13
    display.advance(solved=True, recovered=True)
    assert 'value="1" max="2"' in widget.value
    assert "已耗时 06:40" in widget.value and "预计剩余 06:40" in widget.value
    assert "400.33" not in widget.value
    display.date(pd.Timestamp("2026-01-05"))
    display.advance(solved=False, held=True)
    display.finish("求解完成")
    display.close()
    assert len(created) == 1
    assert "<progress" not in widget.value and '<span class="orp-spinner"' not in widget.value
    assert "运行结束，有未求解期" in widget.value
    assert "有解 1 期 · 无解 1 期 · 沿用旧仓 1 期 · 其中换手恢复 1 期" in widget.value
    assert "首个无解日 2026-01-05 · 最后有解日 2026-01-02" in widget.value
    assert "1/1 期 · 8.93 s" in widget.value


@pytest.mark.parametrize(
    ("exception", "expected_label", "state"),
    [(RuntimeError, "运行出错", "error"), (KeyboardInterrupt, "已中断", "warning")],
)
def test_notebook_preserves_completed_and_pending_stages_on_error(
    exception, expected_label, state, notebook_card
):
    widget, _ = notebook_card

    @_progress._with_progress
    def fail(*, show_progress=False):
        display = _progress._current_progress()
        display.plan("取数", "组装", "逐期求解")
        display.phase("取数")
        display.phase("组装")
        raise exception("<查看异常>")

    with pytest.raises(exception):
        fail(show_progress=True)
    assert f'class="orp-step orp-{state}"' in widget.value
    assert 'class="orp-step orp-done"' in widget.value
    assert 'class="orp-step orp-pending"' in widget.value
    assert f"{expected_label} · 总耗时" in widget.value
    assert "断点 组装" in widget.value and "&lt;查看异常&gt;" in widget.value
    assert "<progress" not in widget.value


@pytest.mark.parametrize("on_failure", ["stop", "hold"])
def test_notebook_tallies_actual_sequence_outcomes(sample_lp_problem, monkeypatch, notebook_card, on_failure):
    widget, created = notebook_card
    optimizer = PortfolioOptimizer()
    native = optimizer.solve(sample_lp_problem)
    later = pd.Timestamp("2026-01-05")
    second = replace(
        sample_lp_problem,
        data=replace(
            sample_lp_problem.data, date=later,
            risk_model=replace(sample_lp_problem.data.risk_model, asof=later),
        ),
    )
    attempts = iter([native, replace(native, status=SolveStatus.INFEASIBLE, weights=None)])
    monkeypatch.setattr(optimizer, "_solve_prevalidated", lambda _: next(attempts))
    optimizer.solve_sequence(
        [sample_lp_problem, second],
        holding_period_returns={later: np.zeros(len(second.data.assets))},
        sequence_policy=SequencePolicy(on_failure=on_failure),
        show_progress=True,
    )
    assert len(created) == 1
    assert "已处理 2 / 2 期 · 有解 1 期 · 无解 1 期" in widget.value
    assert "首个无解日 2026-01-05 · 最后有解日 2026-01-02" in widget.value
    if on_failure == "stop":
        assert "已停止 · 总耗时" in widget.value
        assert "沿用旧仓 0 期" in widget.value and "status=infeasible" in widget.value
    else:
        assert "运行结束，有未求解期" in widget.value and "沿用旧仓 1 期" in widget.value

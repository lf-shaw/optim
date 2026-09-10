"""可选进度显示的生命周期、阶段和结果不变性。"""

import builtins
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optim import PortfolioOptimizer, SolveStatus
from optim import _progress
from optim.data import FactorRiskFrames, InMemoryDataSource, PortfolioSchedule


class RecordingBar:
    """记录 tqdm 稳定接口调用，不依赖终端输出格式。"""

    def __init__(self):
        self.total = None
        self.n = 0
        self.closed = 0
        self.labels = []
        self.dates = []

    def set_description_str(self, text, refresh=True):
        self.labels.append(text)

    def set_postfix_str(self, text, refresh=True):
        if text:
            self.dates.append(text)

    def reset(self, total=None):
        self.total = total
        self.n = 0

    def update(self, n):
        self.n += n

    def close(self):
        self.closed += 1


@pytest.fixture
def bars(monkeypatch):
    created = []

    def create():
        bar = RecordingBar()
        created.append(bar)
        return bar

    monkeypatch.setattr(_progress, "_create_bar", create)
    return created


def test_optional_progress_preserves_results_and_stays_lazy(sample_lp_problem, bars):
    optimizer = PortfolioOptimizer()
    plain = optimizer.solve_sequence([sample_lp_problem])
    assert bars == []
    shown = optimizer.solve_sequence([sample_lp_problem], show_progress=True)
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.n, bar.total, bar.closed) == (1, 1, 1)
    assert "静态预检" in bar.labels
    assert "逐期求解" in bar.labels
    assert bar.labels[-1] == "求解完成"
    assert bar.dates[-1] == "2026-01-02"
    pd.testing.assert_series_equal(plain.final_weight, shown.final_weight)
    assert plain.steps[0].result.metrics == shown.steps[0].result.metrics
    assert _progress._current_progress() is None


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_progress_closes_on_solver_exception(
    sample_lp_problem, bars, monkeypatch, error
):
    optimizer = PortfolioOptimizer()

    def fail(*args, **kwargs):
        raise error("test interrupt")

    monkeypatch.setattr(optimizer, "_solve_prevalidated", fail)
    with pytest.raises(error):
        optimizer.solve_sequence([sample_lp_problem], show_progress=True)
    assert bars[0].closed == 1
    assert bars[0].n == 0
    assert bars[0].labels[-1] == "已中断"
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
    assert (bars[0].n, bars[0].total, bars[0].closed) == (1, 2, 1)
    assert bars[0].labels[-1] == "求解已停止"


def test_range_preparation_reuses_one_progress_bar(sample_lp_problem, bars):
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
    assert len(bars) == 1
    assert all(
        label in bars[0].labels
        for label in ("取数与对齐", "数据对齐", "静态预检", "逐期求解")
    )
    assert bars[0].closed == 1


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
    assert "求解完成" in capsys.readouterr().err


def test_progress_closes_on_precheck_failure(sample_lp_problem, bars):
    from optim import PortfolioValidationError

    bad = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=np.full(4, np.nan)),
    )
    with pytest.raises(PortfolioValidationError):
        PortfolioOptimizer().solve_sequence([bad], show_progress=True)
    assert bars[0].closed == 1
    assert "逐期求解" not in bars[0].labels
    assert _progress._current_progress() is None

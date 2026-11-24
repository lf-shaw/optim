"""新版 tuda2 的表格契约与真实 MOSEK 求解回归。"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tuda2

os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(Path.home() / ".mosek" / "mosek.lic"))

from optim import linopt, opt


@pytest.fixture
def panels(monkeypatch):
    dates = pd.to_datetime(["2026-07-06", "2026-07-08"])
    assets = ["a", "b", "c"]
    index = pd.MultiIndex.from_product([dates, assets], names=["dt", "sid"])
    universe = pd.DataFrame({"alpha": [1.0, 0.1, 0.0] * 2, "member": True, "tradable": True}, index=index)
    benchmark = pd.Series([0.4, 0.3, 0.3] * 2, index=index, name="weight")
    exposure = pd.DataFrame(
        {"size": [0.0, 1.0, -1.0] * 2, "industry": ["old"] * 3 + ["new"] * 3},
        index=index,
    )
    covariance = pd.DataFrame(
        [[0, 0, 0, .02], [.04, 0, 0, 0], [0, 0, .03, 0],
         [0, 0, 0, .02], [.04, 0, 0, 0], [0, .03, 0, 0]],
        index=pd.MultiIndex.from_tuples(
            [(dates[0], factor) for factor in ("country", "size", "old")]
            + [(dates[1], factor) for factor in ("country", "size", "new")],
            names=["dt", "factor"],
        ),
        columns=["size", "new", "old", "country"],
    )
    specific = pd.DataFrame({"spec_risk": .2}, index=index)
    schema = {
        "industry": ["old", "new"],
        "constant_exposures": {"country": 1.0},
        "factor_types": {"country": "country", "size": "style", "old": "industry", "new": "industry"},
    }
    days = pd.date_range(dates[0], dates[-1])
    ret_index = pd.MultiIndex.from_product([days, assets], names=["dt", "sid"])
    calls = []

    def risk(kind, *, dts, model, factor_type=None):
        assert model == "datayes"
        calls.append(kind)
        return {"exposure": exposure, "cov": covariance, "spec_risk": specific}[kind].loc[dts]

    def weights(code, *, dts, type):
        assert (code, type) == ("000852.SH", "daily")
        calls.append("benchmark")
        return benchmark.loc[dts]

    def returns(*, since, until, sids, freq, window, shift, market_side, price_type, price_window):
        assert (since, until, sids) == (dates[0], dates[-1], assets)
        assert (freq, window, shift, market_side, price_type, price_window) == ("D", 1, False, "close", "vwap", 0)
        calls.append("returns")
        return pd.DataFrame({"m0": [0.0] * 3 + [0.10, 0.0, 0.0] + [0.0] * 3}, index=ret_index)

    monkeypatch.setattr(tuda2, "get_index_weight", weights)
    monkeypatch.setattr(tuda2, "get_risk_model", risk)
    monkeypatch.setattr(tuda2, "get_risk_model_schema", lambda *, model: schema)
    monkeypatch.setattr(tuda2, "get_return", returns)
    return dates, universe, benchmark, calls


def test_linear_single_and_chain(panels):
    dates, universe, _, calls = panels
    args = dict(constraint_style={"SIZE": (-.2, .2)}, active_ub=.4, risk_model="datayes", show_progress=False)
    one = linopt.multioptimize(universe.loc[dates[:1]], "000852.SH", **args)
    assert list(calls) == ["benchmark", "exposure"]
    assert one.index.unique(level="dt").equals(dates[:1])
    calls.clear()
    result = linopt.multioptimize(universe, "000852.SH", turnover_limit=.08, **args)
    assert calls == ["benchmark", "exposure", "returns"]
    assert result.index.unique(level="dt").equals(dates)
    assert result.index.is_monotonic_increasing
    previous = result.xs(dates[0]).reindex(["a", "b", "c"], fill_value=0.0)
    previous["a"] *= 1.10
    previous /= previous.sum()
    current = result.xs(dates[1]).reindex(previous.index, fill_value=0.0)
    assert (current - previous).abs().sum() <= .0801


@pytest.mark.parametrize("gamma", [None, .06])
def test_quadratic_and_socp_select_active_factor_coordinates(panels, gamma):
    dates, universe, benchmark, calls = panels
    result = opt.multioptimize(
        universe, "000852.SH", risk_model="datayes", gamma=gamma, active_ub=.4,
        constraint_factor={"ALL": (-.2, .2)}, use_pre_portfolio=True,
        turnover_limit=.10, show_progress=False,
    )
    assert calls == ["benchmark", "exposure", "cov", "spec_risk", "returns"]
    assert result.index.unique(level="dt").equals(dates)
    if gamma is not None:
        for date in dates:
            b = benchmark.xs(date)
            active = result.xs(date).reindex(b.index, fill_value=0.0) - b
            specific = .2 * active.to_numpy()
            factor = active["b"] - active["c"]
            te = np.sqrt(.04 * factor**2 + np.inner(specific, specific))
            assert te <= gamma + 1e-4


def test_quadratic_single_accepts_labelled_initial_weights(panels):
    dates, universe, benchmark, calls = panels
    result = opt.multioptimize(
        universe.loc[dates[:1]], "000852.SH", risk_model="datayes",
        init_portfolio=benchmark.xs(dates[0]), use_pre_portfolio=True,
        turnover_limit=.05, gamma=.06, show_progress=False,
    )
    assert calls == ["benchmark", "exposure", "cov", "spec_risk"]
    weight = result.xs(dates[0])
    assert (weight - benchmark.xs(dates[0])).abs().sum() <= .0501

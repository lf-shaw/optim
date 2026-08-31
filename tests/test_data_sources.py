from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    BenchmarkCoveragePolicy,
    FactorRiskFrames,
    InMemoryDataSource,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioSchedule,
    PortfolioOptimizer,
    PortfolioValidationError,
    SequencePolicy,
    WeightBounds,
)
from optim.integrations.tuda2 import Tuda2DataSource


def _frames(dates=("2026-01-02", "2026-01-05")) -> FactorRiskFrames:
    dts = pd.to_datetime(dates)
    index = pd.MultiIndex.from_product([dts, ["a", "b"]], names=["dt", "sid"])
    exposure = pd.DataFrame(
        {
            "size": [1.0, -1.0] * len(dts),
            "bank": [1.0, 0.0] * len(dts),
        },
        index=index,
    )
    cov_index = pd.MultiIndex.from_product(
        [dts, ["size", "bank"]], names=["dt", "factor"]
    )
    covariance = pd.DataFrame(
        np.tile(np.diag([0.04, 0.02]), (len(dts), 1)),
        index=cov_index,
        columns=["size", "bank"],
    )
    specific = pd.Series(0.10, index=index, name="spec_risk")
    return FactorRiskFrames(
        exposure=exposure,
        covariance=covariance,
        specific_volatility=specific,
        factor_types={"size": "style", "bank": "industry"},
    )


def _schedule(dates=("2026-01-02", "2026-01-05")) -> PortfolioSchedule:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), ["a", "b"]], names=["dt", "sid"]
    )
    return PortfolioSchedule(
        pd.DataFrame(
            {"alpha": [1.0, 0.0] * len(dates), "tradable": True}, index=index
        )
    )


def _benchmark(dates=("2026-01-02", "2026-01-05"), *, outside=False):
    pieces = []
    for date in pd.to_datetime(dates):
        values = {"a": 0.55, "b": 0.45}
        if outside:
            values = {"a": 0.50, "b": 0.40, "c": 0.10}
        series = pd.Series(values, name="weight")
        series.index = pd.MultiIndex.from_product([[date], series.index], names=["dt", "sid"])
        pieces.append(series)
    return pd.concat(pieces)


def _constraints():
    return PortfolioConstraints(asset_weight=WeightBounds(0.0, 1.0))


def test_in_memory_source_uses_exact_dates_and_aligned_coordinate():
    source = InMemoryDataSource(risk_data=_frames(), benchmark=_benchmark())
    problems = source.build_problems(
        _schedule(),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
    )
    assert [problem.data.date for problem in problems] == list(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    np.testing.assert_allclose(problems[0].data.benchmark, [0.55, 0.45])
    assert problems[0].data.risk_model.asof == problems[0].data.date
    np.testing.assert_allclose(problems[1].data.initial_weight, problems[1].data.benchmark)


def test_benchmark_gap_errors_by_default_and_requires_explicit_renormalization():
    schedule = _schedule()
    initial = pd.Series({"a": 0.5, "b": 0.5})
    with pytest.raises(PortfolioValidationError, match="explicit renormalization is disabled") as caught:
        InMemoryDataSource(
            risk_data=_frames(), benchmark=_benchmark(outside=True)
        ).build_problems(
            schedule,
            objective=MaximizeAlpha(),
            constraints=_constraints(),
            alpha_spec=AlphaSpec(),
            initial_weight=initial,
        )
    assert len(caught.value.report.errors) == 2

    source = InMemoryDataSource(
        risk_data=_frames(),
        benchmark=_benchmark(outside=True),
        benchmark_policy=BenchmarkCoveragePolicy(
            action="renormalize_within_tolerance", missing_mass_tolerance=0.10
        ),
    )
    problem = source.build_problems(
        schedule,
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        initial_weight=initial,
    )[0]
    np.testing.assert_allclose(problem.data.benchmark, [5.0 / 9.0, 4.0 / 9.0])
    assert problem.data.provenance.metadata["benchmark_missing_mass"] == pytest.approx(0.10)


def test_manual_data_never_forward_fills_a_missing_risk_date():
    source = InMemoryDataSource(
        risk_data=_frames(("2026-01-02",)), benchmark=_benchmark()
    )
    with pytest.raises(PortfolioValidationError, match="no exact data for 2026-01-05") as caught:
        source.build_problems(
            _schedule(),
            objective=MaximizeAlpha(),
            constraints=_constraints(),
            alpha_spec=AlphaSpec(),
            initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        )
    assert caught.value.report.errors[0].date == pd.Timestamp("2026-01-05")


class _FakeTuda2:
    __version__ = "test"

    def __init__(self):
        self.calls = []

    def get_risk_model(self, kind, *, dts, version):
        self.calls.append(("risk", kind, tuple(pd.DatetimeIndex(dts)), version))
        dates = pd.DatetimeIndex(dts)
        if kind == "exposure":
            index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["dt", "sid"])
            return pd.DataFrame(
                {"size": [1.0, -1.0] * len(dates), "industry": ["bank", "tech"] * len(dates)},
                index=index,
            )
        if kind == "cov":
            index = pd.MultiIndex.from_product(
                [dates, ["size", "bank", "tech"]], names=["dt", "factor"]
            )
            return pd.DataFrame(
                np.tile(np.diag([0.04, 0.02, 0.03]), (len(dates), 1)),
                index=index,
                columns=["size", "bank", "tech"],
            )
        index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["dt", "sid"])
        return pd.DataFrame({"spec_risk": 0.10}, index=index)

    def get_risk_model_factor_names(self, kind, *, model_type):
        self.calls.append(("factor_names", kind, model_type))
        return ["size"] if kind == "style" else ["bank", "tech"]

    def get_index_weight(self, sid, *, dts, type):
        self.calls.append(("benchmark", sid, tuple(pd.DatetimeIndex(dts)), type))
        index = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(dts), ["a", "b"]], names=["dt", "sid"]
        )
        return pd.DataFrame({"weight": [0.5, 0.5] * len(pd.DatetimeIndex(dts))}, index=index)

    def get_return(
        self,
        *,
        since,
        until,
        sids,
        freq,
        shift,
        window,
        market_side,
        price_type,
        price_window,
    ):
        self.calls.append(
            (
                "returns",
                pd.Timestamp(since),
                pd.Timestamp(until),
                sids,
                freq,
                shift,
                window,
                market_side,
                price_type,
                price_window,
            )
        )
        trade_dates = pd.bdate_range(since, until)
        index = pd.MultiIndex.from_product(
            [trade_dates, ["a", "b"]], names=["dt", "sid"]
        )
        values = []
        for position in range(len(trade_dates)):
            a_return = [0.02, 0.05, 0.10][position] if position < 3 else 0.01
            b_return = np.nan if position == len(trade_dates) - 1 else 0.01
            values.extend([a_return, b_return])
        return pd.DataFrame({"m0": values}, index=index)

    def get_universe(self, *, universe, dts):
        self.calls.append(("universe", universe, tuple(pd.DatetimeIndex(dts))))
        index = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(dts), ["a", "b"]], names=["dt", "sid"]
        )
        return pd.DataFrame({"tradable": True}, index=index)


class _MissingMiddleBenchmarkTuda2(_FakeTuda2):
    def __init__(self, missing_date):
        super().__init__()
        self.missing_date = pd.Timestamp(missing_date)

    def get_index_weight(self, sid, *, dts, type):
        frame = super().get_index_weight(sid, dts=dts, type=type)
        keep = frame.index.get_level_values("dt") != self.missing_date
        return frame.loc[keep]


def test_tuda2_adapter_batches_dates_and_expands_industry_once():
    fake = _FakeTuda2()
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    loaded = Tuda2DataSource(module=fake).load(dates=dates, benchmark_sid="000852.SH")
    assert [call[1] for call in fake.calls if call[0] == "risk"] == [
        "exposure",
        "cov",
        "spec_risk",
    ]
    assert list(loaded.risk_data.exposure.columns) == ["size", "bank", "tech"]
    first = loaded.risk_data.materialize(dates[0], pd.Index(["a", "b"], name="sid"))
    np.testing.assert_allclose(first.exposure, [[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    assert first.annualization == "annualized_decimal"


def test_tuda2_daily_close_returns_are_compounded_and_preserve_missing():
    fake = _FakeTuda2()
    dates = pd.to_datetime(["2026-01-02", "2026-01-06"])
    returns = Tuda2DataSource(module=fake).load_close_to_close_returns(
        dates=dates, sids=["a", "b"]
    )
    assert list(returns) == [pd.Timestamp("2026-01-06")]
    assert returns[dates[1]].loc["a"] == pytest.approx((1.05 * 1.10) - 1.0)
    assert np.isnan(returns[dates[1]].loc["b"])
    return_calls = [call for call in fake.calls if call[0] == "returns"]
    assert len(return_calls) == 1
    assert return_calls[0][4:] == ("D", False, 1, "close", "vwap", 0)


def test_optimize_range_reuses_the_same_strict_memory_path():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    source = InMemoryDataSource(risk_data=_frames(), benchmark=_benchmark())
    result = PortfolioOptimizer().optimize_range(
        data_source=source,
        schedule=_schedule(),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        holding_period_returns={dates[1]: pd.Series(0.0, index=["a", "b"])},
        sequence_policy=SequencePolicy(output_weights="none"),
    )
    assert [step.date for step in result.steps] == list(dates)
    assert all(step.result.status.has_solution for step in result.steps)
    assert result.schedule_prepare_s > 0.0


def test_optimizer_range_facade_routes_tuda2_through_common_lazy_core_path():
    fake = _FakeTuda2()
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    result = PortfolioOptimizer().optimize_range(
        data_source=Tuda2DataSource(module=fake),
        schedule=_schedule(),
        benchmark_sid="000852.SH",
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        sequence_policy=SequencePolicy(output_weights="none"),
    )
    assert [step.date for step in result.steps] == list(dates)
    assert all(step.result.status.has_solution for step in result.steps)
    assert len([call for call in fake.calls if call[0] == "returns"]) == 1


def test_tuda2_default_fetches_each_full_range_once_and_reuses_it():
    fake = _FakeTuda2()
    source = Tuda2DataSource(module=fake)
    dates = tuple(pd.bdate_range("2026-01-02", periods=5).strftime("%Y-%m-%d"))
    result = PortfolioOptimizer().optimize_range(
        data_source=source,
        schedule=_schedule(dates),
        benchmark_sid="000852.SH",
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        sequence_policy=SequencePolicy(output_weights="none"),
    )
    assert len(result.steps) == len(dates)
    assert all(step.result.status.has_solution for step in result.steps)
    for kind in ("exposure", "cov", "spec_risk"):
        calls = [call for call in fake.calls if call[0:2] == ("risk", kind)]
        assert len(calls) == 1
        assert len(calls[0][2]) == len(dates)
    assert len([call for call in fake.calls if call[0] == "benchmark"]) == 1
    assert len([call for call in fake.calls if call[0] == "factor_names"]) == 2
    assert len([call for call in fake.calls if call[0] == "returns"]) == 1


def test_tuda2_tradability_is_fetched_once_for_the_full_range():
    fake = _FakeTuda2()
    dates = tuple(pd.bdate_range("2026-01-02", periods=5).strftime("%Y-%m-%d"))
    result = PortfolioOptimizer().optimize_range(
        data_source=Tuda2DataSource(module=fake),
        schedule=_schedule(dates),
        benchmark_sid="000852.SH",
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        sequence_policy=SequencePolicy(output_weights="none"),
        tradable_universe="tradeable_a_share",
    )
    assert len(result.steps) == len(dates)
    calls = [call for call in fake.calls if call[0] == "universe"]
    assert calls == [
        (
            "universe",
            "tradeable_a_share",
            tuple(pd.DatetimeIndex(pd.to_datetime(dates))),
        )
    ]


def test_tuda2_range_stops_after_full_preflight_before_returns_or_solver():
    dates = tuple(pd.bdate_range("2026-01-02", periods=4).strftime("%Y-%m-%d"))
    missing = pd.Timestamp(dates[2])
    fake = _MissingMiddleBenchmarkTuda2(missing)

    with pytest.raises(PortfolioValidationError) as caught:
        PortfolioOptimizer().optimize_range(
            data_source=Tuda2DataSource(module=fake),
            schedule=_schedule(dates),
            benchmark_sid="000852.SH",
            initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
            objective=MaximizeAlpha(),
            constraints=_constraints(),
            alpha_spec=AlphaSpec(),
        )
    assert any(issue.date == missing for issue in caught.value.report.errors)
    assert not [call for call in fake.calls if call[0] == "returns"]
    # The entire malformed range was still fetched in one call per data type.
    assert len([call for call in fake.calls if call[0:2] == ("risk", "exposure")]) == 1


def test_optimizer_single_facade_applies_tuda2_one_off_lists_without_return_fetch():
    fake = _FakeTuda2()
    date = pd.Timestamp("2026-01-02")
    universe = _schedule().day(date)
    result = PortfolioOptimizer().optimize(
        data_source=Tuda2DataSource(module=fake),
        date=date,
        universe=universe,
        benchmark_sid="000852.SH",
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        blacklist=["a"],
    )
    assert result.status.has_solution
    assert result.require_weights()["a"] == pytest.approx(0.0, abs=1e-9)
    assert not [call for call in fake.calls if call[0] == "returns"]


def test_optimizer_single_facade_requires_exactly_one_data_entry():
    optimizer = PortfolioOptimizer()
    common = {
        "objective": MaximizeAlpha(),
        "constraints": _constraints(),
    }
    with pytest.raises(ValueError, match="exactly one"):
        optimizer.optimize(**common)
    one_day_data = (
        InMemoryDataSource(
            risk_data=_frames(("2026-01-02",)),
            benchmark=_benchmark(("2026-01-02",)),
        )
        .build_problem(
            _schedule(("2026-01-02",)),
            objective=MaximizeAlpha(),
            constraints=_constraints(),
            alpha_spec=AlphaSpec(),
            initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        )
        .data
    )
    with pytest.raises(ValueError, match="exactly one"):
        optimizer.optimize(
            data=one_day_data,
            data_source=Tuda2DataSource(module=_FakeTuda2()),
            **common,
        )

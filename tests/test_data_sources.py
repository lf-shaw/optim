from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optim import (
    DataAlignmentError,
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

    def get_risk_model(self, kind, *, dts, model):
        self.calls.append(("risk", kind, tuple(pd.DatetimeIndex(dts)), model))
        dates = pd.DatetimeIndex(dts)
        if kind == "exposure":
            index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["dt", "sid"])
            return pd.DataFrame(
                {"size": [1.0, -1.0] * len(dates), "industry": ["bank", "tech"] * len(dates)},
                index=index,
            )
        if kind == "cov":
            index = pd.MultiIndex.from_product(
                [dates, ["country", "size", "bank", "tech"]],
                names=["dt", "factor"],
            )
            return pd.DataFrame(
                np.tile(np.diag([0.01, 0.04, 0.02, 0.03]), (len(dates), 1)),
                index=index,
                columns=["country", "size", "bank", "tech"],
            )
        index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["dt", "sid"])
        return pd.DataFrame({"spec_risk": 0.10}, index=index)

    def get_risk_model_factor_names(self, kind, *, model):
        self.calls.append(("factor_names", kind, model))
        return ["size"] if kind == "style" else ["bank", "tech"]

    def get_risk_model_schema(self, model):
        self.calls.append(("risk_schema", model))
        return {
            "factor_order": ["country", "size", "bank", "tech"],
            "factor_types": {
                "country": "country",
                "size": "style",
                "bank": "industry",
                "tech": "industry",
            },
            "constant_exposures": {"country": 1.0},
            "physical_exposure_columns": ["size", "industry"],
            "style": ["size"],
            "industry": ["bank", "tech"],
        }

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
    loaded = Tuda2DataSource(module=fake).load(dates=dates, benchmark="000852.SH")
    assert [call[1] for call in fake.calls if call[0] == "risk"] == [
        "exposure",
        "cov",
        "spec_risk",
    ]
    assert list(loaded.risk_data.exposure.columns) == ["size", "bank", "tech"]
    assert dict(loaded.risk_data.constant_exposures) == {"country": 1.0}
    first = loaded.risk_data.materialize(dates[0], pd.Index(["a", "b"], name="sid"))
    assert first.factor_names == ("country", "size", "bank", "tech")
    assert first.factor_types == ("country", "style", "industry", "industry")
    np.testing.assert_allclose(
        first.exposure,
        [[1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]],
    )
    assert first.annualization == "annualized_decimal"


def test_factor_risk_frames_align_exposure_and_specific_risk_by_asset_label():
    date = pd.Timestamp("2026-01-02")
    exposure_index = pd.MultiIndex.from_product(
        [[date], ["a", "b"]], names=["dt", "sid"]
    )
    specific_index = pd.MultiIndex.from_product(
        [[date], ["b", "a"]], names=["dt", "sid"]
    )
    covariance_index = pd.MultiIndex.from_product(
        [[date], ["country", "size"]], names=["dt", "factor"]
    )
    frames = FactorRiskFrames(
        exposure=pd.DataFrame({"size": [1.0, -1.0]}, index=exposure_index),
        covariance=pd.DataFrame(
            np.diag([0.01, 0.04]),
            index=covariance_index,
            columns=["country", "size"],
        ),
        specific_volatility=pd.Series(
            [0.20, 0.10], index=specific_index, name="spec_risk"
        ),
        factor_types={"country": "country", "size": "style"},
        constant_exposures={"country": 1.0},
    )

    model = frames.materialize(date, pd.Index(["b", "a"], name="sid"))

    np.testing.assert_allclose(model.exposure, [[1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(model.specific_volatility, [0.20, 0.10])


class _PhysicalCountryTuda2(_FakeTuda2):
    def get_risk_model(self, kind, *, dts, model):
        frame = super().get_risk_model(kind, dts=dts, model=model)
        if kind == "exposure":
            frame.insert(0, "country", 1.0)
        return frame


class _InvalidCountryTuda2(_FakeTuda2):
    def get_risk_model(self, kind, *, dts, model):
        frame = super().get_risk_model(kind, dts=dts, model=model)
        if kind == "exposure":
            frame.insert(0, "country", [1.0, 0.0] * len(pd.DatetimeIndex(dts)))
        return frame


class _HistoricDataYesTuda2(_FakeTuda2):
    """模拟跨越行业分类变更日的 DataYes 批量协方差布局。"""

    def get_risk_model(self, kind, *, dts, model):
        self.calls.append(("risk", kind, tuple(pd.DatetimeIndex(dts)), model))
        dates = pd.DatetimeIndex(dts)
        if kind == "exposure":
            index = pd.MultiIndex.from_product(
                [dates, ["a", "b"]], names=["dt", "sid"]
            )
            return pd.DataFrame(
                {
                    "size": [1.0, -1.0, 0.5, -0.5],
                    "industry": pd.Categorical(
                        ["old", "old", "new_a", "new_b"],
                        categories=["old", "new_a", "new_b"],
                    ),
                },
                index=index,
            )
        if kind == "cov":
            columns = ["new_b", "old", "country", "new_a", "size"]
            old = pd.DataFrame(
                [
                    [np.nan, 0.002, 0.01, np.nan, 0.001],
                    [np.nan, 0.003, 0.001, np.nan, 0.04],
                    [np.nan, 0.02, 0.002, np.nan, 0.003],
                ],
                index=pd.MultiIndex.from_product(
                    [[dates[0]], ["country", "size", "old"]],
                    names=["dt", "factor"],
                ),
                columns=columns,
            )
            new = pd.DataFrame(
                [
                    [0.004, np.nan, 0.01, 0.002, 0.001],
                    [0.006, np.nan, 0.001, 0.005, 0.04],
                    [0.007, np.nan, 0.002, 0.03, 0.005],
                    [0.05, np.nan, 0.004, 0.007, 0.006],
                ],
                index=pd.MultiIndex.from_product(
                    [[dates[1]], ["country", "size", "new_a", "new_b"]],
                    names=["dt", "factor"],
                ),
                columns=columns,
            )
            return pd.concat([old, new])
        index = pd.MultiIndex.from_product(
            [dates, ["a", "b"]], names=["dt", "sid"]
        )
        return pd.DataFrame({"spec_risk": 0.10}, index=index)

    def get_risk_model_factor_names(self, kind, *, model):
        self.calls.append(("factor_names", kind, model))
        return ["size"] if kind == "style" else ["old", "new_a", "new_b"]

    def get_risk_model_schema(self, model):
        self.calls.append(("risk_schema", model))
        return {
            "factor_order": ["country", "size", "old", "new_a", "new_b"],
            "factor_types": {
                "country": "country",
                "size": "style",
                "old": "industry",
                "new_a": "industry",
                "new_b": "industry",
            },
            "constant_exposures": {"country": 1.0},
            "physical_exposure_columns": ["size", "industry"],
            "style": ["size"],
            "industry": ["old", "new_a", "new_b"],
        }


def test_tuda2_adapter_accepts_but_does_not_virtualize_physical_country():
    date = pd.Timestamp("2026-01-02")
    loaded = Tuda2DataSource(module=_PhysicalCountryTuda2()).load(
        dates=[date], benchmark="000852.SH"
    )

    assert list(loaded.risk_data.exposure.columns) == [
        "country",
        "size",
        "bank",
        "tech",
    ]
    assert not loaded.risk_data.constant_exposures
    model = loaded.risk_data.materialize(date, pd.Index(["a", "b"], name="sid"))
    np.testing.assert_allclose(model.exposure[:, 0], 1.0)


def test_virtual_country_is_numerically_identical_to_physical_country():
    date = pd.Timestamp("2026-01-02")
    assets = pd.Index(["b", "a"], name="sid")
    virtual = Tuda2DataSource(module=_FakeTuda2()).load(
        dates=[date], benchmark="000852.SH"
    ).risk_data.materialize(date, assets)
    physical = Tuda2DataSource(module=_PhysicalCountryTuda2()).load(
        dates=[date], benchmark="000852.SH"
    ).risk_data.materialize(date, assets)

    np.testing.assert_allclose(virtual.exposure, physical.exposure)
    np.testing.assert_allclose(virtual.covariance, physical.covariance)
    np.testing.assert_allclose(
        virtual.specific_volatility,
        physical.specific_volatility,
    )
    active = np.array([0.60, -0.20])
    virtual_factor = virtual.exposure.T @ active
    physical_factor = physical.exposure.T @ active
    assert virtual_factor[0] == pytest.approx(0.40)
    np.testing.assert_allclose(virtual_factor, physical_factor)
    assert (
        virtual_factor @ virtual.covariance @ virtual_factor
        == pytest.approx(physical_factor @ physical.covariance @ physical_factor)
    )


def test_tuda2_adapter_rejects_nonconstant_country_exposure():
    with pytest.raises(ValueError, match="constant exposure 'country' must equal 1.0"):
        Tuda2DataSource(module=_InvalidCountryTuda2()).load(
            dates=[pd.Timestamp("2026-01-02")], benchmark="000852.SH"
        )


def test_tuda2_adapter_uses_daily_covariance_rows_across_industry_regimes():
    dates = pd.to_datetime(["2019-12-02", "2019-12-03"])
    loaded = Tuda2DataSource(module=_HistoricDataYesTuda2()).load(
        dates=dates,
        benchmark="000852.SH",
    )

    old = loaded.risk_data.materialize(dates[0], pd.Index(["a", "b"], name="sid"))
    new = loaded.risk_data.materialize(dates[1], pd.Index(["a", "b"], name="sid"))

    assert old.factor_names == ("country", "size", "old")
    assert new.factor_names == ("country", "size", "new_a", "new_b")
    np.testing.assert_allclose(
        old.covariance,
        [[0.01, 0.001, 0.002], [0.001, 0.04, 0.003], [0.002, 0.003, 0.02]],
    )
    np.testing.assert_allclose(
        new.covariance,
        [
            [0.01, 0.001, 0.002, 0.004],
            [0.001, 0.04, 0.005, 0.006],
            [0.002, 0.005, 0.03, 0.007],
            [0.004, 0.006, 0.007, 0.05],
        ],
    )
    np.testing.assert_allclose(old.exposure[:, 2], [1.0, 1.0])
    np.testing.assert_allclose(new.exposure[:, 2:], [[1.0, 0.0], [0.0, 1.0]])


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
        benchmark="000852.SH",
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
        benchmark="000852.SH",
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
    assert len([call for call in fake.calls if call[0] == "risk_schema"]) == 1
    assert not [call for call in fake.calls if call[0] == "factor_names"]
    assert len([call for call in fake.calls if call[0] == "returns"]) == 1


def test_tuda2_tradability_is_fetched_once_for_the_full_range():
    fake = _FakeTuda2()
    dates = tuple(pd.bdate_range("2026-01-02", periods=5).strftime("%Y-%m-%d"))
    result = PortfolioOptimizer().optimize_range(
        data_source=Tuda2DataSource(module=fake),
        schedule=_schedule(dates),
        benchmark="000852.SH",
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
            benchmark="000852.SH",
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
        benchmark="000852.SH",
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(),
        constraints=_constraints(),
        alpha_spec=AlphaSpec(),
        blacklist=["a"],
    )
    assert result.status.has_solution
    assert result.require_weights()["a"] == pytest.approx(0.0, abs=1e-9)
    assert not [call for call in fake.calls if call[0] == "returns"]


def test_single_custom_benchmark_aligns_without_index_io():
    fake = _FakeTuda2()
    date = pd.Timestamp("2026-01-02")
    result = PortfolioOptimizer().optimize(
        data_source=Tuda2DataSource(module=fake), date=date,
        universe=_schedule().day(date), benchmark=pd.Series({"b": 0.3, "a": 0.7}),
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(), constraints=_constraints(), alpha_spec=AlphaSpec(),
    )
    assert result.status.has_solution
    np.testing.assert_allclose(result.problem.data.benchmark, [0.7, 0.3])
    assert not any(call[0] == "benchmark" for call in fake.calls)


def test_range_custom_benchmark_without_index_io():
    fake = _FakeTuda2()
    result = PortfolioOptimizer().optimize_range(
        data_source=Tuda2DataSource(module=fake), schedule=_schedule(),
        benchmark=_benchmark(), initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(), constraints=_constraints(), alpha_spec=AlphaSpec(),
    )
    assert len(result.steps) == 2
    assert all(step.result.status.has_solution for step in result.steps)
    assert not any(call[0] == "benchmark" for call in fake.calls)
    assert len([call for call in fake.calls if call[0] == "risk"]) == 3


def test_single_period_all_dated_inputs():
    fake = _FakeTuda2()
    date = pd.Timestamp("2026-01-02")
    universe = _schedule().day(date)
    initial = pd.Series({"a": 0.5, "b": 0.5})
    def dated(value):
        return pd.concat({date: value}, names=["dt", "sid"])
    result = PortfolioOptimizer().optimize(
        data_source=Tuda2DataSource(module=fake), date=date,
        universe=dated(universe), benchmark=dated(initial), initial_weight=dated(initial),
        objective=MaximizeAlpha(), constraints=_constraints(), alpha_spec=AlphaSpec(),
    )
    assert result.status.has_solution
    assert result.problem.data.assets.nlevels == 1
    np.testing.assert_allclose(result.problem.data.initial_weight, [0.5, 0.5])


@pytest.mark.parametrize("field", ["universe", "benchmark", "initial_weight"])
@pytest.mark.parametrize("bad", ["multiple", "mismatch", "duplicate", "names", "missing", "string"])
def test_single_period_bad_dates_before_io(field, bad):
    fake = _FakeTuda2()
    date = pd.Timestamp("2026-01-02")
    kwargs = dict(
        data_source=Tuda2DataSource(module=fake), date=date,
        universe=_schedule().day(date), benchmark=pd.Series({"a": 0.5, "b": 0.5}),
        initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
        objective=MaximizeAlpha(), constraints=_constraints(), alpha_spec=AlphaSpec(),
    )
    value = kwargs[field]
    target = {"mismatch": date + pd.Timedelta(days=1), "missing": pd.NaT, "string": str(date)}.get(bad, date)
    value = pd.concat({target: value}, names=["dt", "sid"])
    if bad == "multiple":
        value = pd.concat([value, pd.concat({date + pd.Timedelta(days=1): kwargs[field]}, names=["dt", "sid"])])
    elif bad == "duplicate":
        value = pd.concat([value, value])
    elif bad == "names":
        value.index = value.index.set_names(["date", "sid"])
    kwargs[field] = value
    with pytest.raises(DataAlignmentError):
        PortfolioOptimizer().optimize(**kwargs)
    assert fake.calls == []


@pytest.mark.parametrize("benchmark", [
    pd.Series({"a": 0.5, "b": 0.5}),
    _benchmark(dates=("2026-01-02",)),
    pd.concat([_benchmark(), _benchmark()]),
])
def test_range_custom_benchmark_rejected_before_io(benchmark):
    from optim import DataAlignmentError

    fake = _FakeTuda2()
    with pytest.raises(DataAlignmentError):
        PortfolioOptimizer().optimize_range(
            data_source=Tuda2DataSource(module=fake), schedule=_schedule(),
            benchmark=benchmark, initial_weight=pd.Series({"a": 0.5, "b": 0.5}),
            objective=MaximizeAlpha(), constraints=_constraints(), alpha_spec=AlphaSpec(),
        )
    assert not fake.calls


def test_removed_benchmark_sid_keyword():
    with pytest.raises(TypeError, match="benchmark_sid"):
        PortfolioOptimizer().optimize(benchmark_sid="000852.SH", objective=MaximizeAlpha())


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

"""批量静态对齐与日期切片的行为回归，不绑定 carry2 内部算法。"""

from dataclasses import replace

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
    PortfolioOptimizer,
    PortfolioSchedule,
    RiskCoveragePolicy,
    SequenceDataError,
    SequencePolicy,
)
from optim.data._date_slices import _DateSlices
from optim.data import memory, _reindex


def _case():
    dates = pd.date_range("2026-01-05", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["a", "b", "c", "d"]], names=["dt", "sid"]
    )
    factors = ["SIZE", "country"]
    covariance = pd.DataFrame(
        np.tile(np.diag([0.04, 0.01]), (3, 1)),
        index=pd.MultiIndex.from_product([dates, factors], names=["dt", "factor"]),
        columns=factors,
    )
    risk = FactorRiskFrames(
        exposure=pd.DataFrame(
            {"SIZE": np.tile([-1.0, 0.0, 1.0, 2.0], 3), "country": 1.0}, index=index
        ),
        covariance=covariance,
        specific_volatility=pd.Series(0.2, index=index),
        factor_types={"SIZE": "style", "country": "country"},
    )
    # 日期间样本会变化，批量目标不是日期与全部股票的笛卡尔积。
    target = index.delete([3, 6, 9])
    schedule = PortfolioSchedule(
        pd.DataFrame({"alpha": np.arange(len(target), dtype=float)}, index=target)
    )
    bench = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product([dates, ["a"]], names=["dt", "sid"]),
        name="weight",
    )
    source = InMemoryDataSource(risk_data=risk, benchmark=bench)
    args = dict(
        objective=MaximizeAlpha(),
        constraints=PortfolioConstraints(),
        alpha_spec=AlphaSpec(),
        initial_weight=pd.Series({"a": 1.0}),
    )
    return source, schedule, args


@pytest.mark.parametrize("native", [False, True])
def test_batch_matches_individual_and_reuses_alignment(monkeypatch, native):
    if native:
        monkeypatch.setattr(_reindex, "_carry_reindex", lambda: None)
    source, schedule, args = _case()
    reference = [
        source.materialize_problem(
            schedule,
            date,
            i,
            independent_initial_weights=None,
            extra_attribute_columns=(),
            **args,
        )
        for i, date in enumerate(schedule.dates)
    ]
    calls = []
    original = memory._reindex_rows

    def tracked(value, index, **kwargs):
        if isinstance(index, pd.MultiIndex):
            calls.append(index)
        return original(value, index, **kwargs)

    monkeypatch.setattr(memory, "_reindex_rows", tracked)
    run = source.prepare_run(schedule, **args)
    run.validation.raise_for_errors()
    assert len(calls) == 3
    assert run.data_source is not source
    assert run.data_source.risk_data.exposure.index.equals(schedule.universe.index)
    assert len(source.risk_data.exposure) == 12

    def audit_again(*args, **kwargs):
        pytest.fail("正式物化不应再次执行基准覆盖审计")

    monkeypatch.setattr(memory, "align_benchmark", audit_again)
    for _ in range(2):
        for expected in reference:
            observed = run.problem_at(expected.data.date)
            for name in ("alpha", "benchmark", "initial_weight", "tradable"):
                np.testing.assert_array_equal(
                    getattr(observed.data, name), getattr(expected.data, name)
                )
            for name in ("exposure", "covariance", "specific_volatility"):
                np.testing.assert_array_equal(
                    getattr(observed.data.risk_model, name),
                    getattr(expected.data.risk_model, name),
                )
            assert (
                observed.data.provenance.metadata == expected.data.provenance.metadata
            )
    assert len(calls) == 3


def test_batch_keeps_per_date_coverage_evidence():
    source, schedule, args = _case()
    benchmark = source.benchmark.copy()
    benchmark.iloc[0] = 0.99
    extra = pd.Series(
        [0.01],
        index=pd.MultiIndex.from_tuples(
            [(schedule.dates[0], "outside")], names=["dt", "sid"]
        ),
    )
    benchmark = pd.concat([benchmark, extra]).sort_index()
    source = InMemoryDataSource(risk_data=source.risk_data, benchmark=benchmark)
    run = source.prepare_run(schedule, **args)
    assert any(issue.date == schedule.dates[0] for issue in run.validation.issues)
    policy = BenchmarkCoveragePolicy(
        action="renormalize_within_tolerance", missing_mass_tolerance=0.02
    )
    source = InMemoryDataSource(
        risk_data=source.risk_data, benchmark=benchmark, benchmark_policy=policy
    )
    run = source.prepare_run(schedule, **args)
    run.validation.raise_for_errors()
    data = run.problem_at(schedule.dates[0]).data
    assert data.benchmark.sum() == pytest.approx(1.0)
    assert data.provenance.metadata["benchmark_missing_mass"] == pytest.approx(0.01)
    assert data.provenance.metadata[
        "benchmark_renormalization_factor"
    ] == pytest.approx(1 / 0.99)
    assert benchmark.iloc[0] == 0.99


@pytest.mark.parametrize("kind", ["risk_row", "risk_date", "benchmark_date"])
def test_missing_data_still_reports_the_date(kind):
    source, schedule, args = _case()
    date = schedule.dates[1]
    if kind == "benchmark_date":
        source = InMemoryDataSource(
            risk_data=source.risk_data,
            benchmark=source.benchmark.drop(date, level="dt"),
        )
    else:
        exp = source.risk_data.exposure
        exp = (
            exp.drop((date, "a")) if kind == "risk_row" else exp.drop(date, level="dt")
        )
        source = InMemoryDataSource(
            risk_data=replace(source.risk_data, exposure=exp),
            benchmark=source.benchmark,
        )
    run = source.prepare_run(schedule, **args)
    assert any(issue.date == date for issue in run.validation.issues)


def test_risk_coverage_checks_both_tables_and_stops_before_materialization(monkeypatch):
    source, schedule, args = _case()
    first, second, _ = schedule.dates
    source.risk_data = replace(
        source.risk_data,
        exposure=source.risk_data.exposure.drop([(first, "a"), (second, "b")]),
        specific_volatility=source.risk_data.specific_volatility.drop((first, "a")),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("风险覆盖超限不应逐日物化")

    monkeypatch.setattr(source, "materialize_problem", forbidden)
    run = source.prepare_run(schedule, **args)
    assert run.data_source is source
    first_error, second_error = run.validation.errors
    assert (first_error.date, second_error.date) == (first, second)
    assert first_error.context["missing_count"] == 1
    assert first_error.context["exposure_missing_count"] == 1
    assert first_error.context["specific_missing_count"] == 1
    assert second_error.context["specific_missing_count"] == 0


def test_risk_coverage_crop_checks_benchmark_and_audits_solved_result():
    source, schedule, args = _case()
    date = schedule.dates[1]
    index = source.risk_data.exposure.index
    benchmark = pd.Series(0.0, index=index)
    benchmark.loc[(slice(None), "a")] = 1.0
    benchmark.loc[(date, "a")] = 0.99
    benchmark.loc[(date, "b")] = 0.01
    risk = replace(
        source.risk_data,
        specific_volatility=source.risk_data.specific_volatility.drop((date, "b")),
    )
    policy = RiskCoveragePolicy(missing_asset_fraction_tolerance=1 / 3)
    strict = InMemoryDataSource(risk_data=risk, benchmark=benchmark)
    report = strict.prepare_run(schedule, risk_coverage_policy=policy, **args).validation
    assert any("omits 1.000000% benchmark mass" in error.message for error in report.errors)

    tolerant = InMemoryDataSource(
        risk_data=risk, benchmark=benchmark,
        benchmark_policy=BenchmarkCoveragePolicy(
            action="renormalize_within_tolerance", missing_mass_tolerance=0.01,
        ),
    )
    state = SequencePolicy(mode="independent")
    result = PortfolioOptimizer().optimize_range(
        data_source=tolerant,
        schedule=schedule,
        sequence_policy=state,
        risk_coverage_policy=policy,
        **args,
    ).result_for_date(date)
    assert result.status.has_solution
    assert "b" not in result.require_weights().index
    assert result.alignment.risk_missing_asset_count == 1
    assert result.alignment.risk_missing_asset_fraction == 1 / 3
    assert result.alignment.benchmark_missing_mass == pytest.approx(0.01)
    assert (date, "b") in schedule.universe.index


def test_risk_coverage_does_not_tolerate_entire_day_or_invalid_values():
    source, schedule, args = _case()
    date = schedule.dates[1]
    original = source.risk_data
    source.risk_data = replace(original, exposure=original.exposure.drop(date, level="dt"))
    permissive = RiskCoveragePolicy(missing_asset_fraction_tolerance=1.0)
    run = source.prepare_run(schedule, risk_coverage_policy=permissive, **args)
    assert "no exact data" in run.validation.errors[0].message

    bad = original.specific_volatility.copy()
    bad.loc[(date, "b")] = np.nan
    source.risk_data = replace(original, specific_volatility=bad)
    run = source.prepare_run(schedule, risk_coverage_policy=permissive, **args)
    assert run.validation.errors[0].field == "data.risk_model.specific_volatility"
    assert not run.validation.warnings


def test_risk_coverage_crop_never_drops_explicit_initial_weight():
    source, schedule, args = _case()
    date = schedule.dates[0]
    source.risk_data = replace(
        source.risk_data, exposure=source.risk_data.exposure.drop((date, "b")),
    )
    args["initial_weight"] = pd.Series({"a": 0.9, "b": 0.1})
    run = source.prepare_run(
        schedule, risk_coverage_policy=RiskCoveragePolicy(1 / 3), **args,
    )
    assert "omits 10.000000% initial holding mass" in run.validation.errors[0].message


def test_risk_coverage_crop_does_not_waive_drifted_holdings():
    source, schedule, args = _case()
    first, second, last = schedule.dates
    universe = schedule.universe.copy()
    universe.loc[(first, "b"), "alpha"] = 100.0
    source.risk_data = replace(
        source.risk_data, exposure=source.risk_data.exposure.drop((second, "b")),
    )
    with pytest.raises(SequenceDataError, match="omit 100.000000% of drifted holdings"):
        PortfolioOptimizer().optimize_range(
            data_source=source,
            schedule=replace(schedule, universe=universe),
            holding_period_returns={
                second: pd.Series(0.0, index=["a", "b", "c"]),
                last: pd.Series(0.0, index=["a", "d"]),
            },
            risk_coverage_policy=RiskCoveragePolicy(1 / 3),
            **args,
        )


def test_date_slices_match_xs_even_with_interleaved_dates():
    source, schedule, args = _case()
    frame = source.risk_data.exposure.iloc[[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]
    cache = _DateSlices(frame, "sid", "test")
    for date in schedule.dates:
        pd.testing.assert_frame_equal(cache._day(date), frame.xs(date, level="dt"))
        assert cache._day(date).index is cache._day(date).index


def test_batch_carry_rejects_nonmonotonic_input():
    pytest.importorskip("carry2")
    source, schedule, args = _case()
    source = InMemoryDataSource(risk_data=source.risk_data, benchmark=source.benchmark.iloc[::-1])
    with pytest.raises(ValueError, match="monotonic"):
        source.prepare_run(schedule, **args)


def test_prepared_chained_results_match_explicit_problems():
    source, schedule, args = _case()
    # 使用固定样本，单独比较全链求解及漂移结果，不引入持仓覆盖冲突。
    schedule = PortfolioSchedule(
        pd.DataFrame(
            {"alpha": np.tile([0.0, 1.0, 2.0, 3.0], 3)},
            index=source.risk_data.exposure.index,
        )
    )
    problems = [
        source.materialize_problem(
            schedule,
            date,
            i,
            independent_initial_weights=None,
            extra_attribute_columns=(),
            **args,
        )
        for i, date in enumerate(schedule.dates)
    ]
    run = source.prepare_run(schedule, **args)
    returns = {
        date: pd.Series([0.01, -0.02, 0.03, 0.04], index=["a", "b", "c", "d"])
        for date in schedule.dates[1:]
    }
    optimizer = PortfolioOptimizer()
    expected = optimizer.solve_sequence(problems, holding_period_returns=returns)
    actual = optimizer.solve_sequence(run, holding_period_returns=returns)
    for left, right in zip(expected.steps, actual.steps):
        pd.testing.assert_series_equal(
            left.result.require_weights(), right.result.require_weights()
        )
        pd.testing.assert_series_equal(left.pretrade_weight, right.pretrade_weight)

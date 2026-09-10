"""手工单期数据入口的标签、单位和实际求解回归。"""

from dataclasses import replace
import inspect

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    BenchmarkCoverageError,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    DataProvenance,
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioConstraints,
    PortfolioOptimizer,
    RiskAdjustedAlpha,
    TrackingErrorLimit,
    make_factor_risk_model,
    make_portfolio_data,
)


DAY = pd.Timestamp("2026-08-14")


@pytest.fixture
def universe():
    return pd.DataFrame(
        {
            "alpha": [0.7, 0.1, -0.3],
            "tradable": [True, True, True],
            "score": [2.0, 1.0, 3.0],
        },
        index=pd.Index(["B", "A", "C"], name="sid"),
    )


def risk_inputs(assets):
    """股票和因子列故意错序，附带无效的历史因子列。"""
    return dict(
        date=DAY,
        assets=assets,
        exposure=pd.DataFrame({"SIZE": [1.0, -1.0, 0.0]}, index=["A", "C", "B"]),
        factor_covariance=pd.DataFrame(
            [[0.001, np.nan, 0.01], [0.04, np.nan, 0.001]],
            index=["country", "SIZE"],
            columns=["SIZE", "old", "country"],
        ),
        specific_volatility=pd.Series([0.3, 0.1, 0.2], index=["C", "A", "B"]),
        factor_types={"country": "country", "SIZE": "style"},
        constant_exposures={"country": 1.0},
    )


def dated(table, coordinate="sid", day=DAY):
    return pd.concat({day: table}, names=["dt", coordinate])


def test_universe_and_sparse_weights(universe):
    original = universe.copy(deep=True)
    data = make_portfolio_data(
        date=DAY,
        universe=universe,
        benchmark=pd.Series({"A": 0.4, "B": 0.6}),
        initial_weight=pd.Series({"A": 0.2}),
        alpha_spec=AlphaSpec(units="standardized_score"),
        extra_attribute_columns=("score",),
    )
    assert data.assets.equals(universe.index)
    np.testing.assert_array_equal(data.alpha, [0.7, 0.1, -0.3])
    np.testing.assert_array_equal(data.benchmark, [0.6, 0.4, 0])
    np.testing.assert_array_equal(data.initial_weight, [0, 0.2, 0])
    np.testing.assert_array_equal(data.extra_attributes["score"], [2, 1, 3])
    assert data.provenance.metadata["benchmark_source_date"] == DAY
    assert np.shares_memory(data.alpha, universe["alpha"].to_numpy())
    pd.testing.assert_frame_equal(universe, original)


def test_optional_columns_and_custom_names(universe):
    data = make_portfolio_data(date=DAY, universe=universe.iloc[:, 0:0])
    assert (
        data.alpha is data.benchmark is data.initial_weight is data.risk_model is None
    )
    assert data.tradable.all()
    renamed = universe.rename(columns={"alpha": "signal", "tradable": "can_trade"})
    custom = make_portfolio_data(
        date=DAY, universe=renamed, alpha_column="signal", tradable_column="can_trade"
    )
    np.testing.assert_array_equal(custom.alpha, universe.alpha)


def test_coverage_policy_is_explicit(universe):
    benchmark = pd.Series({"A": 0.999, "outside": 0.001})
    with pytest.raises(BenchmarkCoverageError):
        make_portfolio_data(date=DAY, universe=universe, benchmark=benchmark)
    data = make_portfolio_data(
        date=DAY,
        universe=universe,
        benchmark=benchmark,
        benchmark_policy=BenchmarkCoveragePolicy(
            action="renormalize_within_tolerance", missing_mass_tolerance=0.002
        ),
    )
    assert data.benchmark.sum() == pytest.approx(1)
    assert data.provenance.metadata["benchmark_missing_mass"] == pytest.approx(0.001)
    with pytest.raises(DataAlignmentError, match="nonzero holdings"):
        make_portfolio_data(date=DAY, universe=universe, initial_weight=benchmark)


@pytest.mark.parametrize("field", ["universe", "benchmark", "initial_weight"])
def test_exact_date_required(universe, field):
    kwargs = dict(
        date=DAY,
        universe=universe,
        benchmark=pd.Series({"A": 1.0}),
        initial_weight=pd.Series({"A": 1.0}),
    )
    kwargs[field] = dated(kwargs[field])
    make_portfolio_data(**kwargs)
    kwargs["date"] = DAY + pd.Timedelta(days=1)
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(**kwargs)


@pytest.mark.parametrize("field", ["universe", "benchmark", "initial_weight"])
def test_single_period_rejects_multiple_dates(universe, field):
    kwargs = dict(date=DAY, universe=universe, benchmark=pd.Series({"A": 1.}), initial_weight=pd.Series({"A": 1.}))
    kwargs[field] = pd.concat([dated(kwargs[field]), dated(kwargs[field], day=DAY + pd.Timedelta(days=1))])
    with pytest.raises(DataAlignmentError, match="exactly the requested date"):
        make_portfolio_data(**kwargs)


@pytest.mark.parametrize("bad", ["False", 1, None, pd.NA])
def test_tradable_must_be_boolean(universe, bad):
    universe["tradable"] = pd.Series(
        [True, bad, True], index=universe.index, dtype=object
    )
    with pytest.raises(TypeError, match="bool"):
        make_portfolio_data(date=DAY, universe=universe)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1 + 2j])
def test_alpha_numeric(universe, bad):
    universe["alpha"] = [0.7, bad, 0.1]
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(date=DAY, universe=universe)


def test_duplicate_and_missing_labels(universe):
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(date=DAY, universe=pd.concat([universe, universe]))
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(date=DAY, universe=pd.concat([universe, universe], axis=1))
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(date=DAY, universe=universe.iloc[:0])
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(
            date=DAY, universe=universe, extra_attribute_columns=("absent",)
        )
    with pytest.raises(TypeError):
        make_portfolio_data(date=DAY, universe=universe, benchmark=np.ones(3) / 3)
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(
            date=DAY, universe=universe, benchmark=pd.Series({"A": 1 + 1j})
        )
    with pytest.raises(DataAlignmentError):
        make_portfolio_data(
            date=DAY,
            universe=universe,
            provenance=DataProvenance(source_date=DAY - pd.Timedelta(days=1)),
        )


def test_risk_alignment_and_units(universe):
    kwargs = risk_inputs(universe.index)
    risk = make_factor_risk_model(**kwargs)
    assert risk.factor_names == ("country", "SIZE")
    np.testing.assert_array_equal(risk.exposure, [[1, 0], [1, 1], [1, -1]])
    np.testing.assert_allclose(risk.covariance, [[0.01, 0.001], [0.001, 0.04]])
    np.testing.assert_allclose(risk.specific_volatility, [0.2, 0.1, 0.3])
    for field, coordinate in (
        ("exposure", "sid"),
        ("specific_volatility", "sid"),
        ("factor_covariance", "factor"),
    ):
        kwargs[field] = dated(kwargs[field], coordinate)
    dated_risk = make_factor_risk_model(**kwargs)
    np.testing.assert_array_equal(risk.exposure, dated_risk.exposure)
    kwargs["date"] = DAY + pd.Timedelta(days=1)
    with pytest.raises(DataAlignmentError):
        make_factor_risk_model(**kwargs)


def test_risk_factory_default_assets_and_direct_data_validation(universe):
    kwargs = risk_inputs(universe.index)
    kwargs.pop("assets")
    risk = make_factor_risk_model(**kwargs)
    assert risk.assets.equals(kwargs["exposure"].index)
    data = make_portfolio_data(date=DAY, universe=universe, risk_model=risk)
    assert data.risk_model.assets.equals(universe.index)
    invalid = replace(data, risk_model=risk)
    from optim import PortfolioProblem
    report = PortfolioOptimizer().validate(PortfolioProblem(invalid, MaximizeAlpha(), PortfolioConstraints()))
    assert any(issue.code == "asset_order_mismatch" for issue in report.issues)


@pytest.mark.parametrize(
    "field", ["exposure", "factor_covariance", "specific_volatility"]
)
def test_missing_and_complex_risk(universe, field):
    kwargs = risk_inputs(universe.index)
    kwargs[field] = (
        kwargs[field].drop(columns="SIZE")
        if field == "factor_covariance"
        else kwargs[field].iloc[:1]
    )
    with pytest.raises(DataAlignmentError):
        make_factor_risk_model(**kwargs)
    kwargs = risk_inputs(universe.index)
    kwargs[field] = kwargs[field].astype(complex) + 1j
    with pytest.raises(DataAlignmentError, match="real-valued"):
        make_factor_risk_model(**kwargs)


def test_risk_object_checks(universe):
    risk = make_factor_risk_model(**risk_inputs(universe.index))
    for changed in (
        replace(risk, asof=DAY + pd.Timedelta(days=1)),
        replace(risk, specific_volatility=np.array([-0.1, 0.1, 0.2])),
        replace(risk, exposure=np.ones((2, 2))),
        replace(risk, annualization="daily"),
    ):
        with pytest.raises(DataAlignmentError):
            make_portfolio_data(date=DAY, universe=universe, risk_model=changed)
    kwargs = risk_inputs(universe.index)
    kwargs["constant_exposures"] = {"absent": 1.0}
    with pytest.raises(DataAlignmentError):
        make_factor_risk_model(**kwargs)


@pytest.mark.parametrize(
    "objective,limit",
    [
        (MaximizeAlpha(), None),
        (RiskAdjustedAlpha(), None),
        (MaximizeAlpha(), 0.02),
        (MinimizeTrackingError(), None),
    ],
)
def test_factory_data_solves_all_objectives(universe, objective, limit):
    risk = make_factor_risk_model(**risk_inputs(universe.index))
    if isinstance(objective, MinimizeTrackingError):
        universe = universe.drop(columns="alpha")
    data = make_portfolio_data(
        date=DAY,
        universe=universe,
        risk_model=risk,
        benchmark=pd.Series({"A": 0.4, "B": 0.6}),
        initial_weight=pd.Series({"A": 0.4, "B": 0.6}),
        alpha_spec=AlphaSpec(units="standardized_score"),
    )
    result = PortfolioOptimizer().optimize(
        data=data,
        objective=objective,
        constraints=PortfolioConstraints(
            tracking_error=None if limit is None else TrackingErrorLimit(limit)
        ),
    )
    weights = result.require_weights()
    assert weights.sum() == pytest.approx(1, abs=1e-6)
    active = weights.reindex(data.assets).to_numpy() - data.benchmark
    factor_active = risk.exposure.T @ active
    variance = (
        factor_active @ risk.covariance @ factor_active
        + np.square(risk.specific_volatility * active).sum()
    )
    if limit is not None:
        assert np.sqrt(variance) <= limit + 1e-6
    if isinstance(objective, MinimizeTrackingError):
        # 此目标最小化方差；按方差量纲检查数值误差，而非要求精确零权重差。
        assert variance <= 1e-8


def test_notebook_signature_and_docs():
    assert "universe" in inspect.signature(make_portfolio_data).parameters
    assert "assets" not in inspect.signature(make_portfolio_data).parameters
    for function in (make_portfolio_data, make_factor_risk_model):
        assert all(
            name in inspect.getdoc(function)
            for name in inspect.signature(function).parameters
        )

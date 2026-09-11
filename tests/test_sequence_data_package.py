from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd

from optim import (
    AlphaSpec,
    FactorRiskFrames,
    InMemoryDataSource,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioSchedule,
    SequencePolicy,
    SolverPolicy,
    WeightBounds,
)
from optim.devtools.sequence_data_package import (
    export_tuda2_sequence,
    load_sequence_package,
    performance_frame,
)


def _inputs():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["dt", "sid"])
    exposure = pd.DataFrame({"size": [1.0, -1.0] * 2}, index=index)
    covariance = pd.DataFrame(
        [[0.04], [0.04]],
        index=pd.MultiIndex.from_product([dates, ["size"]], names=["dt", "factor"]),
        columns=["size"],
    )
    risk = FactorRiskFrames(
        exposure=exposure,
        covariance=covariance,
        specific_volatility=pd.Series(0.1, index=index),
        factor_types={"size": "style"},
    )
    benchmark = pd.Series([0.5, 0.5] * 2, index=index, name="weight")
    schedule = PortfolioSchedule(
        pd.DataFrame(
            {"alpha": [1.0, 0.0, 0.8, 0.2], "tradable": [1, 1, 1, 1]}, index=index
        )
    )
    return dates, InMemoryDataSource(risk_data=risk, benchmark=benchmark), schedule


class _PreparedSource:
    def __init__(self, source, returns):
        self.source = source
        self.returns = returns

    def prepare_sequence(self, **kwargs):
        kwargs.pop("benchmark")
        kwargs.pop("holding_period_returns")
        kwargs.pop("require_holding_returns")
        kwargs.pop("tradable_universe")
        kwargs.pop("benchmark_policy")
        run = self.source.prepare_run(**kwargs)
        return SimpleNamespace(run=run, holding_period_returns=self.returns)


def test_sequence_package_round_trip_and_solve(tmp_path):
    dates, source, schedule = _inputs()
    returns = {dates[1]: pd.Series({"a": 0.01, "b": -0.02})}
    path = tmp_path / "sequence.zip"
    export_tuda2_sequence(
        path,
        data_source=_PreparedSource(source, returns),
        schedule=schedule,
        benchmark="000300.SH",
        objective=MaximizeAlpha(),
        constraints=PortfolioConstraints(asset_weight=WeightBounds(0.0, 1.0)),
        alpha_spec=AlphaSpec(),
        solver_policy=SolverPolicy(backend="auto"),
        sequence_policy=SequencePolicy(output_weights="none"),
    )
    with ZipFile(path) as archive:
        assert "manifest.json" in archive.namelist()
        assert not any(
            "result" in name or "weights/" in name for name in archive.namelist()
        )
    case = load_sequence_package(path)
    pd.testing.assert_frame_equal(case.schedule.universe, schedule.universe)
    assert (
        case.holding_period_returns[dates[1]].to_dict() == returns[dates[1]].to_dict()
    )
    result = case.solve(backend="auto", show_progress=False)
    assert len(result.steps) == 2
    assert all(step.result.status.has_solution for step in result.steps)
    assert set(performance_frame(result)["backend"]) == {"highs"}

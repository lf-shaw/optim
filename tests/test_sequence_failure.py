"""收益缺失的股票定位、异常上下文和显式导出。"""

from dataclasses import replace
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from optim import PortfolioOptimizer, SequenceDataError, SequencePolicy
from optim.sequence import _mark_to_market


def _second(first):
    day = pd.Timestamp("2026-01-05")
    return replace(
        first,
        data=replace(
            first.data, date=day, risk_model=replace(first.data.risk_model, asof=day)
        ),
    )


def test_failure_preserves_partial_results_without_disk(sample_lp_problem):
    first, second = sample_lp_problem, _second(sample_lp_problem)
    with pytest.raises(SequenceDataError) as caught:
        PortfolioOptimizer().solve_sequence(
            [first, second],
            holding_period_returns={second.data.date: pd.Series({"b": np.nan})},
            sequence_policy=SequencePolicy(output_weights="none"),
        )
    error = caught.value
    assert "2026-01-02 -> 2026-01-05" in str(error)
    assert "a=" in str(error)
    assert "b=" in str(error)
    assert error.evidence.loc["a", "reason"] == "missing_sid"
    assert error.evidence.loc["b", "reason"] == "missing_value"
    assert error.date == second.data.date
    assert len(error.partial_result.steps) == 1
    assert error.partial_result.steps[0].result.weights is None
    assert error.previous_weight is not None
    assert error.dump_path is None


def test_auto_dump_and_manual_dump(sample_lp_problem, tmp_path):
    second = _second(sample_lp_problem)
    directory = tmp_path / "failures"
    with pytest.raises(SequenceDataError) as caught:
        PortfolioOptimizer().solve_sequence(
            [sample_lp_problem, second],
            holding_period_returns={second.data.date: pd.Series(dtype=float)},
            failure_dump_dir=directory,
        )
    error = caught.value
    assert error.dump_path.is_file()
    with gzip.open(error.dump_path, "rt") as stream:
        payload = json.load(stream)
    assert payload["kind"] == "optim.sequence_failure"
    assert payload["format_version"] == 1
    assert len(payload["completed_steps"]) == 1
    assert payload["missing_returns"]
    assert payload["holdings"]
    assert "risk_model" not in payload
    manual = tmp_path / "manual.json"
    error.dump(manual)
    assert json.loads(manual.read_text()) == payload
    with pytest.raises(FileExistsError):
        error.dump(manual)


def test_success_does_not_create_dump_directory(sample_lp_problem, tmp_path):
    directory = tmp_path / "unused"
    PortfolioOptimizer().solve_sequence([sample_lp_problem], failure_dump_dir=directory)
    assert not directory.exists()


def test_dump_failure_does_not_mask_original_error(
    sample_lp_problem, tmp_path, monkeypatch
):
    second = _second(sample_lp_problem)

    def fail_dump(*args, **kwargs):
        raise PermissionError("test write failure")

    monkeypatch.setattr(SequenceDataError, "dump", fail_dump)
    with pytest.raises(SequenceDataError) as caught:
        PortfolioOptimizer().solve_sequence(
            [sample_lp_problem, second],
            holding_period_returns={second.data.date: pd.Series(dtype=float)},
            failure_dump_dir=tmp_path / "failures",
        )
    assert caught.value.dump_error is not None
    assert caught.value.evidence is not None


def test_missing_mass_uses_absolute_holdings_and_ignores_zero_positions():
    held = pd.Series({"long": 0.2, "short": -0.2, "cashlike": 1.0, "zero": 0.0})
    with pytest.raises(SequenceDataError) as caught:
        _mark_to_market(
            held, pd.Series({"cashlike": 0.0}), held.index, SequencePolicy()
        )
    assert "40.000000%" in str(caught.value)
    assert set(caught.value.evidence.index) == {"long", "short"}

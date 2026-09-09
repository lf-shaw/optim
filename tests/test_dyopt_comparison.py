"""独立验算口径的回归；无需供应商软件或许可证。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "dyopt_comparison", Path(__file__).parents[1] / "benchmarks/dyopt_comparison.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def sample():
    return dict(
        benchmark=np.array([0.5, 0.5]),
        initial=np.array([0.5, 0.5]),
        alpha=np.array([1.0, 2.0]),
        tradable=np.array([True, True]),
        exposure=np.ones((2, 1)),
        covariance=np.array([[0.04]]),
        specific_volatility=np.array([0.2, 0.2]),
        factors=np.array(["country"]),
        types=np.array(["country"]),
    )


def test_annualized_risk_and_two_sided_turnover():
    d = sample()
    r = benchmark.validate(d, np.array([0.51, 0.49]), "original")
    assert r["te"] == pytest.approx(np.sqrt(2) * 0.002)
    assert r["turnover_l1"] == pytest.approx(0.02)
    assert r["max_violation"] < 1e-12


def test_freeze_checked_separately():
    d = sample()
    d["tradable"][0] = False
    r = benchmark.validate(d, np.array([0.51, 0.49]), "lp")
    assert r["violations"]["frozen"] == pytest.approx(0.01)


def test_risk_budget_not_applied_to_lp():
    d = sample()
    w = np.array([0.7, 0.3])
    lp = benchmark.validate(d, w, "lp")
    socp = benchmark.validate(d, w, "te2")
    assert "tracking_error" not in lp["violations"]
    assert socp["violations"]["tracking_error"] > 0

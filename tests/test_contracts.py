from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    FactorRiskModel,
    MaximizeAlpha,
    PortfolioData,
    PortfolioProblem,
    PortfolioValidationError,
)
from optim.validation import validate_problem


def test_valid_problem_has_no_validation_issues(sample_lp_problem):
    report = validate_problem(sample_lp_problem)
    assert report.is_valid
    assert report.errors == ()


def test_validation_aggregates_independent_input_errors(sample_lp_problem):
    data = sample_lp_problem.data
    malformed_risk = replace(
        data.risk_model,
        asof=pd.Timestamp("2026-01-01"),
        specific_volatility=np.array([0.1, -0.2, 0.1, 0.1]),
    )
    malformed = replace(
        data,
        assets=pd.Index(["a", "a", "c", "d"], name="sid"),
        alpha=np.array([np.nan, 0.5, -0.2, 0.1]),
        benchmark=np.array([0.2, 0.2, 0.2, 0.2]),
        risk_model=malformed_risk,
    )
    report = validate_problem(replace(sample_lp_problem, data=malformed))
    codes = {issue.code for issue in report.errors}
    assert {"duplicate", "non_finite", "not_normalized", "date_mismatch", "negative"} <= codes
    with pytest.raises(PortfolioValidationError) as caught:
        report.raise_for_errors()
    assert caught.value.report is report


def test_alpha_objective_requires_explicit_units(sample_lp_problem):
    problem = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha_spec=None),
    )
    report = validate_problem(problem)
    assert any(issue.field == "data.alpha_spec" for issue in report.errors)


def test_portfolio_data_accepts_none_alpha_for_non_alpha_objective(sample_data):
    data = replace(sample_data, alpha=None, alpha_spec=None)
    assert data.alpha is None


def test_problem_data_objects_do_not_import_solver_or_data_source_packages():
    # The side-effect property is also exercised in a clean interpreter by
    # test_import.py; this assertion documents the public contract here.
    assert PortfolioData is not None
    assert FactorRiskModel is not None
    assert AlphaSpec is not None
    assert MaximizeAlpha is not None
    assert PortfolioProblem is not None

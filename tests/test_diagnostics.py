from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    SolveStatus,
    TurnoverLimit,
    WeightBounds,
)


def test_deep_diagnostic_reports_minimum_required_turnover():
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=pd.Index(["old", "new"], name="sid"),
        alpha=np.array([0.0, 1.0]),
        benchmark=np.array([0.0, 1.0]),
        initial_weight=np.array([1.0, 0.0]),
        tradable=np.ones(2, dtype=bool),
        alpha_spec=AlphaSpec(),
    )
    constraints = PortfolioConstraints(
        asset_weight=WeightBounds(
            lower=np.array([0.0, 1.0]),
            upper=np.array([0.0, 1.0]),
        ),
        turnover=TurnoverLimit(0.10),
    )
    problem = PortfolioProblem(data, MaximizeAlpha(), constraints)
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(problem)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.diagnostics is None

    report = optimizer.diagnose(problem, prior_result=result, level="deep")
    assert report.linear_feasible is False
    assert np.isclose(report.turnover_linear_lower_bound, 2.0, atol=1e-9)
    assert report.turnover_limit == 0.10
    assert "最小双边换手率" in report.summary_text
    assert report.relaxations


def test_failure_returns_result_and_require_weights_raises():
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=pd.Index(["a", "b"], name="sid"),
        alpha=np.array([1.0, 0.0]),
        benchmark=np.array([1.0, 0.0]),
        initial_weight=np.array([1.0, 0.0]),
        tradable=np.ones(2, dtype=bool),
        alpha_spec=AlphaSpec(),
    )
    problem = PortfolioProblem(
        data,
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(lower=np.array([0.0, 0.5]), upper=1.0),
            total_active=0.1,
        ),
    )
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(problem)
    assert result.status is SolveStatus.INFEASIBLE
    with pytest.raises(RuntimeError, match="did not produce usable weights"):
        result.require_weights()


def test_diagnostic_rejects_prior_result_for_different_problem(sample_lp_problem):
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(sample_lp_problem)
    result = replace(result, status=SolveStatus.NUMERICAL_ERROR, weights=None)
    changed = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=sample_lp_problem.data.alpha + 0.1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.diagnose(changed, prior_result=result)


def test_single_result_keeps_problem_without_automatic_diagnosis(
    sample_lp_problem, monkeypatch
):
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(sample_lp_problem)
    assert result.problem is sample_lp_problem
    assert result.diagnostics is None
    with pytest.raises(ValueError, match="successful"):
        optimizer.diagnose(result)
    # 单独传入问题时不追踪求解历史，仍允许显式诊断。
    report = optimizer.diagnose(sample_lp_problem)
    assert report.linear_feasible is True
    # 无风险预算的 LP 不需要额外执行最小风险 QP。
    assert report.minimum_tracking_error is None
    assert all(attempt.backend == "highs" for attempt in report.attempts)


@pytest.mark.parametrize("failure", ["limit_reached", "numerical_error", "infeasible"])
def test_failed_phase_one_does_not_claim_infeasibility(
    sample_lp_problem, monkeypatch, failure
):
    from optim import _diagnostic_engine as engine
    from optim._core import CoreBackendResult, CoreSolveStatus

    failed = CoreBackendResult("highs", CoreSolveStatus(failure), None, None, failure)
    monkeypatch.setattr(engine, "_solve_phase_one", lambda *args: (failed, ()))
    report = PortfolioOptimizer().diagnose(sample_lp_problem)
    assert report.linear_feasible is None
    assert "尚未确定" in report.summary_text


def test_full_candidate_checks_are_independent_of_display_slacks(sample_lp_problem):
    from optim._diagnostic_engine import _linear_feasibility
    from optim._core import CoreBackendResult, CoreSolveStatus
    from optim.model import compile_problem

    domain = compile_problem(sample_lp_problem).model.domain
    invalid = CoreBackendResult(
        "highs", CoreSolveStatus.OPTIMAL, np.zeros(domain.n_variables), 0.0, "optimal"
    )
    assert _linear_feasibility(invalid, domain, PortfolioOptimizer().policy) is None


def test_high_risk_candidate_alone_cannot_prove_infeasible():
    from optim._diagnostic_engine import _summary

    summary = _summary(True, (), None, None, 0.03, 0.02)
    assert "不能单凭它确认" in summary
    assert "至少需增加" not in summary


def test_optional_evidence_reader_failure_is_isolated():
    from optim._core.backends.base import capture_infeasibility

    diagnostics = {}

    def broken():
        raise RuntimeError("native evidence unavailable")

    assert capture_infeasibility(broken, diagnostics) is None
    assert diagnostics["infeasibility_evidence_error"] == "RuntimeError"


@pytest.mark.parametrize("backend", ["piqp", "clarabel", "mosek"])
def test_native_qp_evidence_maps_conflicting_rows(
    sample_data, sample_constraints, backend
):
    import scipy.sparse as sp
    from optim import RiskAdjustedAlpha
    from optim._core import ConstraintRecord, CoreSolveStatus
    from optim._core.backends.base import BackendOptions
    from optim._core.backends.piqp import PIQPBackend
    from optim._core.backends.clarabel import ClarabelBackend
    from optim._core.backends.mosek import MosekBackend
    from optim.model import compile_problem

    model = compile_problem(
        PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    ).model
    n = model.domain.n_variables
    domain = replace(
        model.domain,
        A=sp.csc_matrix(([1.0, 1.0], ([0, 1], [0, 0])), shape=(2, n)),
        lower=np.array([0.6, -np.inf]),
        upper=np.array([np.inf, 0.4]),
        variable_lower=np.full(n, -np.inf),
        variable_upper=np.full(n, np.inf),
        constraints=(
            ConstraintRecord("min", "test", "row", 0),
            ConstraintRecord("max", "test", "row", 1),
        ),
    )
    result = {"piqp": PIQPBackend, "clarabel": ClarabelBackend, "mosek": MosekBackend}[
        backend
    ]().solve(replace(model, domain=domain), BackendOptions())
    if backend == "piqp" and result.status is CoreSolveStatus.LIMIT_REACHED:
        # 有些不可行模型只能得到 PIQP 迭代上限；不能从末次 dual 伪造不可行证书。
        assert result.infeasibility is None
        return
    if backend == "mosek" and result.status is CoreSolveStatus.SOLVER_ERROR:
        from optim._core import CoreFailureReason

        if result.reason is CoreFailureReason.BACKEND_UNAVAILABLE:
            pytest.skip("MOSEK license unavailable")
    assert result.status is CoreSolveStatus.INFEASIBLE
    assert result.infeasibility is not None, result.diagnostics
    assert result.infeasibility.proof_status == "numerical_estimate"
    rows = {
        (entry.index, entry.side)
        for entry in result.infeasibility.contributors
        if entry.location == "row"
    }
    assert (0, "lower") in rows
    assert (1, "upper") in rows


def test_highs_does_not_generate_a_missing_ray():
    import highspy
    from optim._core.backends.highs import _dual_ray

    class PresolvedInfeasible:
        def getDualRayExist(self):
            return highspy.HighsStatus.kOk, False

        def getDualRay(self):
            raise AssertionError("must not trigger extra LP solve")

    assert _dual_ray(PresolvedInfeasible(), None, highspy, "infeasible") is None


def test_highs_existing_ray_maps_both_conflicting_sides(sample_lp_problem):
    import highspy
    import scipy.sparse as sp
    from optim.model import compile_problem
    from optim._core.backends.highs import _dual_ray

    # 仅测试关闭 presolve 以稳定获得原生 ray；生产不改变求解选项以强求证书。
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("presolve", "off")
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 1, 2
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = [0.0], [-np.inf], [np.inf]
    lp.row_lower_, lp.row_upper_ = [0.6, -np.inf], [np.inf, 0.4]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [0, 2]
    lp.a_matrix_.index_ = [0, 1]
    lp.a_matrix_.value_ = [1.0, 1.0]
    solver.passModel(lp)
    solver.run()
    model = compile_problem(sample_lp_problem).model
    domain = replace(model.domain, A=sp.csc_matrix([[1.0], [1.0]]))
    evidence = _dual_ray(solver, replace(model, domain=domain), highspy, "infeasible")
    assert evidence is not None
    rows = {
        (entry.index, entry.side)
        for entry in evidence.contributors
        if entry.location == "row"
    }
    assert rows == {(0, "lower"), (1, "upper")}


@pytest.mark.parametrize("backend", ["clarabel", "mosek"])
def test_native_factor_qcqp_keeps_risk_cone_components(sample_data, backend):
    from optim import TrackingErrorLimit
    from optim._core import CoreSolveStatus
    from optim._core.backends.base import BackendOptions
    from optim._core.backends.clarabel import ClarabelBackend
    from optim._core.backends.mosek import MosekBackend
    from optim.model import compile_problem
    from optim._solver_adapter import _native_infeasibility

    weight = np.array([1.0, 0.0, 0.0, 0.0])
    problem = PortfolioProblem(
        sample_data,
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(weight, weight),
            tracking_error=TrackingErrorLimit(0.001),
        ),
    )
    compiled = compile_problem(problem)
    result = (ClarabelBackend() if backend == "clarabel" else MosekBackend()).solve(
        compiled.model, BackendOptions()
    )
    assert result.status is CoreSolveStatus.INFEASIBLE, result
    evidence = _native_infeasibility(result, compiled)
    assert evidence is not None, result.diagnostics
    assert evidence.fingerprint == compiled.fingerprint
    assert evidence.scope == "original"
    cone = [e for e in evidence.contributors if e.location == "cone"]
    assert len(cone) > 1
    assert all(e.constraint_id == "tracking_error" for e in cone)
    assert any(e.multiplier < 0 for e in cone)


def test_piqp_compact_dual_mapping_preserves_equal_sign(
    sample_data, sample_constraints
):
    from types import SimpleNamespace
    from optim import RiskAdjustedAlpha
    from optim._core.backends.piqp import _infeasibility
    from optim.model import compile_problem

    model = compile_problem(
        PortfolioProblem(sample_data, RiskAdjustedAlpha(), sample_constraints)
    ).model
    d = model.domain
    equal = (
        np.isfinite(d.lower)
        & np.isfinite(d.upper)
        & np.isclose(d.lower, d.upper, rtol=0, atol=1e-14)
    )
    native = SimpleNamespace(
        y=-np.ones(equal.sum()),
        z_l=np.ones((~equal).sum()),
        z_u=np.ones((~equal).sum()),
        z_bl=np.zeros(d.n_variables),
        z_bu=np.zeros(d.n_variables),
    )
    evidence = _infeasibility(model, native, "compact", "PIQP_PRIMAL_INFEASIBLE")
    assert any(e.side == "equal" and e.multiplier == -1 for e in evidence.contributors)


def test_sequence_only_keeps_stopped_problem(sample_lp_problem):
    from optim import SequencePolicy

    optimizer = PortfolioOptimizer()
    good = optimizer.solve_sequence([sample_lp_problem])
    assert good.stopped_problem is None
    assert good.steps[0].result.problem is None
    bad = replace(
        sample_lp_problem,
        constraints=replace(
            sample_lp_problem.constraints,
            asset_weight=WeightBounds(lower=np.array([0.6, 0.0, 0.0, 0.0]), upper=1.0),
            active_weight=None,
            turnover=TurnoverLimit(0.0),
        ),
    )
    stopped = optimizer.solve_sequence([bad])
    assert stopped.stopped_problem is not None
    assert stopped.steps[0].result.problem is None
    report = optimizer.diagnose(
        stopped.stopped_problem, prior_result=stopped.steps[0].result
    )
    assert report.linear_feasible is False
    held = optimizer.solve_sequence(
        [bad], sequence_policy=SequencePolicy(on_failure="hold")
    )
    assert held.stopped_problem is None
    assert held.steps[0].result.problem is None


def test_diagnose_result_detects_mutated_input(sample_lp_problem):
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(sample_lp_problem)
    result = replace(result, status=SolveStatus.NUMERICAL_ERROR, weights=None)
    sample_lp_problem.data.alpha[0] += 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.diagnose(result)


@pytest.mark.parametrize(
    "status", [SolveStatus.OPTIMAL, SolveStatus.OPTIMAL_INACCURATE]
)
@pytest.mark.parametrize("via_prior", [False, True])
def test_successful_diagnosis_is_blocked_before_prepare(
    sample_lp_problem, monkeypatch, status, via_prior
):
    optimizer = PortfolioOptimizer()
    result = replace(optimizer.solve(sample_lp_problem), status=status)

    def forbidden(*args, **kwargs):
        raise AssertionError("successful result must be rejected before preparation")

    monkeypatch.setattr(optimizer, "prepare", forbidden)
    with pytest.raises(ValueError, match="result.metrics"):
        if via_prior:
            optimizer.diagnose(sample_lp_problem, prior_result=result)
        else:
            optimizer.diagnose(result)


def test_regular_lp_does_not_compute_diagnostic_bound(sample_lp_problem, monkeypatch):
    from optim._core.backends import highs

    def forbidden(*args):
        raise AssertionError("ordinary LP must not compute diagnostic dual bound")

    monkeypatch.setattr(highs, "_dual_lower_bound", forbidden)
    assert PortfolioOptimizer().solve(sample_lp_problem).status.has_solution


@pytest.mark.parametrize(
    "suffix,indent", [(".json", 2), (".json", None), (".json.gz", None)]
)
def test_dump_complete_report_with_descriptions(tmp_path, suffix, indent):
    import gzip
    import json
    from dataclasses import fields
    from optim import InfeasibilityReport, NativeInfeasibilityEvidence, ProofStatus

    certificate = NativeInfeasibilityEvidence(
        "highs",
        "dual_ray",
        ProofStatus.NUMERICAL_ESTIMATE,
        "infeasible",
        metadata={"signed_vector": np.array([1.0, -2.0]), "missing": np.nan},
    )
    report = InfeasibilityReport(
        "deep",
        False,
        "中文诊断摘要",
        native_certificates=(certificate,),
        native_evidence={"date": pd.Timestamp("2026-01-02"), "bound": np.float64(0.02)},
    )
    path = tmp_path / ("diagnosis" + suffix)
    assert report.dump(path, indent=indent) == path
    text = (
        gzip.open(path, "rt", encoding="utf-8").read()
        if suffix.endswith(".gz")
        else path.read_text(encoding="utf-8")
    )
    payload = json.loads(text)
    assert list(payload) == ["format_version", "field_descriptions", "report"]
    assert payload["format_version"] == 1
    assert set(payload["field_descriptions"]) == {item.name for item in fields(report)}
    assert all(payload["field_descriptions"].values())
    assert (
        "数值对偶下界" in payload["field_descriptions"]["turnover_linear_lower_bound"]
    )
    assert set(payload["report"]) == {item.name for item in fields(report)}
    dumped_certificate = payload["report"]["native_certificates"][0]
    assert dumped_certificate["metadata"]["signed_vector"] == [1.0, -2.0]
    assert dumped_certificate["metadata"]["missing"] == "NaN"
    assert dumped_certificate["proof_status"] == "numerical_estimate"
    assert payload["report"]["native_evidence"]["date"] == "2026-01-02T00:00:00"
    assert "中文诊断摘要" in text
    assert "native_certificates" not in str(report)
    assert "native_certificates" not in repr(report)
    if indent is None:
        assert text.count("\n") == 1
    else:
        assert '\n  "field_descriptions"' in text
    with pytest.raises(FileExistsError):
        report.dump(path)
    report.dump(path, overwrite=True)


def test_dump_rejects_unsupported_data_before_creating_file(tmp_path):
    from optim import InfeasibilityReport

    path = tmp_path / "bad.json"
    report = InfeasibilityReport(
        "deep", None, "", native_evidence={"unsupported": object()}
    )
    with pytest.raises(TypeError, match="unsupported report value"):
        report.dump(path)
    assert not path.exists()
    with pytest.raises(ValueError, match="indent"):
        report.dump(path, indent=-1)


def test_turnover_lower_bound_can_resolve_unknown_phase(sample_lp_problem, monkeypatch):
    from optim import _diagnostic_engine as engine
    from optim._core import CoreBackendResult, CoreSolveStatus

    unknown = CoreBackendResult(
        "highs", CoreSolveStatus.LIMIT_REACHED, None, None, "limit"
    )
    monkeypatch.setattr(engine, "_solve_phase_one", lambda *args: (unknown, ()))
    monkeypatch.setattr(
        engine, "_minimum_linear_turnover", lambda *args: (unknown, 1.62)
    )
    problem = replace(
        sample_lp_problem,
        constraints=replace(
            sample_lp_problem.constraints, turnover=TurnoverLimit(0.05)
        ),
    )
    report = PortfolioOptimizer().diagnose(problem)
    assert report.linear_feasible is False
    assert report.native_evidence["phase_one_linear_feasible"] is None
    assert report.native_evidence["conclusion_basis"] == "numerical_estimate"
    assert "尚未确定" not in report.summary_text
    assert "157.0000 个百分点" in report.summary_text


def test_phase_turnover_summary_explains_units_and_other_slacks():
    from optim import RequiredRelaxation
    from optim._diagnostic_engine import _summary

    relaxations = (
        RequiredRelaxation("turnover:l1", "turnover", "upper", 0.15, 0.05, 1.0),
        RequiredRelaxation("asset:a", "asset_bound", "lower", 0.20, 0.30, 1.0),
    )
    summary = _summary(False, relaxations, 1.62, 0.05, None, None)
    assert "15.0000 个百分点" in summary
    assert "从 5.0000% 变为 20.0000%" in summary
    assert "162.0000%" in summary
    assert "同时松弛了 asset_bound" in summary


def test_contradictory_bound_is_quarantined_using_full_candidate(
    sample_lp_problem, monkeypatch
):
    from optim import _diagnostic_engine as engine
    from optim._core import CoreBackendResult, CoreSolveStatus
    from optim.model import compile_problem
    from optim.solution import lift_weights

    problem = replace(
        sample_lp_problem,
        constraints=replace(
            sample_lp_problem.constraints, turnover=TurnoverLimit(0.05)
        ),
    )
    compiled = compile_problem(problem)
    # 候选实际换手率 20%，其他约束保持可行。它直接反驳最小换手率下界 162%。
    weight = problem.data.initial_weight.copy()
    weight[0] += 0.1
    weight[1] -= 0.1
    vector = lift_weights(problem, compiled, weight)
    phase = CoreBackendResult("highs", CoreSolveStatus.OPTIMAL, vector, 0.15, "optimal")
    # 展示列表故意为空，保证冲突判断不依赖被过滤的松弛记录。
    monkeypatch.setattr(engine, "_solve_phase_one", lambda *args: (phase, ()))
    monkeypatch.setattr(engine, "_minimum_linear_turnover", lambda *args: (phase, 1.62))
    report = PortfolioOptimizer().diagnose(problem)
    assert report.linear_feasible is None
    assert report.turnover_linear_lower_bound is None
    assert report.native_evidence["reported_turnover_lower_bound"] == 1.62
    assert report.native_evidence["conclusion_basis"] == "conflicting_evidence"
    assert "20.0000%" in report.summary_text
    assert "暂不采信" in report.summary_text
    assert "至少需增加" not in report.summary_text


def test_joint_relaxation_does_not_falsely_conflict_with_turnover_bound(
    sample_lp_problem,
):
    from optim._diagnostic_engine import _combine_linear_evidence
    from optim._core import CoreBackendResult, CoreSolveStatus
    from optim.model import compile_problem

    compiled = compile_problem(sample_lp_problem)
    # 零持仓违反预算，不能作为“保持其余约束”的换手率可行上界。
    phase = CoreBackendResult(
        "highs",
        CoreSolveStatus.OPTIMAL,
        np.zeros(compiled.model.domain.n_variables),
        1.0,
        "optimal",
    )
    conclusion, conflict = _combine_linear_evidence(
        None, phase, compiled.model.domain, sample_lp_problem, 1.62, 1e-5
    )
    assert conclusion is False
    assert conflict is None

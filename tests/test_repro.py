"""问题派生和跨进程复现的回归检查。"""

from zipfile import ZipFile
import hashlib
import json

import numpy as np
import pytest

from optim import (
    AssetTradeConstraints,
    PortfolioOptimizer,
    SolverPolicy,
    RiskAdjustedAlpha,
    TrackingErrorLimit,
    TurnoverLimit,
    export_repro,
    load_repro,
)
from optim.model.compiler import compile_problem


def test_problem_derivation(sample_lp_problem):
    original = sample_lp_problem
    changed = original.with_constraints(
        turnover=None, tracking_error=TrackingErrorLimit(0.06)
    )
    assert changed.data is original.data
    assert changed.objective is original.objective
    assert original.constraints.turnover is not None
    assert changed.constraints.turnover is None
    with pytest.raises(TypeError):
        original.with_constraints(unknown=1)
    with pytest.raises(TypeError):
        original.with_constraints(budget=None)
    assert original.with_objective(RiskAdjustedAlpha()).data is original.data
    assert original.with_data(alpha=None).data.alpha is None
    assert original.data.alpha is not None


@pytest.mark.parametrize("kind", ["lp", "qp", "qcqp"])
def test_repro_roundtrip(tmp_path, sample_lp_problem, kind):
    problem = sample_lp_problem
    if kind == "qp":
        problem = problem.with_objective(RiskAdjustedAlpha())
    elif kind == "qcqp":
        problem = problem.with_constraints(tracking_error=TrackingErrorLimit(0.06))
    result = PortfolioOptimizer().solve(problem)
    path = export_repro(tmp_path / "case.zip", result=result)
    case = load_repro(path)
    assert not hasattr(case, "theta_seed")
    assert case.policy == result.solver_policy
    assert compile_problem(case.problem).fingerprint == result.fingerprint
    assert not case.problem.data.risk_model.exposure.flags.writeable
    assert case.problem.data.assets.equals(problem.data.assets)
    replay = case.solve()
    assert replay.status == result.status
    assert replay.objective_value == pytest.approx(result.objective_value, abs=1e-5)
    with pytest.raises(FileExistsError):
        export_repro(path, result=result)
    with pytest.raises(ValueError, match="size limit"):
        load_repro(path, max_uncompressed_bytes=1)
    with pytest.raises(ValueError, match="fingerprint"):
        export_repro(
            tmp_path / "wrong.zip",
            result=result,
            problem=problem.with_data(alpha=problem.data.alpha + 1),
        )


def test_failure_repro_and_manual_counterfactual(tmp_path, sample_lp_problem):
    problem = sample_lp_problem.with_constraints(turnover=TurnoverLimit(0.0))
    problem = problem.with_data(initial_weight=np.array([1.0, 0.0, 0.0, 0.0]))
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(problem)
    assert not result.status.has_solution
    report = optimizer.diagnose(result)
    case = load_repro(
        export_repro(tmp_path / "failure.zip", result=result, report=report)
    )
    assert case.original_report["summary_text"] == report.summary_text
    assert not case.solve().status.has_solution
    assert optimizer.solve(
        case.problem.with_constraints(turnover=None)
    ).status.has_solution


def test_repro_checksum(tmp_path, sample_lp_problem):
    path = export_repro(
        tmp_path / "case.zip", result=PortfolioOptimizer().solve(sample_lp_problem)
    )
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["payload.json"] += b" "
    with ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="checksum"):
        load_repro(path)


def test_repro_policy_and_asset_instructions(tmp_path, sample_lp_problem):
    problem = sample_lp_problem.with_constraints(
        asset_trade=AssetTradeConstraints(weight_overrides={"a": (0.1, 0.4)})
    )
    policy = SolverPolicy(lp_prescreen=True)
    result = PortfolioOptimizer(policy).solve(problem)
    case = load_repro(export_repro(tmp_path / "policy.zip", result=result))
    assert case.policy == policy
    assert case.problem.constraints.asset_trade.weight_overrides["a"] == (0.1, 0.4)
    with pytest.raises(ValueError, match="policy differs"):
        export_repro(tmp_path / "wrong.zip", result=result, policy=SolverPolicy())
    # 源数组被就地修改后，不允许把新输入冒充为原始求解输入导出。
    problem.data.alpha[0] += 0.01
    with pytest.raises(ValueError, match="fingerprint"):
        export_repro(tmp_path / "mutated.zip", result=result)


@pytest.mark.parametrize("version", [1, 999, None])
def test_repro_rejects_unsupported_version(tmp_path, sample_lp_problem, version):
    """校验和有效时仍拒绝旧/未知版本，不静默丢弃旧搜索参数。"""
    path = export_repro(
        tmp_path / "version.zip", result=PortfolioOptimizer().solve(sample_lp_problem)
    )
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    payload = json.loads(members["payload.json"])
    assert payload["format_version"] == 2
    payload["format_version"] = version
    members["payload.json"] = json.dumps(payload).encode()
    manifest = json.loads(members["manifest.json"])
    manifest["payload.json"] = hashlib.sha256(members["payload.json"]).hexdigest()
    members["manifest.json"] = json.dumps(manifest).encode()
    with ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="unsupported repro format version"):
        load_repro(path)

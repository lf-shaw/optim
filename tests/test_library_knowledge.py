from __future__ import annotations

import ast
from dataclasses import replace
import importlib
import importlib.util
import inspect
from pathlib import Path
import re
import tomllib
from zipfile import ZipFile

import numpy as np
import pytest

from optim import (
    PortfolioOptimizer,
    PortfolioValidationError,
    RiskAdjustedAlpha,
    SolverPolicy,
    SolverTuning,
    TrackingErrorLimit,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "optim"


def test_manifest_declares_only_public_namespaces():
    manifest = tomllib.loads((PACKAGE / "LIBRARY.toml").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["meta"]["name"] == "optim"
    assert "version" not in manifest["meta"]
    assert set(manifest) == {"schema_version", "meta", "api"}
    for name in manifest["api"]["modules"]:
        assert not any(part.startswith("_") for part in name.split(".")[1:])
        module = importlib.import_module(name)
        assert inspect.ismodule(module)
        for exported in getattr(module, "__all__", ()):
            owner = getattr(getattr(module, exported), "__module__", "")
            assert not owner.startswith(("optim._core", "optim._impl"))


def test_documented_python_examples_have_valid_syntax():
    for path in (PACKAGE / "references").glob("*.md"):
        body = path.read_text()
        assert "<!-- knowledge:" not in body
        for code in re.findall(r"```python\n(.*?)\n```", body, re.DOTALL):
            ast.parse(code)


def test_standalone_optimization_recipe():
    recipe = (PACKAGE / "references/recipes.md").read_text().split("## R1.")[0]
    code = re.search(r"```python\n(.*?)\n```", recipe, re.DOTALL).group(1)
    namespace = {}
    exec(compile(code, "standalone-optimization", "exec"), namespace)
    result = namespace["result"]
    weights = namespace["weights"]
    assert result.backend == "highs"
    assert weights.index.equals(namespace["assets"])
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= -1e-5
    assert weights.max() <= 0.60 + 1e-5
    with pytest.raises(RuntimeError, match="usable weights"):
        replace(result, weights=None).require_weights()


@pytest.mark.parametrize("route", ["lp", "qp", "factor_qcqp"])
def test_documented_solver_families(sample_lp_problem, route):
    problem = sample_lp_problem
    expected_backend = "highs"
    if route == "qp":
        problem = replace(problem, objective=RiskAdjustedAlpha())
        expected_backend = "piqp"
    elif route == "factor_qcqp":
        problem = replace(
            problem,
            constraints=replace(
                problem.constraints, tracking_error=TrackingErrorLimit(0.20)
            ),
        )
        expected_backend = "clarabel_qdldl"
    result = PortfolioOptimizer(SolverPolicy(tuning=SolverTuning(threads=1))).solve(
        problem
    )
    assert result.backend == expected_backend
    weights = result.require_weights()
    assert weights.index.equals(problem.data.assets)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)


def test_invalid_asset_alignment_is_not_a_success(sample_lp_problem):
    problem = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=np.array([1.0, 2.0])),
    )
    with pytest.raises(PortfolioValidationError):
        PortfolioOptimizer().solve(problem)


@pytest.mark.parametrize("internal", ["_core", "_impl"])
def test_wheel_validator_rejects_internal_knowledge_scopes(tmp_path, internal):
    spec = importlib.util.spec_from_file_location(
        "optim_wheel_validator", ROOT / "scripts/validate_wheel.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    wheel = tmp_path / "optim-test.whl"
    with ZipFile(wheel, "w") as archive:
        for name in validator.REQUIRED_RESOURCES:
            if name != "optim/LIBRARY.toml":
                archive.writestr(name, "placeholder")
        archive.writestr(
            "optim/LIBRARY.toml",
            'schema_version = 2\n[meta]\nname = "optim"\n'
            f'[api]\nmodules = ["optim.{internal}"]\n',
        )
        archive.writestr("optim-0.dist-info/METADATA", "Name: optim\nVersion: 0\n")
    with pytest.raises(SystemExit, match="私有 namespace"):
        validator.validate(wheel)

from __future__ import annotations

import subprocess
import sys

import importlib
import importlib.util
import pytest


def test_import_optim_is_solver_and_data_source_side_effect_free():
    code = """
import sys
import optim
blocked = {'mosek', 'highspy', 'piqp', 'clarabel', 'tuda2', 'carry'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(','.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.parametrize("name", ["opt", "linopt", "solver"])
def test_removed_legacy_modules_cannot_be_imported(name):
    """旧模块既不保留源码，也不通过包属性提供兼容入口。"""
    import optim

    assert importlib.util.find_spec(f"optim.{name}") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"optim.{name}")
    with pytest.raises(AttributeError):
        getattr(optim, name)

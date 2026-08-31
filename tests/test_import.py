from __future__ import annotations

import subprocess
import sys


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

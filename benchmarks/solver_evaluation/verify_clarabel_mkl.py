#!/usr/bin/env python3
"""Fail-fast verification for the custom Clarabel Pardiso-MKL wheel."""

from __future__ import annotations

import contextlib
import io
import os

import clarabel
import numpy as np
from scipy import sparse


def main() -> int:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        clarabel.buildinfo()
    build_info = capture.getvalue()
    if "pardiso_mkl" not in build_info:
        raise RuntimeError("Clarabel wheel was not built with pardiso_mkl")

    threads = int(os.environ.get("MKL_NUM_THREADS", "1"))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.direct_solve_method = "mkl"
    settings.max_threads = threads

    p = sparse.csc_matrix(2.0 * np.eye(2))
    q = np.array([-2.0, -4.0])
    a = sparse.csc_matrix(-np.eye(2))
    b = np.zeros(2)
    solver = clarabel.DefaultSolver(
        p, q, a, b, [clarabel.NonnegativeConeT(2)], settings
    )
    solution = solver.solve()
    info = solver.get_info()
    if str(solution.status) != "Solved":
        raise RuntimeError(f"Pardiso-MKL smoke solve failed: {solution.status}")
    if info.linsolver.name != "mkl":
        raise RuntimeError(f"unexpected linear solver: {info.linsolver.name}")
    if not np.allclose(solution.x, [1.0, 2.0], atol=1e-6):
        raise RuntimeError(f"unexpected solution: {solution.x}")

    print(
        "Clarabel Pardiso-MKL verified: "
        f"clarabel={clarabel.__version__}, threads={info.linsolver.threads}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

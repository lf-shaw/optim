#!/usr/bin/env python3
"""Public minimal reproducer for PIQP 0.6.3 dual-index recovery UB."""

from __future__ import annotations

from importlib import metadata

import numpy as np
import piqp
from scipy import sparse


def main() -> int:
    # $\min_x\;\frac12x^2-x$, subject to $-1\le x$, $x\le2$, $x\le3$.
    # The finite lower-index list ends at row 0, while finite upper bounds
    # continue through rows 1 and 2.
    P = sparse.csc_matrix([[1.0]])
    c = np.array([-1.0])
    G = sparse.csc_matrix(np.ones((3, 1)))
    h_l = np.array([-1.0, -np.inf, -np.inf])
    h_u = np.array([np.inf, 2.0, 3.0])

    solver = piqp.SparseSolver()
    solver.setup(P, c, None, None, G, h_l, h_u)
    status = solver.solve()

    np.testing.assert_allclose(solver.result.x, [1.0], atol=1e-7)
    np.testing.assert_allclose(solver.result.z_l, 0.0, atol=1e-7)
    np.testing.assert_allclose(solver.result.z_u, 0.0, atol=1e-7)
    print(f"piqp={metadata.version('piqp')} status={status} x={solver.result.x[0]:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

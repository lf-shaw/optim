# PIQP 0.6.3 dual-index patched wheel

- Upstream repository: `PREDICT-EPFL/piqp`
- Upstream tag/commit: `v0.6.3` / `8279ed73a42ddc55e3a8be561dfab7be9b3146b8`
- Local distribution version: `0.6.3+optim.dualidx1`
- Python/platform tag: `cp311-cp311-linux_x86_64`
- Wheel SHA-256: `bd65aa059441b3131d3e5bdedf5552d8afed76900e588b94821360969e2f2cde`
- Build date: `2026-08-28`
- Compiler: GCC 13.3.1, release build with PIQP AVX2/AVX512 dispatch

The source change is in `include/piqp/kkt_system.tpp`. During inequality-dual
recovery, PIQP 0.6.3 increments `i_l`/`i_u` and dereferences the index vector
before checking whether the increment reached `n_h_l`/`n_h_u`. The patch moves
the bounds check before every index-vector access.

Validation performed before packaging:

- public 1-variable reproducer: official wheel gives Valgrind `22 errors from
  6 contexts`; patched wheel gives `0 errors`;
- private frozen 5,760-variable QP: official wheel gives 6 Valgrind errors;
  patched wheel gives 0;
- native parameterized C++ regression: 4 available KKT backends pass; the
  BLASFEO case is skipped because BLASFEO is not installed;
- same-process frozen-QP stress: 1,000/1,000 solved, no retry or fallback;
- continuation stress: 200 workspaces and 800/800 QPs solved;
- full local v5 chained/cold QP and 6% SOCP: all 139 non-skipped daily problems
  accepted by direct PIQP, no fallback;
- risk-active v5 2% SOCP with 1% active-weight bounds: all 69 non-skipped daily
  problems accepted, no fallback, maximum independently recomputed TE below 2%.

The wheel was a local convenience artifact for the tested x86-64 CPython 3.11
production environment; it is intentionally not tracked or distributed now
that PIQP 0.6.4 contains the upstream fix. The historical source patch and
`repro_piqp_063_dual_index.py` remain available for auditing the diagnosis.

# Clarabel Pardiso-MKL wheel build information

- Clarabel version/tag: `v0.11.1`
- Source commit: `25540f559592068d0c8a80e46ded1b21760212a1`
- Cargo features: `python,pardiso-mkl`
- Rust: `1.98.0`
- maturin: `1.15.0`
- Runtime oneMKL: `2026.1.0`
- Wheel tag: `cp39-abi3-manylinux_2_28_x86_64` (CPython 3.9+, x86-64, glibc 2.28+)
- SHA-256: `475b6569882791145958e0a5f487ee8a21081fae2727e144a833161630adc9da`

The wheel dynamically loads `libmkl_rt.so`; oneMKL is intentionally not embedded in the
wheel. Install `mkl=2026.1.0` into the target mamba environment and expose that
environment's `lib` directory through `LD_LIBRARY_PATH` when running the MKL cases.

Clarabel 0.11.1 applies its `max_threads` setting after PARDISO symbolic analysis. The
analysis, numerical factorization, and solve phases must use the same thread count for
this workload. Therefore every MKL thread case is launched in a separate process with
matching `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and Clarabel `max_threads`. Do not combine
different `CLARABEL_SCALED_MKL_T*` cases in one benchmark process.

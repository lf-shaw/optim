#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [ENV_PREFIX]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
env_prefix=${1:-"$script_dir/.venv_mkl"}
wheel="$script_dir/wheels/clarabel-0.11.1-cp39-abi3-manylinux_2_28_x86_64.whl"
expected_sha256=475b6569882791145958e0a5f487ee8a21081fae2727e144a833161630adc9da

if [ ! -f "$wheel" ]; then
    echo "Missing custom wheel: $wheel" >&2
    exit 1
fi
actual_sha256=$(sha256sum "$wheel")
actual_sha256=${actual_sha256%% *}
if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "Custom wheel checksum mismatch" >&2
    exit 1
fi

mamba_bin=${MAMBA_EXE:-}
if [ -z "$mamba_bin" ]; then
    if command -v mamba >/dev/null 2>&1; then
        mamba_bin=$(command -v mamba)
    elif command -v micromamba >/dev/null 2>&1; then
        mamba_bin=$(command -v micromamba)
    else
        echo "mamba or micromamba is required" >&2
        exit 1
    fi
fi

if [ -x "$env_prefix/bin/python" ]; then
    "$mamba_bin" install -y -p "$env_prefix" -c conda-forge mkl=2026.1.0
else
    "$mamba_bin" create -y -p "$env_prefix" -c conda-forge \
        python=3.11 pip mkl=2026.1.0
fi

"$env_prefix/bin/python" -m pip install -r "$script_dir/requirements_mkl.txt"
"$env_prefix/bin/python" -m pip install --force-reinstall --no-deps "$wheel"

env \
    LD_LIBRARY_PATH="$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MKL_THREADING_LAYER=SEQUENTIAL \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    "$env_prefix/bin/python" "$script_dir/verify_clarabel_mkl.py"

echo "Environment ready: $env_prefix"
echo "Use: BENCH_PYTHON_BIN=$env_prefix/bin/python ./run_production_mkl.sh ..."

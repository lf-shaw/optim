#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [WHEEL_OUTPUT_DIR]" >&2
    exit 2
fi

output_dir=${1:-dist_mkl}
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)

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

build_root=$(mktemp -d "${TMPDIR:-/tmp}/clarabel-mkl-build.XXXXXX")
build_env="$build_root/env"
source_dir="$build_root/Clarabel.rs"

"$mamba_bin" create -y -p "$build_env" -c conda-forge \
    python=3.11 rust=1.98 maturin=1.15 mkl=2026.1.0 mkl-devel=2026.1.0 \
    numpy=2.4.6 scipy=1.17.1 cffi pip patchelf

git clone --depth 1 --branch v0.11.1 \
    https://github.com/oxfordcontrol/Clarabel.rs.git "$source_dir"
source_commit=$(git -C "$source_dir" rev-parse HEAD)
expected_commit=25540f559592068d0c8a80e46ded1b21760212a1
if [ "$source_commit" != "$expected_commit" ]; then
    echo "Unexpected Clarabel source commit: $source_commit" >&2
    exit 1
fi

"$mamba_bin" run -p "$build_env" maturin build \
    --release \
    --features python,pardiso-mkl \
    --interpreter "$build_env/bin/python" \
    --out "$output_dir"

sha256sum "$output_dir"/clarabel-0.11.1-*.whl
echo "Build workspace retained at: $build_root"

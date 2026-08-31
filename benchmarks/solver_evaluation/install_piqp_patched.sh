#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${1:-python3}
wheel="$script_dir/wheels/piqp-0.6.3+optim.dualidx1-cp311-cp311-linux_x86_64.whl"
expected_sha256=bd65aa059441b3131d3e5bdedf5552d8afed76900e588b94821360969e2f2cde

if [[ ! -x "$python_bin" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter not found: $python_bin" >&2
    exit 2
fi
if [[ ! -f "$wheel" ]]; then
    echo "Missing patched PIQP wheel: $wheel" >&2
    exit 1
fi

actual_sha256=$(sha256sum "$wheel")
actual_sha256=${actual_sha256%% *}
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Patched PIQP wheel checksum mismatch" >&2
    exit 1
fi

if [[ "$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
    echo "The bundled PIQP wheel requires CPython 3.11" >&2
    exit 2
fi

"$python_bin" -m pip install --force-reinstall --no-deps "$wheel"
"$python_bin" -c \
    'from importlib import metadata; assert metadata.version("piqp") == "0.6.3+optim.dualidx1"; print("PIQP patch installed:", metadata.version("piqp"))'

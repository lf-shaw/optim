"""同进程交替对照指纹编码，完整入口保持相同模型、后端和校验。"""

import argparse
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

from current_range_mosek_audit import load_v5, run_range, compiler_module

import optim.fingerprint as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = types.ModuleType("optim._audit_old_fingerprint")
    baseline.__package__ = "optim"
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline], text=True
    ).strip()
    exec(subprocess.check_output(
        ["git", "show", f"{revision}:optim/fingerprint.py"], text=True
    ), baseline.__dict__)
    inputs = load_v5(35)
    runs = []
    for repeat in range(3):
        names = ("before", "after") if repeat % 2 == 0 else ("after", "before")
        for name in names:
            method = baseline.fingerprint if name == "before" else current.fingerprint

            def checked(problem, model):
                result = method(problem, model)
                return result

            with patch.object(compiler_module, "fingerprint", checked):
                result = run_range(inputs, False, "stop", None, backend="auto")
            result.update(repeat=repeat, variant=name)
            runs.append(result)
            print(name, repeat, result["public_wall_s"], flush=True)
    with args.output.open("x") as stream:
        json.dump({"baseline": revision, "runs": runs}, stream, indent=2)


if __name__ == "__main__":
    main()

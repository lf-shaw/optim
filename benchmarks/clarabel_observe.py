"""导出真实原生 Clarabel 输入，并把实验版 Rust 日志映射回业务约束类别。

仅用于源码调查；inputs 和 solutions 含原始数值数据，不应作为普通统计报告分享。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clarabel
import numpy as np

from optim import PortfolioOptimizer, SolverPolicy, TrackingErrorLimit, load_repro
from optim._core.backends import clarabel as backend


def row_map(domain):
    """沿生产锥编译器的行序构建标签；仅用于日志解释，不参与求解。"""
    records = {(r.location, r.index): r for r in domain.constraints}
    equal = (
        np.isfinite(domain.lower)
        & np.isfinite(domain.upper)
        & np.isclose(domain.lower, domain.upper, rtol=0, atol=1e-14)
    )
    labels = []
    for location, side, mask in (
        ("row", "equal", equal),
        ("row", "upper", ~equal & np.isfinite(domain.upper)),
        ("row", "lower", ~equal & np.isfinite(domain.lower)),
        ("variable", "upper", np.isfinite(domain.variable_upper)),
        ("variable", "lower", np.isfinite(domain.variable_lower)),
    ):
        for index in np.flatnonzero(mask):
            r = records.get((location, int(index)))
            labels.append(
                dict(
                    location=location,
                    index=int(index),
                    side=side,
                    group=r.group if r else "auxiliary",
                    key=r.key if r else None,
                    fixed=bool(
                        location == "variable"
                        and domain.variable_lower[index] == domain.variable_upper[index]
                    ),
                )
            )
    return labels


def prepare(folder):
    """导出两日期和五档预算，保留官方 wheel 对同输入的参考结果。"""
    inputs = folder / "inputs"
    inputs.mkdir(parents=True)
    sources = {
        "20260824": "tmp/solver_threads_20260909/inputs/pp_te2.zip",
        "20260814": "tmp/pp_20260814.tar.gz",
    }
    build, factory = backend._build_conic_data, clarabel.DefaultSolver
    rows = []
    for date, source in sources.items():
        base = load_repro(source).problem
        for percent in (2, 4, 6, 8, 10):
            name = f"{date}_te{percent:02d}"
            observed = {}

            def builder(*args):
                data = build(*args)
                observed["row_map"] = row_map(data.domain)
                return data

            def solver(P, q, A, b, cones, settings):
                settings.max_threads = 1
                native = factory(P, q, A, b, cones, settings)

                # 当前安装 wheel 未导出 serde API，按锁定版本的 JsonProblemData 契约写出。
                def matrix(value):
                    return dict(
                        m=value.shape[0],
                        n=value.shape[1],
                        colptr=value.indptr.tolist(),
                        rowval=value.indices.tolist(),
                        nzval=value.data.tolist(),
                    )

                config = {
                    key: getattr(settings, key)
                    for key in dir(settings)
                    if not key.startswith("_") and not callable(getattr(settings, key))
                }
                config["time_limit"] = np.finfo(float).max
                payload = dict(
                    P=matrix(P),
                    A=matrix(A),
                    q=q.tolist(),
                    b=b.tolist(),
                    cones=[{type(c).__name__: c.dim} for c in cones],
                    settings=config,
                )
                (inputs / f"{name}.json").write_text(json.dumps(payload))
                observed["native"] = native
                return native

            with (
                patch.object(backend, "_build_conic_data", builder),
                patch.object(clarabel, "DefaultSolver", solver),
            ):
                result = PortfolioOptimizer(SolverPolicy(backend="clarabel")).solve(
                    base.with_constraints(
                        tracking_error=TrackingErrorLimit(percent / 100)
                    )
                )
            native = observed["native"].get_solution()
            (folder / f"{name}_rows.json").write_text(json.dumps(observed["row_map"]))
            rows.append(
                dict(
                    case=name,
                    status=str(native.status),
                    iterations=native.iterations,
                    objective=native.obj_val,
                    x=native.x,
                    primal_residual=native.r_prim,
                    dual_residual=native.r_dual,
                    accepted=result.status.has_solution,
                )
            )
    (folder / "wheel_reference.json").write_text(json.dumps(rows))


def run(folder, binary, observe):
    """使用同一实验二进制开/关观测，验证观测未改变数学结果。"""
    import os

    env = dict(os.environ)
    env.pop("CLARABEL_OBSERVE", None)
    if observe:
        env["CLARABEL_OBSERVE"] = "1"
    name = "observed" if observe else "control"
    with (
        (folder / f"{name}_solutions.jsonl").open("w") as out,
        (folder / f"{name}.log").open("w") as err,
    ):
        subprocess.run(
            [str(binary), *map(str, sorted((folder / "inputs").glob("*.json")))],
            env=env,
            stdout=out,
            stderr=err,
            check=True,
        )


def analyze(folder):
    """核对观测一致性，按自定义结构化日志汇总步长限制及线性求解残差。"""
    import pandas as pd

    def solutions(name):
        return {
            Path(r["case"]).stem: r
            for r in (json.loads(s) for s in (folder / name).read_text().splitlines())
        }

    control, observed = (
        solutions("control_solutions.jsonl"),
        solutions("observed_solutions.jsonl"),
    )
    wheel = {
        r["case"]: r for r in json.loads((folder / "wheel_reference.json").read_text())
    }
    checks = []
    for case, ref in control.items():
        other = observed[case]
        checks.append(
            dict(
                case=case,
                iterations=other["iterations"],
                status=other["status"],
                instrument_max_weight_diff=float(
                    np.max(np.abs(np.asarray(ref["x"]) - other["x"]))
                ),
                wheel_max_weight_diff=float(
                    np.max(np.abs(np.asarray(ref["x"]) - wheel[case]["x"]))
                ),
                iteration_match=ref["iterations"]
                == other["iterations"]
                == wheel[case]["iterations"],
            )
        )
    (folder / "checks.json").write_text(json.dumps(checks, indent=2))
    rows, refinements, pivots = [], [], []
    for line in (folder / "observed.log").read_text().splitlines():
        if not line.startswith("OBS "):
            continue
        _, kind, payload = line.split(" ", 2)
        if kind == "CASE":
            case, iteration = Path(payload).stem, 0
            labels = json.loads((folder / f"{case}_rows.json").read_text())
            continue
        if kind == "STAGE":
            stage, caps = payload, []
            continue
        data = json.loads(payload)
        if kind == "ITER":
            iteration = data["iter"]
            mu = data["mu"]
        elif kind == "HOM":
            caps.extend(
                [
                    (min(data["tau_cap"], 1.0), "tau", None),
                    (min(data["kappa_cap"], 1.0), "kappa", None),
                ]
            )
        elif kind == "CONE":
            for side in ("primal", "dual"):
                caps.append(
                    (data[f"{side}_cap"], f"{data['kind']}:{side}", data[f"{side}_row"])
                )
        elif kind == "STEP":
            smallest = min(c[0] for c in caps)
            winners = (
                [c for c in caps if abs(c[0] - smallest) < 1e-10]
                if smallest < 1 - 1e-10
                else [(1.0, "full", None)]
            )
            for cap, cause, index in winners:
                label = labels[index] if index is not None else {}
                rows.append(
                    dict(
                        case=case,
                        iteration=iteration,
                        stage=stage,
                        alpha=data["alpha"],
                        cause=cause,
                        group=label.get("group"),
                        side=label.get("side"),
                        fixed=label.get("fixed"),
                        row=index,
                        mu=mu,
                        sigma=(1 - data["alpha"]) ** 3 if stage == "Affine" else None,
                    )
                )
        elif kind == "REFINE":
            refinements.append(dict(case=case, iteration=iteration, **data))
        elif kind == "PIVOTS":
            pivots.append(
                dict(case=case, iteration=iteration, dynamic_regularized=data)
            )
    pd.DataFrame(rows).to_csv(folder / "step_limits.csv", index=False)
    pd.DataFrame(refinements).to_csv(folder / "refinement.csv", index=False)
    pd.DataFrame(pivots).to_csv(folder / "pivots.csv", index=False)
    print(json.dumps(checks, indent=2))


def main():
    """准备和运行隔离观测程序。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/clarabel_observe_20260909")
    )
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.output.exists():
            parser.error("输出目录已存在，请换新目录")
        prepare(args.output)
    elif args.mode == "run":
        if args.binary is None:
            parser.error("run 需要 --binary")
        run(args.output, args.binary, False)
        run(args.output, args.binary, True)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()

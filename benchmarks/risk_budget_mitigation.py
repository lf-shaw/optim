"""风险预算扫描与有限缓解实验：仅修改本进程，生产默认值不变。

包括容差、等价正比例缩放、SOC 分组提升。分组提升未完成诊断坐标映射，
只比较原问题独立验收后的可行结果，不可用于生产证书导出。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clarabel
import numpy as np
import scipy.sparse as sp

from optim import PortfolioOptimizer, SolverPolicy, TrackingErrorLimit, load_repro
from optim._core.backends import clarabel as backend
from risk_budget_scan import risk_value


def split_cone(data, group_size):
    r"""以 $\|r_j\|_2\le t_j,\ \|t\|_2\le\rho$ 等价替换单个大 SOC。

    原始变量保持在最前面，辅助变量不返回用户；仅支持最后一个锥为风险 SOC、零 P。
    """
    if data.P.nnz:
        raise ValueError("分组实验只支持线性目标")
    risk_rows = data.cones[-1].dim
    offset = data.A.shape[0] - risk_rows
    n, m = data.A.shape[1], risk_rows - 1
    groups = (m + group_size - 1) // group_size
    blocks = [
        sp.hstack([data.A[:offset], sp.csc_matrix((offset, groups))], format="csc")
    ]
    rhs = [data.b[:offset]]
    root = sp.csc_matrix(
        (-np.ones(groups), (np.arange(1, groups + 1), n + np.arange(groups))),
        shape=(1 + groups, n + groups),
    )
    blocks.append(root)
    rhs.append(np.r_[data.b[offset], np.zeros(groups)])
    cones = list(data.cones[:-1]) + [clarabel.SecondOrderConeT(1 + groups)]
    for j, start in enumerate(range(0, m, group_size)):
        stop = min(start + group_size, m)
        first = sp.csc_matrix(([-1.0], ([0], [n + j])), shape=(1, n + groups))
        body = sp.hstack(
            [
                data.A[offset + 1 + start : offset + 1 + stop],
                sp.csc_matrix((stop - start, groups)),
            ],
            format="csc",
        )
        blocks.append(sp.vstack([first, body], format="csc"))
        rhs.append(np.r_[0.0, data.b[offset + 1 + start : offset + 1 + stop]])
        cones.append(clarabel.SecondOrderConeT(1 + stop - start))
    return replace(
        data,
        P=sp.csc_matrix((n + groups, n + groups)),
        q=np.r_[data.q, np.zeros(groups)],
        A=sp.vstack(blocks, format="csc"),
        b=np.concatenate(rhs),
        cones=tuple(cones),
    )


def main():
    """固定种子交错测试各配置，保存耗时、迭代数和原始问题验收值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro", default="tmp/pp_20260814.tar.gz")
    parser.add_argument(
        "--output", type=Path, default=Path("tmp/risk_mitigation_20260814")
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budgets", default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument(
        "--variants", default="baseline,gap7,gap6,obj2,obj02,soc10,tree64,tree512"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请换新目录")
    args.output.mkdir(parents=True)
    base = load_repro(args.repro).problem
    lp = PortfolioOptimizer().solve(base.with_constraints(tracking_error=None))
    (args.output / "input.json").write_text(
        json.dumps(
            dict(
                date=str(base.data.date),
                assets=len(base.data.assets),
                lp_status=lp.status.value,
                lp_te=risk_value(base, lp.require_weights().to_numpy())
                if lp.status.has_solution
                else None,
                lp_objective=lp.objective_value,
                arguments=vars(args),
            ),
            default=str,
            indent=2,
        )
    )
    variants = args.variants.split(",")
    unknown = set(variants) - {
        "baseline",
        "gap7",
        "gap6",
        "full7",
        "full6",
        "obj2",
        "obj02",
        "soc10",
        "tree64",
        "tree512",
        "step90",
        "step95",
        "step999",
        "reg10",
        "reg6",
        "eqoff",
    }
    if unknown:
        parser.error(str(unknown))
    build = backend._build_conic_data
    factory = clarabel.DefaultSolver
    optimizer = PortfolioOptimizer(SolverPolicy(backend="clarabel"))
    optimizer.solve(base)
    rng = np.random.default_rng(20260909)
    jobs = [
        (budget, variant)
        for budget in map(float, args.budgets.split(","))
        for variant in variants
    ]
    for repeat in range(args.rounds):
        for i in rng.permutation(len(jobs)):
            budget, variant = jobs[i]
            problem = base.with_constraints(tracking_error=TrackingErrorLimit(budget))
            native = []

            def builder(model, options, module):
                data = build(model, options, module)
                if variant.startswith("tree"):
                    data = split_cone(data, int(variant[4:]))
                if variant in ("obj2", "obj02"):
                    scale = 10.0 if variant == "obj2" else 0.1
                    data = replace(
                        data,
                        q=data.q * scale,
                        objective_scale=data.objective_scale * scale,
                    )
                if variant == "soc10":
                    offset = data.A.shape[0] - data.cones[-1].dim
                    scale = np.r_[np.ones(offset), np.full(data.cones[-1].dim, 10.0)]
                    data = replace(
                        data, A=(sp.diags(scale) @ data.A).tocsc(), b=data.b * scale
                    )
                return data

            def solver(P, q, A, b, cones, settings):
                settings.max_threads = 1
                if variant in ("step90", "step95", "step999"):
                    settings.max_step_fraction = {
                        "step90": 0.90,
                        "step95": 0.95,
                        "step999": 0.999,
                    }[variant]
                if variant in ("reg10", "reg6"):
                    settings.static_regularization_constant = (
                        1e-10 if variant == "reg10" else 1e-6
                    )
                if variant == "eqoff":
                    settings.equilibrate_enable = False
                if variant in ("gap7", "gap6"):
                    settings.tol_gap_abs = settings.tol_gap_rel = 10.0 ** -int(
                        variant[-1]
                    )
                if variant in ("full7", "full6"):
                    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = (
                        10.0 ** -int(variant[-1])
                    )
                if variant in ("obj2", "obj02"):
                    # 保持原始目标单位的绝对 gap 阈值，避免把精度放宽当作缩放加速。
                    settings.tol_gap_abs *= 10.0 if variant == "obj2" else 0.1
                result = factory(P, q, A, b, cones, settings)
                native.append(result)
                return result

            start = time.perf_counter()
            with (
                patch.object(backend, "_build_conic_data", builder),
                patch.object(clarabel, "DefaultSolver", solver),
            ):
                result = optimizer.solve(problem)
            row = dict(
                budget=budget,
                variant=variant,
                repeat=repeat,
                wall_s=time.perf_counter() - start,
                solve_s=result.timings.backend_solve_s,
                setup_s=result.timings.backend_setup_s,
                iterations=native[-1].get_info().iterations if native else None,
                status=result.status.value,
                objective=result.objective_value,
                gap=result.certificate.absolute_gap if result.certificate else None,
                max_violation=max((v.amount for v in result.violations), default=0),
                te=risk_value(problem, result.require_weights().to_numpy())
                if result.status.has_solution
                else None,
                diagnostic_mapping_valid=not variant.startswith("tree"),
            )
            with (args.output / "runs.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    import pandas as pd

    frame = pd.DataFrame(
        json.loads(s) for s in (args.output / "runs.jsonl").read_text().splitlines()
    )
    frame.groupby(["budget", "variant"])[
        [
            "wall_s",
            "solve_s",
            "setup_s",
            "iterations",
            "objective",
            "gap",
            "max_violation",
            "te",
        ]
    ].median().to_csv(args.output / "summary.csv")


if __name__ == "__main__":
    main()

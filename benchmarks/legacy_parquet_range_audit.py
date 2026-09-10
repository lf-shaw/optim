"""只读裁剪旧 tuda2 Parquet，用新版完整入口测试真实日收益漂移。

此目录只有成分标记时使用成分等权测试基准，不宣称复现真实指数；alpha 为固定种子合成值。
原始风险单位保留，LP 不使用风险预算。源目录不写入或迁移。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from current_range_mosek_audit import FactorRiskFrames, run_range


def read_panel(root, relative, since, until, key="sid"):
    """只扫描指定年份，按 dt 谓词裁剪后再恢复排序坐标。"""
    paths = [p for year in range(since.year, until.year + 1)
             for p in (root / relative / f"Y={year}").glob("*.parquet")]
    tables = [pq.read_table(str(p), filters=[("dt", ">=", since), ("dt", "<=", until)])
              .to_pandas().drop(columns="Y", errors="ignore") for p in paths]
    if not tables:
        raise ValueError(f"no data: {relative}")
    frame = pd.concat(tables)
    frame[key] = frame[key].astype(str)
    frame = frame.set_index(key, append=True).sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate coordinates: {relative}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/e/works/5.data/tuda2/basic"))
    parser.add_argument("--since", default="2020-07-01")
    parser.add_argument("--until", default="2020-08-31")
    parser.add_argument("--alpha-mode", choices=["stable", "daily"], default="stable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    since, until = pd.Timestamp(args.since), pd.Timestamp(args.until)

    def read(path, key="sid"):
        return read_panel(args.root, path, since, until, key)

    raw = read("risk/barra/exposure")
    cov = read("risk/barra/cov", "factor")
    spec = read("risk/barra/barra_spec")
    universe = read("quote/universe/csiall")
    rtn = read("quote/rtn/d1")["rtn"]
    expanded = raw.drop(columns=["industry", "sector"]).join(
        pd.get_dummies(raw["industry"], dtype=float)
    )
    expanded["country"] = 1.0
    factors = cov.index.get_level_values("factor").unique()
    expanded = expanded.loc[:, factors]
    styles = set(raw.columns) - {"industry", "sector"}
    types = {f: "style" if f in styles else "country" if f == "country" else "industry"
             for f in factors}
    complete = np.isfinite(expanded).all(axis=1) & np.isfinite(spec["spec_risk"].reindex(expanded.index))
    eligible = expanded.index[complete]
    expanded = expanded.loc[eligible]
    specific = spec["spec_risk"].reindex(eligible)
    frames = FactorRiskFrames(exposure=expanded, covariance=cov,
                              specific_volatility=specific, factor_types=types)
    assets = eligible.get_level_values("sid").unique().sort_values()
    rng = np.random.default_rng(20260910)
    stable_score = pd.Series(rng.standard_normal(len(assets)), index=assets)
    alpha = stable_score.reindex(eligible.get_level_values("sid")).to_numpy()
    if args.alpha_mode == "daily":
        alpha = rng.standard_normal(len(eligible))
    dates = eligible.get_level_values("dt").unique()
    # 收益只按日期切片，保留缺失；由序列引擎严格检查实际持仓覆盖，不填零。
    returns = {date: rtn.xs(date, level="dt") for date in dates[1:]}
    schedule = pd.DataFrame({"alpha": alpha,
                             "tradable": universe["status"].reindex(eligible).eq(1)},
                            index=eligible)
    if universe["status"].reindex(eligible).isna().any():
        raise ValueError("missing universe status")
    report = {"source": str(args.root), "since": args.since, "until": args.until,
              "dates": len(dates), "seed": 20260910,
              "alpha": f"synthetic normal score: {args.alpha_mode}",
              "benchmark": "equal weight among eligible marked members; not official index weights",
              "returns": "stored rtn/d1 rtn, no missing fill",
              "excluded_incomplete_risk_rows": int((~complete).sum()),
              "assets_min": int(schedule.groupby(level="dt").size().min()),
              "assets_max": int(schedule.groupby(level="dt").size().max()),
              "nontradable_rows": int((~schedule.tradable).sum()), "runs": []}
    for name in ["csi300", "csi1000"]:
        members = read(f"quote/universe/{name}")["member"].reindex(eligible)
        if members.isna().any():
            raise ValueError("missing benchmark membership")
        benchmark = members.eq(1).astype(float)
        benchmark /= benchmark.groupby(level="dt").transform("sum")
        inputs = (frames, benchmark, schedule, returns, 0.0)
        for repeat in range(2):
            for backend in (["auto", "mosek"] if repeat == 0 else ["mosek", "auto"]):
                try:
                    result = run_range(inputs, False, "stop", None, backend=backend)
                    result.update(benchmark_name=name, repeat=repeat)
                    print(name, backend, repeat, result["statuses"],
                          result["public_periods_per_s"], flush=True)
                except Exception as exc:
                    result = {"benchmark_name": name, "backend": backend, "repeat": repeat,
                              "error": f"{type(exc).__name__}: {exc}"}
                    print(result, flush=True)
                report["runs"].append(result)
    report["experiment_wall_s"] = time.perf_counter() - started
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()

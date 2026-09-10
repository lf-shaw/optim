"""从真实 repro 重放风险数据接口，在隔离 Python 进程中比较版本。

本脚本不连接 tuda2 存储：接口适配器只返回包内真实风险数组还原的表格。
通过 PYTHONPATH 指定基线或当前 optim，输出小型 JSON 审计及本地比较用权重数组。
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import optim
from optim import PortfolioOptimizer, load_repro, make_portfolio_data
from optim.integrations.tuda2 import Tuda2DataSource


class ReplayRiskSource:
    """保持 tuda2 数据契约的真实数据回放，不进行网络或磁盘风险取数。"""

    __version__ = "repro-audit"

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get_risk_model_schema(self, *, model):
        risk = self.data.risk_model
        kinds = dict(zip(risk.factor_names, risk.factor_types))
        constants = {}
        for i, name in enumerate(risk.factor_names):
            if kinds[name] == "country" and np.all(risk.exposure[:, i] == 1.0):
                constants[name] = 1.0
        return dict(
            factor_order=risk.factor_names,
            style=[name for name in risk.factor_names if kinds[name] == "style"],
            industry=[name for name in risk.factor_names if kinds[name] == "industry"],
            factor_types=kinds,
            constant_exposures=constants,
        )

    def get_risk_model(self, kind, *, dts, model):
        assert list(dts) == [self.data.date]
        self.calls.append(kind)
        data, risk = self.data, self.data.risk_model
        index = pd.MultiIndex.from_product(
            [[data.date], data.assets], names=["dt", "sid"]
        )
        if kind == "exposure":
            frame = pd.DataFrame(risk.exposure, index=index, columns=risk.factor_names)
            # 不物理存储常数 country，覆盖真实供应商的此种格式。
            return frame.drop(
                columns=list(
                    self.get_risk_model_schema(model=model)["constant_exposures"]
                )
            )
        if kind == "spec_risk":
            return pd.Series(risk.specific_volatility, index=index, name="spec_risk")
        assert kind == "cov"
        frame = pd.DataFrame(
            risk.covariance,
            index=pd.MultiIndex.from_product(
                [[data.date], risk.factor_names], names=["dt", "factor"]
            ),
            columns=risk.factor_names,
        )
        # 故意把协方差列倒序，验证以行为准，而非仅检查矩阵 shape。
        return frame.iloc[:, ::-1]


def digest(array):
    """同 dtype/shape 的内容摘要用于跨进程逐项精确比较。"""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def data_hashes(data):
    result = {
        name: digest(getattr(data, name))
        for name in ("alpha", "benchmark", "initial_weight", "tradable")
    }
    result.update(
        {
            name: digest(getattr(data.risk_model, name))
            for name in ("exposure", "covariance", "specific_volatility")
        }
    )
    result["assets"] = hashlib.sha256(
        json.dumps(data.assets.tolist()).encode()
    ).hexdigest()
    result["factor_names"] = hashlib.sha256(
        json.dumps(data.risk_model.factor_names).encode()
    ).hexdigest()
    result["factor_types"] = hashlib.sha256(
        json.dumps(data.risk_model.factor_types).encode()
    ).hexdigest()
    for name, values in data.extra_attributes.items():
        result[f"extra:{name}"] = digest(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("packages", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in args.packages:
        case = load_repro(path)
        original = case.problem
        data = original.data
        universe = pd.DataFrame(
            {"alpha": data.alpha, "tradable": data.tradable, **data.extra_attributes},
            index=data.assets,
        )
        benchmark = pd.Series(data.benchmark, index=data.assets)
        initial = pd.Series(data.initial_weight, index=data.assets)
        module = ReplayRiskSource(data)
        source = Tuda2DataSource(
            module=module,
            risk_model=data.risk_model.provenance.metadata.get("risk_model", "datayes"),
        )
        problem = source.build_problem(
            date=data.date,
            universe=universe,
            benchmark=benchmark,
            initial_weight=initial,
            objective=original.objective,
            constraints=original.constraints,
            alpha_spec=data.alpha_spec,
            extra_attribute_columns=tuple(data.extra_attributes),
        )
        hashes = data_hashes(problem.data)
        assert hashes == data_hashes(data), "原入口未保持包内原始数组"
        assert module.calls == ["exposure", "cov", "spec_risk"]
        # 固定 auto，使跨版本比较不混入 license、后端选择和不同参数的影响。
        optimizer = PortfolioOptimizer(replace(case.policy, backend="auto"))
        started = perf_counter()
        result = optimizer.solve(problem)
        elapsed = perf_counter() - started
        weights = (
            result.require_weights().reindex(data.assets, fill_value=0.0).to_numpy()
        )
        record = dict(
            package=path.name,
            date=str(data.date),
            assets=len(data.assets),
            factor_count=len(data.risk_model.factor_names),
            input_hashes=hashes,
            status=result.status.value,
            backend=result.backend,
            objective=result.objective_value,
            metrics=asdict(result.metrics),
            solve_wall_s=elapsed,
        )
        np.savez_compressed(args.output / (path.name + ".npz"), weights=weights)
        if args.snapshot:
            module.calls.clear()
            risk = source.create_risk_model(date=data.date)
            assert module.calls == ["exposure", "cov", "spec_risk"]
            permutations = {
                "same": np.arange(len(data.assets)),
                "reverse": np.arange(len(data.assets))[::-1],
                "random": np.random.default_rng(20260910).permutation(len(data.assets)),
            }
            for name, order in permutations.items():
                built = make_portfolio_data(
                    date=data.date,
                    universe=universe.iloc[order],
                    benchmark=benchmark,
                    initial_weight=initial,
                    risk_model=risk,
                    alpha_spec=data.alpha_spec,
                    extra_attribute_columns=tuple(data.extra_attributes),
                )
                restore = built.assets.get_indexer(data.assets)
                for key in ("alpha", "benchmark", "initial_weight", "tradable"):
                    np.testing.assert_array_equal(
                        getattr(built, key)[restore], getattr(data, key)
                    )
                np.testing.assert_array_equal(
                    built.risk_model.exposure[restore], data.risk_model.exposure
                )
                np.testing.assert_array_equal(
                    built.risk_model.specific_volatility[restore],
                    data.risk_model.specific_volatility,
                )
                np.testing.assert_array_equal(
                    built.risk_model.covariance, data.risk_model.covariance
                )
                if name == "same":
                    snapshot_data = built
            subset = make_portfolio_data(
                date=data.date, universe=universe.iloc[::2], risk_model=risk
            )
            np.testing.assert_array_equal(
                subset.risk_model.exposure, data.risk_model.exposure[::2]
            )
            missing_rejected = False
            try:
                make_portfolio_data(
                    date=data.date,
                    universe=pd.DataFrame(index=["__missing_asset__"]),
                    risk_model=risk,
                )
            except ValueError:
                missing_rejected = True
            assert missing_rejected
            assert module.calls == ["exposure", "cov", "spec_risk"], (
                "复用快照不应再次取数"
            )
            assert data_hashes(snapshot_data) == hashes
            started = perf_counter()
            snapshot_result = optimizer.solve(replace(original, data=snapshot_data))
            snapshot_elapsed = perf_counter() - started
            new_weights = (
                snapshot_result.require_weights()
                .reindex(data.assets, fill_value=0.0)
                .to_numpy()
            )
            np.testing.assert_allclose(new_weights, weights, rtol=0, atol=1e-8)
            record["snapshot"] = dict(
                status=snapshot_result.status.value,
                input_exact=True,
                tested_orders=list(permutations),
                subset_exact=True,
                missing_rejected=True,
                max_weight_difference=float(np.max(np.abs(new_weights - weights))),
                objective_difference=float(
                    snapshot_result.objective_value - result.objective_value
                ),
                solve_wall_s=snapshot_elapsed,
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "package": path.name,
                    "status": record["status"],
                    "backend": record["backend"],
                    "wall_s": elapsed,
                    "snapshot": record.get("snapshot"),
                }
            ),
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps(dict(optim_path=optim.__file__, records=records), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""漂移数据故障的便携 JSON 导出；只在用户显式请求时执行。"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _dump_failure(error, path: Path) -> Path:
    """导出字段说明及关键中间状态，不序列化任意 Python 对象或风险矩阵。"""
    held = error.previous_weight
    frame = pd.DataFrame()
    if held is not None:
        frame = held.rename("previous_weight").to_frame()
        returns = error.holding_return
        if isinstance(returns, pd.Series) and not returns.index.has_duplicates:
            frame["interval_return"] = returns.reindex(held.index)
            frame["return_sid_present"] = held.index.isin(returns.index)
        elif (
            not isinstance(returns, pd.Series)
            and returns is not None
            and np.asarray(returns).shape == (len(held),)
        ):
            frame["interval_return"] = returns

    def records(value):
        if value is None or value.empty:
            return []
        value = value.copy()
        value.index.name = "sid"
        # pandas 的 JSON 转换将非有限数编码成 null，保持标准 JSON 可读性。
        return json.loads(
            value.reset_index().to_json(orient="records", force_ascii=False)
        )

    partial = error.partial_result
    payload = {
        "kind": "optim.sequence_failure",
        "format_version": 1,
        "field_descriptions": {
            "date": "无法推进持仓的当前调仓日，尚未求解",
            "previous_date": "上一调仓日；interval_return 对应 (previous_date, date]",
            "missing_returns": "有持仓的收益缺失股票，按绝对持仓降序；missing_sid 为无股票标签，missing_value 为标签存在但值缺失",
            "holdings": "上一目标持仓与本次区间收益；不包含原始日度收益或风险模型",
            "completed_steps": "已完成步骤摘要，不包含逐日全量权重；最后实际持仓见 holdings",
            "missing_returns.weight": "失败前实际持仓权重（小数、有符号）",
            "missing_returns.absolute_weight": "持仓权重绝对值；缺失质量按此字段合计，避免多空抵消",
            "holdings.interval_return": "上一调仓日到当前调仓日的复合收益，小数；缺失或非有限数在 JSON 中为 null",
            "holdings.return_sid_present": "收益输入是否包含该股票标签；False 表示标签缺口，True 不保证值有效",
            "duplicate_return_input": "仅收益输入存在重复股票标签时保留原始行，避免重排失败掩盖证据",
            "holding_missing_mass_tolerance": "允许缺失的绝对持仓质量上限，小数权重单位",
        },
        "message": str(error),
        "date": str(error.date) if error.date is not None else None,
        "previous_date": str(error.previous_date)
        if error.previous_date is not None
        else None,
        "missing_returns": records(error.evidence),
        "holdings": records(frame),
        "duplicate_return_input": records(
            error.holding_return.rename("interval_return").to_frame()
        )
        if isinstance(error.holding_return, pd.Series)
        and error.holding_return.index.has_duplicates
        else [],
        "completed_steps": []
        if partial is None
        else [
            {
                "date": str(step.date),
                "status": step.result.status.value,
                "backend": step.result.backend,
                "total_s": step.result.timings.total_s,
            }
            for step in partial.steps
        ],
        "holding_missing_mass_tolerance": None
        if partial is None
        else partial.policy.holding_missing_mass_tolerance,
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return path

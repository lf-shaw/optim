# optim — Gotchas

> 这些条目用于阻止调用方或 LLM 生成语义错误、数值单位错误或不必要的低性能代码。

---

## 01. TE 和风险模型使用年化小数单位

2% 年化 TE 必须写作 `TrackingErrorLimit(0.02)`，不是 `2`。因子协方差是年化小数收益
协方差，特异风险是年化小数波动率；不要再次乘除 100、10,000 或 252。

---

## 02. 风险、基准和 alpha 必须严格同日

优化日 $t$ 使用 $t$ 日盘后风险模型、$t$ 日基准和 $t$ 日 alpha。`InMemoryDataSource` 和
`Tuda2DataSource` 不会用前一个可用日期替代缺失数据；第一个日期缺失也不会静默丢弃。

---

## 03. 换手率是完整 L1 范数，不除以 2

`TurnoverLimit(0.05)` 表示：

$$
\lVert x-x_0\rVert_1\le 0.05.
$$

不要在调用前自行乘 2 或除以 2。换手率约束与 TE 约束彼此独立。

---

## 04. alpha 证书容差依赖 alpha 单位和 scale

默认归一化 gap 通过 `AlphaSpec.scale` 换算成原始 $\alpha^{\mathsf T}x$ 单位。它不是权重
1 bp、收益 1 bp、TE 误差或逐股差异。alpha 的标准化方式改变时，必须重新审阅
`AlphaSpec` 和 `ObjectiveTolerance` 的业务含义。

---

## 05. `alpha=None` 不是所有目标都允许

只有 `MinimizeTrackingError(alpha_floor=None)` 允许缺少 alpha。`MaximizeAlpha`、
`RiskAdjustedAlpha` 或带 `alpha_floor` 的 minimum-TE 都要求 alpha 和 `AlphaSpec`。

---

## 06. 基准覆盖缺口默认报错

即使缺口很小也不会默认归一化。必须显式使用：

```python
BenchmarkCoveragePolicy(
    action="renormalize_within_tolerance",
    missing_mass_tolerance=1e-5,
)
```

超过阈值仍报错；结果的 `alignment` 会记录缺失质量和归一化倍数。

---

## 07. 求解失败默认返回结果，不抛异常

不可行、无界、数值失败或迭代上限通过 `OptimizationResult.status` 表达。直接使用
`result.weights` 前先检查 `result.status.has_solution`；若业务要求失败即抛异常，调用
`result.require_weights()`。

输入 shape、日期、单位或非法模型属于调用错误，会在求解前抛异常。

---

## 08. 深度诊断不会自动运行

自动对每个失败日做 Phase-I、最小换手率和最小 TE 会显著拖慢长回测。先保留准确失败问题和
结果，再对选定日期显式调用 `optimizer.diagnose(...)`。

---

## 09. 链式多期必须使用自然漂移后的真实持仓

不能把上一日目标权重原样当作下一期 `initial_weight`。序列入口会用相邻调仓区间的日度
close-to-close 收益复合并归一化，然后再计算 turnover。当前不存在 open 模式。

---

## 10. 不要在 `optimize_range` 中广播黑名单或冻结名单

这些操作性名单依赖当日真实持仓和交易状态，属于单期实盘指令。批量入口拒绝非空静态
`asset_trade`；在实盘日期使用 `optimizer.optimize(..., blacklist=..., frozen=...)`。

---

## 11. tuda2 数据必须整段一次获取

使用 `Tuda2DataSource` 或先自行批量加载。不要在每日优化循环里调用
`get_risk_model_xxx`、`get_index_weight` 或 `get_return`；I/O 延迟会轻易抵消后端性能收益。

---

## 12. Factor-QCQP 的 LP 预筛默认关闭

`SolverPolicy(lp_prescreen=True)` 只在 LP 最优点同时满足 TE 时提供严格快速证书。通过率未知时
它会增加一次 LP 成本，因此默认关闭，必须由用户根据策略历史通过率显式开启。

---

## 13. 不要从公共调用代码手工选择具体后端

公共 API 只描述目标和约束。后端、必要回退和 Factor-QCQP 搜索属于内部行为；通过
`result.backend` 和 `result.route` 审计实际路线。MOSEK 可用时优先用于回退；不可用或失败时
才使用 Clarabel。

---

## 14. 指数调整日的换手率恢复必须显式授权

基准换仓可能使 5% 换手率与其他约束结构性冲突。默认 `on_failure="stop"`，不会自动放宽。
需要时配置 `TurnoverRecoveryPolicy(max_turnover=...)`；恢复只对当日生效，下一期恢复原上限。

---

## 15. 当前不是通用 SOCP 或全协方差优化器

当前高性能路径面向 factor model。`RiskAdjustedAlpha + TrackingErrorLimit`、通用
`FullCovarianceRiskModel` 和任意锥约束尚未实现；不要假设它们会由 Clarabel 自动接管。

---

## 16. 小权重清理不会无条件改变解

默认尝试删除绝对值小于 `1e-5` 的权重并重新归一化，但只有清理后仍满足约束和目标证书时才
采用。多期 `output_weights="sparse"` 主要减少历史结果体积，不改变内部求解精度。

---

## 17. `on_failure="hold"` 可能传播结构性不可行

指数调整造成的换手冲突若只选择继续持有，后续多日可能持续不可行。默认 `stop` 更安全；
确有业务授权时应使用有上限、可诊断的 turnover recovery，而不是盲目 hold。

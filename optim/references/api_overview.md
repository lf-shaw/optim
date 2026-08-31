# optim — API Overview

> optim 是面向因子风险模型组合构造的统一优化器。普通代码只需要描述数据、目标和约束；
> 系统自动选择已经验证的数值路径，并通过统一结果对象报告状态、权重、指标、证书和实际路线。

---

## 1. 推荐入口

```python
from optim import PortfolioOptimizer

optimizer = PortfolioOptimizer()
```

同一个 `PortfolioOptimizer` 可以复用，但它只保存不可变求解策略，不保存样本空间、持仓、
黑名单或多期状态。

| 场景 | 推荐入口 |
|---|---|
| 已有严格对齐的完整单日问题 | `optimizer.solve(problem)` |
| 单日数据或实盘临时名单 | `optimizer.optimize(...)` |
| 数据源驱动的多期调仓 | `optimizer.optimize_range(...)` |
| 显式每日问题的高级研究 | `optimizer.solve_sequence(...)` |
| 静态输入检查 | `optimizer.validate(problem)` |
| 指定失败问题的深度诊断 | `optimizer.diagnose(problem, prior_result=result)` |

普通不可行、迭代上限或数值失败不会自动抛异常，而是返回 `OptimizationResult`。输入 shape、
单位、日期或模型定义错误会在求解前抛出 `PortfolioValidationError`。

---

## 2. 单日问题契约

一个单日请求由以下三个不可变对象组成：

```python
problem = PortfolioProblem(
    data=portfolio_data,
    objective=MaximizeAlpha(),
    constraints=portfolio_constraints,
)
```

`PortfolioData.assets` 是所有资产维数组唯一的坐标：

```text
alpha                    (n_assets,) 或 None
benchmark                (n_assets,) 或 None
initial_weight           (n_assets,) 或 None
tradable                 (n_assets,)
risk_model.exposure      (n_assets, n_factors)
risk_model.covariance    (n_factors, n_factors)
specific_volatility      (n_assets,)
```

所有数组必须使用同一准确日期。手工 `PortfolioData` 不会按标签重新对齐；需要表格对齐时使用
`InMemoryDataSource` 或 `Tuda2DataSource`。

### 风险单位

- 因子协方差：年化小数收益协方差；
- 特异风险：年化小数波动率；
- 跟踪误差：年化小数波动率，例如 2% 写作 `0.02`；
- 因子暴露：无量纲；
- alpha：由 `AlphaSpec(units=..., scale=...)` 明确声明，系统不猜测单位。

风险模型日期代表当日收盘后可知的数据；同日 alpha、基准和风险模型一起用于当日盘后形成的
下一执行机会目标组合。

---

## 3. 目标和问题类型

| 目标 | 风险预算 | 数学问题 |
|---|---|---|
| `MaximizeAlpha()` | 无 | LP |
| `MaximizeAlpha()` | `TrackingErrorLimit(...)` | Factor-QCQP |
| `RiskAdjustedAlpha(...)` | 无 | 凸 QP |
| `MinimizeTrackingError()` | 不需要额外 TE 上限 | 凸 QP |

`MinimizeTrackingError(alpha_floor=None)` 是当前唯一允许 `PortfolioData.alpha=None` 的目标。
若配置 `alpha_floor`，仍必须提供 alpha 和 `AlphaSpec`。

`RiskAdjustedAlpha + TrackingErrorLimit`、通用全协方差模型和任意通用 SOCP 不是当前第一版的
已实现范围，不能假定系统会自动降级求解。

---

## 4. 约束语义

`PortfolioConstraints` 在 LP、QP 和 Factor-QCQP 中共享。常用字段包括：

- `asset_weight=WeightBounds(lower, upper)`：绝对目标权重；
- `active_weight=SymmetricBound(c)`：逐股主动权重满足 $|x_i-b_i|\le c$；
- `total_active=L`：总主动权重满足 $\lVert x-b\rVert_1\le L$；
- `turnover=TurnoverLimit(T)`：换手率满足 $\lVert x-x_0\rVert_1\le T$，不除以 2；
- `benchmark_member_weight=LowerBound(v)`：基准成员目标权重合计下限；
- `style`、`industry`：相对基准的因子主动敞口；
- `tracking_error=TrackingErrorLimit(B)`：年化 TE 不超过 $B$；
- `freeze_nontradable=True`：不可交易证券固定在期初权重；
- `extra_active`、`extra_absolute`：用户提供的逐资产额外属性约束。

单日 `AssetTradeConstraints` 或 `optimize()` 的便利参数支持：

- `blacklist`：目标权重固定为 0；
- `frozen`：固定为实际期初权重；
- `not_buyable`：不得高于期初权重；
- `not_sellable`：不得低于期初权重；
- `weight_overrides`：指定精确权重或闭区间。

这些名单只对当前单日请求有效。`optimize_range()` 不接受静态交易名单，因为未来实盘名单和
真实期初持仓不能在研究开始前可靠预知。

---

## 5. 多期语义

`PortfolioSchedule.universe` 必须使用名称严格为 `("dt", "sid")` 的唯一 `MultiIndex`。
风险模型、基准和 alpha 必须与每个调仓日严格同日，不会用前一个可用日期补齐。

链式模式在日期 $t$ 求解前，先将上一期目标组合按区间 close-to-close 收益自然漂移：

$$
x_t^{\mathrm{pre}}
=
\frac{x_{t-1}^{\mathrm{target}}\odot(1+r_{t-1,t})}
{\mathbf 1^{\mathsf T}[x_{t-1}^{\mathrm{target}}\odot(1+r_{t-1,t})]}.
$$

随后 turnover 相对 $x_t^{\mathrm{pre}}$ 计算。不存在 open 执行模式。独立模式则要求每个日期
提供自己的期初权重。

`SequencePolicy` 默认值：

- `mode="chained"`；
- `on_failure="stop"`；
- `theta_seed="auto"`，链式传播上一成功日 theta，独立模式使用固定初值；
- 不自动放宽换手率；
- `output_weights="sparse"`，仅保留绝对值不小于默认 `1e-5` 的权重。

---

## 6. 数据源

### 已加载内存数据

`InMemoryDataSource` 接受：

- `FactorRiskFrames`：批量 exposure、covariance、specific volatility；
- 严格 `(dt, sid)` 基准权重；
- 显式 `BenchmarkCoveragePolicy`。

### tuda2

```python
from optim.integrations.tuda2 import Tuda2DataSource
```

`Tuda2DataSource` 对完整区间一次读取 exposure、covariance、specific risk、benchmark；链式
模式需要时再一次读取日度 close-to-close 收益。逐日求解循环不会回源 I/O。

基准在样本空间外存在权重缺口时默认报错。只有显式传入
`BenchmarkCoveragePolicy(action="renormalize_within_tolerance", ...)` 才允许阈值内归一化。

---

## 7. 结果和证书

先检查：

```python
if result.status.has_solution:
    weights = result.require_weights()
else:
    print(result.status, result.message, result.route)
```

重要字段：

| 字段 | 含义 |
|---|---|
| `status` | 标准化最终状态 |
| `weights` | 可用目标权重；失败时为 `None` |
| `backend` | 最终产生已验收候选的后端 |
| `route` | 实际尝试顺序、原生状态、耗时和元数据 |
| `metrics` | 独立复算的目标、TE、换手和风险分解 |
| `certificate` | 可用的目标上界与 gap，带 alpha 单位和尺度 |
| `violations` | 独立验收发现的具名约束违约 |
| `alignment` | 基准缺口、归一化和数据来源日期 |
| `timings` | 准备、后端建立、求解、验收和总耗时 |
| `fingerprint` | 当前准确问题身份，用于 fallback/诊断审计 |

目标 gap 衡量的是 $\alpha^{\mathsf T}x$ 的损失，不代表权重、TE 或单股持仓差异。

---

## 8. 不可行诊断

普通求解不会自动运行高成本诊断。对于一个准确的失败 `PortfolioProblem`：

```python
result = optimizer.solve(problem)
if not result.status.has_solution:
    report = optimizer.diagnose(problem, prior_result=result, level="deep")
    print(report.summary_text)
```

deep 诊断可能给出：线性 Phase-I 松弛、满足其他线性约束时的最小换手率、线性域可行时的
最小 TE，以及各诊断尝试。`prior_result` 的 fingerprint 必须与重新诊断的问题完全一致。

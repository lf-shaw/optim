# optim — API Overview

`optimize` / `optimize_range` 的 `benchmark` 接受指数代码或 pandas.Series。
单期 universe/权重 Series 支持 sid 或只含请求日期的 (dt, sid) 索引，双层日期必须为
时间戳且唯一匹配 date；多期要求严格同日的 (dt, sid) 索引；不广播、不补日期。
Series 输入不调用指数权重接口。旧参数 benchmark_sid 已移除，无兼容别名。
PortfolioData / InMemoryDataSource 已绑定基准时，不允许重复传入 benchmark。

> optim 是面向因子风险模型组合构造的统一优化器。普通代码只需要描述数据、目标和约束；
> 系统自动选择已经验证的数值路径，并通过统一结果对象报告状态、权重、指标、证书和实际路线。

---

## 1. 推荐入口

诊断结果中无法验证的下界允许返回 None，不应解释为零；具体原因可查看诊断尝试记录。

`InfeasibilityReport.load(path, max_uncompressed_bytes=268435456)` 读取 v2 JSON/gzip，
拒绝缺失版本、v1 或未知版本。`contributors_complete` 标记贡献完整性；摘要保留
`contributor_summaries`，缺少贡献时 contributors_frame 和 full 导出会报错，不伪造空证据。

`PortfolioOptimizer(SolverPolicy(backend="auto"))` 默认 LP→HiGHS、QP→direct PIQP、Factor-QCQP→Clarabel。
用户应调用公共 API，不直接依赖内部实现模块；数据契约、报告和复现 API 提供稳定入口。
旧 `optim.opt` / `optim.linopt` / `optim.solver` 已删除，不提供兼容别名；单期与多期使用
`PortfolioOptimizer.optimize` / `optimize_range`，不能原样套用旧参数。
QP 失败可转 Clarabel；自动路径不调用 MOSEK。当前不支持 theta 搜索和跨期参数传播。
复现包格式升级为 v2；v1 包需在原版本提取业务输入，再用新版重建问题与策略并导出 v2。
原版本直接重新导出仍是 v1，不能完成迁移；不静默套用已删除的策略参数。
backend 可指定 mosek、clarabel（LP/QP/Factor-QCQP），highs（仅 LP）或 piqp（仅 QP）。
显式指定不预筛、不回退；MOSEK 缺安装或有效授权抛 RuntimeError，模型类型不支持则 prepare 报错。
所有候选仍独立验收；diagnose(..., backend="auto") 默认自动路由辅助问题，也可显式
指定 mosek/clarabel，不继承主后端、不回退。未提供的数值对偶下界返回 None；piqp 不支持
Phase-I LP，highs 不支持最小风险 QP。非默认旧 lp/qp 预留字段会报错。

`RequiredRelaxation.relaxed_bound`、`unit`、`description` 用于展示原边界到放宽后边界的
变化，dump 同时导出这些字段。lower 减去 amount，upper 加上 amount，不代表独立最小修复。

```python
from optim import PortfolioOptimizer

optimizer = PortfolioOptimizer()
```

同一个 `PortfolioOptimizer` 可以复用，但它只保存不可变求解策略，不保存样本空间、持仓、
黑名单或多期状态。

多期 prepare 会准备静态数据并保留逐日覆盖审计，后续求解复用。日程、风险表和准备后的
输入不得原地修改；修改输入需要重新创建日程并 prepare。动态持仓仍按实际结果逐期检查。

| 场景 | 推荐入口 |
|---|---|
| 已有严格对齐的完整单日问题 | `optimizer.solve(problem)` |
| 单日数据或实盘临时名单 | `optimizer.optimize(...)` |
| 数据源驱动的多期调仓 | `optimizer.optimize_range(...)` |
| 显式每日问题的高级研究 | `optimizer.solve_sequence(...)` |
| 静态输入检查 | `optimizer.validate(problem)` |
| 显式诊断单个问题或结果 | `optimizer.diagnose(result)` 或 `optimizer.diagnose(problem)` |

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
`make_portfolio_data`、`InMemoryDataSource` 或 `Tuda2DataSource`。

手工单期优先 `make_portfolio_data(date=..., universe=..., benchmark=..., initial_weight=...,
risk_model=..., alpha_spec=...)`：universe 的股票索引确定资产顺序，alpha/tradable 是列，
可选列名参数为 alpha_column/tradable_column；未提供 tradable 默认全 True，缺 alpha 保留 None。
权重为带标签 Series；默认严格检查样本外基准，不能丢弃样本外非零持仓。单期股票表和权重
的双层索引只能含请求日期，多日直接报错。原始风险表可用 `make_factor_risk_model` 单独装配，
assets 省略时使用完整当日暴露股票范围；协方差和特异波动率必须已经使用年化小数单位。
两个 make 函数不调用数据源或求解器。

`Tuda2DataSource.create_risk_model(date=...)` 只获取准确同日的三类风险数据，返回带自身
assets 标签的 FactorRiskModel。可供多个组合复用，由 make_portfolio_data 根据 universe
对齐；缺股票报错。无标签风险对象仍要求调用方保证同序；直接 PortfolioData 不自动对齐。
原有单期/多期数据源入口不调用此单日工厂，多期仍走批量 I/O，不重复逐日加载。

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
- `total_active=L`：总主动权重满足 $\lVert x-b\rVert_1\le L$，要求有限且 $L>0$，`None` 禁用；
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
- `output_weights="sparse"`，仅保留绝对值不小于默认 `1e-5` 的权重。

---

## 6. 数据源

### 已加载内存数据

`InMemoryDataSource` 接受：

- `FactorRiskFrames`：批量 exposure、covariance、specific volatility；
- `FactorRiskFrames.constant_exposures`：不在批量 exposure 中重复存储的常数因子，
  例如 `{"country": 1.0}`；
- 严格 `(dt, sid)` 基准权重；
- 显式 `BenchmarkCoveragePolicy`。

### tuda2

```python
from optim.integrations.tuda2 import Tuda2DataSource
```

`Tuda2DataSource` 对完整区间一次读取 exposure、covariance、specific risk、benchmark；链式
模式需要时再一次读取日度 close-to-close 收益。逐日求解循环不会回源 I/O。
适配器通过 tuda2 的 `get_risk_model_schema()` 获取完整因子坐标。schema 将 country 声明为
恒为 1 的常数敞口时，批量 exposure 不重复存储该列，每日进入数值核心前才物化。协方差
跨越 DataYes 2019-12-03 行业分类变更时，columns 可以包含全历史因子并集；每个日期严格以
`(dt, factor)` 行索引选择同名列构造当日方阵。

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

普通求解不会自动运行高成本诊断。失败后显式调用示例：

```python
result = optimizer.solve(problem)
if not result.status.has_solution:
    report = optimizer.diagnose(result, level="deep")
    print(report.summary_text)
```

deep 诊断给出一个加权 Phase-I 松弛方案、最小换手率的数值对偶下界、存在风险预算时的
最小风险候选 TE，以及各诊断尝试。`linear_feasible=None` 表示未确定；候选 TE 高于预算
不能独立证明不可行，Phase-I 松弛不是唯一修复方案。`native_certificates` 保存原求解中
已经存在的可选后端证据，乘子不代表业务重要性。

单期结果保留 `problem` 引用；输入数组不可原地修改，诊断前验证 fingerprint。序列只保留
`stopped_problem`，请与停止日结果一起传给 `diagnose(problem, prior_result=...)`。

`diagnose(result)` 或 `diagnose(problem, prior_result=result)` 在 `result.status.has_solution`
为 True 时立即抛出 `ValueError`，包括 `OPTIMAL_INACCURATE`；阻断发生在校验和编译之前。
成功解的指标读取 `result.metrics`。失败结果包括不可行、迭代上限和数值失败。单独传入
`PortfolioProblem` 仍允许诊断，因为接口不查询求解历史；不提供 `force` 绕过参数。

`report.dump("diagnosis.json")` 导出诊断及原生证据的组计数摘要，默认缩进 2 个空格。
`report.dump("diagnosis.json.gz", indent=None)` 导出紧凑压缩文件；覆盖已有文件须显式
`overwrite=True`。完整乘子使用 `evidence="full"`，按列存储全部字段。文件格式版本为 2，
前部的 `field_descriptions` 集中说明字段，后面的 `report` 保存数据；`evidence_mode` 标明模式。
综合证据冲突时 `linear_feasible=None`，冲突下界不参与恢复；换手率松弛 0.15 表示增加
15 个百分点，不是相对增加 15%，也不等于保持其他约束时的最小换手率。

诊断中的放宽方案不自动应用到原问题。`native_evidence.phase_one_turnover_l1` 为原始候选实际 L1 换手率，证据为空的
原因见 `certificate_availability`。完整证书不是 IIS，组计数不是重要性排名。
# 问题派生与复现

`PortfolioProblem.with_constraints(**changes)` 支持一次替换多个约束字段，排查时推荐单项
修改。`with_data(**changes)`、`with_objective(objective)` 返回新问题，未变更数组共享。
单期结果的 `problem` 可直接派生，无需重新取数；求解策略保存在 `solver_policy`。

`export_repro(path, result=..., report=None, problem=None, policy=None, overwrite=False)`
导出单期输入、实际策略、结果快照和可选完整诊断。序列传入 stopped_problem。
`load_repro(path, max_uncompressed_bytes=1073741824)` 返回 `ReproCase`，不运行求解或诊断。
`case.solve()` 独立复跑；`case.problem` 可继续派生。完整例子见 recipes 中的同进程对照。

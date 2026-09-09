# optim

`optim` 是面向 A 股因子风险模型和指数增强策略的统一组合优化器，使用不可变的
`PortfolioProblem` 描述数据、目标和约束，由
`PortfolioOptimizer` 统一处理：

- 线性 alpha 最大化（LP）；
- 因子风险惩罚目标（凸 QP）；
- 带年化跟踪误差预算的 Factor-QCQP；
- 换手率、主动权重、风格/行业敞口、基准成员覆盖和逐资产交易指令；
- 单期实盘优化、链式多期优化和独立冷启动研究；
- 标准化结果、最优性证书、求解路线审计和显式不可行诊断；
- 手工数组、已加载内存数据，以及 tuda2 批量数据适配。

用户只负责表达业务问题。具体数值路径和必要回退由优化器自动处理，并通过结果对象报告实际
行为。

---

## 1. 安装

运行要求：

- Python `>=3.10`；
- Linux x86-64；
- NumPy、SciPy、pandas；
- HiGHS、PIQP、Clarabel；
- MOSEK Python 包。MOSEK license 是可选的；默认路径只使用免费后端，显式选择 MOSEK 才会检查授权。

安装内部平台发布的 wheel：

```bash
python -m pip install optim-3.0.0-cp311-cp311-linux_x86_64.whl
```

wheel 带 CPython ABI 和平台标记。不同 Python 次版本或操作系统需要分别构建对应 wheel。

从仓库安装开发和构建依赖：

```bash
python -m pip install -r requirements.txt
```

核心 wheel 不强制依赖 tuda2；手工 `PortfolioData` 和 `InMemoryDataSource` 可独立使用。
需要标准数据集成时安装可选 extra：

```bash
python -m pip install "optim[tuda2]"
```

该 extra 要求 `tuda2>=2.0.40`，以提供统一的 `model` 参数、风险模型 schema、稳定资产顺序和跨历史行业分类
制度的协方差坐标契约。

---

## 2. 最小单期示例

如果已经获得严格对齐的 `PortfolioData`：

```python
from optim import (
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioOptimizer,
)

optimizer = PortfolioOptimizer()
result = optimizer.optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=PortfolioConstraints(),
)

if result.status.has_solution:
    weights = result.require_weights()
else:
    print(result.status.value, result.message)
    for attempt in result.route:
        print(attempt.backend, attempt.status.value, attempt.native_status)
```

`PortfolioOptimizer` 可以跨请求复用，但只保存不可变的 `SolverPolicy`，不会保存样本空间、
持仓、黑名单、冻结名单或上一期求解状态。

---

## 3. 公共入口

| 使用场景 | 推荐入口 |
|---|---|
| 已有完整单日问题 | `optimizer.solve(problem)` |
| 单日数据或实盘临时名单 | `optimizer.optimize(...)` |
| 数据源驱动的多期优化 | `optimizer.optimize_range(...)` |
| 已显式构造每日问题 | `optimizer.solve_sequence(...)` |
| 只检查静态输入 | `optimizer.validate(problem)` |
| 复用已准备的单日问题 | `optimizer.prepare(problem)` + `solve_prepared(prepared)` |
| 显式诊断单个问题或结果 | `optimizer.diagnose(result)` 或 `optimizer.diagnose(problem)` |

推荐层级：

```text
普通单期用户              optimize(...)
普通多期用户              optimize_range(...)
需要完整模型控制          solve(PortfolioProblem(...))
框架或低延迟重复调用      prepare(...) / solve_prepared(...)
```

旧的 `optim.opt`、`optim.linopt`、`optim.solver` 已移除，不提供兼容别名。原 `optimize` / `multioptimize`
调用请迁移到 `PortfolioOptimizer.optimize(...)` / `optimize_range(...)`，并按新数据及约束契约
重新组装参数；旧 `Solver` 的可变状态接口不再提供。

---

## 4. 数据模型与严格对齐

### 4.1 `PortfolioProblem`

一个单期问题由三个不可变对象完整定义：

```python
from optim import PortfolioProblem

problem = PortfolioProblem(
    data=portfolio_data,
    objective=objective,
    constraints=constraints,
)
```

求解器实例没有隐式可变建模状态，因此调用顺序不会改变问题。

### 4.2 `PortfolioData`

`PortfolioData.assets` 是所有资产维数组唯一的权威位置坐标：

```text
alpha                              (n_assets,) 或 None
benchmark                          (n_assets,) 或 None
initial_weight                     (n_assets,) 或 None
tradable                           (n_assets,)
risk_model.exposure                (n_assets, n_factors)
risk_model.covariance              (n_factors, n_factors)
risk_model.specific_volatility     (n_assets,)
```

手工传入 NumPy 数组时，优化器不会根据证券标签再次重排。带标签数据必须先由数据适配层严格
对齐，或者使用 `InMemoryDataSource`/`Tuda2DataSource`。

### 4.3 日期规则

优化日 $t$ 使用：

- $t$ 日风险模型；
- $t$ 日基准权重；
- $t$ 日 alpha；
- $t$ 日盘后实际可得的交易状态和期初持仓。

这些信息在 $t$ 日收盘后、下一执行机会前可知，不构成未来信息。公共数据层要求严格同日；
不会用前一个可用日期补风险、基准或 alpha，也不会静默丢弃缺失日期。

### 4.4 风险单位

风险单位固定为 annualized decimal：

| 字段 | 单位 | 例子 |
|---|---|---|
| `FactorRiskModel.covariance` | 年化小数收益协方差 | 对角线 `0.04` 对应 20% 年化波动率 |
| `specific_volatility` | 年化小数波动率 | `0.20` 表示 20% |
| `TrackingErrorLimit` | 年化小数波动率 | 2% 写作 `0.02` |
| `exposure` | 无量纲 | 不缩放 |
| `alpha` | 用户声明的业务单位 | 由 `AlphaSpec` 说明 |

不要再次乘除 `100`、`10_000` 或 `252`。优化器不会根据数值大小猜测单位。

---

## 5. 因子风险模型

令目标权重为 $x$、基准权重为 $b$、主动权重为 $a=x-b$、资产因子暴露矩阵为 $E$、因子
协方差为 $F$、特异波动率为 $d$。主动因子暴露为：

$$
f=E^{\mathsf T}a.
$$

年化跟踪误差满足：

$$
\operatorname{TE}(x)^2
=f^{\mathsf T}Ff+\lVert d\odot a\rVert_2^2.
$$

手工构造单日风险模型：

```python
import numpy as np
import pandas as pd

from optim import FactorRiskModel

risk_model = FactorRiskModel(
    asof=pd.Timestamp("2026-08-31"),
    exposure=np.asarray(exposure, dtype=float),
    covariance=np.asarray(factor_covariance, dtype=float),
    specific_volatility=np.asarray(specific_volatility, dtype=float),
    factor_names=tuple(factor_names),
    factor_types=tuple(factor_types),  # 例如 style / industry
)
```

`exposure` 列、`covariance` 两个轴、`factor_names` 和 `factor_types` 必须使用完全一致的因子顺序。

DataYes 协方差包含 country 因子，但批量敞口可以不物理存储全 1 列。
`FactorRiskFrames.constant_exposures={"country": 1.0}` 会在逐日物化时将其按协方差
因子顺序补入；最终 `FactorRiskModel.exposure` 仍是包含 country 的完整数值矩阵。

---

## 6. 支持的目标

### 6.1 最大化 alpha

```python
from optim import MaximizeAlpha

objective = MaximizeAlpha()
```

目标为：

$$
\max_x\ \alpha^{\mathsf T}x.
$$

没有 TE 预算时属于 LP；增加 `TrackingErrorLimit` 后属于 Factor-QCQP。

### 6.2 风险调整 alpha

```python
from optim import RiskAdjustedAlpha

objective = RiskAdjustedAlpha(
    factor_aversion=0.75,
    specific_aversion=0.75,
)
```

目标同时惩罚因子和特异方差。风险厌恶系数的经济意义取决于 alpha 的尺度和年化小数方差
口径，优化器不会自动统一二者量纲。

当前 `RiskAdjustedAlpha` 不支持同时再配置 `TrackingErrorLimit`。

### 6.3 最小化跟踪误差

```python
from optim import MinimizeTrackingError

objective = MinimizeTrackingError(alpha_floor=None)
```

这是当前唯一允许 `PortfolioData.alpha=None`、`alpha_spec=None` 的目标。如果配置
`alpha_floor`，仍必须提供 alpha 和 `AlphaSpec`。

---

## 7. 组合约束

下面是一个接近指数增强实际应用的约束集合：

```python
from optim import (
    ExposureBounds,
    LowerBound,
    PortfolioConstraints,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)

constraints = PortfolioConstraints(
    long_only=True,
    budget=1.0,
    asset_weight=WeightBounds(lower=0.0, upper=1.0),
    active_weight=SymmetricBound(0.01),
    total_active=1.8,
    turnover=TurnoverLimit(0.05),
    benchmark_member_weight=LowerBound(0.81),
    style=ExposureBounds(
        default=(-0.60, 0.60),
        overrides={"size": (-0.30, 0.30)},
    ),
    industry=ExposureBounds(default=(-0.05, 0.05)),
    tracking_error=TrackingErrorLimit(0.02),
)
```

约束含义：

| 字段 | 数学含义 |
|---|---|
| `budget=1.0` | $\mathbf 1^{\mathsf T}x=1$ |
| `asset_weight` | 每只证券绝对目标权重上下限 |
| `active_weight` | $|x_i-b_i|\le c$ |
| `total_active` | $\lVert x-b\rVert_1\le L$，有限正数；`None` 禁用 |
| `turnover` | $\lVert x-x_0\rVert_1\le T$ |
| `benchmark_member_weight` | 基准成员目标权重合计下限 |
| `style`/`industry` | 相对基准的因子主动敞口 |
| `tracking_error` | $\operatorname{TE}(x)\le B$ |
| `freeze_nontradable` | 不可交易证券固定在实际期初权重 |

换手率采用完整 L1 口径，不除以 2。`TurnoverLimit(0.05)` 就表示：

$$
\lVert x-x_0\rVert_1\le0.05.
$$

换手率与跟踪误差是两个独立约束，二者不存在固定大小关系。

---

## 8. 完整手工单期示例

```python
import numpy as np
import pandas as pd

from optim import (
    AlphaSpec,
    FactorRiskModel,
    MaximizeAlpha,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
)

assets = pd.Index(asset_ids, name="sid")
risk_model = FactorRiskModel(
    asof=pd.Timestamp("2026-08-31"),
    exposure=np.asarray(exposure, dtype=float),
    covariance=np.asarray(factor_covariance, dtype=float),
    specific_volatility=np.asarray(specific_volatility, dtype=float),
    factor_names=tuple(factor_names),
    factor_types=tuple(factor_types),
)
data = PortfolioData(
    date=pd.Timestamp("2026-08-31"),
    assets=assets,
    alpha=np.asarray(alpha, dtype=float),
    alpha_spec=AlphaSpec(
        units="standardized_score",
        scale=1.0,
    ),
    benchmark=np.asarray(benchmark_weight, dtype=float),
    initial_weight=np.asarray(pretrade_weight, dtype=float),
    tradable=np.asarray(tradable, dtype=bool),
    risk_model=risk_model,
)
problem = PortfolioProblem(
    data=data,
    objective=MaximizeAlpha(),
    constraints=constraints,
)

optimizer = PortfolioOptimizer()
report = optimizer.validate(problem)
report.raise_for_errors()

result = optimizer.solve(problem)
weights = result.require_weights()
```

`validate()` 不建立求解器，会尽量一次返回所有可以独立发现的输入错误。直接调用 `solve()`
也会先执行同样的前置校验。

---

## 9. 单期实盘交易指令

黑名单、冻结名单、不可买入/卖出名单通常只对一次实盘优化有效，可以直接使用便利参数：

```python
result = optimizer.optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
    blacklist=["delist_candidate"],
    frozen=["suspended_sid"],
    not_buyable=["sell_only_sid"],
    not_sellable=["buy_only_sid"],
    weight_overrides={
        "special_sid": (0.001, 0.003),
    },
)
```

语义如下：

- `blacklist`：目标权重固定为零；
- `frozen`：固定为当前实际期初权重；
- `not_buyable`：目标权重不得高于期初权重；
- `not_sellable`：目标权重不得低于期初权重；
- `weight_overrides`：标量表示精确权重，二元组表示闭区间。

名单会被写入当前不可变问题，不会泄漏到下一次调用。默认情况下，名单包含样本空间之外的证券
会在求解前报错。

`optimize_range()` 不支持把同一静态 `asset_trade` 广播到未来日期，因为未来真实持仓、停牌、
黑名单和冻结状态通常无法在回测开始前准确预知。

---

## 10. 使用 tuda2 单期取数

`benchmark` 接受指数代码或以 `sid` 为索引的自定义权重 Series。自定义权重视为指定
`date` 当日的基准，按股票标签对齐，不请求指数权重接口，仍执行覆盖率和归一化校验。
直接传入已包含基准的 `PortfolioData` 时，不再指定 `benchmark`。

```python
from optim import AlphaSpec, MaximizeAlpha, PortfolioOptimizer
from optim.integrations.tuda2 import Tuda2DataSource

source = Tuda2DataSource(
    risk_model="datayes",
    benchmark_weight_type="daily",
)

result = PortfolioOptimizer().optimize(
    data_source=source,
    date=trade_date,
    universe=today_universe,       # sid 索引，包含 alpha 列
    benchmark="000852.SH",
    initial_weight=pretrade_weight,
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(
        units="standardized_score",
        scale=1.0,
    ),
)
```

单期入口只读取指定日期，不额外获取持仓漂移收益。数据适配器负责把 tuda2 的风险模型和指数
权重转换为严格同日、统一资产坐标的 `PortfolioData`。适配器要求 tuda2 提供
`get_risk_model_schema()`，并从这一接口读取完整因子顺序、因子类型和常数敞口；不会再从
多个因子名称接口自行拼接模型定义。

---

## 11. 链式多期优化

### 11.1 构造调仓计划

`PortfolioSchedule.universe` 必须使用名称严格等于 `("dt", "sid")` 的唯一 `MultiIndex`。
每个日期可以拥有不同的样本空间。

```python
from optim import PortfolioSchedule

schedule = PortfolioSchedule(
    universe=universe.sort_index(),
    alpha_column="alpha",
    tradable_column="tradable",
)
```

### 11.2 tuda2 链式求解

`benchmark` 接受指数代码或以 `(dt, sid)` 为索引的逐日权重 Series。自定义权重必须
覆盖所有调仓日期，不广播单期权重、不前向填充；传入 Series 时跳过指数权重取数。
使用已绑定基准的 `InMemoryDataSource` 时，不再指定 `benchmark`。

```python
from optim import (
    AlphaSpec,
    MaximizeAlpha,
    PortfolioOptimizer,
    SequencePolicy,
)
from optim.integrations.tuda2 import Tuda2DataSource

sequence = PortfolioOptimizer().optimize_range(
    data_source=Tuda2DataSource(risk_model="datayes"),
    schedule=schedule,
    benchmark="000852.SH",
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
    initial_weight=first_day_weight,
    sequence_policy=SequencePolicy(
        mode="chained",
        on_failure="stop",
        output_weights="sparse",
    ),
)
```

tuda2 风险模型、基准权重和日度收益按完整区间一次取足；逐日优化循环不会反复发起 I/O。

如果风险模型和基准已经以 pandas 对象常驻内存，使用相同的多期入口，不需要逐日
构造 `PortfolioProblem`：

```python
from optim import FactorRiskFrames, InMemoryDataSource

risk_frames = FactorRiskFrames(
    exposure=exposure_frame,                  # (dt, sid) 行索引
    covariance=factor_covariance_frame,       # (dt, factor) 行索引
    specific_volatility=specific_risk_frame,  # (dt, sid) 行索引
    factor_types=factor_type_by_name,
    constant_exposures={"country": 1.0},     # 协方差含 country 且敞口恒为 1 时
)
memory_source = InMemoryDataSource(
    risk_data=risk_frames,
    benchmark=daily_benchmark_weight,          # (dt, sid) 行索引
)

sequence = optimizer.optimize_range(
    data_source=memory_source,
    schedule=schedule,
    objective=objective,
    constraints=constraints,
    alpha_spec=alpha_spec,
    initial_weight=first_day_weight,
    holding_period_returns=close_to_close_returns,
    sequence_policy=SequencePolicy(mode="chained"),
)
```

`InMemoryDataSource` 不实施日期回退、forward-fill 或隐式单位转换。链式模式下，
`holding_period_returns` 以“当前调仓日”为键，其值是上一调仓日到当日的逐资产 C2C
复合收益。

### 11.3 持仓自然漂移

链式模式在第 $t$ 日求解前，将上一期目标组合按相邻调仓区间 close-to-close 收益推进：

$$
x_t^{\mathrm{pre}}
=
\frac{x_{t-1}^{\mathrm{target}}\odot(1+r_{t-1,t})}
{\mathbf 1^{\mathsf T}\left[x_{t-1}^{\mathrm{target}}\odot(1+r_{t-1,t})\right]}.
$$

随后用 $x_t^{\mathrm{pre}}$ 计算当日换手率。不能把上一期目标权重原样当作下一期期初权重。
当前接口不存在 `open` 执行模式。

### 11.4 默认序列策略

```text
mode                 chained
holding_update       mark_to_market
on_failure           stop
turnover_recovery    None
output_weights       sparse
```


---

## 12. 独立冷启动序列

独立模式下，每个日期使用调用者提供的独立期初权重，不承接前一天优化结果，也不需要持仓漂移
收益：

```python
from optim import SequencePolicy

sequence = optimizer.optimize_range(
    data_source=data_source,
    schedule=schedule,
    objective=objective,
    constraints=constraints,
    alpha_spec=alpha_spec,
    initial_weight=first_initial_weight,
    independent_initial_weights={
        date: initial_weight_by_date[date]
        for date in schedule.dates
    },
    sequence_policy=SequencePolicy(
        mode="independent",
        output_weights="none",
    ),
)
```

冷启动组合应具备合理的基准覆盖和敞口。随机从全部股票中任取少量证券，可能天然违反换手率、
基准成员覆盖或因子敞口约束。

多期返回值保留每个已尝试日期的输入状态和求解结果：

```python
for step in sequence.steps:
    result = step.result
    print(
        step.date.date(),
        result.status.value,
        result.backend,
        result.timings.total_s,
        result.metrics.tracking_error,
        result.metrics.turnover_l1,
    )

one_day = sequence.result_for_date(target_date)
print(sequence.stopped_date, sequence.final_weight)
```

`output_weights="none"` 不保存每日目标、交易前和最终权重 payload；状态、路由、证书、
违约和耗时仍会保留，适合长区间可靠性与性能测试。

---

## 13. 基准覆盖策略

优化样本空间遗漏任何实质性基准权重时，默认直接报错，不会自动归一化。若业务明确允许极小
缺口，必须显式配置：

```python
from optim import BenchmarkCoveragePolicy

benchmark_policy = BenchmarkCoveragePolicy(
    action="renormalize_within_tolerance",
    missing_mass_tolerance=1e-5,
)
```

超过阈值仍然报错。发生获准归一化时，结果会记录：

```python
result.alignment.benchmark_missing_mass
result.alignment.benchmark_renormalization_factor
```

---

## 14. 结果对象

普通不可行、无界、迭代上限或数值失败返回结构化 `OptimizationResult`，不会自动抛异常。输入
shape、单位、日期或模型定义错误则会在求解前抛出异常。

主要字段：

| 字段 | 含义 |
|---|---|
| `status` | 标准化最终状态 |
| `weights` | 目标权重；没有可用解时为 `None` |
| `objective_value` | 以原始业务单位复算的目标值 |
| `backend` | 最终产生已验收结果的后端 |
| `route` | 所有实际尝试及原生状态、耗时和元数据 |
| `metrics` | 独立复算的目标、TE、换手和风险分解 |
| `certificate` | 可用的目标上界、绝对 gap 和归一化 gap |
| `violations` | 独立验收发现的具名约束违约 |
| `alignment` | 基准缺口、持仓缺口和实际来源日期 |
| `timings` | 准备、建立、求解、验收和总耗时 |
| `fingerprint` | 业务问题及最终数学问题的确定性身份 |
| `message` | 简短状态说明 |

推荐读取方式：

```python
if result.status.has_solution:
    weights = result.require_weights()
    print(result.metrics.tracking_error)
    print(result.metrics.turnover_l1)
else:
    print(result.status.value, result.message)
```

`require_weights()` 为要求“无解即异常”的调用方提供显式异常语义。

默认尝试清理绝对值小于 `1e-5` 的权重；只有清理后仍满足约束和目标证书时才采用清理结果。

---

## 15. alpha 单位与目标证书

多数业务 alpha 是标准化分数，而不是直接收益率。必须用 `AlphaSpec` 明确说明：

```python
from optim import AlphaSpec

alpha_spec = AlphaSpec(
    units="standardized_score",
    scale=1.0,
)
```

目标容差由 `SolverPolicy.objective_tolerance` 控制。原始 alpha 目标单位下的允许 gap 为：

$$
\epsilon_{\mathrm{raw}}
=
\epsilon_{\mathrm{abs}}
+
\epsilon_{\mathrm{normalized}}\times\operatorname{AlphaSpec.scale}.
$$

默认 `normalized=1e-4`。它表示 $\alpha^{\mathsf T}x$ 的目标损失，不表示：

- 单股权重差小于 `1e-4`；
- 组合 L1 差小于 `1e-4`；
- TE 差小于 `1e-4`；
- 收益损失固定为 1 bp。

alpha 的预处理或 scale 改变后，必须重新确认该容差的经济含义。

---

## 16. 不可行诊断

接收他人导出的报告后，可直接恢复报告对象：

```python
from optim import InfeasibilityReport

report = InfeasibilityReport.load("diagnosis.json.gz")
if report.contributors_complete:
    frame = report.contributors_frame()
else:
    print(report.contributor_summaries)  # 原贡献数量、约束组计数；请对方提供 full 报告
```

仅支持整数 `format_version=2`，缺少版本、v1 或未知版本明确拒绝。full 文件恢复全部
贡献；摘要无法恢复省略内容，调用 contributors_frame 或导出 full 会报错。真正没有证据
时则返回空表。默认解压大小上限 256 MiB，可通过 `max_uncompressed_bytes` 调整。
加载不求解、不恢复原模型，重新优化仍需复现包。metadata 保持 JSON 值，不恢复任意 Python 类型。

使用 `report.contributors_frame()` 可直接把内存中的原生贡献转换为 DataFrame，无需解析
dump。例如 `report.contributors_frame(groups=("asset_bound", "turnover"))` 仅查看指定
约束组；`include_metadata=True` 可附带后端专用详情。默认保留证书编号、后端、证据范围和
原始乘子，不按乘子排名；无证据时返回相同列结构的空表。此操作不重新诊断或求解。

松弛条目提供 `relaxed_bound`、`unit` 和 `description`：下界从原值减去 `amount`，
上界加上 `amount`。例如下界 2% 松弛 0.005，表示降低到 1.5%，不是提高到 2.5%。
权重按百分比展示，风格及自定义敞口保留原单位。JSON 导出包含这些计算字段，原始数值
保持完整精度。它们是一个同时松弛方案，不是各约束必须放宽的最小值，也不保证满足风险预算。

> 开发者调试支持直接从 `result.problem.with_constraints(turnover=None)` 派生对照问题。
> `export_repro("case.zip", result=result)` 与 `load_repro("case.zip")` 用于跨进程传递完整
> 单期输入；加载后 `case.solve()` 复跑、`case.problem` 继续派生。不会自动启动诊断。
> 诊断摘要、完整证据、模型复现包的区别和示例见
> [调试与复现范例](optim/references/recipes.md#同进程对照与跨进程复现)。


深度诊断可能包含额外 LP/QP，因此不会在普通求解路径自动运行。下面是失败后显式诊断的例子：

```python
result = optimizer.solve(problem)

if not result.status.has_solution:
    report = optimizer.diagnose(
        result,
        level="deep",
    )
    print(report.summary_text)
    print(report.turnover_linear_lower_bound)
    print(report.minimum_tracking_error)

    for item in report.relaxations[:10]:
        print(item.group, item.key, item.side, item.amount)
```

deep 诊断会按问题结构尝试：

1. 线性 Phase-I，报告一个加权松弛方案及边界的配置来源；
2. 配置了换手率时，删除该上限并计算最小 L1 换手率的数值对偶下界；
3. 线性域可行且配置了风险预算时，求解最小风险问题，报告已验收候选的 TE；
4. 保留所有诊断尝试及原生状态。

`prior_result` 必须来自完全相同的 `PortfolioProblem`，fingerprint 不一致会被拒绝。

诊断入口聚焦失败原因：传入成功结果，或通过 `prior_result` 提供成功结果时，会在校验、编译
和额外求解之前立即抛出 `ValueError`。成功包括 `OPTIMAL` 和 `OPTIMAL_INACCURATE`，按
`status.has_solution` 判断。当前解的换手率、TE 等指标请直接读取 `result.metrics`。
失败结果不限于不可行，也允许诊断迭代上限或数值失败。单独传入 `PortfolioProblem` 时无法
知道历史求解状态，仍允许显式诊断；不提供 `force` 绕过选项。若只诊断不可行结果，可用
`if result.status is SolveStatus.INFEASIBLE:` 控制。报告类型名 `InfeasibilityReport` 不意味着
问题必然不可行，具体结论以报告字段为准。

诊断会综合 Phase-I 和最小换手率检查：即使 Phase-I 未确定，可信的数值换手率下界高于
原上限仍支持线性不可行。若下界与已检查的原始候选冲突，报告返回 `linear_feasible=None`，
暂不采信该下界，避免自动恢复使用它；原值和原因记录在 `native_evidence` 中。

换手率松弛使用小数权重单位。例如 `amount=0.15` 表示增加 15 个百分点，5% 上限变为 20%。
这是一组可能同时松弛其他约束的 Phase-I 方案，不等于保持其他约束时的最小换手率。
摘要会列出其他受松弛的约束组，完整数值见 `relaxations`。

单期 `solve`、`optimize` 的结果保留 `result.problem`，因此可直接 `optimizer.diagnose(result)`。
它保留输入引用而非深复制，求解后不要原地修改输入数组；诊断前会重新检查 fingerprint。
多期为控制内存仅保留停止时的 `sequence_result.stopped_problem`：

```python
if sequence_result.stopped_problem is not None:
    failed = sequence_result.result_for_date(sequence_result.stopped_date)
    report = optimizer.diagnose(sequence_result.stopped_problem, prior_result=failed)
```

`linear_feasible` 为 `True` 表示找到了满足容差的线性候选，`False` 表示数值对偶下界支持
不可行，`None` 表示证据不足。超时、数值失败不会自动解释成不可行。Phase-I 松弛取决于
惩罚尺度，可能有多个最优方案，不代表每条边界都必须放宽该数值。最小风险候选 TE 是最小值
的上界；即使高于预算，也不能单凭它证明预算不可行。

`result.native_infeasibility` 和 `report.native_certificates` 是可选的原生证据。只有实际到达的
后端在原求解中留下可读取证据时才有内容；没有证据不代表可行。接口保留原生带符号乘子和
请求 fingerprint，锥转换会标记坐标来源。当前证据均标为数值估计，不承诺独立严格验证或 IIS。
乘子大小受约束缩放影响，不是业务重要性或所需放宽量。原生读取失败只记录简短错误类别，
不改变求解状态，不解析日志，也不会为补充证据额外求解。

完整报告可导出为 UTF-8 JSON。文件前部集中放置从 Attributes 文档生成的
`field_descriptions` 字典，之后的 `report` 保留全部 Phase-I 松弛、检查结论和尝试记录。
默认原生证据仅汇总来源、质量、贡献数量及约束组计数，不把数万条乘子解释成数万个冲突原因。
需要完整坐标时显式使用 `evidence="full"`。导出不会重新诊断或求解：

```python
report.dump("diagnosis.json")                         # 默认 indent=2，便于阅读
report.dump("diagnosis.json.gz", indent=None)         # 紧凑格式 + gzip，便于传输
report.dump("native-full.json.gz", evidence="full", indent=None)  # 完整原生坐标
report.dump("diagnosis.json", indent=4, overwrite=True)
```

默认不覆盖已有文件，父目录需存在。导出文件包含 `format_version=2` 和 `evidence_mode`；枚举使用其值、日期使用
ISO 格式、数组使用列表，非有限数值表示为字符串 `NaN` / `Infinity` / `-Infinity`。
完整模式的 `contributors` 使用 `encoding="columns"`，`columns` 各列的第 i 项共同表示
一个贡献对象，保留符号与全部字段。摘要模式不会构造逐项 JSON，仅统计计数。
两种模式都不包含重放求解所需的原始风险模型等完整输入。

诊断专用线性域保留完整 L1 换手率和全部显式总主动/基准覆盖约束，不沿用依赖原上限的
省略规则。Phase-I 默认保护非负底线及操作指令资产边界，避免建议卖空或解除冻结；报告记录
`phase_one_turnover_l1` 实际换手率和 `certificate_availability` 证据可用性。正常优化仍使用
原有加速简化。Phase-I 是线性松弛方案，不承诺满足风险预算，也不保证非线性风险冲突的最小修复。

Factor-QCQP 直接使用锥求解器，原生证书来自实际运行的 MOSEK /
Clarabel 回退求解完整问题时，才可产生完整问题的锥证据。各后端的能力无需完全一致。

冻结、黑名单等指令依照既定优先级可能覆盖常规主动权重边界。诊断针对最终有效模型，
`sources` / `metadata` 提供配置来源与覆盖记录，不会把已被覆盖的规则报告为数学冲突。

---

## 17. 指数调整日与换手率恢复

指数成分调整可能使严格换手率与基准覆盖、主动权重或因子约束发生结构性冲突。默认策略为停止，
不会自动放宽约束。

业务明确授权时：

```python
from optim import SequencePolicy, TurnoverRecoveryPolicy

sequence_policy = SequencePolicy(
    on_failure="stop",
    turnover_recovery=TurnoverRecoveryPolicy(
        max_turnover=0.20,
        buffer=1e-5,
    ),
)
```

恢复满足以下原则：

- 必须由用户显式提供最大允许换手率；
- 不把基准自身换手率直接当作组合最小换手率；
- 只对当前异常日期有效；
- 下一日期自动恢复原配置；
- 结果记录配置上限、诊断下界和实际有效上限。

简单设置 `on_failure="hold"` 可能把一次结构性冲突传播到后续很多日期，因此默认仍为 `stop`。

---

## 18. 求解行为与默认策略

实现布局分为公共 Python 接口、`optim._impl` 业务实现和 `optim._core` 数学核心。
`_impl` 集中编译模型编译器、求解适配、诊断引擎、资产边界合并及结果验收五个模块；
它可以依赖公共数据契约和 `_core`，但 `_core` 不反向依赖它或上层类型。
两层实现均以扩展模块发布，不包含实现 `.py`、`.pyi` 或生成的 C 文件。
公共接口和数据契约仍保留 Python 类型注解；实现路径不属于稳定用户接口。

当前主路径（不再包含 theta 搜索）：

旧的 `theta_seed`、theta 数值调参、前沿策略和商业回退配置已移除，不接受旧参数。
`export_repro` / `load_repro` 使用 v2 复现包格式；旧 v1 包不再直接加载，需在原环境迁移。

| 问题 | 默认数值路径 |
|---|---|
| LP | HiGHS |
| 凸 QP | direct PIQP |
| Factor-QCQP / SOCP 风险预算 | Clarabel（QDLDL） |
| QP 失败复核 | Clarabel；不自动调用 MOSEK |

调用方通常不应根据问题类型手工选择后端；通过 `result.backend` 和 `result.route` 审计实际路线。

需要独立对比时，可以显式选择后端：

```python
auto = PortfolioOptimizer(SolverPolicy(backend="auto"))
mosek = PortfolioOptimizer(SolverPolicy(backend="mosek"))
baseline = auto.solve(problem)
comparison = mosek.solve(problem)
```

`mosek`、`clarabel` 支持 LP、QP 和 Factor-QCQP；`highs` 仅支持 LP，`piqp` 仅支持 QP。
显式指定时不执行 LP 预筛、不自动回退；MOSEK 缺安装、无授权或授权过期时抛出 `RuntimeError`。
普通不可行或数值失败仍返回状态，不会悄悄换后端。不支持的模型在 prepare 阶段报错，独立结果验收仍执行。默认 `auto` 保留上述路线。
`lp`、`qp` 等旧预留选择字段仅允许原默认值，非默认值会报错，应使用 `backend` 选择。
`diagnose()` 的辅助 LP/QP 默认自动路由，因为它们可能与原问题类型不同。
需要交叉验证时使用 `optimizer.diagnose(result, backend="mosek")` 或 `backend="clarabel"`，
只影响辅助问题，原始证书不变。显式选择不回退，不支持的辅助模型会报错；piqp 不支持
必需的 Phase-I LP，highs 遇到最小风险 QP 会报错。报告记录实际尝试。
HiGHS/MOSEK/Clarabel 的辅助 LP 均按原矩阵统一复算数值对偶下界；内点残差遇到无穷变量
边界时，尝试在已验收候选的目标子水平集推导有限盒界后计入残差影响，不能忽略残差。
无法验证时下界仍为 None，attempts.metadata 的 dual_bound_status/reason/method 说明原因和方法。
这是浮点数值估计，不是区间算术严格证明；候选目标只提供子水平集上界，绝不冒充最优值下界。

后端适配器只负责原生调用及坐标、符号和缩放转换；共享数值证据算法位于独立的
`_core/dual_bounds` 模块，诊断层负责构造辅助问题并解释证据。新增后端无需复制下界算法。
统一的是证据标准和失败语义，不保证不同后端产生相同乘子、松弛分配或同样完整的结论。

Factor-QCQP 的 LP 预筛默认关闭：

```python
from optim import PortfolioOptimizer, SolverPolicy

optimizer = PortfolioOptimizer(
    SolverPolicy(lp_prescreen=True)
)
```

只有当 LP 全局最优点同时满足 TE 预算时，LP 结果才是 Factor-QCQP 的严格全局最优解。若历史
通过率未知，预筛会增加一次 LP 成本，因此必须由用户显式开启。

---

## 19. 性能原则

组合优化通常会在十年回测中运行约 2,500 次，前端处理同样属于关键路径：

- tuda2 按完整区间、每种数据类型一次取足；
- 每个日期在昂贵求解前完成严格预检；
- 多期全区间预检完成后，逐日不重复完整静态校验；
- 日循环的求解路径不执行 pandas merge/groupby 或外部 I/O；
- NumPy/CSC 数值结构进入求解路径后不转换回表格；
- 多期默认保存稀疏权重，避免输出完整的 `date × universe` dense 历史；
- 深度诊断只对选定问题手工执行。

编译器会根据最终有效资产边界与固定预算消除已证明冗余的 L1 约束，而不是根据某次解上
约束不活跃就删除它。只做多的总主动权重采用直接生成的紧凑布局：利用
$\lVert w-b\rVert_1=B-\sum_i b_i+2\sum_i(b_i-w_i)_+$，只对有效区间跨过基准的资产
分配辅助列；卖空暂保留完整表达。此过程不消除固定资产本身，诊断仍保留完整约束。
换手率、总主动权重和基准覆盖率均从最终持仓重新验收，包括已经省略冗余行的约束。
有关实现对照与测试口径见 [紧凑 L1 实施报告](docs/compact_l1_implementation_20260909.md)。

开发机参考结果，不能替代目标生产机基准：

| 场景 | 35 日中位数 |
|---|---:|
| LP / HiGHS | 约 0.098 s/日 |
| 风险惩罚 QP / PIQP | 约 0.095 s/日 |
| 历史 2% Factor-QCQP / 已移除的 PIQP 搜索 | 约 0.258 s/日（非当前路线） |

5200 资产 × 47 因子的合成 Factor-QCQP，当前前置 `prepare()` 开发机中位数约 0.041 秒。
生产判断应使用相同数据、依赖版本和硬件上的 p50/p95/P99，并分别统计数据 I/O、准备、后端
求解、验收和结果输出。

详细性能记录见 [`docs/v5_unified_optimizer_real_benchmark.md`](docs/v5_unified_optimizer_real_benchmark.md)。

---

## 20. 当前能力边界

当前版本明确不包含：

1. `RiskAdjustedAlpha + TrackingErrorLimit` 的联合问题；
2. `FullCovarianceRiskModel` 的生产编译路径；
3. 任意通用 SOCP/锥约束建模接口；
4. 近似 alpha 最优集合内最小化 $\lVert x-x_0\rVert_2$ 的二阶段持仓稳定性目标；
5. 多期未来日期的动态黑名单/冻结名单声明；
6. 自动使用前一个日期补齐风险、基准或 alpha；
7. GPU 路径。

遇到未实现的模型组合会在求解前明确报错，不会静默改变数学问题。

---

## 21. 构建与发布

发布 wheel 不包含 `_core` / `_impl` 的算法 docstring；源码仍保留审阅说明，公共 Python
API 的参数、单位与使用文档不受影响。AI 知识库保留后端选择和诊断使用方法，不包含内部
表达变换细节。README 会进入 wheel 元数据，因此 README 本身不是保密载体。

项目采用公共 Python facade 和选择性 Cython 数值核心，只发布平台 wheel，不发布 sdist。

版本由 Git tag 和 `setuptools-scm` 自动生成：

```bash
git tag v3.0.1
./build.sh
```

正式上传：

```bash
./build.sh --push -r local
```

发布约束：

- tag 必须严格为 `vX.Y.Z`；
- `--push` 要求已跟踪工作树和暂存区干净；
- `HEAD` 必须正好具有发布 tag；
- wheel 版本、`optim.__version__` 和包内 `LIBRARY.toml` 必须一致；
- wheel 不得包含数值核心 `.py/.pyi/.c/.cpp` 实现文件；
- LP、QP 和 Factor-QCQP 安装后 smoke 必须通过。

tag 之间的本地构建会自动生成带提交距离和 commit id 的开发版本。

详细发布规则见 [`docs/core_wheel_distribution.md`](docs/core_wheel_distribution.md)。

---

## 22. 进一步文档

- [当前实现架构](docs/current_implementation_architecture.md)
- [统一接口设计](docs/portfolio_optimizer_api_design.md)
- [当前 API 使用与审计示例](docs/current_api_usage_audit_example.md)
- [前端性能契约](docs/frontend_performance_contract.md)
- [求解器后端决策](docs/solver_backend_decision.md)
- [Factor-QCQP / PIQP benchmark](docs/factor_model_qcqp_piqp_benchmark.md)
- [不可行与模块校验审计](docs/current_module_validation_audit.md)
- [构建与 wheel 分发](docs/core_wheel_distribution.md)

安装后的包还包含适合离线知识抽取的：

```text
optim/LIBRARY.toml
optim/references/api_overview.md
optim/references/recipes.md
optim/references/gotchas.md
```

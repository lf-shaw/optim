# 统一组合优化器架构与接口设计

> 本文保留目标态设计与已冻结的产品 policy。当前代码已经实现到哪里、实际调用链和仍未接线
> 的 contract，请以 [`current_implementation_architecture.md`](current_implementation_architecture.md)
> 为准。

状态：contract 已冻结，核心实现进行中。本文描述目标架构，不要求沿用当前
`opt/linopt/Solver` 实现；已落地范围和漂移检查以 `implementation_status.md` 为准。

## 1. 设计结论

新版不再公开区分 `opt` 与 `linopt`，统一提供一个 `PortfolioOptimizer`。调用方描述
投资目标、约束和数据，系统自动识别 LP、QP 或 factor-QCQP，并按
[后端定型方案](solver_backend_decision.md) 路由求解器。

前端本身也是性能关键路径。数据对齐、canonical compilation、验证和结果转换遵守
[前端性能 contract](frontend_performance_contract.md)，不能因为增加抽象层而抵消
HiGHS/direct PIQP 的性能收益。

数据访问与数学求解分层：

- `optim.core` 接受已经标准化的内存数据，不依赖 tuda2；
- `optim.integrations.tuda2` 批量获取风险模型、基准、股票池和收益率，再转换为同一个
  核心数据协议；
- 自动取数和手动传数最终都生成相同的 `PortfolioProblem`，因此不会形成两套求解逻辑。

```text
手动 DataFrame/ndarray              tuda2
          |                           |
          v                           v
  InMemoryDataSource           Tuda2DataSource
          |                           |
          +------------+--------------+
                       v
              标准 PortfolioData
                       |
        objective + constraints + policies
                       v
              PortfolioProblem compiler
                       |
             canonical sparse problem
                       |
        HiGHS / PIQP / MOSEK / Clarabel
                       |
       OptimizationResult + Diagnostics
```

## 2. 当前模块需要替换的设计

当前实现可以作为业务语义参考，但不适合继续叠加新后端：

1. `optim.__init__` 导入模块时立即检查 MOSEK license，使免费后端和纯数据功能也依赖
   商业环境。
2. `Solver` 同时负责数学建模、MOSEK Fusion 对象和状态判断，无法复用同一模型到
   HiGHS、PIQP 与 Clarabel。
3. `set_benchmark()` 必须先于若干主动约束调用，公共 API 存在隐式顺序依赖。
4. `opt.py` 与 `linopt.py` 重复实现数据校验、tuda2 取数、日期循环、约束组装和持仓推进，
   但命名、默认值和异常策略不同。
5. 单期函数只返回权重；solver 状态、fallback、残差、风险、对齐来源和耗时全部丢失。
6. 多期函数会捕获异常、静默跳日或自动放宽换手率，调用方很难区分数据问题、真实不可行
   和数值失败。
7. 风险单位、alpha 缺失处理和日期 forward-fill 没有形成显式数据契约。
8. 现有代码还存在应由迁移回归测试覆盖的实现风险，例如 scalar `asset_ub` 分支没有实际
   添加约束、`x0` 维数校验引用了 `mu.ndim`、定制单股固定权重分支引用错误变量，以及
   `10e-5` 与期望 `1e-5` 不一致。

因此建议新建核心实现，通过兼容层调用新接口；不在旧 `Solver` 上逐步增加 backend 分支。

## 3. 统一问题模型

### 3.1 数据对象

单日核心数据建议使用不可变对象：

```python
@dataclass(frozen=True)
class FactorRiskModel:
    asof: pd.Timestamp
    exposure: np.ndarray                 # (n_assets, n_factors)
    covariance: np.ndarray               # (n_factors, n_factors)
    specific_volatility: np.ndarray       # (n_assets,)
    factor_names: tuple[str, ...]
    factor_types: tuple[str, ...]
    provenance: DataProvenance
    annualization: str = "annualized_decimal"


@dataclass(frozen=True)
class FullCovarianceRiskModel:
    asof: pd.Timestamp
    covariance: np.ndarray               # (n_assets, n_assets)
    provenance: DataProvenance
    annualization: str = "annualized_decimal"


@dataclass(frozen=True)
class PortfolioData:
    date: pd.Timestamp
    assets: pd.Index
    alpha: np.ndarray | None              # (n_assets,)
    alpha_spec: AlphaSpec | None
    benchmark: np.ndarray | None          # (n_assets,)
    initial_weight: np.ndarray | None     # (n_assets,)
    tradable: np.ndarray                  # (n_assets,)
    risk_model: FactorRiskModel | FullCovarianceRiskModel | None
    extra_attributes: Mapping[str, np.ndarray]
    provenance: DataProvenance
```

`assets[i]` 是所有资产维数组唯一的 positional coordinate；输入适配层完成 sid 对齐后才能
构造 `PortfolioData`。任何后端不得各自重新 merge/reindex。除上面写明的 shape 外，
`extra_attributes[name]` 也必须为 `(n_assets,)`，factor name 数量必须等于 exposure 和
covariance 的 factor 维。所有数组应只读或按 immutable contract 使用。

风险数据收进独立 `RiskModel`，避免 `PortfolioData` 随风险模型种类不断膨胀。第一版只需
完整支持 `FactorRiskModel`；`FullCovarianceRiskModel` 可以只冻结协议和显式报 unsupported，
不能错误地落入 factor-QCQP 快速路径。

核心层统一使用：

- 权重和 alpha 为 decimal/score 原始值；
- factor covariance 为年化 decimal 协方差；
- specific volatility 为年化 decimal 波动率；
- TE 的公共参数和结果均为年化 decimal，例如 `0.02` 表示 2%。

tuda2 当前输出已经符合上述风险单位。手动入口也应转换成这个协议；对于历史 DataYes
百分数数据必须显式声明单位后转换，禁止根据数值大小猜测单位。

`alpha` 是按目标函数条件必填：

- `MaximizeAlpha`、`RiskAdjustedAlpha` 或带 alpha floor 的问题必须提供，并且当日所有
  优化资产的 alpha 都必须有限；
- 纯 `MinimizeTrackingError`/`MinimizeRisk` 问题允许 `alpha=None`；
- 编译器根据 objective 验证字段，`PortfolioData` 本身不强制所有问题都有 alpha。

多期入口继续支持以 `(dt, sid)` MultiIndex universe 同时表达优化日、资产池和 alpha：

```python
@dataclass(frozen=True)
class PortfolioSchedule:
    universe: pd.DataFrame  # index=(dt, sid), columns include member/tradable
```

当 objective 需要 alpha 时，它可以直接来自 `universe["alpha"]`，无需拆成另一份参数。
这也是月度、不规则调仓策略的推荐手动入口。

### 3.2 目标函数

```python
MaximizeAlpha()

RiskAdjustedAlpha(
    factor_aversion=0.75,
    specific_aversion=0.75,
)

MinimizeTrackingError(alpha_floor=None)  # 后续扩展
```

风险预算不是 LP 的隐含属性，而是独立约束：

```python
TrackingErrorLimit(annualized=0.02)
```

自动分类规则：

| 目标 | TE 约束 | 编译结果 |
|---|---|---|
| `MaximizeAlpha` | 无 | LP |
| `RiskAdjustedAlpha` | 无 | QP |
| `MinimizeTrackingError` | 无 | QP |
| `MaximizeAlpha` | 有 | factor-QCQP |

二阶段持仓稳定性目标可表示为
`MinimizeDistance(reference="initial", metric="l2")`，但必须通过明确的 alpha floor 或
剩余 certificate 预算实现 lexicographic 语义，不能悄悄改变第一阶段目标。

设第一阶段最大化 alpha 后得到可行 incumbent 下界 `LB` 和已验证对偶上界 `UB`。若用户
允许相对真正最优值最多损失 `epsilon_select`，第二阶段应使用：

$$
\begin{aligned}
\min_x\quad & \lVert x-x_0\rVert_2^2 \\
\text{s.t.}\quad & \alpha^{\mathsf T}x\ge UB-\varepsilon_{\mathrm{select}}.
\end{aligned}
$$

只有当 $\varepsilon_{\mathrm{select}}\ge UB-LB$ 时，第一阶段 incumbent 才保证属于这个集合。
使用 $\alpha^{\mathsf T}x\ge LB-\varepsilon_{\mathrm{select}}$ 只能保证“不比当前 incumbent
差太多”，不能声称相对
未知严格最优值的 lexicographic 保证；API 应把这两种 policy 分开命名。

第二阶段不是 theta continuation。theta 用于计算第一阶段 frontier；distance 是第一阶段
近似最优集合内的显式选择规则。对于只有线性约束的第一阶段，加入 alpha floor 后第二阶段
是一个 PIQP 可直接求解的凸 QP；若仍有 TE 二次约束，第二阶段是凸 QCQP，不是单次 PIQP
调用，应走通用锥 fallback，或未来扩展专用 factor-QCQP strategy。第一版可冻结接口但默认
关闭，避免阻塞核心后端迁移。

#### Alpha 单位与目标最优性容差

`objective_gap_abs=1e-4` 不是无量纲的通用精度，也不天然表示收益率 1 bp。它约束的是
用户传入 alpha 单位下的组合目标差：

$$
\alpha^\star-\alpha^{\mathsf T}x
\le\varepsilon_{\mathrm{abs}}.
$$

因此它与 alpha 的量纲、时间口径和缩放直接相关。例如，把全部 alpha 乘以 100 不改变
精确最优组合，却会使固定的 `1e-4` 相对严格 100 倍；对于 decimal 收益率 alpha，
`1e-4` 才在相同时间口径下等于 1 bp，而对于 z-score alpha，它只是 `0.0001` 个 score。
给全部 alpha 加常数在满仓约束下也不改变问题，所以不应只用
`eps_rel * abs(alpha_optimal)` 定义相对误差，该表达不具有平移不变性。

公共接口不应长期暴露一个语义不明的裸 float，建议使用：

```python
AlphaSpec(
    units="standardized_score",
    scale=1.0,
)

ObjectiveTolerance(
    absolute=None,       # 原始 alpha 单位；需要调用方明确选择
    normalized=1e-4,     # 相对于 AlphaSpec.scale
)
```

验收条件为：

$$
g_{\mathrm{raw}}
\le\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{normalized}}s_\alpha.
$$

其中 `alpha_scale` 必须是显式给定或平移不变的尺度，例如约定标准化 score 的 `1.0`、
优化样本空间内的横截面标准差，或稳健离散尺度。不能根据 alpha 最大绝对值自动赋予业务
含义。当前应用的 alpha 通常已经按 $(z-\mu)/\sigma$ 标准化，因而
`scale=1.0, normalized=1e-4` 近似保持现有 `objective_gap_abs=1e-4` 的验收语义；如果标准化
发生在另一个样本空间，调用方应传入实际约定的 scale，而不是由优化器再次偷偷标准化。

近似常数 alpha 的 scale 需要单独处理：在满仓问题中常数部分不影响选择；scale 低于阈值
时应把问题识别为目标退化，而不是直接除以接近零的数。

必须在接口和结果中区分：

- `AlphaSpec.scale` / `ObjectiveTolerance`：业务层目标近似程度；
- `alpha_target`：factor-QP 内部改善数值条件的缩放，不改变业务容差；
- PIQP/Clarabel 的 `eps_abs`、`eps_rel`：KKT 数值停止条件，不是 alpha 最优性保证。

为兼容现有 benchmark，可继续保留 `objective_gap_abs=1e-4`，但应标记为“原始输入 alpha
单位”，只对已明确采用标准化 score 的调用设置默认值。任意手动 alpha 输入应要求提供
`AlphaSpec` 或显式接受原始单位容差。`alpha=None` 的最小 TE 问题不适用该 certificate。

### 3.3 约束对象

建议把当前长参数列表合并为有类型的配置，同时保留按字段覆盖能力：

```python
constraints = PortfolioConstraints(
    long_only=True,
    budget=1.0,
    asset_weight=WeightBounds(upper=0.01),
    active_weight=SymmetricBound(0.004),
    total_active=1.8,
    turnover=TurnoverLimit(0.05, convention="two_way"),
    benchmark_member_weight=LowerBound(0.81),
    style=ExposureBounds(
        default=(-0.6, 0.6),
        overrides={"size": (-0.3, 0.3)},
    ),
    industry=ExposureBounds(default=(-0.05, 0.05)),
    tracking_error=TrackingErrorLimit(0.02),
)
```

blacklist、freeze、only-sell、单股固定/区间权重最终都编译为统一的资产上下限，并在
`ConstraintRegistry` 中保留来源和优先级。冲突必须在编译期报出，不能依赖调用顺序覆盖。

这些交易指令是单个 `PortfolioProblem` 的不可变输入，不是 `PortfolioOptimizer` 的状态。
实盘单期便利接口为：

```python
result = optimizer.optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
    blacklist=[...],
    frozen=[...],
    not_buyable=[...],
    not_sellable=[...],
    weight_overrides={"sid": (lower, upper)},
)
```

核心默认严格拒绝未知证券、同一证券出现在多类指令中，以及冻结/单边交易但缺少
`initial_weight`。旧 API 的 `freeze > blacklist > customized` 静默覆盖只允许留在未来的
legacy adapter 中；新核心不依赖调用顺序。强制 blacklist/freeze 与个股 active bound 冲突
时，交易指令只对该证券优先，并把原区间与豁免来源写入 registry；行业、风格、总主动和
风险等组合级约束仍然有效。

`optimize_range` 不允许把同一个非空一次性清单静态复制到所有日期。多期若需要每日不同的
清单，应输入带日期的 `PortfolioProblem` 序列；freeze 等清单不会被 optimizer 隐式延续。

## 4. 两种数据入口

### 4.1 手动入口：核心、稳定、无 tuda2 依赖

```python
from optim import PortfolioOptimizer, PortfolioProblem
from optim.data import PortfolioData

problem = PortfolioProblem(
    data=PortfolioData(...),
    objective=MaximizeAlpha(),
    constraints=constraints,
)

result = PortfolioOptimizer().solve(problem)
```

适合单元测试、离线文件、其他数据平台和上层系统已经完成对齐的场景。

多期手动入口允许保留原有 universe 用法：

```python
series = optimizer.optimize_range(
    universe=universe_with_alpha,  # MultiIndex (dt, sid) 决定调仓日
    benchmark=benchmark_frame,
    risk_data=InMemoryRiskModel(...),
    initial_weight=initial,
    objective=MaximizeAlpha(),
    constraints=constraints,
)
```

所有输入必须存在于 universe 指定的同一日期；手动入口不做前值替代。

### 4.2 tuda2 入口：便利适配层

```python
from optim import PortfolioOptimizer
from optim.integrations.tuda2 import Tuda2DataSource

optimizer = PortfolioOptimizer()
source = Tuda2DataSource(
    risk_model="datayes",
    benchmark_weight_type="daily",
)

result = source.optimize(
    optimizer,
    date="2025-04-30",
    universe=universe_with_alpha_for_one_date,
    benchmark_sid="000852.SH",
    initial_weight=previous_weight,
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
    blacklist=[...],
    frozen=[...],
)
```

月度/不规则调仓时建议直接传原有 universe：

```python
series = optimizer.optimize_range(
    universe=universe_with_alpha,   # dt 即唯一调仓日历
    benchmark="000852.SH",
    initial_weight=initial,
    objective=MaximizeAlpha(),
    constraints=constraints,
)
```

适配器严格使用 `universe` 中的日期作为 `dts` 参数，不扩展到其他风险模型日期。

`Tuda2DataSource` 应批量调用：

```python
tuda2.get_risk_model("exposure", ..., version="datayes")
tuda2.get_risk_model("cov", ..., version="datayes")
tuda2.get_risk_model("spec_risk", ..., version="datayes")
tuda2.get_risk_model_factor_names(...)
tuda2.get_index_weight(...)
tuda2.get_universe(...)
```

协方差必须按日期区间批量读取，再按 `(dt, factor)` 行索引选择同名列得到每日方阵，
不能在日循环中重复访问 tuda2。

alpha 通常是调用方的策略信号，因此既可直接传 `Series/DataFrame`，也可实现
`AlphaSource` 协议；不应把 alpha 获取硬编码进风险模型适配器。

如果调用方传入带 alpha 的 universe，tuda2 适配器只负责按 universe 的日期精确读取
风险模型、基准、交易状态和持仓漂移所需收益，不再要求单独传 alpha。

tuda2 必须延迟导入。没有安装 tuda2 时，手动入口和全部核心求解功能仍应正常工作。

tuda2 顶层接口读取本地 FizzDB，生产路径减少 I/O 次数：一个优化区间内 exposure、cov、
spec_risk、benchmark 分别一次取足，close-to-close 日收益也一次读取完整区间；同一批数据用于
全量预检和随后逐日物化，不在求解循环中重复访问 tuda2。因子名称元数据在 data source 内
缓存。风险模型规模由日期数、资产上限、因子数和 dtype 预先决定，核心接口不提供 chunk/LRU
分支；内存容量不足属于运行前资源 gate，不用重复 I/O 隐藏。

## 5. 日期、数据对齐和序列求解

### 5.1 对齐规则

默认 `AlignmentPolicy` 使用严格同日口径：

- 优化日由 universe `(dt, sid)` 中的唯一日期明确指定；
- t 日使用 t 日盘后生成的风险模型、指数权重和 alpha；
- alpha、benchmark、exposure、cov 和 spec_risk 都必须存在同一个 t 日；
- 任一必需数据缺失时禁止使用前一可用日期，也严禁未来数据；
- 缺失按显式 `error` 或 `skip_date` 处理，默认 `error`；
- 每个结果记录精确 source date、风险宇宙覆盖率和 benchmark 覆盖率，用于证明没有发生
  日期替代。

当 objective 需要 alpha 时，alpha 缺失默认直接报错，不再隐式 `fillna(0)`。纯最小化
TE 问题的 `alpha=None` 是合法模型，不属于数据缺失。

benchmark 逐日校验至少包括：原始权重和、非负/有限/重复 sid、与当日 universe 的覆盖
权重、缺失资产清单及最大单股缺失。默认 policy 为任何非零覆盖缺口都抛
`BenchmarkCoverageError`；只有调用方显式选择 `renormalize_within_tolerance` 时，才允许在
$m_{\mathrm{missing}}\le\varepsilon_{\mathrm{coverage}}$ 的日期将 universe 内剩余 benchmark 权重归一化为 1。
超过阈值始终报错，并在任何昂贵求解前终止。每个日期记录缺失权重、
缺失 sid、最大缺失单股权重和归一化因子，不能使用全区间平均覆盖率代替逐日检查。

该阈值使用独立字段 `benchmark_missing_mass_tolerance`，不能复用权重清理阈值或 solver
feasibility tolerance。调用方可以设为 0 取得严格全集覆盖，也可以显式选择更小/更大的业务
阈值；但不能配置超过阈值后仍静默归一化。

数据装配时需要区分“策略候选 universe”和“实际求解资产集合”。后者至少应覆盖当日
候选资产、benchmark 成分和自然漂移后的实际持仓。逐日分别计算：

```text
benchmark_missing_mass
holding_missing_mass
risk_model_missing_benchmark_mass
risk_model_missing_holding_mass
```

不能只报告证券数量重叠率，因为 1 只高权重成分的影响可能大于数百只零权重资产。
无法纳入求解集合的已有持仓不得静默丢弃；超过容差必须报错。对于仅需强制卖出的资产，
编译器可将其作为只参与初始持仓和换手率的退出变量，但若资产冻结/不能卖出，则必须具备
完整风险数据才能继续持有并计算 TE。

### 5.2 多期引擎

```python
series = optimizer.optimize_range(
    since="2025-01-01",
    until="2025-12-31",
    alpha=alpha_frame,
    benchmark="000852.SH",
    initial_weight=initial,
    state=SequencePolicy(
        mode="chained",
        holding_update="mark_to_market",
        on_infeasible="stop",
        turnover_recovery=None,
        theta_seed="previous",
    ),
    objective=MaximizeAlpha(),
    constraints=constraints,
)
```

连续多期优化强制使用 `mark_to_market`：上一期实际目标持仓先按两次调仓之间的复权收益
自然漂移、归一化，再作为本期 `initial_weight`。月度策略应计算整个调仓间区间的累计
收益，不能直接把上月目标权重原样带入本月。

满仓组合的基本更新为：

```text
pretrade_weight[t, i]
    = target_weight[previous, i] * (1 + holding_period_return[i])
      / sum_j(target_weight[previous, j] * (1 + holding_period_return[j]))
```

如果允许现金或 `budget < 1`，归一化分母必须包含现金头寸及其区间收益。t 日换手率相对
这个 `pretrade_weight[t]` 计算，而不是相对上一调仓日的目标权重。

`previous_target` 只允许用于明确标记的 benchmark/research 测试，不作为连续组合优化的
生产选项。优化器只保留 `close_to_close` 快照语义：风险模型、alpha、benchmark、调仓前
持仓和约束都定义在 t 日 close；公共 API 不提供 `open`、`next_open_to_next_open`、开盘
成交或滑点模式。缺失收益不能直接填 0：必须计算对应持仓质量，超过容差时报错。

执行时点和订单成交属于优化器外部。链式研究接口明确采用“上一目标组合已在上一 close
形成”的简化假设后做 close-to-close 漂移；实际交易系统若在下一开盘成交，应由独立账户/
执行模块跟踪真实成交持仓，并在下一次求解时把当日 close 的真实 `initial_weight` 传回
优化器。优化器不根据开盘价格推演或修正持仓。

求解失败、指数大调整跳日和自动放宽约束也必须是独立策略。默认不静默放松约束；诊断
只能解释问题，不能自动改变原问题。优化失败或跳日后，真实组合仍然存在，序列引擎必须
继续漂移上一实际持仓；不得像旧实现一样把持仓设为 `None` 并在下一日取消换手率约束。

### 5.3 Theta 传播

`theta_seed` 支持：

- `fixed`：每天从 `SolverTuning.theta_initial` 开始；
- `previous`：使用上一成功日最终 theta；多期链式默认；
- `predict`：时间序列外推，暂列实验功能；
- `auto`：链式解析成 `previous`，独立模式解析成 `fixed`。

建议公共默认写作 `auto`，并在结果元数据中记录实际解析值。跨日只传 theta 标量，每日
重新构建 workspace；当日 frontier 搜索内部才复用 workspace。发生跳日、问题配置变化、
theta 非法或没有历史成功值时自动重置到 `16384`。

固定 theta 与 previous theta 都必须满足相同的独立约束和 alpha certificate。该选项是
性能/搜索路径策略，不是持仓稳定性定义。近似最优集合内的持仓选择应由显式二阶段目标
解决。

### 5.4 求解前完整预检

默认执行两阶段流程：

```text
load + align + validate entire schedule
                |
                v
         PreparedPortfolioRun
                |
                v
       sequential solver execution
```

在建立 HiGHS/PIQP/MOSEK/Clarabel workspace 前，尽可能扫描完整调仓区间，汇总所有数据
错误，而不是遇到第一个错误立即停止。公共接口建议为：

```python
prepared = optimizer.prepare(request)       # 只取数、对齐、编译与预检
report = prepared.validation
report.raise_for_errors()

result = optimizer.solve_prepared(prepared)

# 或仅做生产数据验收
report = optimizer.validate(request)
```

`optimize/optimize_range` 内部默认执行同样的 `prepare()`，调用方不需要手动拆成两步。

完整预检至少包括：

1. **Schema/index**：输入类型、`(dt,sid)` 名称、唯一性、排序、调仓日非空、重复 sid。
2. **日期完整性**：每个调仓日的 alpha（目标需要时）、benchmark、exposure、cov、
   spec_risk 必须同日存在；调仓间收益区间完整。
3. **数值**：NaN、inf、权重负值、specific volatility 非负、矩阵维度和因子名一致。
4. **风险矩阵**：协方差逐日对称、有限并满足允许容差内的 PSD/Cholesky 条件；修复轻微
   数值负特征值必须是显式 policy 并记录，不能静默修改。
5. **逐日权重覆盖**：benchmark 原始权重和、benchmark/holding/risk-model missing mass、
   最大缺失单股权重和处理动作。
6. **目标字段**：例如 `MinimizeTrackingError` 允许 alpha 为空，而 `MaximizeAlpha` 不允许。
7. **静态约束**：上下限顺序、预算是否落在资产上下限总和内、active 约束是否有 benchmark、
   turnover 是否有初始持仓、因子约束 key 是否存在。
8. **编译结果**：canonical 矩阵维度、稀疏索引、上下界、目标系数全部有限。
9. **持仓漂移数据**：收益锚点、区间覆盖及缺失收益对应的潜在持仓资产；超过容差提前报错。

预检错误使用结构化 `ValidationIssue`，至少包含 `date/field/code/severity/message/context`，
并一次返回完整清单，便于生产数据批量修复。

部分检查无法在全区间开始前确定，例如未来实际持仓取决于前期优化结果。序列引擎必须在
每日 solver 调用前对漂移后的真实持仓再做轻量增量检查，但这不能替代全区间数据预检。

预检不应为每个 factor-QCQP 默认再解一个昂贵的通用锥问题。预算与 bounds、理论最小
强制换手等便宜条件可以前置；完整 Phase-I、最小 TE 和冲突松弛仍在真实不可行时由诊断
接口触发。

对 2500 日数据可按日期块扫描并生成磁盘/内存 `PreparedPortfolioRun`，避免一次复制全部
dense exposure，同时保证开始回测前已经验证所有调仓日的必要数据和风险矩阵。

`PreparedPortfolioRun` 正式采用 manifest/reference 设计，而不是 2500 个完整单日对象的
tuple：

```text
PreparedPortfolioRun
├── immutable schedule metadata
├── validation report
├── date -> RiskDataRef / AlphaDataRef / BenchmarkDataRef
├── content hashes and cache keys
└── per-day lazy materializer
```

预检可以流式读取每个日期块、完成 shape/PSD/覆盖率检查并保存摘要与内容 hash；求解时再
按日物化数组。只允许显式大小受控的 LRU cache，不能因“预检完成”而把全区间 dense risk
model 再复制一份常驻内存。

### 5.5 链式不可行恢复与指数调整

连续优化中，`skip` 不是通用恢复方案。指数调整可能使新 benchmark-relative 可行域与旧
持仓相距超过基础换手率上限：如果本日跳过，真实组合仍在旧可行域外，后续日期即使
benchmark 不再变化也可能持续不可行。

新版把这类情况识别为 `STRUCTURAL_TURNOVER_INFEASIBILITY`，并提供显式恢复策略：

```python
state = SequencePolicy(
    mode="chained",
    on_infeasible="recover_or_stop",
    turnover_recovery=TurnoverRecoveryPolicy(
        mode="minimum_feasible",
        max_turnover=0.25,           # 用户明确授权的硬上限
        buffer=1e-5,
        scope="current_rebalance",
        reset_next_period=True,
        require_turnover_only=True,
    ),
)
```

处理流程为：

```text
原问题不可行
    |
    v
诊断：移除 turnover 后是否可行？
    |
    +-- 否 -> 不是单纯换手率冲突，不自动放宽，输出诊断并停止/交由其他 policy
    |
    +-- 是 -> 求保持其他约束不变时的最小可实现 turnover
                  |
                  +-- 不超过原上限 -> 数值/其他原因，不能按换手率恢复
                  |
                  +-- 位于原上限与用户授权上限之间
                  |      -> effective limit = minimum turnover + buffer
                  |      -> 只重解当前调仓日
                  |      -> 下一调仓日恢复 configured limit
                  |
                  +-- 超过用户授权上限
                         -> 用户授权范围内无法桥接，停止并报告
```

这里必须求“最小必要换手率”，不能像旧实现一样按固定步长重复求解，也不应默认完全删除
换手率约束。这样既减少异常日重复求解，也能明确报告基础上限、最低需求、实际采用上限和
超出量。

对于 LP 和风险惩罚 QP，可用 HiGHS 在全部其他线性约束下最小化 L1 turnover，得到精确
的 `Tmin`。对于带 TE 预算的 factor-QCQP，线性 Phase-I 只能提供下界；最终 `Tmin` 必须
在保留 TE 约束的完整凸问题上求得，可在这个低频恢复路径使用 MOSEK/Clarabel SOCP，或
将同一结构化 frontier 编译器扩展为最小 turnover 目标。

诊断结果必须分别保存 `linear_lower_bound`、`convex_minimum` 和各自 certificate。如果
线性下界已经高于配置上限，可以直接证明原问题违反换手率约束并跳过昂贵的完整 QCQP
诊断；如果线性下界没有超限，则不能据此宣称含 TE 的问题可行。

benchmark 本身的逐日 L1 变化可作为低成本预警：变化超过阈值时，在主 alpha 优化前主动
运行最小 turnover 检查，避免先支付一次必然失败的昂贵完整 QCQP。但 benchmark turnover 只
是触发器，不是 `Tmin`，二者不能直接等同。

自动恢复只允许修改用户显式授权的 turnover。风格、行业、active weight、TE、预算等
投资约束不随之自动放宽。若 `max_turnover` 仍低于 `Tmin`，要分多日逐步迁移就必须同时
暂时放宽 benchmark-relative 约束；这是另一种投资政策，不能由 solver 自行决定。

每次恢复在结果中记录：

```text
configured_turnover_limit
minimum_feasible_turnover
effective_turnover_limit
benchmark_turnover
trigger/reason
changed_constraints
recovery_solver/attempts
```

没有启用 `TurnoverRecoveryPolicy` 时，结构性换手率不可行的默认行为应是诊断后停止该链，
而不是在后续每个日期反复执行昂贵求解。若用户选择继续持有，序列引擎仍维护并漂移真实
组合，但应进入明确的 `BLOCKED_BY_HARD_TURNOVER` 状态，直到 benchmark/约束改变或用户
提供新的恢复授权。

## 6. 后端与编译器分层

建议目录职责如下：

```text
optim/
  api.py                    # PortfolioOptimizer 公共入口
  types.py                  # request/result/status
  data/
    contracts.py            # PortfolioData 与 source protocols
    alignment.py            # 严格同日对齐、覆盖率与审计
    memory.py               # 手动数据适配
  integrations/
    tuda2.py                # 可选 tuda2 适配器
  model/
    objectives.py
    constraints.py
    compiler.py             # 命名约束 -> canonical sparse model
    risk.py                 # factor risk 与单位
  backends/
    base.py
    router.py
    highs.py
    piqp.py
    mosek.py
    clarabel.py
  engine/
    single.py
    sequence.py
    validate.py
    diagnostics.py
  compat/
    linopt.py
    opt.py
```

编译器一次生成带名字的 canonical model。后端只负责把 canonical model 转成自己的矩阵
或锥形式并求解，不再暴露 `set_*` 调用顺序。

`CanonicalModel` 不是强行用一套 `P/q/A/l/u` 表达所有问题，而是一个有类型的不可变
联合：

```python
CanonicalModel = (
    LinearProgram
    | QuadraticProgram       # P, q, A, lower, upper
    | FactorQCQP             # linear domain + FactorRiskOperator + risk bound
    | ConicProgram           # explicit cone blocks
)
```

其中 `P/q/A/l/u` 只覆盖 LP/QP。`FactorQCQP` 保留 low-rank + diagonal 风险算子，不能先
展开成资产数维 dense covariance；`ConicProgram` 用于真正的通用 SOCP fallback。每个
canonical constraint 都携带 `ConstraintRegistry` id，使后端 dual、残差和诊断能映射回
业务约束。

还需要区分“求解策略”和“单次后端”：

```text
LPStrategy                -> one HiGHS backend call
QPStrategy                -> one PIQP backend call
FactorQCQPFrontierStrategy-> multiple parametric QP backend calls
                              -> MOSEK/Clarabel conic fallback
GenericConicStrategy      -> one MOSEK/Clarabel backend call
```

PIQP 因而是 QP backend，不是 SOCP backend。factor-QCQP 的专用性属于
`FactorQCQPFrontierStrategy` 的 reformulation、theta bracket/refinement 和 certificate。
底层协议保持很薄：

```python
class SolverBackend(Protocol):
    def solve(
        self,
        model: BackendModel,
        options: BackendOptions,
    ) -> BackendResult: ...
```

workspace setup/update/rebuild、PIQP 生命周期恢复和原生状态转换全部封装在 backend 内部；
上层 strategy/router 不调用 `backend.update()` 等状态方法。

### 6.1 Problem fingerprint 与可审计 fallback

建议把评审提出的 `ProblemFingerprint` 拆成两个稳定 hash，避免把数学问题与运行路线混在
一起：

```python
@dataclass(frozen=True)
class ProblemFingerprint:
    semantic_hash: str       # 对齐后的数据、目标、约束、单位、初始持仓
    canonical_hash: str      # 标准化稀疏模型、risk operator、registry
    compiler_version: str

@dataclass(frozen=True)
class RunFingerprint:
    problem: ProblemFingerprint
    solver_policy_hash: str  # backend 次序、容差、theta/retry policy
    package_versions: Mapping[str, str]
```

primary 和 fallback 的 `semantic_hash` 必须相同。不同 backend 的 conic/QP payload 表示可
不同，因此每个 attempt 另存 `backend_payload_hash`，不能要求它们与 canonical hash 字节
相同。hash 使用确定性的稀疏索引排序、固定 dtype/endianness 和加密内容摘要；禁止使用进程
相关的 Python `hash()`。任何 recovery 真正修改 turnover 或其他约束时，必须生成新的
problem fingerprint，并在结果中以 `derived_from` 连接原问题。

旧 `optim.linopt.optimize/multioptimize` 和 `optim.opt.multioptimize` 可在过渡期变成兼容
包装器，将旧参数翻译成新对象并发出 deprecation warning。核心实现只保留一份。

## 7. 统一结果协议

```python
@dataclass(frozen=True)
class OptimizationResult:
    status: SolveStatus
    weights: pd.Series | None
    objective_value: float | None
    backend: str | None
    route: tuple[SolverAttempt, ...]
    metrics: PortfolioMetrics
    certificate: OptimalityCertificate | None
    violations: tuple[ConstraintViolation, ...]
    alignment: AlignmentReport
    diagnostics: InfeasibilityReport | None
    timings: SolveTimings
    fingerprint: ProblemFingerprint
```

状态至少区分：

```text
OPTIMAL
OPTIMAL_INACCURATE
INFEASIBLE
UNBOUNDED
LIMIT_REACHED
NUMERICAL_ERROR
RESOURCE_ERROR
SOLVER_ERROR
SKIPPED
```

只要问题已经通过 prepare/compiler，`solve()` 对数学或求解器结局统一返回结构化结果，
不能再把所有非成功状态都压缩成 `ProblemInfeasible`。`weights` 只在可接受成功状态下
存在。希望立即取得权重或抛异常的单日调用方使用 `result.require_weights()`，不再给
`solve()` 增加改变基本返回协议的 `raise_on_failure` 分支。

返回失败状态主要用于：多期引擎按 `SequencePolicy` 决定停止/保持真实持仓/显式恢复；记录
fallback 已耗尽、超时或数值失败的完整 route；以及稍后只挑某个不可行日期运行 deep
诊断。它不是用来吞掉调用错误。非法参数、schema/date/shape/单位错误和编译失败应在
`prepare()` 抛带完整 `ValidationReport` 的异常，正常情况下不进入 solver result。

`INFEASIBLE/UNBOUNDED` 是数学模型结论；`LIMIT_REACHED/NUMERICAL_ERROR/RESOURCE_ERROR`
是求解器失败；非法模型和数据错误不属于 `OptimizationResult`，由 prepare/compiler
异常及 `ValidationReport` 表达。每个 `SolverAttempt` 同时记录统一 reason、原生 status、
是否完成独立确认：

```text
UPDATE_FAILURE
MAX_ITER
TIME_LIMIT
INVALID_NUMERICS
NUMERICAL_FAILURE
INFEASIBLE_REPORTED
INFEASIBLE_CONFIRMED
UNBOUNDED_REPORTED
UNBOUNDED_CONFIRMED
```

PIQP 报 infeasible 时可以转独立 Phase-I 或 MOSEK/Clarabel 确认，但这属于“模型状态
确认”，不是把所有 infeasible 都当成普通数值失败机械重试。确认结果及不一致必须进入
route 和 diagnostics。

### 7.1 OptimalityCertificate contract

```python
@dataclass(frozen=True)
class OptimalityCertificate:
    kind: str
    proof_status: str             # VERIFIED / NUMERICAL_ESTIMATE / UNAVAILABLE
    primal_value: float
    dual_bound: float | None
    absolute_gap: float | None
    normalized_gap: float | None
    objective_units: str
    objective_scale: float | None
    components: Mapping[str, float]
```

最大化问题使用 feasible primal value 作为下界、dual bound 作为上界；最小化问题方向相反。
certificate 由 strategy/backend 提供必要原始 dual 信息，统一 validator 在返回前重新计算
原始单位的 objective、约束和 bound arithmetic。validator 不能仅从一组权重凭空证明全局
最优，但不得信任 backend 字符串状态代替独立复算。

PIQP factor frontier 的 certificate 至少把风险边界松弛、参数化 QP 子问题 gap 和输出
清理/投影影响分项记录；具体公式见后端定型文档。只满足约束而没有有效 dual bound 时，
可以返回 feasible result，但不能把 certificate 标为 `VERIFIED`。

## 8. 不可行性诊断接口

### 8.1 公共接口

```python
result = optimizer.solve(problem)  # 默认不运行额外不可行诊断

if result.status is SolveStatus.INFEASIBLE:
    # 只对选中的这个问题执行可能较慢的诊断
    report = optimizer.diagnose(
        problem,
        prior_result=result,
        level="deep",
    )

# 多期结果也可按日期定位到同一个 prepared problem
report = series.diagnose(date="2025-06-16", level="deep")
```

`solve()` 不提供自动 diagnostics 选项。solver 原生状态、route 和求解过程中已经产生的
残差仍正常记录，但 Phase-I、逐行冲突、最小 turnover/TE 和替代求解只由显式
`diagnose(..., level="deep")` 触发。

`diagnose()` 必须校验 prior result 与 problem/prepared problem 的 fingerprint 一致，避免
重新装配数据后诊断了另一个问题。不自动诊断不影响返回前必要的解可行性验证；validator
与 infeasibility diagnostics 是两套成本和目的不同的流程。

### 8.2 约束登记表

所有编译行都必须带稳定元数据：

```text
constraint_id
group                 budget/turnover/asset/active/style/industry/risk/...
key                   sid、因子或属性名
side                  lower/upper/equality
bound
unit
source                 用户、tradable、blacklist、benchmark 等
relaxable
diagnostic_scale
```

这是跨后端统一解释的基础，也使 solver 原生证书可以映射回业务约束。

### 8.3 分层诊断算法

标准诊断不依赖某个求解器是否提供 IIS：

1. **静态检查**：NaN、维度、重复证券、`lb > ub`、预算与资产上下限矛盾、缺基准却有
   主动约束等。
2. **线性 Phase-I**：给约束组引入带尺度的非负 slack，用 HiGHS 最小化总/最大松弛，
   找出必须放松的约束组。
3. **逐行定位**：只在可疑约束组内增加逐行 slack，报告具体 sid、风格因子或行业及最小
   所需放松量。
4. **换手率专项**：在其他线性约束下求最小可实现双边换手率。如果最小值为 8.2%、用户
   上限为 5%，直接报告至少还需 3.2 个百分点。这能够解释指数调整日的真实冲突。
5. **风险预算专项**：先确认线性可行；再以 factor-model 风险为目标求最小可实现 TE。
   如果最小 TE 为 2.37%、预算为 2%，报告风险预算至少需增加 0.37 个百分点。
6. **原生证据补充**：附加 HiGHS/MOSEK 能提供的 ray、certificate 或原生状态，但不把
   它作为统一接口的唯一来源。

Phase-I 的 slack 结果是“最小所需放松”和“冲突候选”，不应错误宣称为唯一业务原因。
不同归一化或优先级可能产生不同的等价最小冲突集，因此报告必须给出尺度和目标。

### 8.4 诊断结果

```python
@dataclass(frozen=True)
class InfeasibilityReport:
    stage: str
    linear_feasible: bool | None
    summary_text: str
    turnover_linear_lower_bound: float | None
    turnover_convex_minimum: float | None
    turnover_certificate: OptimalityCertificate | None
    turnover_limit: float | None
    minimum_tracking_error: float | None
    tracking_error_limit: float | None
    relaxations: tuple[RequiredRelaxation, ...]
    suspected_conflicts: tuple[ConstraintRef, ...]
    native_evidence: Mapping[str, Any]
    attempts: tuple[SolverAttempt, ...]
```

示例人类可读输出：

```text
问题的线性部分不可行。
在保持预算、行业、风格和个股上限不变时，最小双边换手率为 8.21%，
高于约束 5.00%，至少需要放宽 3.21 个百分点。
主要强制交易来自 37 个退出 benchmark/不可交易标的；详见 relaxations 表。
```

诊断与自动修复必须分开。如果用户希望放宽换手率或风险预算，应根据报告构造一个新的
`PortfolioProblem` 并再次求解，原结果中完整保留修改前后的问题指纹。

## 9. Solver policy

```python
policy = SolverPolicy(
    lp="highs",
    qp="piqp",
    factor_qcqp_strategy="frontier",
    factor_qcqp_subproblem_backend="piqp",
    licensed_fallback="mosek",
    free_fallback="clarabel_qdldl",
    lp_prescreen=False,
    validate_solution=True,
    rebuild_after_update_failure=True,
    repeat_failed_cold_solve=False,
    tuning=SolverTuning(
        alpha_target=0.2,
        theta_initial=16384,
        theta_growth=4.0,
        theta_max=1e12,
        max_outer_iters=30,
        intermediate_eps=1e-5,
        final_eps=1e-8,
        piqp_max_iter=1000,
        piqp_inequality_form="auto",
        polish=True,
    ),
    objective_tolerance=ObjectiveTolerance(
        normalized=1e-4,
        absolute=None,
    ),
)
```

普通接口只暴露带 `AlphaSpec` 语义的 `objective_tolerance`、超时和 theta policy。兼容层
仍可接受 `objective_gap_abs`，但必须注明它使用原始 alpha 单位。其余参数属于高级调优
对象，默认值来自当前生产测试，不应散落在 `optimize()` 的长参数列表中。

license 探测只发生在需要 fallback 时。找不到 MOSEK 包、环境变量、默认 license 目录或
有效授权时，路由器记录原因后使用 Clarabel；不能导致 `import optim` 失败。

## 10. 依赖与发布

建议将依赖拆分为 extras：

```text
optim                 NumPy/SciPy/pandas/HiGHS/受控 PIQP build
optim[clarabel]       Clarabel fallback
optim[mosek]          MOSEK fallback
optim[tuda2]          tuda2 数据集成；数据对齐工具链由该集成层独立维护
optim[all]            上述全部
```

当前 patched PIQP wheel 需要形成受控的内部构建/发布来源，并在运行元数据中记录完整版本
和补丁标记。不能依赖用户恰好安装某个同名官方 wheel。

## 11. 迁移顺序

1. 先建立 contract/准确性测试和前端分阶段计时、RSS benchmark；保留当前实现作为整链
   性能基线，避免重构完成后才发现包装开销。
2. 将 benchmark 中已经验证的 factor-QP、约束复算和路由逻辑迁入正式模块，并建立
   canonical problem 与 `ConstraintRegistry`。
3. 实现统一单日 `PortfolioProblem -> OptimizationResult`，覆盖 LP/QP/factor-QCQP。
4. 实现标准不可行诊断：静态检查、HiGHS Phase-I、最小换手率和最小 TE。
5. 实现 `Tuda2DataSource`、批量缓存和 provenance，并对接数据工具链提供的对齐 contract；
   pandas/carry/carry2 的算法选型与性能由对应项目维护，optim 日循环只消费 offsets/arrays。
6. 实现链式/冷启动多期引擎和显式持仓推进策略。
7. 将旧 `linopt/opt` 改为兼容包装器，完成准确性、性能和内存回归后逐步弃用。

## 12. 实现 gate：冻结的八个 contract

正式实现不再继续扩展顶层概念，先冻结并分别建立 contract test：

1. **`PortfolioData`**：唯一资产坐标、shape、风险/alpha 单位、只读语义和 provenance。
2. **`PortfolioProblem`**：`data + objective + constraints`；solver policy 不属于数学模型。
3. **`CanonicalModel`**：LP/QP/factor-QCQP/conic 的 tagged union、变量/约束 registry。
4. **`SolverBackend`**：单次不可变 model 输入和结构化 result 输出；workspace 生命周期内部化。
5. **`OptimizationResult`**：状态、route、metrics、violations、timings 和原始单位结果。
6. **`ProblemFingerprint`**：semantic/canonical/run hash 分层，fallback 与 recovery 可审计。
7. **`OptimalityCertificate`**：primal/dual bound、gap 组成、单位和 proof status。
8. **`SequencePolicy`**：持仓推进、theta seed、失败处理和 turnover recovery，不进入 backend。

第一批实现以这些对象及其序列化/相等性/validation contract 为起点，再迁移求解代码。
尤其在完整 QP subproblem gap 尚未纳入前，不把 benchmark 的
`frontier_gap_bound_abs` 对外标成 `VERIFIED` optimality certificate。

以上 contract 之外还有一个横切 gate：每个迁移阶段必须通过前端整链性能与 peak RSS
回归，不能只报告 solver time。日循环出现 pandas merge/reindex、重复 canonicalization
或 fallback 重新取数/对齐，均视为架构回归。

## 13. 已冻结的产品 policy

1. benchmark 与逐日 universe 的任何 missing mass 默认报错；仅在调用方显式选择
   `renormalize_within_tolerance` 后才于配置阈值内归一化并审计，超过阈值始终报错。阈值
   独立配置，不按全区间平均。
2. 已编译问题的数学/solver 失败返回 `OptimizationResult.status`；数据、参数和非法模型在
   prepare/compiler 抛异常。需要异常式取权重时调用 `result.require_weights()`。
3. 优化器只支持 close snapshot/close-to-close 持仓语义，移除全部 open execution 模式。
4. 自动不可行诊断默认关闭；用户针对某个 fingerprint/date 显式运行 `deep`。
5. 第一版只保留二阶段持仓稳定性 contract 和扩展位置，不阻塞后端统一实现。
6. turnover recovery 对所有策略默认关闭；只有用户显式提供
   `TurnoverRecoveryPolicy(max_turnover=...)` 后才允许修改原换手率约束，指数跟踪模板也不
   自动代替用户授权。

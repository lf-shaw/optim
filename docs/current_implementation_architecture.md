# 统一组合优化器当前实现架构

更新日期：2026-08-30。

本文面向代码审阅，描述仓库当前实现，而不是未来目标态。后端选择依据见
[`solver_backend_decision.md`](solver_backend_decision.md)，接口最初设计见
[`portfolio_optimizer_api_design.md`](portfolio_optimizer_api_design.md)，实现进度与未完成项见
[`implementation_status.md`](implementation_status.md)。当三者不一致时，本文以当前源码行为为准。

## 1. 当前实现的核心原则

当前实现围绕六个不变量组织：

1. 一个优化请求由不可变的 `PortfolioProblem(data, objective, constraints)` 完整描述，求解器
   对象不保存黑名单、冻结名单或其他可跨请求泄漏的业务状态。
2. 输入先经过不调用 solver 的聚合校验，再编译为 solver-independent canonical model。
3. LP、QP、单一 factor-model TE 约束分别走 HiGHS、direct PIQP、PIQP frontier strategy。
4. backend 报告 `solved` 后仍需由公共验收层复算约束、风险和目标；backend 状态不是直接的
   返回证书。
5. 普通求解失败返回结构化 `OptimizationResult`；输入/schema/模型定义错误在建立 backend
   之前抛异常。昂贵的不可行诊断不会自动运行。
6. 多期优化只使用 close-to-close 自然漂移；tuda2 按完整区间批量取数，日循环不做 I/O。

## 2. 分层与依赖方向

```text
用户 / 策略
    │
    ├── 手动 PortfolioData / PortfolioProblem
    │
    └── PortfolioSchedule + InMemoryDataSource / Tuda2DataSource
                         │
                         ▼
                PortfolioProblem
                         │
                         ▼
        validate_problem（便宜、聚合、无 solver）
                         │
                         ▼
         compile_problem（语义 → 稀疏 canonical model）
                         │
                         ▼
                  PortfolioOptimizer
                         │
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
       HiGHS LP       PIQP QP       factor-QCQP strategy
                                          │
                              PIQP 参数 QP / 可选 HiGHS LP
                         │
                         ▼
             evaluate_solution 独立验收
                         │
                  失败且允许 fallback
                         │
               MOSEK → Clarabel QDLDL
                         │
                         ▼
               OptimizationResult
```

模块责任如下：

| 层 | 主要模块 | 当前责任 |
|---|---|---|
| 公共 contract | `portfolio_types.py` | 数据、目标、约束、policy、状态、证书和结果对象 |
| 数据对齐 | `data/alignment.py`、`data/contracts.py`、`data/memory.py` | 严格同日 `(dt, sid)` 对齐、benchmark 覆盖、风险模型物化、序列预检 |
| 外部数据适配 | `integrations/tuda2.py` | 风险、benchmark、可交易性及 C2C 收益的批量 I/O |
| 静态校验 | `validation.py` | shape、单位、日期、有限性、PSD、约束组合和交易清单预检 |
| 单股域解析 | `asset_bounds.py` | 合并绝对/主动边界、黑名单、冻结、单边交易和非交易冻结 |
| canonical model | `model/canonical.py` | $l\le Az\le u$、变量边界、QP 矩阵、风险算子和审计 registry |
| 编译器 | `model/compiler.py` | 问题分类、辅助变量布局、稀疏约束与目标构造、fingerprint |
| backend | `backends/*` | 极薄的 HiGHS/PIQP/MOSEK/Clarabel 调用及状态标准化 |
| factor-QCQP | `factor_qcqp.py` | LP 严格预筛、参数 QP、theta 搜索、PIQP workspace 生命周期和 gap |
| 统一入口 | `api.py` | prepare、路由、fallback、独立验收、清理、证书与结果组装 |
| 结果验收 | `solution.py` | 辅助变量重建、全部 canonical 约束与业务指标复算 |
| 多期 | `sequence.py` | C2C 漂移、theta 传播、失败策略和显式换手恢复 |
| 显式诊断 | `diagnostics.py` | Phase-I、线性最小换手、最小 TE 诊断 |
| 审计 | `fingerprint.py` | semantic/canonical hash 与 compiler version |

旧的 `opt.py`、`linopt.py`、`solver.py` 仅由 `optim.__getattr__` 延迟加载。新版正常 import 和
求解路径不依赖它们。

## 3. 公共对象模型

### 3.1 `PortfolioData`

所有数组共用唯一 positional coordinate：`assets[i]` 对应 alpha、benchmark、initial weight、
tradable、specific volatility 和 exposure 第 `i` 行。核心 shape 为：

```text
alpha                  (n_assets,) 或 None
benchmark              (n_assets,) 或 None
initial_weight          (n_assets,) 或 None
tradable                (n_assets,)
exposure                (n_assets, n_factors)
factor covariance       (n_factors, n_factors)
specific volatility     (n_assets,)
```

风险公开单位固定为 annualized decimal：协方差为年化小数收益协方差，specific volatility 和
TE 为年化小数波动率。核心不会根据数值量级猜单位。

`alpha=None` 只适用于不需要 alpha 的目标，例如纯 `MinimizeTrackingError()`；最大化 alpha、
风险调整 alpha 以及 `alpha_floor` 都要求 alpha 和 `AlphaSpec`。

### 3.2 目标与问题分类

`classify_problem()` 当前行为为：

| 目标 | TE 约束 | 分类 | 当前可求解性 |
|---|---|---|---|
| `MaximizeAlpha` | 无 | LP | HiGHS |
| `MaximizeAlpha` | 有 | Factor-QCQP | PIQP frontier，失败后 MOSEK/Clarabel |
| `RiskAdjustedAlpha` | 无 | QP | direct PIQP，失败后 MOSEK/Clarabel |
| `RiskAdjustedAlpha` | 有 | CONIC | 已分类，但当前 compiler 明确报 unsupported |
| `MinimizeTrackingError` | 任意 | QP | direct PIQP |

`FullCovarianceRiskModel` 目前只是公共 contract；v1 compiler 只实现 factor risk model。

### 3.3 约束语义

- turnover 使用 $\lVert x-x_0\rVert_1$ 的双边口径；没有除以 2。
- active weight、风格、行业和 extra active 都相对 benchmark。
- `blacklist` 强制为零；`frozen` 和 non-tradable 固定在期初权重；`not_buyable`/`not_sellable`
  分别给出相对期初权重的单边边界。
- 操作性交易指令可对指定证券局部覆盖普通 active bound，但仍受 absolute/long-only domain
  约束；覆盖来源写入 constraint registry。
- 批量 `optimize_range()` 不允许把同一份非空 `asset_trade` 复制到所有日期。逐日清单需要
  显式构造逐日问题。

## 4. 数据入口与日期规则

### 4.1 手动单期入口

最底层入口是：

```python
optimizer.solve(PortfolioProblem(...))
```

`optimizer.optimize(...)` 只是单期便利 facade：它把 blacklist、frozen、not-buyable、
not-sellable 和 weight override 组装成当次问题的 `AssetTradeConstraints`，随后调用同一个
`solve()`，不保存任何名单状态。

### 4.2 已加载批量数据

`PortfolioSchedule` 要求 universe 使用名称严格为 `("dt", "sid")` 的 MultiIndex。
`InMemoryDataSource` 对 risk、benchmark、alpha 和 initial holdings 使用精确日期与精确证券
坐标，不会取前一个可用日期补齐。

benchmark 在 universe 外存在任何非容差内的质量缺口时默认报错。只有调用者显式选择
`renormalize_within_tolerance` 且缺口不超过独立阈值时才归一化，并把缺失质量和归一化倍数
写入 provenance/result alignment。

`prepare_run()` 会逐日物化并校验所有静态输入，聚合错误后返回轻量 manifest；它不会长期
保存每天的 dense NumPy 风险数组。求解时 `problem_at()` 再物化当日数组，因此当前实现用
一次额外的内存计算换取“昂贵序列开始前发现全部静态错误”。底层 pandas/tuda2 frame 已经
一次性载入，不发生逐日远程 I/O。

### 4.3 tuda2 入口

`Tuda2DataSource.load()` 对一个区间分别调用一次 exposure、covariance、specific risk 和
benchmark 接口。行业标签只展开一次。链式序列若没有手工提供持有期收益，则再一次性读取
日度 close-to-close return，并对相邻调仓日之间的区间复合：

```text
R(t-1, t) = product(1 + r_daily) - 1
```

缺失收益保持 NaN，由序列层按实际持仓质量判断是否可接受；不会在复合时静默跳过。

## 5. 校验、编译与 canonical model

### 5.1 两种校验不能混淆

`validate_problem()` 是求解前静态校验：不建立 backend，尽量一次返回所有独立输入问题。
它检查数组、日期、风险单位、PSD、目标依赖、factor 名称、资产边界、名单冲突等。

`evaluate_solution()` 是求解后动态验收：使用 backend 返回的完整 canonical vector，重新计算
$Az$、列边界、TE、目标、换手、主动权重和敞口。只有最大残差不超过
`feasibility_tolerance` 才允许返回权重。

### 5.2 统一线性域

所有问题共享线性域：

$$
\underline z\le z\le\overline z,
\qquad
l\le Az\le u.
$$

`z` 的前 `n_assets` 列始终是组合权重，之后按需增加 turnover epigraph、total-active
epigraph 和 factor-active 变量。`VariableRecord`/`ConstraintRecord` 把矩阵行列映射回业务
约束，是独立验收、fingerprint 和 deep diagnosis 的共同基础。

### 5.3 换手率的精确稀疏展开

一般情形使用每只证券一个绝对值辅助变量。若满足 long-only、期初权重非负且总和等于预算，
则利用满仓恒等式将 L1 换手精确写为：

若 $S=\{i:x_{0,i}>0\}$，$p_i\ge x_i-x_{0,i}$ 且 $p_i\ge0$，则

$$
\lVert x-x_0\rVert_1
=2\left(\sum_{i\in S}p_i+\sum_{j\notin S}x_j\right).
$$

因此只需为期初非零支持集建立辅助变量；原零持仓证券直接使用权重列。该优化记录为
`exact_sparse_turnover`，不是近似，也没有把 turnover 与 TE 建立关系。

### 5.4 factor QP

令主动权重 $a=x-b$，factor active exposure $f=E^{\mathsf T}a$。风险为：

$$
R(x)=f^{\mathsf T}Ff+\lVert d\odot a\rVert_2^2.
$$

QP 显式增加 $f$，通过等式 $f=E^{\mathsf T}(x-b)$ 连接权重。风格/行业约束随后直接约束
$f$ 的单列，不再重复一份 dense $E^{\mathsf T}x$，从而减小 KKT 稀疏结构。

风险调整 alpha 的编译目标在 budget 等式下对 alpha 做 benchmark-centered translation。
整体平移只改变 objective constant，不改变最优权重，使 PIQP scaling 依赖 alpha 横截面离散
程度，而不是 alpha 任意的绝对水平。

### 5.5 fingerprint

每个 compiled problem 有：

- `semantic_hash`：覆盖对齐后的业务输入、目标和约束，但排除 provenance；
- `canonical_hash`：覆盖稀疏矩阵、边界、目标、风险算子以及完整 registry；
- `compiler_version`：当数学编译语义改变时显式升级。

其用途是证明 fallback 和诊断处理的是同一个数学问题，而不是用于缓存求解结果。

## 6. 单期求解主流程

```text
prepare(problem)
    ├── validate_problem
    └── compile_problem（仅在 valid 时）
             │
solve_prepared
    ├── LP             -> HiGHS
    ├── QP             -> PIQP
    └── Factor-QCQP    -> factor frontier strategy
             │
    _audit_backend_result
        ├── evaluate_solution
        ├── 1e-5 小权重清理尝试
        ├── 清理后重建辅助变量
        └── 再次验收约束与 objective certificate
             │
    未通过且为 QP/QCQP
        ├── MOSEK（配置为 licensed fallback 时）
        └── Clarabel QDLDL（仍未通过时）
             │
    _result -> OptimizationResult
```

LP 当前没有 fallback。QP/QCQP fallback 每次都使用同一个 `CompiledProblem`；每个 backend
结果都单独经过相同的独立验收。

小权重清理阈值默认为 `1e-5`，即 0.1 bp 权重。清理流程先置零再按预算归一化，然后重建
turnover/factor 等辅助变量。如果破坏可行性或使已认证 objective gap 超限，则丢弃清理结果，
返回原始可行解。

## 7. Factor-QCQP strategy

原问题是：

$$
\begin{aligned}
\max_x\quad & \alpha^{\mathsf T}x \\
\text{s.t.}\quad & x\in\mathcal X, \\
& R(x)\le B^2.
\end{aligned}
$$

其中 `X` 是统一线性域，`R` 是 low-rank factor + diagonal specific risk。当前 strategy 增加
factor exposure 变量，把不同 theta 对应的问题写成共享 `P/A/bounds`、只更新线性项 `q` 的
参数 QP：

$$
\begin{aligned}
\min_x\quad & \frac12s_RR(x)-\theta s_\alpha\alpha^{\mathsf T}x \\
\text{s.t.}\quad & x\in\mathcal X.
\end{aligned}
$$

`risk_scale=10000` 只延续已经生产验证的 theta 数值尺度；公开 TE 和 certificate 仍回到
annualized decimal risk 与原始 alpha 单位。

### 7.1 搜索过程

1. 从显式 `theta_seed` 或 `theta_initial` 开始；
2. 以 `theta_growth` 扩张/收缩，寻找风险可行与超预算的 bracket；
3. 在 bracket 内使用有保护的线性插值/二分逼近风险边界；
4. 用最终精度重解；若最终解因数值误差越过风险边界，再执行有界恢复；
5. 计算 frontier slack gap、QP subproblem gap 和 total gap；
6. total gap 超过业务 tolerance 时返回 `LIMIT_REACHED`，由统一路由进入 fallback。

同一天的搜索复用一个 PIQP workspace，只更新 `q`。跨日期不复用 workspace。

### 7.2 PIQP 生命周期

项目强制依赖官方 `piqp>=0.6.4`；该版本已包含上游 dual-recovery 越界修复，`auto`
固定使用 compact 双边约束。one-sided 展开仅保留为显式诊断/性能对照选项，不承担旧版本兼容。

首次 cold solve 失败不会用完全相同参数盲重试。只有 workspace 已经成功求解、随后 update
路径失败，并且 policy 允许时，才销毁并用当前 QP 冷重建一次；再失败则返回统一 fallback。

### 7.3 可选 LP prescreen

`lp_prescreen=False` 是默认值。显式开启后，HiGHS 先求相同线性域上的全局 alpha 最优点。
如果该点同时满足 $\operatorname{TE}(x)\le B-m_R$，它就是 QCQP 的严格全局最优解，返回
`VERIFIED` 证书；否则其耗时累计到主路线并继续 PIQP frontier。

frontier strategy 当前证书标记为 `NUMERICAL_ESTIMATE`，因为 total gap 包含 PIQP 报告的
数值 primal-dual gap；LP prescreen 通过时才是该路线的严格 `VERIFIED` 证书。

## 8. 结果、失败与诊断

### 8.1 失败返回值

输入错误通过 `PortfolioValidationError` 等异常前置暴露。backend infeasible、数值失败、
迭代上限或 fallback 失败返回没有权重的 `OptimizationResult`，以便批量任务统计状态、route、
耗时和 fingerprint。调用者若要求异常式语义，使用 `result.require_weights()`。

`OptimizationResult.route` 记录每次 backend attempt；最终 `backend` 只在某次结果通过独立
验收时设置。`metrics` 和 `violations` 来自最后一次验收，不直接相信 solver 原生日志。

### 8.2 显式 deep diagnosis

`optimizer.diagnose(problem, prior_result=...)` 不在普通失败路径自动运行。当前 deep 诊断包括：

1. 对标记为 relaxable 的 canonical row/variable bound 增加带尺度 slack，用 HiGHS 解 Phase-I；
2. 去掉 turnover 上界后求满足其余线性域的最小 L1 turnover；
3. 若线性域可行且有风险模型，再求线性域中的 minimum tracking error；
4. 输出所需松弛、最小换手下界、最小 TE、求解 attempt 和中文摘要。

`prior_result` 必须与重新编译问题的 fingerprint 完全一致，避免拿另一个数学问题的失败状态
作为诊断依据。

## 9. 多期引擎

链式模式中，第 `t` 日求解前持仓由上一期目标权重按 close-to-close return 漂移：

```text
x_pretrade(t) = normalize(x_target(t-1) * (1 + return(t-1, t)))
```

随后把它写入第 `t` 日 `PortfolioData.initial_weight`，因此 turnover 始终相对真实漂移后的
期初持仓。冷启动/independent 模式则使用调用者提供的逐日初始权重。

`theta_seed="auto"` 在 chained 模式等价于 `previous`，在 independent 模式等价于 `fixed`。
只传播上一成功日的最终 theta 数值，不传播 workspace；失败或没有 theta 时回到固定起点。

普通不可行默认 `stop`；显式 `on_failure="hold"` 才继续持有漂移后组合。换手恢复还需要显式
`TurnoverRecoveryPolicy(max_turnover=...)`：先运行 deep diagnosis，只有线性最小换手处于授权
区间内才尝试放宽；含 TE 时可能进一步在授权区间内二分寻找可行上界。恢复只对当日生效，
下一日恢复原配置。

## 10. 性能边界

- 数据 I/O 与数学求解分层；tuda2 区间数据一次取足，日循环不访问 tuda2。
- canonical 和 backend 使用 NumPy/CSC 数组；pandas 主要留在边界对齐与结果标签层。
- risk QP/factor strategy 直接利用 factor + diagonal 结构，不构造资产维度 dense covariance。
- factor bounds 复用 factor-active 变量，避免重复 dense exposure block。
- 大型 registry 使用 immutable metadata flyweight 和批量 fingerprint 编码。
- 不引入 chunk/LRU 分支；序列规模应在启动前做容量判断。
- 当前不跨日期复用 canonical sparse template，也不跨日期复用 PIQP workspace。

真实 v5 35 日开发机结果见
[`v5_unified_optimizer_real_benchmark.md`](v5_unified_optimizer_real_benchmark.md)。当前 5200×47
合成 factor-QCQP 的 warm `prepare()` 中位数约 0.046 秒。

从用户调用角度审计单期、C2C 链式求解、结果读取和显式 deep diagnosis 的可运行范例见
[`current_api_usage_audit_example.md`](current_api_usage_audit_example.md)。该范例只把 v5 当作真实
输入样本，不把 v5 文件格式纳入公共 API 契约。

## 11. 当前已知边界与审阅关注点

以下项目是当前真实边界，不能从已冻结的 contract 推断为已经实现：

1. `FullCovarianceRiskModel` 尚未编译；通用 CONIC/SOCP canonical compiler 尚未实现。
2. `RiskAdjustedAlpha + TrackingErrorLimit` 会分类为 CONIC，随后明确报 unsupported。
3. `SolverPolicy.lp/qp/factor_qcqp_strategy/factor_qcqp_subproblem_backend` 当前是目标态 contract；
   `solve_prepared()` 仍固定路由 HiGHS、PIQP 和 frontier。
4. `SolverPolicy.validate_solution` 当前不关闭验收；实现始终独立验证，这是现阶段正确性不变量。
5. `SolverTuning.polish`、`repeat_failed_cold_solve` 当前没有接入 backend 行为。
6. `RunFingerprint`、`SolverAttempt.backend_payload_hash` 已定义但尚未由主流程填充；当前结果使用
   `ProblemFingerprint`。
7. `SolveTimings.compile_s/postprocess_s` 尚未单独拆分，prepare 和 total 字段已经可用。
8. 二阶段“近似 alpha 最优集合内最小化与上一期距离”尚未实现。
9. `PreparedPortfolioProblem` 当前公开持有 `CompiledProblem`；未来做 Cython `_core` 保护时需要
   改为 opaque handle，但本轮不改接口。
10. 当前 `setup.py` 仍是旧单层 Cython 构建配置，不覆盖新子包；源代码保护改造尚未开始。

这些条目应作为审阅后的实施清单，而不是在本轮文档整理中顺手改变。

## 12. 建议的审阅顺序

为了先验证语义再看数值细节，建议按以下顺序阅读：

1. `portfolio_types.py`：公共数据、目标、约束、policy 和结果语义；
2. `validation.py`、`asset_bounds.py`：前置校验和交易约束优先级；
3. `model/canonical.py`、`model/compiler.py`：统一数学模型；
4. `api.py`：路由、fallback、独立验收和结果；
5. `factor_qcqp.py`、`backends/piqp.py`：专用快速路径；
6. `solution.py`：solver-independent 正确性边界；
7. `data/*`、`integrations/tuda2.py`、`sequence.py`：数据与多期状态推进；
8. `diagnostics.py`：显式不可行诊断。

# 组合优化器求解器后端定型方案

状态：已定型，作为新版组合优化器的后端实现基线。

## 1. 最终决定

| 问题类型 | 默认主后端 | 商业 fallback | 无 MOSEK license 的 fallback |
|---|---|---|---|
| 线性目标、线性约束（LP） | HiGHS | 不需要 | 不需要 |
| 凸风险惩罚目标（QP） | direct PIQP | MOSEK | Clarabel scaled + QDLDL |
| 单一 factor-model TE 约束（凸 QCQP，可锥表示） | Factor-QCQP frontier strategy（PIQP 子问题） | MOSEK SOCP | Clarabel scaled + QDLDL |
| 其他通用 QCQP/SOCP | 不走 factor-QP | MOSEK | Clarabel scaled + QDLDL |

`CLARABEL_SCALED_QDLDL` 是免费通用安全阀。生产测试中 FAER 在部分链式
SOCP 上出现明显长尾，因此不选 FAER 作为默认 fallback。

PIQP 是 QP 的默认主后端，也是结构化 factor-QCQP strategy 内部的参数化 QP 后端；它
本身不是 SOCP solver。MOSEK 仅在 license 可用时作为优先安全阀，不是正常主路径。
MOSEK 必须按需加载，模块导入时不得检查 license。

## 2. 问题分类

新版公共接口不再以 `opt` 与 `linopt` 区分问题。编译器根据目标和约束自动分类：

```text
maximize alpha，只有线性约束
    -> LP -> HiGHS

maximize alpha - factor/specific risk penalty
    -> convex QP -> direct PIQP

minimize factor-model tracking error/variance
    -> convex QP -> direct PIQP（alpha 可为空）

maximize alpha，且存在 TE 风险预算
    -> factor-model convex QCQP -> specialized frontier strategy -> PIQP QP subproblems

存在 factor-QP 不支持的锥约束
    -> generic SOCP -> MOSEK / Clarabel scaled QDLDL
```

LP 与带风险预算的 QCQP 是两个独立问题。换手率是线性可行域约束，与 TE 大小没有
直接关系；不得根据换手率推断风险约束是否活跃。

## 3. Factor-QP 的适用边界

factor-QP 仅用于以下风险结构：

$$
R(a)=\operatorname{TE}(a)^2
=(E^{\mathsf T}a)^{\mathsf T}F(E^{\mathsf T}a)
+\lVert d\odot a\rVert_2^2.
$$

其中 $a=x-b$，$F$ 为因子协方差，$d$ 为特异风险
年化波动率。实现利用低秩因子风险与对角特异风险构造参数化稀疏 QP，并通过一维
theta 搜索获得风险可行 frontier 点和基于 Lagrangian bound 的 alpha 最优性证书。

这不是通用 SOCP 求解器。出现多个一般锥约束、非 factor-model 二次约束或当前编译器
不支持的模型时，应直接转通用 MOSEK/Clarabel 后端。

LP prescreen 默认关闭。只有用户明确开启时，才先用 HiGHS 求解线性放松；LP 最优解
满足 TE 预算时可作为严格最优性证明直接返回，否则继续 factor-QP。

### 3.1 Theta 参数化与 certificate

令 `X` 为全部线性约束可行域，原问题为：

$$
\begin{aligned}
\max_x\quad & \alpha^{\mathsf T}x \\
\text{s.t.}\quad & x\in\mathcal X, \\
& R(x)\le B.
\end{aligned}
$$

当前实现内部求解的参数化 QP 可写为：

$$
\begin{aligned}
\min_x\quad & \frac12R(x)-\theta s_\alpha\alpha^{\mathsf T}x \\
\text{s.t.}\quad & x\in\mathcal X.
\end{aligned}
$$

其中 `s=alpha_scale>0` 只是数值缩放。对精确的参数化 QP 解 `x_theta`，令
$\lambda=(2\theta s_\alpha)^{-1}$。如果 $x_\theta$ 对原 QCQP 风险可行，则 Lagrangian dual
给出：

$$
\begin{aligned}
LB &= \alpha^{\mathsf T}x_\theta, \\
UB &= LB+\frac{B-R(x_\theta)}{2\theta s_\alpha}.
\end{aligned}
$$

所以风险 slack 项确实是原始 alpha 单位下的最优性 gap 上界，而不只是“TE 靠近边界”的
经验判断。风险约束不活跃时也可通过增大 theta 逼近线性目标最优点；若启用 LP prescreen，
可直接取得更简单的严格证书。

但是上述简式假设参数化 QP 被精确求解。若 backend 只证明该最小化 QP 的 suboptimality
不超过 `delta_qp`，保守上界还必须加入：

$$
\begin{aligned}
g_{\mathrm{subproblem}}
&=\frac{\delta_{\mathrm{QP}}}{\theta s_\alpha}, \\
g_{\mathrm{total}}
&=g_{\mathrm{frontier}}+g_{\mathrm{subproblem}}.
\end{aligned}
$$

输出权重清理、投影或归一化造成的 objective 变化也要再计入，且清理后重新验证可行性。
因此 benchmark 中现有字段 `dual_gap_abs` 更准确的名字是
`frontier_gap_bound_abs`；只有加上可验证的 QP primal-dual gap 等组成项后，正式 API 才把
总量放入 `OptimalityCertificate.absolute_gap`。若 solver 的 dual 不可行或无法给出有效
bound，只能标记为 `NUMERICAL_ESTIMATE` 并继续细化或 fallback，不能声明 `VERIFIED`。

对精确参数化 QP，theta 增大时 alpha 权重增加、风险通常单调不减（等价的风险乘子
`lambda` 下降），因此 bracket + safeguarded secant/bisection 有凸对偶基础。但是风险映射
可能有平台或非严格单调，固定 5--7 次只是当前数据的经验表现，不是理论保证。唯一的正常
成功停止条件是总 certificate 达标；达到 `max_outer_iters/theta_max/time_limit` 仍不达标
必须进入 fallback。

### 3.2 多期 theta 策略

theta 是 factor-QP Pareto frontier 搜索的种子，不是投资约束。它应作为显式选项：

- 单日求解和每日独立冷启动：默认 `fixed`，从 `theta_initial` 开始；
- 多期链式求解：默认 `previous`，以上一成功日的最终 theta 作为本日搜索种子；
- `predict`（根据前两日外推）只作为实验选项，积累证据前不设为默认。

跨日只传递 theta 数值，不跨日复用 PIQP workspace。每日问题的资产、稀疏结构和约束都
可能变化，必须新建当日 workspace；当日 theta 搜索内部可以正常 update/warm-start。

以下情况重置为固定 `theta_initial`：上一日无成功解、发生共同不可行跳日、风险预算/
主动上限/目标尺度等问题签名变化、theta 非有限或超出安全范围。指数大调整导致跳日后也
应重置，避免把调整前的搜索尺度带入新的可行域。

生产 183 日结果中，previous-theta 明显降低了链式 factor-QP 时间，因此适合作为多期
默认。它不降低验证标准；每一天仍需满足相同 TE 和 alpha certificate。若业务希望从
近似最优解集合中选择更接近上一期持仓的解，应使用显式二阶段稳定性目标，不能依赖固定
theta 来替代该业务语义。

## 4. PIQP 版本与故障恢复

生产 fast path 强制依赖官方 `piqp>=0.6.4`，默认使用 compact 双边不等式。
0.6.4 已包含上游 dual-recovery 越界修复；项目不再支持 0.6.3，也不再以 one-sided
作为旧版本兼容路径。one-sided 仅用于显式诊断和性能对照。

修复生命周期 bug 后，不再默认进行“相同参数、完整 workspace 盲重试”。故障策略为：

```text
首次 setup + cold solve 失败
    -> 直接进入 fallback

workspace update 后求解失败
    -> 用当前 QP 完整重建 workspace，cold solve 一次
       -> 成功：返回并记录 recovered
       -> 失败：进入 fallback
```

首次 cold solve 的完全相同重试通常只会放大异常日尾延迟，也可能掩盖补丁回归。
全量盲重试只保留为默认关闭的诊断选项。

## 5. 返回前的独立验证

任何后端的 `solved/optimal` 状态都不能直接等价为可返回组合。统一验证器至少检查：

- 权重和、绝对权重上下限及 long-only；
- 个股主动权重、总主动权重；
- 风格、行业和额外属性敞口；
- 双边换手率；
- 基准成分股持仓比例；
- factor-model TE 及其单位；
- factor-QP 的原始 alpha 目标上下界、frontier slack、QP 子问题 gap 和总 certificate。

默认验收阈值为：

- 对外最大约束残差：`1e-5`；
- factor-QP alpha 容差：当前 benchmark 用原始标准化 alpha score 的 `1e-4` 控制
  frontier slack bound；正式实现以包含 QP 子问题 gap 的 total certificate 验收。它不是
  通用收益率 1 bp，公共接口必须携带 alpha 单位/尺度语义；
- 输出权重清理阈值：`1e-5`，即 0.1 bp 权重。

注意 Python 中 `10e-5 == 1e-4`，不能用它表达 `1e-5`。清理后必须记录累计删除
权重，并按显式策略决定是否归一化和再次验证，不能只看单股阈值。

如果最终高精度 factor-QP 重解后 total certificate 超限，应继续执行有边界保护的 theta
细化；仍不合格则进入通用 fallback，不能静默接受为 `optimal`。

### 5.1 已验证的默认数值参数

这些参数应收进高级 `SolverTuning`，普通用户不必逐项传入：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `alpha_target` | `0.2` | alpha 数值缩放目标，不改变业务目标 |
| `objective_gap_abs` | `1e-4` | 当前 benchmark 的原始标准化 alpha score 绝对容差；任意量纲输入不得直接套用 |
| `theta_initial` | `16384` | 无可用历史 theta 时的起点 |
| `theta_growth` | `4` | 构造风险上下界时的扩张倍数 |
| `theta_max` | `1e12` | 搜索安全上限 |
| `max_outer_iters` | `30` | theta 搜索最大外层次数 |
| `intermediate_eps` | `1e-5` | frontier 搜索阶段 PIQP 精度 |
| `final_eps` | `1e-8` | 最终组合重解精度 |
| `piqp_max_iter` | `1000` | 单个 PIQP 子问题迭代上限 |
| `piqp_inequality_form` | `auto` | PIQP 0.6.4+ 固定使用 compact；one-sided 仅显式诊断 |
| `polish` | `True` | PIQP polishing |
| `lp_prescreen` | `False` | 用户明确开启才使用 |

`eps=1e-7/max_iter=5000` 的所谓 robust profile 在生产可靠性测试中没有改善成功率，反而
放大失败尾延迟，因此不作为默认重试参数。Clarabel fallback 使用正目标缩放和 QDLDL，
其余容差先采用经过测试的 solver 默认值并接受统一独立验证。

实现中的 `alpha_target=0.2` 只是数值缩放。certificate 在除回 `alpha_scale` 后仍以原始
输入 alpha 为单位，所以该内部缩放不会使裸 `objective_gap_abs` 自动获得跨 alpha 数据源
的可比性。公共接口应使用 `ObjectiveTolerance(absolute, normalized)` 与显式
`AlphaSpec.scale`；推荐验收式为
$g_{\mathrm{raw}}\le\varepsilon_{\mathrm{abs}}+
\varepsilon_{\mathrm{normalized}}s_\alpha$。尺度必须显式约定或采用平移不变的离散
尺度，不能只用 `abs(alpha_optimal)`，因为满仓问题对 alpha 整体平移不敏感。

公共 API 使用年化 decimal 风险单位，因此历史 benchmark 中 `risk_margin_pct=1e-5`
个百分点等价于公共接口的 `1e-7` decimal。实现中应采用带单位的字段名或统一在编译器
内部换算，避免把百分数容差直接暴露给用户。

## 6. 统一 fallback 路由

```text
primary backend
    |
    +-- 状态、独立约束复算和证书全部通过 -> return
    |
    +-- PIQP update 路径失败 -> cold rebuild 当前 QP 一次
    |                               |
    |                               +-- 通过 -> recovered return
    |                               +-- 未通过 -> fallback
    |
    +-- 其他失败 -> fallback
                    |
                    +-- 尝试 MOSEK
                            |
                            +-- 解通过独立验收 -> return
                            +-- 明确 infeasible / unbounded -> 终止回退并返回该状态
                            +-- 未安装、无 license 或求解失败 -> Clarabel scaled + QDLDL
```

每次尝试都必须保留 backend、版本、状态、耗时、迭代、残差、证书和失败原因。
fallback 成功是可用结果，但不能计为 primary backend 直接成功。最终状态不能简单采用“最后一个
attempt 覆盖此前证据”：MOSEK 给出的确定数学状态优先于后续免费后端的数值故障。

## 7. 依赖策略

- HiGHS、PIQP、NumPy 和 SciPy 属于核心求解依赖；
- MOSEK 是可选商业 extra，运行到对应路由时才导入并检查 license；
- Clarabel 是可选免费 fallback extra；
- tuda2 是可选数据集成，不属于数学求解核心；需要时通过 `optim[tuda2]` 安装，最低版本为
  2.0.33；
- CVXPY 不进入 LP/QP/factor-QP fast path。第一版可作为低频通用 fallback 的建模层，
  后续再决定是否改为 MOSEK/Clarabel 原生锥接口。

## 8. 证据边界

当前生产证据支持：

- HiGHS 是 LP 明确主后端；
- direct PIQP 风险惩罚 QP 相对通用接口和 MOSEK 有数量级优势；
- 修补后的 compact PIQP 在 183 日、8225 个子问题中全部 `SOLVED`；
- factor-QP 冷启动通常显著快于 MOSEK；链式性能依赖风险活跃度、支持集和 theta
  搜索次数，因此必须保留 fallback、P95/P99 和逐日诊断；
- Clarabel scaled + QDLDL 比 FAER 更适合作为当前生产 CPU 上的通用免费安全阀。

相关测试见：

- [patched PIQP 183 日生产验收](piqp_patched_production_183d_analysis.md)
- [direct PIQP 183 日可靠性分析](production_direct_piqp_reliability_183d.md)
- [v5.4 生产全量结果](v5_4_production_full_results_analysis.md)

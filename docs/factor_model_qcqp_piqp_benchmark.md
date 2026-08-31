# Factor-model QCQP 专用求解器测试报告

测试日期：2026-08-26。数据区间：2026-06-01 至 2026-08-24，共 60 个交易日。

## 结论

可以用免费开源组件替代本问题中的 MOSEK 主路径。当前最值得上生产机复测的组合是：

1. HiGHS 解去掉风险约束的精确 LP；若 LP 最优解满足 6% TE，直接返回；
2. 否则用 PIQP 解 factor-model 参数 QP；
3. PIQP 状态、约束或 `1e-4` 原始 alpha 最优性证书不合格时，回退 Clarabel SOCP；已有 MOSEK license 时也可把 MOSEK 作为最终保险。

PIQP 是 BSD-2-Clause 的稀疏近端内点 QP 求解器，支持只更新线性项后重新求解。开发机 60 日实测中，纯 PIQP 在独立冷启动口径比 MOSEK 平均快 20%，LP→PIQP 平均快 51%；链式口径分别快 40% 和 67%。所有最终 PIQP 结果都通过约束复算及 `1e-4` alpha 误差证书。

这里的“专用”应理解为组合问题的结构化 reformulation，而不是 PIQP 本身是 factor-model 专用 solver。专用性来自 low-rank + diagonal 风险表示、稀疏换手率建模、参数 QP、证书和路由；PIQP 是其中适合本约束结构的通用凸 QP 内核。另需区分建议架构与当前测试实现：本报告分别测试了 PIQP、Clarabel 和 MOSEK 路径，但测试脚本尚未实现 PIQP 失败后自动调用 Clarabel/MOSEK。

这不是把换手率和跟踪误差建立关系。两者仍是独立约束；LP 预筛只利用“线性问题的全局最优解若碰巧也满足风险约束，则它也是 QCQP 的全局最优解”这一包含关系。

## 专用模型

设主动权重为 $a=x-b$，因子暴露矩阵为 $E$，$47\times47$ 因子协方差为 $\Sigma$，
特异风险为 $d$：

$$
\operatorname{TE}(a)^2
= (E^{\mathsf T}a)^{\mathsf T}\Sigma(E^{\mathsf T}a)
+ \lVert d\odot a\rVert_2^2.
$$

引入 47 维主动因子暴露 $f=E^{\mathsf T}a$ 后，对固定 $\theta>0$ 求解：

$$
\begin{aligned}
\min_{x,f}\quad
& \frac12\left(f^{\mathsf T}\Sigma f+\lVert d\odot a\rVert_2^2\right)
  -\theta\,\widetilde{\alpha}^{\mathsf T}x \\
\text{s.t.}\quad
& f=E^{\mathsf T}(x-b), \\
& x\in\mathcal X,
\end{aligned}
$$

其中 $\mathcal X$ 表示全部原线性约束的可行域。

QP Hessian 只有 5200 维特异风险对角块和 47×47 因子块，不形成 5200×5200 稠密协方差。一天内 `P/A/l/u` 不变，搜索风险乘子时只更新线性项 `q`。

换手率使用精确稀疏形式。对初始持仓非零集合 $S$，令
$p_i\ge x_i-x_{0,i}$ 且 $p_i\ge0$；预算守恒时：

$$
\lVert x-x_0\rVert_1
=2\left(\sum_{i\in S}p_i+\sum_{j\notin S}x_j\right)
\le 0.05.
$$

风险不活跃时，不需要先知道 LP 是否会通过。参数 QP 点的原始 alpha 上界误差为：

$$
\operatorname{gap}(x_\theta)
\le
\frac{B^2-R(x_\theta)}{2\theta s_\alpha},
$$

其中 $B$ 是风险预算，$R(x_\theta)$ 是方差，$s_\alpha$ 是 alpha 数值缩放。

风险活跃时在安全点与超限点之间对 $\theta$ 做 safeguarded secant。最终返回的每个点都是真实
QP 最优解，不使用权重插值，因此证书仍成立。

## 测试设置

- 基准：中证1000权重；首日/独立日初始持仓为权重前500只归一化。
- `chained`：第一日 top500，此后使用本 case 上一日结果。
- `independent_top500`：每个日期都重新从相同 top500 开始，用来消除前后依赖并比较同一个优化问题。
- 换手率 5%，单股主动权重上限 0.4%，基准内权重至少 81%，总主动权重上限 1.8。
- 风格、行业约束与用户示例一致；SOCP/QCQP 年化 TE 预算 6%。
- PIQP 原始 alpha 绝对误差证书阈值 `1e-4`，固定 `theta_initial=65536`。
- LP 筛选时间上限 0.75 秒，风险安全边际 `1e-6` 个百分点，独立约束残差阈值 `1e-7`。
- 所有数据加载、建模、求解和约束检查均为 CPU；未使用 GPU。

## 60 日速度结果

下表 `均值/P50/P95/最大` 都是单日求解调用秒数；“2500次”按均值线性外推，不含一次性 Excel 加载。

### 链式持仓

| 路径 | 均值 | P50 | P95 | 最大 | 2500次 | 相对 MOSEK | LP通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOSEK | 1.099 | 1.120 | 1.265 | 1.308 | 45.8 分 | 1.00× | — |
| Clarabel FAER | 1.885 | 1.983 | 2.397 | 2.607 | 78.5 分 | 0.58× | — |
| LP→Clarabel | 0.472 | 0.275 | 2.088 | 4.218 | 19.7 分 | 2.33× | 54/60 |
| 纯 PIQP | 0.663 | 0.591 | 1.354 | 1.811 | 27.6 分 | 1.66× | — |
| LP→PIQP | **0.358** | **0.222** | **1.188** | 2.602 | **14.9 分** | **3.08×** | 53/60 |

加入矩阵组装和当日数据准备后，LP→PIQP 的 2500 次外推分别为 15.2 和 16.4 分钟；MOSEK 为 46.1 和 47.1 分钟。

### 每日 top500 独立冷启动

| 路径 | 均值 | P50 | P95 | 最大 | 2500次 | 相对 MOSEK | LP通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOSEK | 0.683 | 0.684 | 0.761 | 0.814 | 28.5 分 | 1.00× | — |
| Clarabel FAER | 0.950 | 0.937 | 1.301 | 1.354 | 39.6 分 | 0.72× | — |
| LP→Clarabel | 0.384 | 0.200 | 1.238 | 1.479 | 16.0 分 | 1.78× | 48/60 |
| 纯 PIQP | 0.544 | 0.534 | 0.827 | 1.034 | 22.7 分 | 1.26× | — |
| LP→PIQP | **0.334** | **0.196** | 1.023 | 1.238 | **13.9 分** | **2.05×** | 48/60 |

冷启动是更公平的同问题求解器比较。链式模式中，各 solver 前一日的微小差异会改变下一日的初始持仓、LP 通过率和问题难度，不能把跨 solver 权重差全部解释为单次求解误差。

### 分月均值

| 模式 | 路径 | 6月 | 7月 | 8月 |
|---|---|---:|---:|---:|
| chained | MOSEK | 1.067 | 1.168 | 1.042 |
| chained | 纯 PIQP | 0.530 | 0.619 | 0.900 |
| chained | LP→PIQP | 0.267 | 0.223 | 0.669 |
| independent | MOSEK | 0.684 | 0.662 | 0.714 |
| independent | 纯 PIQP | 0.513 | 0.576 | 0.538 |
| independent | LP→PIQP | 0.196 | 0.318 | 0.537 |

8月风险有效日较多，因此筛选路径均值上升；这说明不能只报告 LP 大量通过月份的中位数。

## 精度与可行性

每日 top500 模式保证候选与 MOSEK 使用完全相同的初始持仓。60 日比较如下：

| 指标 | 纯 PIQP | LP→PIQP |
|---|---:|---:|
| 原始 alpha 平均绝对差 | `8.20e-6` | `8.16e-6` |
| 原始 alpha 最大损失 | `9.124e-5` | `9.124e-5` |
| 参数 QP 证书最大值（全部模式） | `9.8e-5` | `9.1e-5` |
| 最大风险超限 | 0 | 0 |
| 最大独立约束残差 | `4.64e-9` | `4.88e-15` |
| 60日状态 | 60 optimal | 60 optimal |

链式纯 PIQP 的全区间最大约束残差为 `2.41e-7`，仍远小于同批 MOSEK 的 `1.47e-5`；LP→PIQP 为 `6.01e-8`。

纯 PIQP 与 MOSEK 的最大单股权重差曾达到 0.3583%（2026-08-18），但该日 alpha 只差 `2.80e-6`，TE 仍低于预算。这是风险不活跃、线性目标存在近似多解时，风险正则化的 QP 点与 LP/MOSEK 选择不同最优面位置，不是 1bp 目标误差失控。LP→PIQP 在风险不活跃日直接接受 LP，可避免这类不必要的持仓漂移；因此生产上优先推荐筛选路径。

## 架构定位、能力边界与外部审阅核对

相对直接横向比较通用 SOCP solver，本方案的实质升级是将当前问题分解为“精确 LP 筛选 + 结构化 factor-QP + 连续凸锥安全阀”。但它不是 Barra Optimizer 全功能替代品。对《Barra Optimizer 8.7 User Guide》所列功能按类别作不加权估计：当前模块对本报告这类单期、连续、long-only、单因子模型、线性 alpha、一个整体 TE 上限的问题直接覆盖；对指南全部功能的直接覆盖约为 15%–25%。通用建模层完善后，底层 solver 对单期连续凸类别的理论覆盖可接近 85%，但基数、门槛、整手、固定费用、5/10/40、复杂税务、非凸风险贡献、多期和多账户仍不属于当前模块。这个估计只用于认识能力边界，不构成开发范围或路线承诺。

第三方审阅中关于 LP 最优性证明、结构化 reformulation、目标证书与权重相似性必须分离等判断成立；以下几点需要按现有实现校正：

- 当前 `theta` 搜索已经不是 blind search：同一天内复用一个 PIQP workspace，只更新线性项，并使用 warm start、单调 bracket 和 safeguarded secant。
- `--factor-carry-theta` 已支持链式模式继承上一日 `theta`，但正式对照默认固定 `theta_initial=65536`，以隔离跨日状态并保持可复现。上一日预测能否降低总耗时尚未 A/B 测试，因此“再降低 30%–50%”不能作为本报告结论。
- `0.0477` 秒、26 次迭代是一个相同硬 QP 的诊断结果，不是完整单日 QCQP 成本；正式样本平均仍需要约 6.3 个 QP 和 279–321 次累计内点迭代。
- 本批 PIQP case 全部通过证书和约束复算，说明 Clarabel/MOSEK 有成为低频安全阀的潜力；由于自动回退尚未接线，当前不能报告真实的 PIQP→Clarabel fallback rate。

## 目标证书与持仓稳定性

$\mathtt{factor\_dual\_gap\_abs}\le10^{-4}$ 只约束原始 alpha 目标损失，不约束权重
$L^1$ 距离、最大单股差、TE 差或跨日持仓变化。已有 0.3583% 最大单股差的观测正好说明：
线性目标在近似多解最优面上很平时，目标几乎不变而持仓仍可能明显跳动。硬换手率上限只
限制“最多变多少”，不会在多个近似最优解中自动选择“变动最少”的一个。因此，当相邻日期
alpha 变化不大时，优先选择靠近上一期持仓的组合具有明确的持仓稳定性、实际换手和可解释性意义。

若以后需要这种行为，数学上应采用显式二阶段 lexicographic/epsilon-constraint 定义，而不能任意在原目标中加入一个很小的惩罚系数：

第一阶段求原问题，得到 $x_1$、$\alpha_1=\alpha^{\mathsf T}x_1$ 及最优性上界
$\alpha^\star-\alpha_1\le g$。第二阶段求：

$$
\begin{aligned}
\min_x\quad & d(x,x_{\mathrm{previous}}) \\
\text{s.t.}\quad
& x\in\mathcal F, \\
& \alpha^{\mathsf T}x\ge \alpha_1-\eta,
\end{aligned}
$$

其中 $\mathcal F$ 是第一阶段的完整可行域。

由 $\alpha^\star-\alpha_1\le g$ 可得阶段二结果满足
$\alpha^\star-\alpha_2\le g+\eta$。若仍要求总 alpha 损失不超过 $10^{-4}$，就必须显式保证
$g+\eta\le10^{-4}$；当前阶段一若已经用满证书预算，便没有足够空间再牺牲 alpha，需要把
阶段一求得更紧。$L^1$ 距离直接对应降低实际换手，平方 $L^2$ 距离
$\lVert x-x_0\rVert_2^2$ 更倾向于唯一、平滑的权重解；参考组合应为真实上一期持仓，而不是
MOSEK 解。该二阶段定义属于投资组合行为选择，会增加一次优化并改变回测路径，未包含在本
报告性能数字中，也不作为当前模块的开发指导。

## 未知 LP 通过率与 workspace 成本

独立冷启动中 LP 通过率为 48/60=80%。实测：

- LP 平均成本约 0.195 秒；
- PIQP 回退主阶段平均约 0.694 秒；
- 不筛选的纯 PIQP 平均约 0.544 秒；
- PIQP workspace 建立平均约 0.012 秒，`q` 更新约 0.2 毫秒量级。

因此是否筛选主要取决于 LP 成本，不再取决于“是否值得建立 QP workspace”。按

```text
T_screen + (1-pass_rate) * T_fallback < T_pure
```

估算，独立模式通过率高于约 50% 时 LP→PIQP 有利；链式样本的经验阈值约 57%，实际通过率 88%。对未知新策略，可先用纯 PIQP，或用前20日/滚动窗口估计通过率和耗时后动态启停 LP 筛选。不要根据换手率猜测 TE 是否会通过。

## OSQP、ProxQP 与结构化尝试的负面结果

同一真实 QP 上，HiGHS 提供的 warm start 最大残差为 `2.22e-15`，因此 OSQP 失败不是模型不可行：

- 默认 OSQP 2万次、2.91秒后仍未收敛，最大违约 `4.20e-4`，定位到5%换手率汇总行；
- 10万次约15秒后仍达到最大迭代；
- 将换手率行放大1000倍后违约降到 `1.77e-5`，但2万次仍未收敛；
- 将换手率对偶化可让单个 QP 收敛，但需要嵌套搜索交易乘子和风险乘子，接近换手率边界的单点仍约2.9秒；
- ProxQP sparse 在首个真实 QP 上超过60秒，已中止；
- 把因子暴露白化会把稀疏行业暴露混成稠密矩阵，OSQP 10万次时间从约15秒恶化到约50秒。

结论是：low-rank + diagonal 结构确实重要，但必须和适合窄换手率边界的内点 QP 算法结合。PIQP 在相同硬约束 QP 上为26次迭代、0.0477秒、最大残差 `3.22e-15`，而 OSQP/ProxQP 不适合作为本问题的专用内核。

## 生产建议与限制

- 建议的生产路由是：HiGHS LP → PIQP factor-QP → Clarabel SOCP 安全阀；当前测试包尚未接入自动安全阀。
- 有 MOSEK license 时，可将最后一级换成 MOSEK，作为极端日期保险。
- 准入必须同时检查：solver status、TE、换手率/全部线性残差、
  $\mathtt{factor\_dual\_gap\_abs}\le10^{-4}$；不能只相信 solver 状态。
- 记录 LP 通过率、fallback 比例、QP 次数、theta、证书和 P95/P99；不能只看中位数。
- 本报告是开发机60日数据，不应直接替代生产机2500次结论。生产机应运行新测试包，尤其验证 PIQP 的 CPU/编译指令集表现和8月长尾。
- 若需要降低近似多解导致的跨日持仓跳动，应采用上述带 alpha 预算的二阶段 tie-break，并以真实上一期持仓为 reference；不应把“接近 MOSEK 权重”本身作为生产目标。

原始结果位于 `tmp/results/factor_qcqp_v4/combined_final/`，生产测试入口为 `benchmarks/solver_evaluation/run_factor_qcqp_suite.sh`。

参考： [PIQP 官方仓库](https://github.com/PREDICT-EPFL/piqp)、[ProxSuite/ProxQP 官方文档](https://simple-robotics.github.io/proxsuite/md_doc_22-ProxQP__api.html)、[OSQP 官方文档](https://osqp.org/docs/)。

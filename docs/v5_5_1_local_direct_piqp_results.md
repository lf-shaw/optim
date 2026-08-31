# v5 direct PIQP 本机全矩阵结果

## 结论

v5 本机 35 个风险模型日的全矩阵已完成。direct PIQP 的通用风险惩罚 QP
在 280 个逐日结果中全部直接成功；风险预算 QCQP 的 factor-QP 路径只出现
1 次 PIQP 子问题失败，当日由 Clarabel QDLDL 回退成功。返回的所有组合都满足
`1e-5` 约束容差，factor-QP 的原始 alpha 绝对 gap 全部不超过 `1e-4`。

因此，direct PIQP 接口和 factor-model reformulation 已可进入长区间生产机审计，
但 35 日样本还不足以单独确认 2500 次回测的尾部可靠性。

## 覆盖范围

| 维度 | 本机实测 |
|---|---|
| 风险模型日 | 35 |
| 单股主动权重上限 | 0.4%、1.0% |
| 模型 | 风险惩罚 QP、风险预算 QCQP/factor-QP |
| QCQP 年化 TE 预算 | 2%、6% |
| 初始持仓 | 链式、逐日独立冷启动 |
| theta 策略 | 每日固定起点、链式跨日 continuation |
| PIQP profile | baseline: `eps=1e-8/max_iter=1000`；robust: `eps=1e-7/max_iter=5000` |
| 对照 | CVXPY→PIQP、MOSEK SOCP/QP |
| LP 筛选 | 关闭 |

共审计 1104 个有效 direct 逐日结果，另有 16 个链式场景在 `2025-06-16`
被一致跳过。该日指数调整的基准 L1 换手为 23.71%，共同线性可行性检查
证明在 5% 换手约束下不可行，因而属于已约定的序列断点，不是 PIQP 失败。

## 求解时间

下表为 baseline 的 `optimization_total_s`，包含当日稀疏矩阵/工作区组装和求解，
不含一次性 Excel 加载。

| 模型 | 主动上限 | 序列 | direct PIQP 均值 | direct PIQP P50 | 参照均值 |
|---|---:|---|---:|---:|---:|
| QP | 0.4% | 链式 | 0.119 s | 0.097 s | CVXPY→PIQP 0.483 s；MOSEK 0.858 s |
| QP | 0.4% | 冷启动 | 0.081 s | 0.081 s | CVXPY→PIQP 0.516 s；MOSEK 0.668 s |
| QP | 1.0% | 链式 | 0.101 s | 0.094 s | CVXPY→PIQP 0.521 s；MOSEK 0.679 s |
| QP | 1.0% | 冷启动 | 0.087 s | 0.078 s | CVXPY→PIQP 0.503 s；MOSEK 0.611 s |
| factor-QP, TE 2% | 0.4% | 链式/冷启动 | 0.582 / 0.320 s | 0.597 / 0.295 s | MOSEK 0.789 / 0.753 s |
| factor-QP, TE 2% | 1.0% | 链式/冷启动 | 0.584 / 0.297 s | 0.559 / 0.260 s | MOSEK 0.785 / 0.731 s |
| factor-QP, TE 6% | 0.4% | 链式/冷启动 | 0.525 / 0.571 s | 0.528 / 0.498 s | MOSEK 0.800 / 0.759 s |
| factor-QP, TE 6% | 1.0% | 链式/冷启动 | 0.535 / 0.422 s | 0.517 / 0.412 s | MOSEK 0.827 / 0.732 s |

0.4%/6% 冷启动的 direct 均值被唯一一次 3.329 s 回退拉高。排除该点时，
其余 34 日均为 direct 成功。链式 continuation 在 6% 预算下的均值约为
0.262–0.263 s，但这是利用前一日 theta 的时序特化路径，不能代表独立冷启动成本。

direct QP 与 CVXPY→PIQP 的独立冷启动解几乎相同：每日权重 L1 差中位数为
`1.8e-8`–`2.9e-8`，最大 L1 差不超过 `8.6e-7`。

## 唯一的 PIQP 失败

- 场景：baseline、0.4% 主动上限、6% TE、固定 theta、冷启动；
- 日期：`2025-06-24`；
- 子问题：首个 `theta=16384`；
- 结果：`PIQP_MAX_ITER_REACHED`，1000 次迭代，1.981 s；
- 处置：同日 Clarabel QDLDL fallback，整体 3.329 s，最大约束残差 `1.54e-9`。

robust profile 的完全相同输入在首个 theta 仅用 46 次迭代成功。随后对 baseline
完全相同的 35 日设置重复 3 遍，105 次逐日求解均未再失败。这说明该点是
暂时未能稳定复现的 native/数值状态事件，不能简单归因为 `max_iter=1000` 太小。

生产实现建议保留显式 fallback，并评测一次“新建 workspace、原参数重试”，再决定
是否进入 robust 参数或 Clarabel。不应仅提高迭代上限并隐藏原始失败。

## baseline 与 robust

两组参数的经济结果一致：

- 552 个可配对 direct 逐日结果中，alpha 最大绝对差为 `1.32e-7`，TE 最大差为
  `2e-6` 个百分点；
- factor-QP 除 fallback 日外权重完全一致；
- QP 的每日权重 L1 差中位数约 `4.8e-7`–`5.5e-7`，最大 `1.94e-5`。

robust 未带来系统性速度损失，但因唯一失败不可复现，本样本也不能证明是
`eps=1e-7/max_iter=5000` 修复了它。

## 内存

冷启动场景连续 35 日的 RSS 通常只增加 2–7 MB。链式场景显示约 90–101 MB
的首尾差，但主要跳变发生在 `2025-06-16` 指数调整后：状态重置到约 1000 只的
完整基准，促使换手稀疏工作区扩容，而不是 35 日内持续单调泄漏的证据。
仍需用 200–250 日生产样本观察稳态 RSS 和 P95/P99。

## 结果位置

- 完整归集：`tmp/v5/direct_piqp_reliability_v551_local/`
- 可靠性计数：`RELIABILITY_REPORT.md`
- 逐场景耗时/残差：`reliability_summary.csv`
- 失败子问题：`failed_subproblems.csv`
- 全部 theta trace：`piqp_subproblems.csv.gz`

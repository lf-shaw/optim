# direct PIQP 生产机 183 日可靠性分析

## 数据和环境

- 回传包：`tmp/results_direct_piqp_reliability_ex_weights.tar.gz`
- SHA256：`0471c43db2a5eb0593dbd13a1bf68fce1d9d5cb67fd0590e936200213803f70f`
- 交易日：183 日，`2023-04-03`–`2023-12-29`
- 生产 CPU：Intel Xeon Gold 6126，64 逻辑 CPU
- Python 3.11.16，PIQP 0.6.3，CVXPY 1.9.2，MOSEK 11.2.3，Clarabel 0.11.1
- 运行包版本：`5.5.1-direct-piqp-reliability`

压缩包未包含原始权重，但保留了删除权重前生成的
`solution_comparisons.csv`，因此仍能审计已计算的权重 L1/L∞ 差、alpha 差和 TE 差。
不能再独立重算这些比较，也不能定位差异最大的具体股票。

## 先修正 v5.5.1 的归集口径

v5.5.1 归集器把链式共同不可行的跳过日也纳入 direct 成功率和耗时。
使用已修正的归集器重算后：

- 有效 direct 逐日结果：5824；
- 共同不可行跳过：32 行；
- direct PIQP 失败并当日 fallback：43；
- 最终返回的可行组合：5824/5824；
- 最终约束残差超过 `1e-5`：0；
- factor-QP alpha certificate 超过绝对 `1e-4`：0；
- `optimal_inaccurate`：0。

两个跳过日是 `2023-06-12` 和 `2023-12-11`，当日基准 L1 换手分别为
20.49% 和 19.78%。这与已约定的指数调整断点处理一致，不是 PIQP 错误。

## direct 可靠性

| profile | 序列 | 有效求解 | direct 成功 | fallback | direct 成功率 |
|---|---|---:|---:|---:|---:|
| baseline | 链式 | 1814 | 1814 | 0 | 100.000% |
| baseline | 冷启动 | 1098 | 1083 | 15 | 98.634% |
| baseline | 全部 | 2912 | 2897 | 15 | 99.485% |
| robust | 链式 | 1814 | 1814 | 0 | 100.000% |
| robust | 冷启动 | 1098 | 1070 | 28 | 97.450% |
| robust | 全部 | 2912 | 2884 | 28 | 99.038% |
| 合计 | 全部 | 5824 | 5781 | 43 | 99.262% |

链式路径包含 QP、factor-QP 固定 theta 和 theta continuation，3628 个有效结果
全部 direct 成功。所有 43 次 fallback 都发生在每日独立冷启动：

| profile | 冷启动模型 | 求解数 | fallback | direct 成功率 |
|---|---|---:|---:|---:|
| baseline | penalty-QP | 366 | 1 | 99.727% |
| baseline | factor-QP | 732 | 14 | 98.087% |
| robust | penalty-QP | 366 | 9 | 97.541% |
| robust | factor-QP | 732 | 19 | 97.404% |

## 43 次失败的性质

所有失败都是 `PIQP_MAX_ITER_REACHED`，没有 infeasible、数值非法或残差超限后
被误接受的情形。

- penalty-QP 失败 10 次；baseline 1 次，robust 9 次；
- factor-QP 失败 33 次；baseline 14 次，robust 19 次；
- 33 次 factor-QP 失败全部发生于首个 `theta=16384` 子问题；
- baseline 失败尝试约 4.06–4.26 秒，回退后总耗时均值约 6.21 秒；
- robust 失败尝试约 20.09–21.52 秒，回退后总耗时均值约 22.40 秒。

最重要的证据是：失败不是由固定的“困难问题”决定的。

factor-QP 冷启动的首个 QP 不含 TE 预算，所以同一 `(active_ub, date)` 在
2%/6%、baseline/robust 中实际有 4 次数学上相同的首 QP。在出现失败的
28 个 `(active_ub, date)` 中：

- 没有一个问题 4 次全失败；
- 最多只有 2/4 次失败；
- 失败问题的其他独立运行通常只用 29–69 次迭代就成功。

penalty-QP 的 10 个失败 `(active_ub, date)` 在另一 profile 中全部成功，且只用
18–22 次迭代。这与本机之前的偶发不可复现失败一致，更像 PIQP native
工作区/数值路径的偶发状态，而不是数学问题需要 5000 次迭代。

`robust: eps=1e-7/max_iter=5000` 没有改善可靠性，失败数反而从 15 增加到 28，
且将失败尾部放大约 4 倍。不应将 robust profile 用作生产默认或第一重试。

## 求解效率

时间口径为 `optimization_total_s`，包含当日矩阵/工作区构建和求解，不含一次性
Excel 加载。下表使用 baseline 并排除共同不可行跳过日。

### penalty-QP

| 主动上限 | 序列 | direct PIQP 均值 | direct P50 | CVXPY→PIQP 均值 | MOSEK 均值 |
|---:|---|---:|---:|---:|---:|
| 0.4% | 链式 | 0.145 s | 0.140 s | 0.990 s | 1.171 s |
| 0.4% | 冷启动 | 0.152 s | 0.123 s | 1.026 s | 1.078 s |
| 1.0% | 链式 | 0.149 s | 0.145 s | 1.024 s | 1.212 s |
| 1.0% | 冷启动 | 0.124 s | 0.123 s | 1.001 s | 1.112 s |

0.4% 冷启动均值包含一次 5.462 秒 MOSEK fallback。direct QP 正常日的耗时约为
0.12–0.15 秒，相对通用 CVXPY→PIQP/MOSEK 约快 7–9 倍。

### 链式 factor-QP / SOCP 参照

| 主动上限 | TE | 固定 theta 均值 | 跨日 theta 均值 | MOSEK 均值 | 2500 日跨日 theta 估算 |
|---:|---:|---:|---:|---:|---:|
| 0.4% | 2% | 1.324 s | 0.936 s | 1.312 s | 39.0 min |
| 0.4% | 6% | 1.100 s | 0.506 s | 1.434 s | 21.1 min |
| 1.0% | 2% | 1.425 s | 0.927 s | 1.295 s | 38.6 min |
| 1.0% | 6% | 1.651 s | 0.746 s | 1.620 s | 31.1 min |

2500 日估算只累加优化耗时，不包含数据处理和回测其他逻辑。不使用跨日 theta 时，
factor-QP 并不是所有场景都快于 MOSEK；对真实时序回测，theta continuation 是主要性能来源。

## 数学结果一致性

所有返回组合的最大约束残差低于 `1e-5`。factor-QP 的最大直接残差约
`1.79e-8`，最大 alpha certificate 为 `9.99e-5`。QP 中最大的 `3.07e-7`
残差来自 MOSEK fallback，仍显著小于验收容差。

独立冷启动下，direct QP 与 CVXPY→PIQP 在排除唯一 fallback 日后：

- raw objective 最大差 `2.57e-8`；
- alpha 最大差 `3.22e-8`；
- TE 最大差 `4.18e-8` 个百分点。

生产机预先生成的权重对比也保留了下来。独立冷启动下：

- direct QP vs MOSEK：最大权重 L1 差 `4.93e-5`，最大单股差 `7.04e-6`，
  最大 raw objective 差 `7.58e-6`；
- factor-QP vs MOSEK：最大 alpha 差 `3.42e-6`，最大 TE 差 0.0553 个百分点；
- factor-QP vs MOSEK 最大权重 L1 差为 1.42%，最大单股差约 0.40%。

factor-QP 的权重差显著大于 alpha/TE 差，说明多近似最优解现象在长样本上仍然存在。
这支持未来单独设计“在 alpha 近似最优可行集内最小化与前期持仓距离”的二阶段
tie-break，但不影响当前后端选型。

## 数据对齐和输入覆盖

- 183 个风险日的基准和 alpha 都使用同日数据，没有向前填充或未来数据；
- 风险宇宙每日约 4947–5096 只资产；
- 每日 alpha 缺失 1085–1236 只，占风险宇宙 21.9%–24.3%，当前按零 alpha 处理；
- 记录的基准缺失质量最高为 0.284%。

alpha 缺失比例是数据/前端宇宙政策问题，不是 solver 错误，但在正式封装前应确认
“缺失 alpha = 0”确实是预期业务语义。

## 内存和警告

183 日运行没有 OOM，但 RSS 尚不能判定已完全进入平台：

- direct 链式进程通常从约 438 MB 增长到 686–714 MB；
- 首次 direct workspace 通常带来约 87 MB 跳变；
- `2023-06-12` 首次加载 HiGHS 做调仓可行性预检时，代表性进程 RSS 增加约 71 MB；
- 无 fallback 的后续冷启动 183 日通常再增加 62–75 MB；
- penalty-QP 首次触发 MOSEK fallback 时 RSS 一次增加约 463 MB，主要是 MOSEK 延迟加载/工作区。

这不像每日等量失控的泄漏，但仍有 Python/native allocator 保留和工作区高水位。
2500 日回测建议按年或固定日数回收 worker，并序列化当前持仓和 theta 后续跑，
同时记录 P95/P99 RSS。

所有 8784 个逐日记录中只有 1 行产生警告：robust、1% 主动上限、2% TE、
`2023-06-20` 冷启动的 Clarabel fallback 内出现 8 条 NumPy overflow/invalid warning。
回退解最终为 `optimal`，最大残差约 `2.03e-9`，警告未污染最终结果。

## 后端定型建议

1. 使用 baseline direct PIQP：`eps=1e-8/max_iter=1000`；不采用 robust 5000 次路径。
2. LP 筛选仍默认关闭；真实时序 factor-QP 默认传递前一日 theta。
3. PIQP 返回非 solved 时，丢弃工作区，用完全相同参数新建 workspace 快速重试一次。
4. 重试仍失败时，factor-QP 使用 Clarabel QDLDL fallback；penalty-QP 可用 MOSEK
   或后续单独评测 Clarabel QP fallback。
5. 原始失败、重试结果、两次 trace 和 fallback 都必须保留，不能将重试成功伪装成首次成功。
6. 在生产机上对本次失败日做专项重试后，再决定 direct PIQP 是否正式成为默认快路。

当前证据支持的定位是：**direct PIQP 可作为默认快路，但不能作为无重试、
无 fallback 的唯一后端。**

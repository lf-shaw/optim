# MOSEK 替代求解器评估与组合回测性能报告

日期：2026-08-25

## 1. 结论

当前项目可以用免费/开源求解器替代 MOSEK，但应按模型类型选择后端，而不是用一个求解器覆盖全部场景：

- 纯 LP：首选 HiGHS（MIT）。生产机链式 16 日中位总耗时 0.389 秒，精度约为 `1e-15`；MOSEK 为 0.630 秒，Clarabel 为 2.319 秒。
- 风险惩罚 QP：Clarabel（Apache-2.0）是当前最稳妥的开源方案。HiGHS 首日很快，但在链式第二日失败，暂不能用于生产 QP。
- 6% 风险预算 QCQP/SOCP：单独使用 Clarabel 时，目标缩放加 FAER 是当前最稳的开源配置，16 日总耗时比未缩放基线下降约 41.1%，但仍比 MOSEK 慢约 40.8%。若允许精确 LP 预筛，`缩放+预筛+Clarabel 回退` 在本批数据上比 MOSEK 总耗时还低约 16.6%，同时保持更小约束残差；其尾延迟较大，仍需增加预筛超时或风险活跃期旁路。
- 开发机上的自编译 Clarabel + Pardiso-MKL 没有提速：在默认等价化简和目标缩放口径下，8 线程均值 2.552 秒，慢于同一 wheel 的 FAER 1.648 秒和 QDLDL 1.735 秒；canonical、未缩放同口径下 MKL 也比 QDLDL 慢约 33%。生产 Xeon 仍保留一次独立进程线程扫描，以确认硬件是否改变排序。
- ECOS、SCS、QOCO-CPU、OSQP 均不建议用于本项目的 SOCP 主路径。COSMO 值得作为后续 Julia/热启动专项实验，但尚无本项目数据上的证据证明其更快或足够精确。

对于约 2500 次串行优化的 10 年回测，按生产 v2 的逐日均值外推：HiGHS LP 约 16.2 分钟，Clarabel QP 约 76.2 分钟，Clarabel SOCP 基线约 209.5 分钟，缩放+FAER 约 123.5 分钟，缩放+预筛组合约 73.1 分钟。最后一个数字依赖风险约束活跃日占比，不能用其 0.486 秒中位数机械外推成 20.2 分钟。

## 2. 模型口径

三个模型互相独立，不共享求解结果链：

1. LP：最大化线性 alpha，只施加线性约束。跟踪误差是事后统计，不施加 6% 风险预算，也不会因 TE 超过 6% 回退 SOCP。
2. QP：最大化 `alpha - 0.75 × common_variance - 0.75 × specific_variance`，不施加 6% 上限。
3. SOCP：最大化线性 alpha，同时施加年化跟踪误差不超过 6%。

风险定义为主动权重 $a=x-b$ 的因子风险与特异风险：

$$
\begin{aligned}
F &= LL^{\mathsf T}, \\
\operatorname{TE}(a)
&=\left\lVert
\begin{bmatrix}
L^{\mathsf T}E^{\mathsf T}a \\
D\odot a
\end{bmatrix}
\right\rVert_2.
\end{aligned}
$$

数据中的因子协方差单位为年化百分比平方、特异风险单位为年化百分比，因此 6% 在模型中写作
$\gamma=6.0$，不是 $0.06$。

## 3. 数据与约束

测试数据：

- `tmp/datayes_data_20260821.xlsx`：单日风险模型。
- `tmp/datayes_data_202608.xlsx`：2026-08-03 至 2026-08-24，共 16 个交易日。
- `tmp/000852.SH.csv`：中证1000权重，1000 只股票。

风险模型每日约 5203–5211 只股票、47 个有效因子，其中 16 个风格因子、30 个行业因子和 1 个 country 因子。16 个日期的 47×47 协方差矩阵均为对称正定矩阵。2026-08-21 有 1 只中证1000成分股缺少特异风险，原始指数权重为 0.053%；风险测试按当日有效成分重新归一化基准。

为保证求解器目标完全一致，测试 alpha 使用文件内因子收益和特异收益构造：
$\alpha=100E r_f+r_s$。这是数值等价性和性能测试目标，不是可用于真实回测的预测信号。

约束与用户示例一致：

- SIZE 主动暴露：`[-0.3, 0.3]`。
- 其他 15 个风格因子主动暴露：`[-0.5, 0.5]`。
- 30 个行业主动暴露：`[-0.03, 0.03]`。
- 总主动权重 L1：不超过 1.8。
- 双边换手率 L1：不超过 0.05。
- 单股主动权重：`[-0.004, 0.004]`。
- 基准成分股内权重：不低于 0.81。
- 权重和为 1，禁止做空，单股绝对权重上限为 1。
- SOCP 风险预算：年化 6%。

链式测试不再使用随机初始持仓：首日取中证1000权重最大的 500 只股票并归一化，其总原始基准权重为 75.545%，归一化后最大单股权重为 1.4071%；之后每天使用同一模型、同一 case 上一日的最终权重。

## 4. 链式 16 日性能结果

以下 `总耗时` 包含 CVXPY canonicalization 与求解器调用，不包含 Excel 首次载入。开发机为 Intel Core Ultra 5 225H，14 个逻辑 CPU，Python 3.11.11。

### 4.1 LP

| 后端 | 中位数/日 | 均值/日 | 范围 | 最大约束残差 | 结果 |
|---|---:|---:|---:|---:|---|
| HiGHS | 0.256 s | 0.263 s | 0.235–0.315 s | 约 `1.3e-15` | 16/16 optimal |
| MOSEK（CVXPY） | 0.956 s | 1.108 s | 0.612–2.363 s | `5.70e-7` | 16/16 optimal |
| Clarabel | 1.869 s | 1.906 s | 1.317–2.653 s | `4.07e-7` | 16/16 optimal |

LP 的事后 TE 范围为 4.779%–7.283%，中位数 6.099%。这不构成 LP 约束违例，只说明纯 alpha 线性规划可能产生较高主动风险。

这里 MOSEK 通过 CVXPY 的锥接口运行，不能替代项目当前 Fusion 接口的基线。此前直接使用项目 Fusion 模型、随机固定初始持仓的 16 日冷启动中位数为 0.427 秒；同一批测试中 HiGHS 为 0.524 秒、Clarabel 为 2.078 秒。无论采用哪组口径，LP 的开源首选都是 HiGHS。

### 4.2 风险惩罚 QP

| 后端 | 中位数/日 | 均值/日 | 范围 | 最大约束残差 | 结果 |
|---|---:|---:|---:|---:|---|
| MOSEK（CVXPY） | 1.112 s | 1.316 s | 0.919–2.637 s | `3.16e-5` | 16/16 optimal |
| Clarabel | 1.524 s | 1.565 s | 1.013–2.272 s | `3.61e-9` | 16/16 optimal |
| HiGHS | 首日 1.233 s | — | — | 首日约 `1e-15` | 第二日求解失败 |

Clarabel 比本次 MOSEK 链慢约 37%，但可行性残差小约四个数量级。HiGHS 虽然理论上支持凸 QP，但本项目第二日即失败，不能根据单日速度用于生产。

QP 链的 TE 范围约为 0.759%–4.429%，中位数约 0.815%；这些数值来自风险惩罚目标，不是 6% 风险上限。

### 4.3 年化 6% 风险预算 SOCP

| 后端 | 中位数/日 | 均值/日 | 范围 | 最大约束残差 | 状态 |
|---|---:|---:|---:|---:|---|
| MOSEK（CVXPY） | 1.764 s | 1.834 s | 1.304–2.604 s | `8.45e-6` | 16 optimal |
| Clarabel | 3.630 s | 3.758 s | 2.721–4.917 s | `2.46e-8` | 14 optimal，2 optimal_inaccurate |
| ECOS | 4.250 s | 4.508 s | 2.965–7.894 s | `4.10e-5` | 15 optimal，1 optimal_inaccurate |

Clarabel 的 TE 范围为 4.177%–6.000%，中位数 5.358%。风险约束在 08-04 至 08-07 连续 4 天活跃；其余日期风险约束不活跃。Clarabel 的两次 `optimal_inaccurate` 发生在风险约束活跃区间，但独立重算后的 TE 未超过 6%，最大总残差来自换手率，约 `2.46e-8`。

ECOS 不仅更慢，而且出现约 `4.10e-5` 的换手率超限和 `3.28e-4` 个百分点的 TE 超限。MOSEK 更快，但默认容差下最大换手率超限约 `8.45e-6`。生产系统必须对所有后端统一做解后验算，不能只看 `optimal` 状态。

### 4.4 生产机复测结果

用户提供的 `tmp/results.tar.gz` 包含同一 16 日链式测试。环境为 Linux、Python 3.11.16、64 个逻辑 CPU，求解器与开发测试包版本一致；旧版元数据没有记录 CPU 具体型号，第二版已补充 `/proc/cpuinfo` 的 model name。

| 模型/后端 | 中位总耗时/日 | 均值/日 | 内部求解中位数 | 最大约束残差 | 状态 |
|---|---:|---:|---:|---:|---|
| LP / HiGHS | 0.389 s | 0.391 s | 0.258 s | `1.26e-15` | 16 optimal |
| LP / MOSEK | 0.630 s | 0.643 s | 0.455 s | `2.34e-8` | 16 optimal |
| LP / Clarabel | 2.319 s | 2.294 s | 2.195 s | `2.94e-7` | 16 optimal |
| QP / MOSEK | 1.481 s | 1.479 s | 1.105 s | `3.16e-5` | 16 optimal |
| QP / Clarabel | 1.773 s | 1.837 s | 1.514 s | `4.52e-9` | 16 optimal |
| SOCP / MOSEK | 2.076 s | 2.099 s | 1.702 s | `1.02e-5` | 16 optimal |
| SOCP / Clarabel | 5.160 s | 5.032 s | 4.872 s | `2.32e-8` | 15 optimal，1 optimal_inaccurate |
| SOCP / ECOS | 6.433 s | 6.620 s | 6.156 s | `2.80e-4` | 16 optimal |

LP 结果是一个独立的无风险预算问题。其事后 TE 中位数为 6.099%，范围 4.779%–7.283%；超过 6% 的日期并不是 LP 违约。生产机上 HiGHS 比 MOSEK 快约 38%，比 Clarabel 快约 83%，所以 LP 结论非常明确：开源替代应使用 HiGHS，而不是 Clarabel。

生产机的 Clarabel SOCP 只有约 0.264 秒中位时间花在 CVXPY 接口与 canonicalization，约占总时间 5%；主要成本仍是 Clarabel 内部稀疏 KKT 求解。因此仅改为原生 API 的预期收益有限，数值尺度、是否需要进入锥求解、以及线性代数后端更值得优先测试。

### 4.5 生产机第二版 Clarabel 优化复测

第二版结果归档为 `tmp/results_production_v2.tar`。生产机 CPU 是 Intel Xeon Gold 6126 2.60GHz，64 个逻辑 CPU；全部 case 完成，无脚本错误。LP、QP 与第一版结果基本复现。SOCP 结果如下：

| SOCP case | 中位数/日 | 均值/日 | 16 日总耗时 | 预筛通过/回退 | 按均值外推 2500 次 |
|---|---:|---:|---:|---:|---:|
| MOSEK | 2.034 s | 2.105 s | 33.680 s | — | 87.7 分钟 |
| Clarabel 基线 | 5.077 s | 5.028 s | 80.440 s | — | 209.5 分钟 |
| Clarabel 缩放 | 3.033 s | 3.071 s | 49.141 s | — | 128.0 分钟 |
| Clarabel 缩放 + FAER | 2.973 s | 2.963 s | 47.412 s | — | 123.5 分钟 |
| Clarabel 未缩放预筛 | 5.707 s | 5.293 s | 84.693 s | 1 / 15 | 220.6 分钟 |
| Clarabel 缩放 + 预筛 | 0.486 s | 1.755 s | 28.082 s | 12 / 4 | 73.1 分钟 |

目标缩放将 Clarabel 基线总耗时降低约 38.9%；FAER 在缩放基础上再降低约 3.5%，16 天中有 12 天更快，属于小幅但较稳定的收益。未缩放预筛只有首日通过，反而比基线慢约 5.3%，应删除。

缩放+预筛组合在 08-03、08-10 至 08-24 共 12 天直接接受 HiGHS 的风险可行 LP 最优解；08-04 至 08-07 四个风险约束活跃日回退完整 Clarabel SOCP。16 日总耗时比 Clarabel 基线下降约 65.1%，也比 MOSEK 下降约 16.6%。但 08-05 和 08-06 的 LP 预筛分别耗时 4.40 秒和 8.35 秒，使组合 case 最大单日耗时达到 10.21 秒、95 分位约 7.33 秒。因此对这类混合策略必须按均值或总耗时外推，不能使用被 12 个快速通过日主导的中位数。

解质量没有因提速发生实质退化：缩放+预筛 16/16 为 `optimal`，独立重算无风险超限，最大总残差为 `3.01e-8`，而 MOSEK 最大残差为 `1.02e-5`。相对 Clarabel 基线，逐日权重 L1 差中位数 `2.50e-6`、最大值 `3.12e-6`，单股最大差 `9.33e-7`，原始 alpha 目标最大绝对差 `8.89e-6`，TE 最大差 `4.37e-5` 个百分点。相对 MOSEK 的权重差与 Clarabel 基线相当，说明组合策略没有引入新的量级偏差。

从回测应用出发，下一步不应优先折腾 NumPy MKL，而应控制预筛尾延迟。两个安全方向都不会改变最终数学模型：为 HiGHS 预筛设置约 0.75–1.0 秒独立时限，未取得 `optimal` 证书就回退；或者上一日风险约束活跃时直接旁路预筛、求解缩放后的 SOCP。按本 16 日结果静态估算，0.75 秒预筛上限可把组合总耗时从约 28.1 秒降到约 16.8 秒，但这只是反事实估算，需下一轮实测验证。

### 4.6 开发机 Pardiso-MKL 定向复测

Clarabel 官方 PyPI/conda-forge wheel 不含 `pardiso-mkl` feature。本次从官方
`v0.11.1`、commit `25540f559592068d0c8a80e46ded1b21760212a1` 构建了
`cp39-abi3-manylinux_2_28_x86_64` wheel，并动态加载免费 oneMKL 2026.1.0。

先按当前生产测试包的默认口径（等价约束化简、Clarabel case 目标缩放）比较，
所有 case 均为独立持仓链，16/16 `optimal`：

| 开发机 SOCP case | 中位数/日 | 均值/日 | 范围 |
|---|---:|---:|---:|
| MOSEK（等价化简、未缩放） | 0.991 s | 1.006 s | 0.768–1.266 s |
| Clarabel 缩放 + FAER | 1.564 s | 1.648 s | 0.786–2.406 s |
| Clarabel 缩放 + QDLDL | 1.700 s | 1.735 s | 0.833–2.590 s |
| Clarabel 缩放 + MKL 1 线程 | 2.952 s | 2.861 s | 1.183–4.747 s |
| Clarabel 缩放 + MKL 4 线程 | 2.751 s | 2.658 s | 1.105–4.537 s |
| Clarabel 缩放 + MKL 8 线程 | 2.615 s | 2.552 s | 0.998–4.194 s |

这张表不能直接替换 4.3 的开发机旧表，也不能与 4.5 的生产机表直接求比例。
4.3 是早期 canonical 完整约束、未缩放口径；4.5 虽已使用等价化简，但机器是
Xeon Gold 6126。为隔离后端影响，又在当前开发机复跑 canonical、未缩放模型：

| 开发机 canonical、未缩放 | 中位数/日 | 均值/日 | 最大约束残差 |
|---|---:|---:|---:|
| Clarabel + QDLDL（本轮） | 4.232 s | 4.293 s | `8.18e-9` |
| Clarabel + Pardiso-MKL 8 线程 | 5.776 s | 5.729 s | `5.31e-9` |

历史 4.3 的官方 wheel Clarabel 均值为 3.758 秒；当前同轮 QDLDL 因构建与运行
环境波动为 4.293 秒。判断 MKL 后端效果应使用同轮比较：MKL 比 QDLDL 慢约
33.4%，不是提速。优化口径下 MKL-8 也分别比 QDLDL 和 FAER 慢约 47.0% 和
54.9%。这符合该模型的 KKT 特征：每次分解的并行工作量不足以覆盖 PARDISO
通用不定稀疏分解的线程与数据结构开销。

测试还定位了一个必须规避的线程阶段问题：Clarabel 0.11.1 在 PARDISO 符号分析
完成后才调用 `mkl_set_num_threads_local(max_threads)`。若进程环境在分析阶段允许
8 个线程、case 却把后续阶段改为 1 个线程，PARDISO 会产生错误线性解；自动
线程路径还曾在大型真实 KKT 的 oneMKL 数值分解中触发原生崩溃。第三版脚本把
每个线程数放到独立进程，并令 `MKL_NUM_THREADS`、`OMP_NUM_THREADS` 和 Clarabel
`max_threads` 完全一致。

## 5. 2500 次串行回测耗时估算

按生产机第二版逐日均值外推，只计求解调用与 canonicalization。均值适用于包含快速通过和慢速回退的混合策略：

| 模型/后端 | 单次均值 | 2500 次估算 |
|---|---:|---:|
| LP / HiGHS | 0.389 s | 16.2 分钟 |
| QP / MOSEK | 1.489 s | 62.0 分钟 |
| QP / Clarabel | 1.829 s | 76.2 分钟 |
| SOCP / MOSEK | 2.105 s | 87.7 分钟 |
| SOCP / Clarabel 基线 | 5.028 s | 209.5 分钟 |
| SOCP / Clarabel 缩放 | 3.071 s | 128.0 分钟 |
| SOCP / Clarabel 缩放 + FAER | 2.963 s | 123.5 分钟 |
| SOCP / Clarabel 缩放 + 预筛 | 1.755 s | 73.1 分钟 |

这是单条持仓链的串行估算。风险数据读取、alpha 生成、行情对齐、结果落盘等回测成本未包含；生产机 CPU、内存带宽和 BLAS/稀疏线性代数实现也会改变比例，因此最终选型应以生产测试包结果为准。

## 6. Clarabel 为什么解 LP 慢

这个差距不主要来自 CVXPY：在同一个 2026-08-21 普通 LP 上，Clarabel 总耗时 2.437 秒、内部求解 2.244 秒；HiGHS 总耗时 0.719 秒、内部求解 0.496 秒。差距主要发生在求解器内部。

原因是：

- Clarabel 把 LP 作为锥规划，用内点法反复分解大型稀疏 KKT 系统。
- 换手率与总主动权重的 L1 约束会引入数千个辅助变量和成对不等式。
- 该组合模型高度退化，很多权重和绝对值辅助变量落在边界；单日 Clarabel 需要约 93 次锥内点迭代。
- HiGHS 是专用 LP/QP 求解器，presolve 与单纯形实现更适合这种大规模稀疏线性模型。虽然链式 LP 每日约有 429–566 次单纯形迭代，但每次迭代远比 KKT 分解便宜。

因此 Clarabel 支持 LP 不等于它适合作为 LP 主后端。按模型分流是合理设计：LP 用 HiGHS，QP/SOCP 用 Clarabel。

## 7. 约束建模优化

测试包默认使用几项可证明等价的化简：

1. $x\ge0$ 且 $\mathbf1^{\mathsf T}x=1$ 时，$x\le\mathbf1$ 自动成立。
2. 在预算守恒下，$\lVert x-x_0\rVert_1$ 等于正向交易量之和的两倍。初始权重为零的股票不需要绝对值辅助变量。
3. 若 $\lVert x_0-b\rVert_1+T_{\max}\le A_{\max}$，由三角不等式可知总主动权重约束必然满足，可以省略。
4. 若初始基准成分权重减去最大单边卖出量后仍高于 81%，基准成分权重下限必然满足，可以省略。

这些化简没有把换手率转换为跟踪误差。换手率是 $x-x_0$ 的 $L^1$ 距离，TE 是 $x-b$
在风险协方差下的二范数，两者仍独立计算和约束。

同一 08-21 LP 中，化简后 HiGHS 三次总耗时为 0.174–0.244 秒；未化简为 0.719 秒。Clarabel 化简后为 1.431 秒，未化简为 2.437 秒。相比单纯更换求解器，减少不必要的 L1 辅助变量同样重要。

## 8. 其他开源 SOCP 候选

- Clarabel：支持 LP/QP/SOCP，Apache-2.0；当前真实数据上的最佳开源 CPU 方案。[官方文档](https://clarabel.org/stable/)
- ECOS：轻量 SOCP 内点法，GPL-3.0；本项目比 Clarabel 慢且残差更差，不推荐。[官方仓库](https://github.com/embotech/ecos)
- SCS：支持 SOCP，但单日完整 LP 已需 30.7 秒，SOCP 超过 90 秒后终止，不适合 2500 次回测。[官方文档](https://www.cvxgrp.org/scs/api/cones.html)
- QOCO 0.3.2：BSD-3-Clause，新版 CPU/GPU 二次目标锥求解器；当前 CPU 单日 SOCP 超过 30 秒后终止，不进入默认矩阵。[Python 官方仓库](https://github.com/qoco-org/qoco-python)
- COSMO：Apache-2.0、Julia、ADMM，官方明确提供数据更新、KKT 分解复用和 warm start，并将组合回测列为适用场景。它可能在 2500 个相似问题上受益，但需要独立 Julia 工程、容差和残差测试，目前没有证据优于 Clarabel。[数据更新](https://oxfordcontrol.github.io/COSMO.jl/stable/getting_started/)、[性能建议](https://oxfordcontrol.github.io/COSMO.jl/dev/performance/)
- OSQP：Apache-2.0，只处理凸 QP，不能直接处理 6% 二次风险约束；本项目带 L1 辅助变量的单日 QP 超过 90 秒，不推荐。[官方站点](https://osqp.org/)
- HiGHS：MIT，支持 LP/QP，但不支持 SOCP；本项目 QP 链第二日失败，因此目前只推荐 LP。[官方文档](https://ergo-code.github.io/HiGHS/stable/)

Clarabel 原生接口允许在维度和稀疏结构不变时更新 `P/q/A/b`。若将股票池固定为全期并对当日无数据股票设置零上限，可以研究持久化模型，减少 2500 次重复 canonicalization 和内存分配；这可能节省每次约 0.1–0.3 秒，但不会自动消除内点法本身约 3 秒的求解成本。[Clarabel 数据更新文档](https://clarabel.org/stable/user_guide_data_updating/)

### 8.1 Clarabel SOCP 的第二轮优化方向

开发机上的定向实验显示两个优先级更高且保持数学问题不变的优化：

1. 目标数值缩放：原测试 alpha 大致在 `[-20, 20]`。先减去基准组合 alpha，再按正数缩放到最大绝对系数约 0.2，16 日 Clarabel SOCP 中位耗时从 3.630 秒降到 2.114 秒、均值从 3.758 秒降到 2.216 秒，约下降 42%；16 日均为 `optimal`。解后独立重算最大换手率超限约 `1.31e-7`，因此必须继续比较原始单位目标与权重误差，不能只比较状态。
2. 精确 LP 预筛：先用 HiGHS 解相同目标及全部线性约束。若该全局 LP 最优解已满足 6% 风险预算，则它必然也是 SOCP 全局最优解；否则再调用 Clarabel。早期开发实验中未缩放版本曾有 12/16 日直接接受，但生产 v2 的完整链只有首日接受；目标缩放后才稳定恢复为 12/16。数学上的正比例缩放不改变最优解，但会改变有限精度单纯形的路径、停止判断以及退化 LP 返回的最优极点，因此生产实现必须保留缩放和原始目标验算。这是 SOCP 内部的精确求解策略，不是用换手率推导 TE，也不是把独立 LP 策略与 SOCP 策略混为一谈。

生产机完整链已经验证目标缩放与 LP 预筛可以组合，但主要剩余问题是风险活跃日的 LP 预筛尾延迟。Clarabel `pardiso-mkl` 自定义 wheel 已完成开发机测试且更慢，现只保留一次生产 Xeon 线程扫描作为硬件交叉验证；NumPy 使用 MKL 仍不会替换 Clarabel 的稀疏 KKT 后端。

## 9. 当前代码中的阻断问题

在替换后端前应先修复以下问题，否则高层接口测试结果可能不代表预期模型：

1. [optim/opt.py](../optim/opt.py#L206) 无论 `gamma` 是否存在都使用 `is_linear_obj=False`。当 `gamma` 非空时，[optim/solver.py](../optim/solver.py#L519) 只添加风险上限，没有约束目标中的 epigraph 变量 `s`，导致高层 6% 风险预算模型无界。正确方向是 `gamma is not None` 时使用线性目标，或彻底拆开 QP 与 SOCP 目标构造。本报告 SOCP 使用的是修正后的预期数学模型，不是这个有缺陷的高层路径。
2. [optim/solver.py](../optim/solver.py#L435) 的换手率和总主动权重辅助变量无界，并使用
   $\mathbf1^{\mathsf T}z=u_b$。数学上可表达 $L^1$ 上限，但引入无意义松弛和数值退化；
   应使用非负辅助变量及 $\mathbf1^{\mathsf T}z\le u_b$，并应用本报告的稀疏换手建模。
3. [optim/solver.py](../optim/solver.py#L168) 的标量 `set_asset_ub` 分支只生成数组，没有实际添加约束。
4. [optim/solver.py](../optim/solver.py#L569) 将所有非可行/非最优状态都转换成 `ProblemInfeasible`，混淆无界、数值失败、超时和真正不可行。
5. [optim/linopt.py](../optim/linopt.py#L644) 的 float 自定义权重错误地写成 `(w,w)`，可能引用旧变量或未定义变量，应为 `(ww,ww)`。
6. [optim/linopt.py](../optim/linopt.py#L669) 删除小于 `1e-4` 的权重后不重新归一化，会改变预算并污染下一期换手率。
7. [optim/linopt.py](../optim/linopt.py#L1109) 放松换手率时捕获所有异常，会把程序错误误当成不可行。
8. [optim/__init__.py](../optim/__init__.py#L5) 导入包即加载所有模块并检查 MOSEK license，[setup.py](../setup.py#L110) 又把 MOSEK 固定为必需依赖。应改成延迟加载和可选后端，否则无法部署真正的开源版本。

此外，`optim/opt.py` 的风险因子列表是旧版 10 风格/中文行业，而当前 DataYes 文件是 16 风格/英文行业；需要先做数据适配层，再谈统一后端。

## 10. 推荐落地顺序

1. 修复上述模型与状态处理问题，并为每类约束加入统一解后验算。
2. 抽象 `backend="auto|mosek|highs|clarabel"`：`auto` 对 LP 选 HiGHS，对 QP/SOCP 选 Clarabel。
3. 采用测试包中的等价约束化简，特别是稀疏换手率建模。
4. LP 生产后端使用 HiGHS；QP 开源后端使用 Clarabel。
5. SOCP 稳定保守方案使用 Clarabel 目标缩放 + FAER；追求回测总吞吐时使用目标缩放 + 精确 LP 预筛 + Clarabel 回退，并保留统一解后验算。
6. 为预筛加入独立时限或风险活跃期旁路，生产复测尾延迟后再决定是否完全移除 MOSEK。
7. Pardiso-MKL 只再做一次生产 Xeon 交叉验证；若仍慢于 FAER/QDLDL，就停止该方向。后续纯开源 SOCP 优化优先研究固定维度持久化原生模型；愿意引入 Julia 时再单独评估 COSMO warm start。

## 11. 可复现测试包

生产机测试包位于 [benchmarks/solver_evaluation](../benchmarks/solver_evaluation/README.md)：

- [benchmark.py](../benchmarks/solver_evaluation/benchmark.py)
- [requirements.txt](../benchmarks/solver_evaluation/requirements.txt)
- [生产机一键脚本](../benchmarks/solver_evaluation/run_production.sh)
- [Pardiso-MKL 生产线程扫描](../benchmarks/solver_evaluation/run_production_mkl.sh)
- [Pardiso-MKL 环境安装](../benchmarks/solver_evaluation/install_mkl_env.sh)
- [自定义 wheel 构建信息](../benchmarks/solver_evaluation/BUILD_INFO_MKL.md)
- [运行说明](../benchmarks/solver_evaluation/README.md)

测试包不包含两份 Excel、中证1000 CSV 或 license，只通过命令行读取外部路径。第二版一键脚本默认保存权重；运行后请返回 `summary.csv`、`runs.csv`、`solution_comparisons.csv`、`metadata.json` 和 `weights.csv.gz`。

本次没有修改 `optim/` 下的生产源码，只新增报告和独立测试包。

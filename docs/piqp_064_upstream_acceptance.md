# PIQP 0.6.4 上游版本验收

测试日期：2026-08-31；Python 3.11；Linux x86-64 AVX2；本机 v5 数据 35 个交易日。

## 结论

官方 PyPI `piqp==0.6.4` 可以替代项目自维护的 `0.6.3+optim.dualidx1` wheel。项目现强制
依赖 `piqp>=0.6.4`，`piqp_inequality_form=auto` 固定使用 compact 双边不等式；0.6.3
不再受支持。one-sided 仅保留为显式诊断/性能对照选项。

上游 `v0.6.4` tag 为提交 `5d849590...`，其直接父提交是修复
`98d788f18b2a7de7ca49edea7c1efd29e7a86270`：`fix dual_recovery out-of-bounds memory bug`。
修复内容与本项目此前定位一致：在读取 `h_l_idx` / `h_u_idx` 前先检查有限边界索引是否
耗尽。

## 原缺陷与压力测试

| 测试 | 官方 0.6.4 结果 |
|---|---:|
| 1 变量公开最小复现 | `PIQP_SOLVED`, `x=1` |
| 最小复现 Valgrind | `0 errors / 0 contexts` |
| 历史故障冻结 QP，compact + 首次 `update(c=...)` | 500/500 solved，0 retry |
| 同一冻结 QP，多 theta continuation | 200/200 workspace 成功，800/800 QP solved |
| 两组压力测试的每组解哈希 | 各自唯一 |

冻结问题数值指纹为
`6379c3a0926382674c90333b6a5d2ed304fba4b60c9dfd15cc7b10766070b557`，与此前产生偶发失败的
2025-06-24 冷启动 top-560 样本完全相同：5153 个资产、5760 个变量、48 个等式和 6320 个
一般不等式。测试没有换成更容易的矩阵。

在 `eps=1e-8` 下，官方 0.6.4 与自维护修补版对同一冻结 QP 的目标值分别为
`-34.881007017381286` 和 `-34.88100701737285`，绝对差约 `8.44e-12`；两者均为 45 次迭代，
一般不等式违反量为 0。

## v5 35 日正确性

使用每日固定种子的 benchmark top-500~600 冷启动、主动权重上限 1%、换手率 5%、行业
5%、风格 0.6、2% 年化 TE，关闭 LP prescreen，且不保存权重：

| 场景 | 状态 | fallback / rebuild | 最大约束违反 | 最大 TE | 最大 certificate gap |
|---|---:|---:|---:|---:|---:|
| 风险惩罚 QP | 35/35 optimal | 0 / 0 | `2.54e-13` | `1.35875%` | 不适用 |
| factor-QCQP | 35/35 optimal | 0 / 0 | `9.50e-9` | `1.99054%` | `9.9818e-5` |

与自维护修补版逐日比较，最大 objective、TE 和 turnover 绝对差分别处于 `1e-14`、`1e-13`
和 `1e-14` 量级。

## 紧邻运行性能对照

为降低跨时段机器负载影响，在同一数据缓存上紧邻运行两版完整 35 日：

| 场景 | 指标 | 修补版 0.6.3 | 官方 0.6.4 | 0.6.4 相对变化 |
|---|---|---:|---:|---:|
| QP | backend solve 均值 | 0.03462 s | 0.03343 s | -3.4% |
| QP | 优化总耗时均值 | 0.09465 s | 0.09507 s | +0.4% |
| factor-QCQP | backend solve 均值 | 0.19222 s | 0.19286 s | +0.3% |
| factor-QCQP | 优化总耗时均值 | 0.25719 s | 0.26246 s | +2.0% |

差异落在本机短样本运行噪声范围，没有观察到求解器层面的实质性能回退。生产机长区间仍应
复跑一次完整包，用 P50/P95/P99 确认目标 CPU 上的吞吐。

## 维护决策

- 正常安装仅使用 PyPI `piqp>=0.6.4`；`setup.py` 与两份 requirements 均声明最低版本。
- 删除所有针对 0.6.3 的运行时降级分支；版本过低应在依赖安装/生产预检阶段直接失败。
- 历史 patch、0.6.3 wheel 和构建说明暂留作问题复现与审计证据，但不再维护为生产依赖。
- Clarabel/MOSEK fallback、独立残差校验和 certificate 保持不变；上游修复只消除了已知
  越界缺陷，不替代未知问题上的数值安全网。

上游参考：
[修复提交](https://github.com/PREDICT-EPFL/piqp/commit/98d788f18b2a7de7ca49edea7c1efd29e7a86270)、
[v0.6.4 tag](https://github.com/PREDICT-EPFL/piqp/tree/v0.6.4)、
[PyPI piqp](https://pypi.org/project/piqp/)。

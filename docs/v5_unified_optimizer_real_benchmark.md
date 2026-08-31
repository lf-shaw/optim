# 统一组合优化器 v5 真实数据测试

更新日期：2026-08-29。

## 1. 数据与口径

使用仓库已有数据：

- `tmp/v5/datayes_data_v5.xlsx`
- `tmp/v5/000852.SH_v5.csv`
- `tmp/v5/alpha_v5.csv`

共同区间为 2025-05-06 至 2025-06-24，共 35 个交易日、约 5,146--5,153 只股票、47 个
风险因子。每天独立冷启动，初始权重为当日中证 1000 权重最大的 N 只股票归一化，其中
`N ~ Uniform{500,...,600}`，固定种子 `20260826`。

约束为：L1 换手率 5%、单股主动权重 ±1%、总主动权重 1.8、基准成分权重至少 81%、
风格 ±0.6、行业 ±0.05；绝对个股权重使用历史 v5 默认区间 `[0, 1]`。factor-QCQP 的
年化 TE 预算为 2%。QP 使用 decimal variance 下的风险惩罚 `7500`，等价于历史 DataYes
$\%^2$ 口径下的 `0.75`。

风险单位按原始接口定义显式转换：factor covariance 除以 10,000，specific volatility
除以 100。风险、基准和 alpha source date 全部要求与优化日一致；本数据全部满足。历史 v5
alpha 文件未覆盖风险宇宙的股票按该 benchmark 原有语义设为零信号，而不是借用其他日期。

计时不含 Excel/CSV I/O；35 日 DataYes frame 物化约 0.74 秒。结果不保存权重。

## 2. 默认路线结果

`SolverPolicy.lp_prescreen=False`，即用户冻结的默认策略：

| 场景 | 后端 | 状态 | prepare 中位数 | 总耗时中位数 | 总耗时均值 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|
| LP | HiGHS | 35/35 optimal | 0.0373 s | 0.0982 s | 0.1090 s | 0.1433 s | 0.1638 s |
| 风险惩罚 QP | direct PIQP | 35/35 optimal | 0.0409 s | 0.0945 s | 0.0998 s | 0.1264 s | 0.1339 s |
| 2% factor-QCQP | PIQP frontier | 35/35 optimal | 0.0394 s | 0.2577 s | 0.2788 s | 0.3617 s | 0.7534 s |

按均值外推 2,500 次独立优化：LP 约 4.54 分钟，QP 约 4.16 分钟，默认 factor-QCQP
约 11.62 分钟。开发机外推仅用于架构对比，生产结论仍应使用生产机 p50/p95/P99。

数值质量：

- 三条路线均无 fallback；PIQP workspace rebuild 为 0；
- LP、QP、factor-QCQP 最大报告残差分别为 `1.55e-15`、`2.54e-13`、`9.50e-9`；
- LP 的事后最大 TE 为 2.2625%，QP 为 1.3588%；LP/QP 均没有风险预算约束；
- factor-QCQP 最大 TE 为 1.99054%，certificate gap 最大 `9.982e-5`，小于配置的 `1e-4`；
- factor-QCQP 平均 3.46 次、P99 13.94 次、最多 17 次 PIQP 子问题，长尾来自少数风险边界日。

## 3. 显式 LP 筛选 A/B

现在 `SolverPolicy(lp_prescreen=True)` 已正式接线，但默认仍为 `False`。HiGHS 求的是与
factor-QCQP 完全相同的线性域；只有当全局 LP 最优点同时满足 TE 时才直接返回，并给出
`lp_global_optimum_feasible_for_factor_qcqp` 全局证书，否则继续原 PIQP frontier。

本独立冷启动样本中 31/35 日通过 LP 证书，4 日进入 PIQP：

| 路线 | 总耗时中位数 | 总耗时均值 | P95 | P99 | 2500 次均值外推 |
|---|---:|---:|---:|---:|---:|
| 默认纯 frontier | 0.2577 s | 0.2788 s | 0.3617 s | 0.7534 s | 11.62 分钟 |
| 显式 LP 筛选 | 0.0989 s | 0.1554 s | 0.4073 s | 0.8624 s | 6.48 分钟 |

LP 筛选使均值下降约 44.2%、中位数下降约 61.6%，但没有消除少数真正进入 frontier 的
长尾，因此 P95/P99 没有改善。本结果支持“冷启动且近期 LP 通过率高时显式开启”，不支持
把它改成通用默认值；链式序列的历史 LP 通过率较低，默认关闭仍合理。

## 4. 本轮结构优化

真实数据剖析发现 QP 和 factor-QCQP 的参数 QP 已经建立
$f=E^{\mathsf T}(x-b)$ 因子变量，但又用 dense $E^{\mathsf T}x$ 重复表达风格/行业约束。
现在有因子变量时直接约束 47 维 $f$；LP 和通用
factor-QCQP canonical model 仍保留原表达，SOCP fallback 语义不变。

同时，风险惩罚 QP 使用满仓等式对 alpha 做 benchmark-centered translation，并将明确的
alpha dispersion 作为 PIQP objective scaling reference。平移只改变 objective constant，
不改变最优权重；测试覆盖 alpha 整体加常数后的权重和 scale 不变。

相对本轮优化前的同口径 35 日结果：

| 场景 | 优化前均值 | 优化后均值 | 变化 |
|---|---:|---:|---:|
| QP | 0.1386 s | 0.0998 s | -28.0% |
| factor-QCQP | 0.5066 s | 0.2788 s | -45.0% |
| factor-QCQP + LP 筛选 | 0.1990 s | 0.1554 s | -21.9% |

变量列顺序也做过一次隔离 A/B，但 PIQP solve 时间没有改善，因此已撤回，避免无收益地改变
canonical fingerprint。

## 5. 复现与结果

```bash
PYTHONNOUSERSITE=1 python benchmarks/frontend_prepare_v5_real.py \
  --cache-dir /tmp/optim_v5_real_data_cache \
  --summary-only \
  --output tmp/v5/new_core_real_35d_final.json

PYTHONNOUSERSITE=1 python benchmarks/frontend_prepare_v5_real.py \
  --cache-dir /tmp/optim_v5_real_data_cache \
  --cases factor_qcqp \
  --lp-prescreen \
  --summary-only \
  --output tmp/v5/new_core_real_35d_factor_qcqp_lp_prescreen_structured_bounds.json
```

脚本所有业务参数均可通过 CLI 修改，没有把 v5 case 硬编码进 solver。正式输出为：

- `tmp/v5/new_core_real_35d_final.json`
- `tmp/v5/new_core_real_35d_factor_qcqp_lp_prescreen_structured_bounds.json`

两份 JSON 仅含配置、分日计时、状态、残差、TE、certificate 和路由元数据，不含权重。

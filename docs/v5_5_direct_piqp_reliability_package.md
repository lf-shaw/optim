# v5.5.2 direct PIQP 生产可靠性专项测试

## 目的和边界

本专项包只回答 direct PIQP 能否作为 factor-model QP/QCQP 主后端的问题，不重新进行
LP、Clarabel、ECOS 等完整横向评测。LP 筛选在所有命令中默认关闭，也没有前端开关设计。

测试包含两个彼此独立的模型：

- 风险惩罚 QP：direct PIQP、CVXPY→PIQP 和 MOSEK 基线；
- 风险预算 QCQP：direct PIQP factor-QP 与 MOSEK SOCP 基线。

factor-QP 失败时使用 Clarabel scaled QDLDL 保留当日结果和链式状态；direct penalty-QP
失败时使用 MOSEK 保留当日结果。fallback 后的组合算“整体路径成功”，但不会算作
“direct PIQP 成功”。

## 默认矩阵

| 维度 | 默认值 |
|---|---|
| 单股主动权重上限 | 0.4%、1.0% |
| 年化 TE 预算 | 2%、6% |
| 初始序列 | 链式、每日独立冷启动 |
| factor-QP theta | 固定初值；链式另测跨日延续 |
| LP 筛选 | 关闭 |
| alpha certificate | 原始 alpha 单位绝对 gap `1e-4` |
| 风格/行业约束 | ±0.6、±0.05 |
| 换手率 | 5% L1，保持原模型口径 |
| 随机种子 | `20260826` |

同一数据会运行两个 PIQP 设置 profile：

- `baseline`：最终精度 `1e-8`、每个 PIQP 子问题最多 1000 次迭代；
- `robust`：最终精度 `1e-7`、最多 5000 次迭代。

`robust` 只是独立实验，用来判断能否恢复 baseline 数值失败及其耗时影响，不代表已经实现
自动重试。MOSEK/CVXPY→PIQP 参照只在 baseline 完整运行；robust 只有 direct PIQP，发生
失败时仍会调用同日 fallback。

## 安装和运行

建议 Python 3.11 新建环境：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

先做一个交易日、单进程的数据冒烟测试：

```bash
SMOKE_ONLY=1 SAVE_WEIGHTS=0 \
BENCH_PYTHON_BIN=.venv/bin/python \
./run_direct_piqp_reliability.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/direct_piqp_smoke
```

`SMOKE_ONLY=1` 只启动一个 benchmark 进程，同时测试 direct QP、factor-QP 及其
MOSEK 参照；不运行完整主动上限/TE/profile 矩阵。首次使用一份输入时仍需完整解析 Excel，
但会逐张 sheet 显示进度，并将解析结果写入
`/tmp/optim_direct_piqp_data_cache_<uid>`。后续 smoke 和完整矩阵会直接读取该缓存，不再为
每个子场景重复解析 Excel。可用 `DATA_CACHE_DIR=/fast/local/path` 指向空间充足的本地盘。

确认完成后运行全部日期：

```bash
BENCH_PYTHON_BIN=.venv/bin/python \
./run_direct_piqp_reliability.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/direct_piqp_reliability
```

输出权重默认开启。若磁盘空间紧张，可以设置 `SAVE_WEIGHTS=0`，但这样无法进行逐股票
MOSEK 对照。建议优先提供更多不同交易日而不是对相同日期做很多 repeats。至少 250 个日期
较有价值；若只有较短区间，也可以先运行。

可覆盖的环境变量：

| 变量 | 默认值 | 用途 |
|---|---:|---|
| `ACTIVE_UB_LIST` | `0.004 0.01` | 主动权重上限列表 |
| `RISK_BUDGET_LIST` | `2 6` | 年化 TE 预算列表，单位 `%` |
| `MAX_DATES` | 空 | 完整矩阵只运行前 N 个有效风险日；不会减少首次 Excel 解析量 |
| `RUN_ROBUST_MATRIX` | `1` | 是否运行 robust profile |
| `REPEATS` | `1` | 完整序列重复次数 |
| `SAVE_WEIGHTS` | `1` | 是否保存 `weights.csv.gz` |
| `SMOKE_ONLY` | `0` | 单进程、单风险日的数据冒烟测试 |
| `DATA_CACHE_DIR` | `/tmp/optim_direct_piqp_data_cache_<uid>` | 跨进程解析数据缓存 |
| `INITIAL_RANDOM_SEED` | `20260826` | 日期确定的冷启动种子 |
| `CHAINED_STATE_WEIGHT_TOL` | `1e-5` | 进入下一日状态前的权重尘埃阈值 |
| `FACTOR_THETA_INITIAL` | `16384` | 固定 theta 搜索起点 |

## 日期和数据对齐

风险模型日期是主交易日。风险日 `t` 直接使用 `t` 日风险模型、指数权重和 alpha；指数或
alpha 缺少整个 `t` 日截面时，只能向前使用最近可用截面。若第一批风险日没有任何历史
指数或 alpha，则丢弃这些开头日期。实际来源日期、滞后天数、缺失 alpha 数量以及丢弃日期
都写入结果。

链式首日和冷启动每日都从当日指数权重前 500–600 名中确定性选择一个 N 并归一化；N 由
固定种子和日期生成。链式后续使用该 case 自己的前一日最终权重，solver case 之间不会共享
状态。指数调整造成共同线性模型确实不可行时，沿用既有规则统一跳过并重置到当日基准。

风险单位保持 DataYes 定义：因子协方差为年化 $\%^2$，特异风险为年化 $\%$，不再次乘
$\sqrt{252}$。

## 关键输出

根目录自动生成：

- `RELIABILITY_REPORT.md`：快速验收计数；
- `reliability_summary.csv`：逐场景 direct 成功率、fallback、P50/P95/P99、残差和 RSS；
- `piqp_subproblems.csv.gz`：每个日期、每次 theta 子问题的状态、迭代数、时间、PIQP 残差；
- `failed_subproblems.csv`：非 solved 的 PIQP 子问题；
- `direct_failures.csv`：发生同日 fallback 的完整逐日记录；
- `all_runs.csv.gz`：所有逐日结果；
- `solution_comparisons.csv.gz`：有同目录 MOSEK 参照时的权重、alpha 和 TE 差异；
- `scenario_manifest.csv`、`metadata_errors.csv`：运行完整性和 benchmark 级异常。

每个原始结果目录仍保存 `runs.csv`、`summary.csv`、`metadata.json` 以及可选的
`weights.csv.gz`。`runs.csv.factor_qp_trace_json` 保留原始 trace，即使采集脚本中途失败也能
恢复诊断。

重点检查：

1. baseline 的首次 direct 成功率，以及失败是否集中在特定日期、theta 或迭代上限；
2. robust 是否恢复这些日期，且 P95/P99 没有显著恶化；
3. 所有返回结果的最大约束残差是否不超过 `1e-5`；
4. factor-QP 的原始 alpha certificate 是否不超过 `1e-4`；
5. 进程 RSS 是否随日期持续单调增长；
6. 冷启动 direct 与 MOSEK 的目标、TE 和权重差异。链式 case 的可行域会因前期持仓路径
   分化，跨 solver 权重差不能解释成同一优化问题的求解误差。

## 回传

不要包含输入数据也可以直接压缩结果目录：

```bash
tar -czf direct_piqp_reliability_results.tar.gz \
  -C results direct_piqp_reliability
sha256sum direct_piqp_reliability_results.tar.gz
```

请同时提供 tar.gz 和 SHA256。结果目录包含完整运行参数、软件版本、CPU、线程环境和日期对齐
记录，因此不需要另行手工整理机器信息。

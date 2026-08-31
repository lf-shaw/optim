# v5.4 完整生产测试包说明

## 目的

本入口用于生产机器上的完整验收，不只测试直接 PIQP。LP、风险惩罚 QP 和风险预算
SOCP 是三类相互独立的问题，汇总时不会把 QP 风险厌恶参数或 SOCP 风险预算错误地
附加到其他模型。

推荐入口：

```text
benchmarks/solver_evaluation/run_v5_full_suite.sh
```

## 默认测试矩阵

| 模型 | 求解路径 |
|---|---|
| LP | HiGHS、MOSEK、Clarabel、Clarabel scaling + FAER |
| QP | 直接 PIQP factor workspace、CVXPY→PIQP、MOSEK、Clarabel、Clarabel scaling + FAER |
| SOCP | MOSEK、ECOS、Clarabel 基线、scaled QDLDL、scaled FAER、HiGHS LP→Clarabel、直接 PIQP factor-QP、HiGHS LP→factor-QP |
| SOCP continuation | 直接 PIQP factor-QP 和 LP→factor-QP 的跨日 theta；单独输出，不和固定 theta 混淆 |

所有主要模型测试以下维度：

- 链式序列：首日随机 top 500–600，之后使用上一日优化结果；
- 独立冷启动：每日使用日期相关、固定种子的随机 top 500–600；
- 个股主动权重上限 0.4% 和 1.0%；
- 风格 ±0.6，行业 ±0.05，换手率 5%；
- SOCP 同时测试 2% 年化 TE 风险活跃场景和 6% LP 高通过率场景；
- 链式状态删除绝对权重低于 `1e-5` 的持仓并归一化；
- 风险模型日期为主日历，基准和 alpha 使用同日或前一个可用日期；
- 2025-06-16 一类经共同线性模型证明不可行的调仓日统一跳过并重置基准状态。

默认 35 日样本理论上产生 3780 个逐日 case 记录，包括统一不可行跳过记录。某一个
对照求解器失败时，benchmark 将错误写入 metadata 后继续其余 case，避免损失整批
生产结果。

## 运行

```bash
tar -xzf solver_evaluation_v5_4_full_suite.tar.gz
python3.11 -m venv solver_env
source solver_env/bin/activate
python -m pip install -r benchmarks/solver_evaluation/requirements.txt

BENCH_PYTHON_BIN="$PWD/solver_env/bin/python" \
benchmarks/solver_evaluation/run_v5_full_suite.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_full_suite
```

先做一日端到端冒烟：

```bash
MAX_DATES=1 \
ACTIVE_UB_LIST="0.004" \
RISK_BUDGET_LIST="2" \
BENCH_PYTHON_BIN="$PWD/solver_env/bin/python" \
benchmarks/solver_evaluation/run_v5_full_suite.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_full_suite_smoke
```

该缩小入口已在开发机用 v5 数据实际执行：3 个场景目录、36 个逐日 case 全部成功，
统一汇总文件正常生成。

## 输出

每个场景保留独立的：

- `runs.csv`：逐日时间、状态、来源日期、约束和风险残差；
- `summary.csv`、`summary_by_month.csv`；
- `weights.csv.gz` 和 `solution_comparisons.csv`；
- `metadata.json`：完整命令、版本、CPU、线程环境、风险单位和错误。

根目录另外生成：

- `all_summary.csv`；
- `all_runs.csv.gz`；
- `scenario_manifest.csv`；
- `all_errors.csv`。

生产结果回传时应打包整个输出根目录。优先分析 `scenario_manifest.csv` 是否完整，
再比较 `optimization_total_s`、P95/P99、约束残差、目标差、fallback rate 和链式
持仓差异。

## 可配置项

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `ACTIVE_UB_LIST` | `0.004 0.01` | 个股主动权重场景 |
| `RISK_BUDGET_LIST` | `2 6` | SOCP 年化 TE 预算场景；不影响 LP/QP |
| `COMMON_RISK_AVERSION` | `0.75` | QP 公共风险惩罚 |
| `SPECIFIC_RISK_AVERSION` | `0.75` | QP 特异风险惩罚 |
| `INITIAL_RANDOM_SEED` | `20260826` | 可复现初始持仓 |
| `CHAINED_STATE_WEIGHT_TOL` | `1e-5` | 链式状态清理 |
| `MAX_DATES` | 空 | 限制测试日期数，冒烟使用 |
| `LP_CASES` / `QP_CASES` / `SOCP_CASES` / `CARRY_CASES` | 完整默认矩阵 | 覆盖求解器 case |

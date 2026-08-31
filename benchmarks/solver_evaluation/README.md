# 组合优化求解器生产机测试包

## PIQP 0.6.4+ upstream compact 路径（当前推荐）

Valgrind 和公开最小复现已经把历史偶发 `PIQP_MAX_ITER_REACHED` 定位为 PIQP 0.6.3
`include/piqp/kkt_system.tpp` 的 dual recovery 未初始化索引读取。旧代码在有限下界或上界
索引已经耗尽时，先递增索引再读取 `h_l_idx(n_h_l)` / `h_u_idx(n_h_u)`，边界判断发生得
太晚。真实组合矩阵恰好包含“双边行之后跟随仅上界行”的结构，因此会进入该路径。

上游已在 PIQP 0.6.4 修复该问题。测试包现在强制依赖 `piqp>=0.6.4`，默认
`--factor-piqp-inequality-form auto` 固定使用 compact 双边表示；不再支持 0.6.3，也不再
维护生产用自定义 wheel。one-sided 仍可显式选择，但只用于诊断和性能对照。

首个 theta cost 仍直接传给 `setup`，后续 continuation 使用 `update(c=...)`；这是消除一次
不必要 update 的正常生命周期优化，不再把它表述为 bug 修复。隔离工具
`isolate_direct_piqp.py` 可冻结 QP 并测试同进程/跨进程、双边/单边、预条件器和 cost
lifecycle。冻结快照含衍生输入，只写临时目录，不随生产包分发。源码位置、公开复现、
Valgrind、压力测试和完整 v5 结果见 `docs/piqp_063_first_update_isolation.md`。

### PIQP compact 生产机专项验收（当前应先运行）

验证本次源码修复不需要先运行包含慢速 ECOS/Clarabel/CVXPY 对照的完整套装。使用：

```bash
BENCH_PYTHON_BIN="$PWD/.venv/bin/python" \
./run_piqp_patched_acceptance.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  results/piqp_patched_acceptance
```

该入口文件名保留历史命名，但会强制确认 PIQP 版本至少为 0.6.4 且 `auto` 已启用 compact，
关闭 LP 筛选，只运行 direct
penalty-QP 和 factor-QP，并覆盖链式/每日冷启动、0.4%/1% 主动权重上限、2%/6% 年化
TE。QP 与风险预算相互独立，因此每个主动权重配置只运行一次 QP；SOCP 分别运行 2% 和
6%。同日 Clarabel fallback 只作为不中断长序列的安全网，归集器在任何 fallback、失败
子问题、证书超限或元数据错误出现时最终返回非零。

专项入口不会传 `--save-weights`，因此不产生 `weights.csv.gz` 或
`solution_comparisons.csv`。返回目录只包含逐日指标、压缩的 PIQP theta trace、
P50/P95/P99 汇总、失败表和元数据，适合长区间传输。首次解析输入会写共享缓存；后续 5
个场景进程复用缓存。`MAX_DATES=2` 可先做快速端到端冒烟，但完整验收时不要设置。

## v5.5.2 direct PIQP 可靠性专项

后端定型前，使用 `run_direct_piqp_reliability.sh` 单独审计 direct PIQP。该入口默认关闭
LP 筛选，覆盖风险惩罚 QP、2%/6% factor-QP、链式/冷启动、固定/延续 theta，并逐次保存
PIQP 子问题状态和残差。它还运行 baseline 与 robust 两组 PIQP 参数，单日 direct 失败会
显式记录后由 MOSEK（penalty-QP）或 Clarabel QDLDL（factor-QP）接管，不会丢失整个序列。

```bash
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_direct_piqp_reliability.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/direct_piqp_reliability
```

先用 `SMOKE_ONLY=1 SAVE_WEIGHTS=0` 做单进程、单风险日冒烟。首次完整解析 Excel 后会写入
`/tmp/optim_direct_piqp_data_cache_<uid>`，后续完整矩阵跨进程复用，不再重复加载。完整说明见随包文档
`v5_5_direct_piqp_reliability_package.md`。

该目录不包含风险模型 Excel 或中证1000权重 CSV。脚本读取外部路径，分别测试三个互不混用的模型：

- LP：线性目标与线性约束；跟踪误差只作为事后指标，不施加 6% 风险预算。
- QP：风险调整目标
  $\alpha^{\mathsf T}x-0.75R_f(x-b)-0.75R_s(x-b)$。
- SOCP：线性 alpha 目标，并施加年化 6% 跟踪误差上限。

每个 `(模型, case)` 都建立自己的持仓链。首日初始持仓为中证1000权重最大的 500 只股票归一化，之后使用该模型、该 case 上一日的最终权重。

## v5 每日基准、外部 alpha 与随机初始持仓

v5 支持三份独立输入：

- 风险模型 Excel：交易日主日历；使用 `asset_exposure`、`asset_data` 和 `covariance`；
- 每日中证1000 CSV：`dt/date,sid,weight`；
- 每日 alpha CSV：`dt/date,sid,fv`。

日期对齐只允许向过去查找。风险日 `t` 的优化直接使用 `t` 日风险模型、`t` 日指数权重和 `t` 日 alpha，含义是这些数据在 `t` 日盘后生成并用于 `t+1` 开盘交易，不额外滞后。只有基准或 alpha 的整个 `t` 日截面缺失时，才使用不晚于 `t` 的最近可用截面；不得使用未来数据。如果风险模型最前面的日期还没有历史基准或 alpha，则连续丢弃这些开头日期，直到两类数据都可用。逐日实际使用的来源日期、滞后天数及丢弃日期写入 `runs.csv` 和 `metadata.json`。

DataYes 原始定义中，`covariance` 是年化协方差乘 10000（单位 $\%^2$），`spec_risk` 是
年化特异波动率乘 100（单位 $\%$），敞口和持仓权重无量纲。令 $a=x-b$、
$f=E^{\mathsf T}a$，脚本直接计算：

$$
\operatorname{TE}_{\%}(a)^2
=f^{\mathsf T}\Sigma_{\%}f+\lVert d_{\%}\odot a\rVert_2^2.
$$

6% 上限写作 $6.0$，不再乘 $\sqrt{252}$。`runs.csv` 同时输出 `factor_risk_pct`、
`specific_risk_pct`、两项方差和实际最大风格/行业敞口，`metadata.json.risk_units` 固化这一口径。

日期截面存在但个别风险资产没有 alpha 时，该资产保留在风险宇宙中并使用零 alpha；`alpha_missing_count/fraction` 会记录覆盖情况。基准成分只有在当天风险模型中有完整风险数据时才允许进入随机初始组合，少于500只会直接报错。链式持仓在后续风险宇宙消失时默认报错，不会静默删除；只有明确接受重新归一化时才使用 `--missing-holding-policy renormalize`。

两个 v5 初始模式为：

- `chained_topn_random`：首日确定性地随机生成 $N\in[500,600]$，取当天中证1000指数权重最大的 $N$ 只并归一化；以后使用本 case 前一日优化权重；
- `independent_topn_random`：每天重新生成日期确定的 $N\in[500,600]$，取当日指数权重最大的 $N$ 只并归一化。

对固定 N，选择指数权重最大的 N 只能最大化保留的基准权重，也就最小化截断归一化相对完整基准的 L1 偏离；这比任意随机抽取成分股更不容易在 5% 换手以及风格/行业约束下产生偶发不可行。随机数由 `base seed + YYYYMMDD` 构造，因此同一日期的 N 和初始组合不受输入区间起点、缺失日期或运行顺序影响；两个模式首日使用相同初始组合。默认 seed 为 `20260826`，逐日的 `generated_initial_n` 和 `initial_fingerprint` 会写入 `runs.csv`。范围可通过 `--initial-top-n/--initial-top-n-max` 调整；旧的随机成分模式仍保留供诊断，但不进入 v5 生产脚本。

指数定期调整可能使链式组合在保持原约束和 5% 换手上限时不可行。脚本只在指数自身的单日 L1 换手超过 5% 时触发 HiGHS 完整线性可行性预检；确认共同线性模型不可行后才跳过该链式日期。这个日期被视为序列断点：所有 case 统一用当日完整指数权重重置状态，使下一交易日可以继续，但重置本身不算优化结果或求解耗时。不能简单保留调整前持仓，否则随后日期仍可能无法在 5% 换手内进入调整后的主动权重区间。冷启动序列若当天可行则仍保留。`runs.csv` 的 `state_reset/state_reset_to` 明确记录这一例外；`--require-consistent-skips` 要求同一序列所有 solver case 的跳日集合完全一致，否则生产测试返回失败。其他约束不会自动放宽。

生产机完整对照：

```bash
chmod +x run_v5.sh
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5
```

脚本同时运行 MOSEK、Clarabel FAER、LP→Clarabel、纯 PIQP 和 LP→PIQP，并为两个初始模式分别维护互不混合的持仓链。可通过 `INITIAL_RANDOM_SEED=12345` 显式覆盖 seed。factor-QP 默认从更稳健的 `theta=16384` 开始（可用 `FACTOR_THETA_INITIAL` 覆盖）；任何 PIQP 数值失败都会在同日自动回退 Clarabel QDLDL（可通过 benchmark 参数修改），失败尝试与回退耗时、原因和实际路径分别写入 `factor_failed_attempt_s`、`factor_failure`、`factor_fallback_used` 和 `solver_path`，不会静默伪装成 PIQP 成功。

v5.3 对链式状态增加数值尘埃清理。求解器返回的绝对权重小于 `1e-5`
（单股小于 0.1 bp）时，不再把它当作下一日真实持仓；清理后才归一化并进入下一日换手率约束。生产样本中这一设置把数值意义上的数千只“持仓”恢复到约 330
只；Clarabel/PIQP 单日累计清理质量最多约 0.096 bp，MOSEK 最多约 0.237 bp。`runs.csv` 会记录
`next_state_cleanup_mass`、`next_state_nonzero_count` 和归一化因子，原始求解权重仍完整保存在
`weights.csv.gz`。可用 `CHAINED_STATE_WEIGHT_TOL` 覆盖；设为 `1e-12` 可复现旧口径。

此外，当初始非零持仓超过 1200 只时，生产脚本直接跳过 LP 预筛并进入
PIQP/Clarabel 主求解器。原因是部分 HiGHS 版本的 `time_limit` 不是严格 wall-clock
上限，大支持集换手率 LP 可能在 0.75 秒设置下仍运行数秒并产生
`user_limit`/`Solution may be inaccurate` 警告。跳过行为及阈值记录在
`screen_status`、`screen_skipped_large_support` 和 `screen_max_initial_n`；可用
`SCREEN_MAX_INITIAL_N` 覆盖。未跳过的预筛若仍由 CVXPY 产生警告，终端不再刷屏，
但原文和次数会保存在 `screen_warnings`、`screen_warning_count`。
主求解阶段的同类警告保存在 `main_warnings`、`main_warning_count`。

v5 脚本的默认约束为：所有风格因子（包括 SIZE）主动敞口 `[-0.6, 0.6]`，所有行业主动敞口 `[-0.05, 0.05]`，单股主动权重上限 0.4%，年化 TE 预算 6%，其余约束保持原规格。风格、行业和风险预算分别可用 `STYLE_BOUND`、`INDUSTRY_BOUND`、`RISK_BUDGET_PCT` 环境变量覆盖，并会完整写入结果元数据。例如设置 `RISK_BUDGET_PCT=2` 即使用 2% 年化 TE 上限，风险模型单位不需要其他转换。

默认单股主动权重上限为 0.4%。要对风险预算更容易生效的 1% 版本做逐日配对测试，保持其他参数不变并使用新的输出目录：

```bash
ACTIVE_UB=0.01 BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_active_1pct
```

实际值写入 `metadata.json` 的 `config.active_ub`，不要把两个口径归集到同一输出目录。

生产机建议直接运行严格配对入口；它依次生成 `active_004/`、`active_010/`，再验证日期、非目标约束、seed、每日随机 N 和初始持仓指纹完全一致，并输出 `paired/paired_summary.csv` 与逐日差异：

```bash
chmod +x run_v5.sh run_v5_paired.sh
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5_paired.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_paired
```

## v5.4 完整生产验收矩阵（推荐）

生产机提交结果时优先运行 `run_v5_full_suite.sh`。它不是只测直接 PIQP 的微基准，
而是默认覆盖完整的模型和序列矩阵：

| 维度 | 默认范围 |
|---|---|
| 模型 | 独立 LP、风险惩罚 QP、风险预算 SOCP |
| 序列 | 链式、每日独立冷启动 |
| 个股主动上限 | 0.4%、1.0% |
| SOCP 年化 TE 预算 | 2%、6% |
| LP | HiGHS、MOSEK、Clarabel、Clarabel scaling + FAER |
| QP | 直接 PIQP、CVXPY→PIQP、MOSEK、Clarabel、Clarabel scaling + FAER |
| SOCP 固定 theta | MOSEK、ECOS、Clarabel 基线/QDLDL/FAER、LP→Clarabel、直接 factor-QP、LP→factor-QP |
| SOCP continuation | 直接 factor-QP、LP→factor-QP；仅链式，单独输出 |

运行方式：

```bash
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5_full_suite.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_full_suite
```

默认输出层级为：

```text
active_004|active_010/
├── lp_qp/
└── socp_te_2|socp_te_6/
    ├── fixed_theta/
    └── carry_theta/
```

完成后自动生成：

- `all_summary.csv`：所有场景汇总，显式包含主动上限、风险预算和 theta 策略；
- `all_runs.csv.gz`：逐日明细；
- `scenario_manifest.csv`：各场景成功、跳过、错误数和总耗时；
- `all_errors.csv`：未安装或求解失败的 case。某个对照求解器失败不会阻止后续
  case，保证生产样本尽量完整。

可用 `ACTIVE_UB_LIST` 和 `RISK_BUDGET_LIST` 缩小或扩展矩阵，例如快速只跑 2%：

```bash
RISK_BUDGET_LIST="2" ./run_v5_full_suite.sh ...
```

`MAX_DATES=1` 可用于生产环境安装后的端到端冒烟测试。`LP_CASES`、`QP_CASES`、
`SOCP_CASES` 和 `CARRY_CASES` 允许显式覆盖 case 列表；不设置时使用上表的完整默认值。

## v5.3 LP / 风险惩罚 QP 配对测试

LP 不施加 TE 预算；QP 使用
$\alpha^{\mathsf T}x-\gamma_fR_f(x-b)-\gamma_sR_s(x-b)$，
也不施加 TE 上限。两个 gamma 默认都是 0.75，可分别用
`COMMON_RISK_AVERSION`、`SPECIFIC_RISK_AVERSION` 环境变量或 benchmark.py 的
同名命令行参数修改。生产配对入口分别运行 0.4% 和 1.0% 单股主动权重上限：

```bash
chmod +x run_v5_lp_qp.sh run_v5_lp_qp_paired.sh
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5_lp_qp_paired.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_lp_qp_paired
```

LP 测试 HiGHS、MOSEK、Clarabel；QP 测试直接 PIQP factor workspace、
CVXPY→PIQP、MOSEK、Clarabel 和 Clarabel scaling + FAER。
`FACTOR_PENALTY_QP_PIQP` 直接构造通用的 `P/q/A/l/u`，风险厌恶、资产/因子规模和
全部线性约束均来自运行时配置，不包含 v5 参数常量。HiGHS QP 在 v5 数据的链式
第二日再次失败，因此不进入生产入口。脚本默认显式使用
`MISSING_HOLDING_POLICY=renormalize`，用于风险宇宙中
资产消失的日期，并把删除质量写入 `initial_missing_mass`；若要求任何缺失都立即失败，
设置 `MISSING_HOLDING_POLICY=error`。

只测试本版直接 PIQP QP、CVXPY→PIQP/MOSEK 对照，以及 SOCP 固定/跨日 theta
的生产入口为：

```bash
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_v5_direct_piqp.sh \
  /path/to/risk_model.xlsx \
  /path/to/000852.SH.csv \
  /path/to/alpha.csv \
  /path/to/mosek.lic \
  results/v5_direct_piqp
```

脚本分别测试 0.4% 和 1.0% 个股主动权重上限；SOCP 的 TE 预算默认 2%，可用
`RISK_BUDGET_PCT` 修改。

## Factor-QCQP 专用求解器与独立冷启动测试（第四版）

第四版增加两个不经过通用 SOCP canonicalization 的 case：

| case | 路径 |
|---|---|
| `FACTOR_PENALTY_QP_PIQP`（QP） | 直接构建 low-rank + diagonal 风险惩罚 QP 并调用 PIQP sparse API；单次求解，不经过 CVXPY canonicalization |
| `FACTOR_QP_PIQP` | 直接用一族 factor-model QP 搜索风险乘子；同一天只更新线性 alpha，复用 PIQP workspace |
| `FACTOR_QP_PIQP_SCREENED` | 先运行受限时且带大支持集保护的精确 HiGHS LP；通过则接受，否则延迟建立 factor-QP workspace |
| `FACTOR_QP_OSQP[_SCREENED]` | 保留的诊断 case；真实换手率边界上收敛很慢，不进入完整套件 |

参数 QP 使用 47 维主动因子敞口变量，二次矩阵只有资产特异风险对角块和
47×47 因子协方差块。PIQP 是 BSD-2-Clause 的近端内点 QP 求解器。风险约束有效时搜索到 6% 边界；风险不活跃时使用拉格朗日
对偶上界判断原始 alpha 目标误差。`factor_dual_gap_abs` 使用未缩放的原始 alpha
单位，默认阈值为 `1e-4`。所有结果仍会独立重算换手率、风险和每类约束残差。

`--initial-modes` 可以同时运行两种互不混合的口径：

- `chained`：首日 top500，之后使用本 case 上一日优化权重；默认每天使用固定且可复现的 theta 起点。只有显式指定 `--factor-carry-theta` 才继承上一日 theta。
- `independent_top500`：每个日期都重新使用同一份中证1000前500权重归一化；重新建立 workspace，并使用固定 theta 起点，不继承任何上一日求解状态。

完整运行 2026 年 6–7 月和 8 月两份外部数据：

```bash
chmod +x run_factor_qcqp_suite.sh
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_factor_qcqp_suite.sh \
  /path/to/datayes_data_202606_202607.xlsx \
  /path/to/datayes_data_202608.xlsx \
  /path/to/000852.SH.csv \
  /path/to/mosek.lic \
  results/factor_qcqp_v4
```

该脚本同时跑 MOSEK、Clarabel FAER、0.75 秒 LP→Clarabel、纯 PIQP factor-QP
和 LP→PIQP，并同时输出 chained/independent 两种模式。除了全区间 `summary.csv`，
还生成 `summary_by_month.csv`，避免 6–7 月合并均值掩盖单月难例。
两段数据完成后，脚本会调用 `aggregate_results.py` 生成无重复日期的 `combined/`；
如手工归集目录存在重叠 `(initial_mode, model, case, repeat, date)`，脚本会拒绝合并。

时间字段口径：

- `solve_total_s`：求解调用，不含模型/稀疏矩阵组装；
- `optimization_total_s`：模型组装 + 求解，是比较独立优化问题的主要口径；
- `end_to_end_s`：再加入当日 DataFrame 合并、Cholesky 和初始持仓映射；不含一次性 Excel 加载；
- `data_load_elapsed_s`：一次性 Excel/CSV 加载，记录在 `metadata.json`；
- `factor_workspace_setup_s`、`factor_search_s`、`factor_final_s`：factor-QP 的 setup、参数搜索和最终高精度 polish 分项。

为了让不同求解器具有相同且可复现的目标，本测试用风险模型文件中的因子收益与特异收益构造
$\alpha=100E r_f+r_s$。这只是数值基准目标，不应作为真实回测预测信号。

## 安装

建议使用 Python 3.11，并新建独立虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_piqp_install.py
```

`requirements.txt` 强制安装官方 `piqp>=0.6.4`。生产入口会在正式加载大文件前验证版本并
打印 `auto` 的实际选择；版本过低或不是 compact 会直接失败。仓库中的 0.6.3 patch、构建
说明和 wheel 仅保留为历史复现/审计材料，不再进入安装或生产路径。

`requirements.txt` 固定了开发机实际测试版本。MOSEK Python 包可以安装，但 MOSEK 用例仍需要有效 license。若只跑免费求解器，可从安装命令中排除 `Mosek`，脚本会将未安装的 MOSEK 记录为 skipped。

## Pardiso-MKL 自定义版（第三版）

普通的 `mamba install clarabel mkl` 只会安装 PyPI/conda-forge 的标准 Clarabel，
不会启用 Cargo 的 `pardiso-mkl` feature。第三版附带一个从 Clarabel 官方
`v0.11.1` 源码构建的 x86-64 wheel；mamba 负责安装免费 oneMKL 运行时和 Python
环境，随后 pip 只安装这个自定义 wheel。

生产机为 glibc 2.28，和附带 wheel 的 `manylinux_2_28` 标签匹配。安装命令：

```bash
chmod +x install_mkl_env.sh run_production_mkl.sh
./install_mkl_env.sh /path/to/solver_mkl_env
```

然后运行 SOCP 对照和 Pardiso-MKL 线程扫描：

```bash
BENCH_PYTHON_BIN=/path/to/solver_mkl_env/bin/python \
./run_production_mkl.sh \
  /path/to/datayes_data_202608.xlsx \
  /path/to/000852.SH.csv \
  /path/to/mosek.lic \
  results/production_mkl_v3
```

该脚本先验证 wheel 确实包含 `pardiso_mkl`，再测试 MOSEK、QDLDL、FAER，并把
MKL 的 1/4/8/16/32 线程 case 分别放到独立进程中执行，最后汇总到
`combined/`。不同线程数不能在同一个进程内扫描：Clarabel 0.11.1 在 PARDISO
符号分析之后才应用 `max_threads`；阶段间线程数不一致会导致错误线性解，甚至
在 oneMKL 并行分解中崩溃。运行脚本已经强制三个阶段使用相同线程数。

默认扫描 1/4/8/16/32 线程；如只需快速验证，可设置例如
`MKL_THREAD_LIST="1 4 8"`。FAER 对照默认限制为 8 个 Rayon 线程，可通过
`FAER_NUM_THREADS` 修改，这两个值都会记录在各 component 的元数据中。

wheel 的源码 commit、构建特性、平台标签和校验和见 `BUILD_INFO_MKL.md`。如需
自行重建，可运行 `./build_clarabel_mkl.sh dist_mkl`。

## 推荐的生产机命令（第二版）

第二版在原始 LP/QP/SOCP 矩阵之外，加入 Clarabel SOCP 的等价数值优化和精确 LP 预筛。直接运行：

```bash
chmod +x run_production.sh
./run_production.sh \
  /path/to/datayes_data_202608.xlsx \
  /path/to/000852.SH.csv \
  /path/to/mosek.lic \
  results/production_v2
```

如需指定虚拟环境解释器：

```bash
BENCH_PYTHON_BIN=/path/to/.venv/bin/python ./run_production.sh \
  /path/to/datayes_data_202608.xlsx \
  /path/to/000852.SH.csv \
  /path/to/mosek.lic \
  results/production_v2
```

该命令测试：

- LP：HiGHS、MOSEK、Clarabel。
- QP：MOSEK、Clarabel。已删除会在链式第二日失败的 HiGHS QP，避免重复浪费时间。
- SOCP：MOSEK、Clarabel 基线，以及下面四个 Clarabel 优化 case。

| SOCP case | 含义 |
|---|---|
| `CLARABEL_SCALED` | alpha 先减去基准组合 alpha，再按正比例缩放，使最大绝对系数为 0.2 |
| `CLARABEL_SCREENED` | 先用 HiGHS 解去掉风险约束后的 LP；LP 最优解满足 6% 时直接接受，否则回退 Clarabel SOCP |
| `CLARABEL_SCREENED_SCALED` | 同时使用目标缩放和精确 LP 预筛 |
| `CLARABEL_SCALED_FAER` | 缩放目标，并把 Clarabel 直接线性求解后端指定为 FAER，用于和默认 QDLDL 做生产 CPU A/B |

目标变换不改变最优持仓：预算约束 $\mathbf1^{\mathsf T}x=1$ 下，从所有 alpha 系数减去
同一个常数只改变目标常数项；再乘正数也不改变最优解。脚本仍用原始 alpha 重算并输出
`raw_objective`。

LP 预筛也不是把 LP 和 SOCP 两类策略混用。SOCP 可行域是相同线性约束可行域的子集；如果线性问题的全局最优解已经满足 6% 风险约束，它同时就是 SOCP 的全局最优解。若不满足，脚本仍求解完整 SOCP。为控制数值误判，默认还要求 TE 至少低于上限 `1e-6` 个百分点，且独立重算最大约束残差不超过 `1e-7`。`runs.csv` 中的 `screen_accepted`、`fallback_used` 和 `solver_path` 会逐日记录实际路径。

生产命令不设置 GPU，也不安装 GPU 依赖。

## 原始默认矩阵

先运行默认完整 CPU 矩阵：

```bash
python benchmark.py \
  --risk-model-xlsx /path/to/datayes_data_202608.xlsx \
  --benchmark-csv /path/to/000852.SH.csv \
  --mosek-license /path/to/mosek.lic \
  --output-dir results/default
```

默认求解器如下：

- LP：HiGHS、MOSEK、Clarabel
- QP：HiGHS、Clarabel、MOSEK
- SOCP：MOSEK、Clarabel、ECOS

QOCO、SCS、OSQP 没有进入默认矩阵：当前开发机上 QOCO/SCS 的 SOCP 很慢，OSQP 不支持二次风险约束。仍可显式补测，例如：

```bash
python benchmark.py \
  --risk-model-xlsx /path/to/datayes_data_202608.xlsx \
  --benchmark-csv /path/to/000852.SH.csv \
  --models SOCP \
  --socp-solvers QOCO,SCS \
  --output-dir results/slow_candidates
```

只做快速冒烟测试：

```bash
python benchmark.py \
  --risk-model-xlsx /path/to/datayes_data_202608.xlsx \
  --benchmark-csv /path/to/000852.SH.csv \
  --models LP \
  --lp-solvers HIGHS \
  --max-dates 2 \
  --output-dir results/smoke
```

## 输出

- `summary.csv`：按模型和 case 汇总中位数、均值、范围、2500 次外推、预筛次数及最大约束残差。对预筛/回退混合 case 应使用 `projected_2500_mean_min`，不能使用可能被大量快速通过日主导的中位数外推。
- `runs.csv`：逐日耗时、原始目标值、TE、换手率、预筛/回退路径及每一类约束残差。
- `metadata.json`：CPU、Python、包版本、完整命令和失败用例。
- `weights.csv.gz`：仅在指定 `--save-weights` 时输出逐日权重。
- `solution_comparisons.csv`：指定 `--save-weights` 时，逐日比较各 case 与参考解的权重 L1、单股最大差、原始目标差和 TE 差。

把整个结果目录压缩后发回即可分析。第二版请返回全部五个文件，尤其是 `weights.csv.gz`，这样才能判断提速是否以解偏差为代价。

例如：

```bash
tar -czf production_v2_results.tar.gz -C results production_v2
```

## 口径说明

默认使用 `--formulation optimized`，但所有化简都是可证明等价的：

- 在预算守恒下，L1 换手率用正向交易量表示；初始权重为零的股票不创建多余绝对值辅助变量。
- 当 $\lVert x_0-b\rVert_1+T_{\max}\le A_{\max}$ 时，总主动权重约束必然满足，省略其辅助变量。
- 当初始基准内权重扣除最大单边卖出量后仍高于 81% 时，基准成分权重下限必然满足。
- $x\ge0$ 且 $\mathbf1^{\mathsf T}x=1$ 时，$x\le\mathbf1$ 是冗余的。

这些化简不使用换手率推导跟踪误差；TE 始终由当日因子协方差、因子暴露和特异风险独立计算。若需核对原始直观建模，可加 `--formulation canonical`，但速度会更慢。

若上一日实际持仓中有超过 `1e-8` 的权重在下一日风险数据中消失，脚本默认报错，避免静默改变回测持仓。只有明确接受“删除缺失持仓后归一化”时才使用 `--missing-holding-policy renormalize`。

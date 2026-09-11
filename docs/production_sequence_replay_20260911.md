# 生产中证1000多期数据开发机重放

测试日期：2026-09-11。输入为生产机通过 `Tuda2DataSource` 导出的
`tmp/prod_csi1000_20260705_20260824.zip`。包内环境为 Python 3.12.11、optim
`3.2.9.dev3+g211585be9.d20260911`；开发机使用 Python 3.11.11，并分别测试工作区源码和
本机重新构建的 Cython wheel。

## 输入确认

- 36 个交易日：2026-07-06—2026-08-24；每日 4881–4888 只股票。
- 175851 行 alpha、tradable、风险敞口和特异风险；42 个风险因子。
- 35 段生产 close-to-close 区间收益，共 161 个缺失数值。没有预填零；序列仍按实际持仓
  检查，包内缺失持仓质量容差为 5%。
- 没有显式初始持仓；链式、首期免换手、失败停止、输出稀疏权重。
- 约束与生产报告一致：主动权重 0.004、总主动 1.8、换手 0.05、风格默认 ±0.3、
  size ±0.2、行业 ±0.01；无 TE 预算。
- 包中保存的后端为 MOSEK；重放时只覆盖 backend，其他业务输入和策略不变。

## 完整入口性能

每个重放命令使用全新 Python 进程；ZIP/Parquet 加载发生在 wall-clock 计时之前。开发机
普通运行不开 profiler 和进度条。首轮预热后交换顺序，各正式运行两次。

| 运行形态 | 后端 | 完整入口吞吐 | 逐期 prepare 均值 | 原生 solve 均值 | 逐期 total 均值 |
|---|---|---:|---:|---:|---:|
| 生产 wheel 原调用 | MOSEK | 未提供完整 wall | 0.2474 s | 0.1742 s | 0.4742 s |
| 生产新进程重放 | auto / HiGHS | 逐期 2.61 期/s | 0.2547 s | 0.1029 s | 0.3831 s |
| 生产新进程重放 | MOSEK | 逐期 2.12 期/s | 0.2287 s | 0.1735 s | 0.4727 s |
| 开发机源码 | auto / HiGHS | 9.88 期/s | 0.0357 s | 0.0477 s | 0.0904 s |
| 开发机源码 | MOSEK | 6.80–7.28 期/s | 0.0363–0.0380 s | 0.0772–0.0832 s | 0.1259–0.1345 s |
| 开发机 Cython wheel | auto / HiGHS | 10.13 期/s | 0.0342 s | 0.0471 s | 0.0883 s |
| 开发机 Cython wheel | MOSEK | 7.46 期/s | 0.0340 s | 0.0782 s | 0.1238 s |

按逐期 `total_s`，最新生产 MOSEK 为 2.11 期/s；开发机同数据 Cython wheel 为 8.08
期/s。生产 prepare 比开发机 wheel 约慢 7.3 倍，原生 MOSEK solve 约慢 2.2 倍，验收约慢
10 倍。慢化横跨 Python 编译/哈希、原生求解和结果验收，不符合单一业务日期或一个模型分支
独自导致的特征。

开发机 Cython wheel 没有复现退化，且比源码略快，因此当前证据排除“_core/_impl Cython
封装本身导致生产 prepare 退化”。真实生产数据在开发机也没有复现退化，因此同时排除数据
规模、CSI1000 基准、真实收益漂移本身是充分原因。

生产机随后通过 shell 全新 Python 进程重放，仍与原 Notebook 基本一致，因此也排除长期
Notebook kernel 状态是充分原因。auto 确实快于 MOSEK，但 prepare 在两个后端中都约
0.23–0.25 秒；HiGHS 和 MOSEK 原生求解相对开发机也均慢约 2.2 倍。慢化跨越不同原生库和
Python 验收，下一步应定位生产 CPU 配额、调度、运行时 hook、GC 和单核实际执行效率。

运行时探针进一步确认，Jupyter 容器使用 Xeon Gold 6126，采样频率约 2.59 GHz；开发机为
Core Ultra 5 225H，采样频率约 3.69 GHz。生产微基准的进程 CPU 时间几乎等于 wall time，
没有 trace/profile hook、tracemalloc、主缺页或持续上下文切换，因此不是文件 I/O 或 Python
调试插桩造成。生产 semantic hash、canonical hash 和 canonical 编译分别约慢 4.3、2.7、
2.7 倍，与旧服务器 CPU 的主频、IPC、缓存和指令集差距相符。

容器的最终探针确认了独立的线程超额订阅问题：进程可见 64 个 CPU，但 cgroup `cpu.max`
只允许每 100 ms 使用 1200 ms CPU，即 12 核总配额；Conda 运行时实际加载的
`/opt/conda/lib/libmkl_rt.so.2` 和 Intel OpenMP 默认各使用 64 线程。未限制线程时，36 期
运行有 86 个 cgroup period 被限流，累计 `throttled_usec=356050483`；进程 CPU/墙钟比达到
7.93。显式设为单线程后，限流次数和时间均为零，CPU/墙钟比降至 1.02。

同一容器、数据、auto/HiGHS 路线的结果如下。带 profiler 的 64 线程运行与无 profiler 的
单线程运行不能直接用 wall time 做严格微小差异比较，但两倍量级变化和 cgroup 证据足以确认
主因；阶段计时则显示 HiGHS 本身不受影响。

| 容器线程状态 | 36 期 wall | 逐期 compile | HiGHS solve | 逐期 validation | cgroup 限流 |
|---|---:|---:|---:|---:|---:|
| MKL/OpenMP 默认 64 线程 | 15.803 s | 0.2445 s | 0.1031 s | 0.0120 s | 86 次 |
| MKL/OpenMP 显式 1 线程 | 7.854 s | 0.0769 s | 0.1039 s | 0.0029 s | 0 次 |

单线程容器按完整 wall 为 4.58 期/s，按逐期 `total_s` 为 5.08 期/s；相同输出指标与默认
线程及宿主机结果的最大差异在约 $10^{-15}$。宿主机不受 cgroup quota 限制，默认运行按逐期
计时为 4.40 期/s，因此“容器比宿主机慢约一倍”并非 CPU 主频单独造成。

开发机使用相同数据做了无 profiler 的串行 A/B：默认线程为 10.80 期/s，显式单线程为
10.74 期/s，仅相差 0.6%；CPU/墙钟比均约 1.01，且没有 cgroup 限流。线程限制在开发机没有
性能收益，也没有实质退化。这说明环境设置的价值在于防止“可见 CPU 数大于实际 quota”时的
超额订阅。当前实现不会在导入 optim 时修改环境，而是在每次优化的数值作用域内动态限制并
在退出时恢复；多期循环只设置一次。

实现侧改为用单线程逐元素归约计算低维因子敞口，并复用验收阶段已经计算的因子主动敞口，
避免小型 GEMV 唤醒进程级 BLAS 线程池。开发机改动前后端到端差异小于 0.3%，数值指标差异
小于 $1.5\times10^{-15}$；生产容器未限制线程时的收益仍需用新 wheel 复测确认。

线程策略进入公共 `SolverTuning` 后，又使用同一 36 期包做了完整回放。`threads="auto"`
将 BLAS 作用域限制为一，但 HiGHS 保留原生 `threads=0` 自动调度；36 期 wall 为 3.373 秒，
即 10.67 期/秒，逐期 total/compile/HiGHS solve/validation 均值分别为
0.0827/0.0288/0.0472/0.0011 秒，与此前显式环境变量基准相当。若把 HiGHS 也机械固定为一，
同机回放会退化到约 4.75–4.95 秒，因此 `auto` 必须是按后端校准的策略，而不是同一个整数
广播给所有数值库。线程作用域退出后 OpenBLAS 从运行中 1 线程恢复到调用前 14 线程；单期
MOSEK 与 Clarabel 路线也分别报告了实际原生线程数。

## 数值一致性

开发机分别完整链式求解 auto 和 MOSEK。36 期均 optimal：

- 最大目标值绝对差：`7.40e-8`；
- 最大单股权重绝对差：`7.30e-7`；
- 最大 TE 绝对差：`1.37e-7`；
- 最大换手率绝对差：`8.74e-8`；
- 两条路线的最大公共报告约束残差：`8.33e-6`。

因此本例 auto 的性能收益不是通过改变业务解或跳过失败日期得到的。

## 运行时探针

使用 `benchmarks/production_runtime_probe.py` 在生产机采集不含权重的运行时证据：

```bash
python benchmarks/production_runtime_probe.py \
  prod_csi1000_20260705_20260824.zip \
  --output-dir runtime_probe_auto \
  --repeats 7 \
  --measure-backend auto
```

探针记录 wall/进程 CPU 时间、上下文切换、page fault、CPU affinity、cgroup quota、已知线程
变量、实际线程池（安装 `threadpoolctl` 时）、trace/profile hook、tracemalloc、GC 开关和
同一真实问题的 semantic hash、canonical hash、完整编译微基准；还会记录整个未插桩重放
期间 `cpu.stat` 的限流增量。只读取指定的无敏感系统状态，不导出通用环境变量。需要函数级
热点时可另加 `--profile-backend auto`；`sequence.prof` 有额外开销，
`performance_profiled.csv` 不能用于吞吐比较。把整个输出目录压缩后传回即可，无需再次
传权重或风险数据。

旧 wheel 或同进程中不经过 optim 的其他数值程序，仍可在专用 worker 容器中显式配置：

```bash
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
KMP_BLOCKTIME=0
OMP_WAIT_POLICY=PASSIVE
```

这些变量影响整个进程，不由库在导入时写入。新版默认使用
`SolverTuning(threads="auto")`：数值阶段限制 BLAS 并设置可配置后端的实例线程参数，退出后
恢复 BLAS 原状态。由于部分 BLAS 运行库的动态限制本身是进程级，同一进程有其他并发数值
任务时仍建议使用独立优化进程。

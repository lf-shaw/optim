# Clarabel 源码观测与 total_active 消融实验

## 结论

2026-09-09 在开发机对两个实际复现问题测试，确认本批样本的主要性能问题是：
**当前 `total_active` 的绝对值辅助变量表达显著增加 Clarabel 的迭代成本，并改变其迭代路径。**
删除该约束后，随风险预算扩大而快速增加的迭代数大幅下降，与 MOSEK 的耗时差距明显缩小。
这比仅从风险 SOC 维度或风险预算大小解释现象更准确。

本次仅增加实验脚本和源码观测补丁；没有修改生产编译器、后端默认参数或路由。
`auto` 对这些问题仍为 Clarabel；没有重新引入 theta 搜索。

后续已完成保留约束含义的稀疏表达及正负分解对照，见
[total_active 精确稀疏表达实验](total_active_reformulation_20260909.md)。下文“下一步”描述保留原消融实验的时点，最新结果以该后续报告为准。

## 1. 对照设计

- 日期：2026-08-14、2026-08-24，分别来自 `tmp/pp_20260814.tar.gz` 和此前转换的 `tmp/solver_threads_20260909/inputs/pp_te2.zip`。
- 两个输入均不含换手率约束。本报告不外推为有换手率场景的实测结论。
- 风险预算：年化 2%、4%、6%、8%、10%。
- 变体：原问题保留 `total_active=1.8`；对照仅移除 `total_active`。
- 后端：官方 Clarabel 0.11.1（QDLDL）、MOSEK；均固定单线程，数值库也固定单线程。
- 每组重复三次、固定种子交错排列、独立冷启动；预热不计入。共 120 次正式求解。
- 耗时为 `PortfolioOptimizer.solve()` 总墙钟时间的中位数，包括编译和结果验收，不包括复现包加载。

### 总耗时与迭代数

耗时单位为秒。每格 Clarabel 的括号内为迭代数。

| 日期 | TE 预算 | Clarabel 保留 | Clarabel 移除 | MOSEK 保留 | MOSEK 移除 |
|---|---:|---:|---:|---:|---:|
| 08-14 | 2% | 0.346（25） | 0.163（20） | 0.227 | 0.126 |
| 08-14 | 4% | 0.464（36） | 0.180（23） | 0.207 | 0.114 |
| 08-14 | 6% | 0.682（55） | 0.162（19） | 0.231 | 0.126 |
| 08-14 | 8% | 0.765（69） | 0.167（20） | 0.251 | 0.122 |
| 08-14 | 10% | 0.868（74） | 0.195（22） | 0.269 | 0.163 |
| 08-24 | 2% | 0.377（25） | 0.172（21） | 0.215 | 0.116 |
| 08-24 | 4% | 0.475（36） | 0.181（23） | 0.221 | 0.130 |
| 08-24 | 6% | 0.669（56） | 0.182（19） | 0.224 | 0.146 |
| 08-24 | 8% | 0.843（74） | 0.208（26） | 0.245 | 0.139 |
| 08-24 | 10% | 0.878（84） | 0.212（30） | 0.269 | 0.141 |

不是仅有前端开销下降。例如 08-14、10% 的 Clarabel 原生求解时间从 0.729 秒降至 0.115 秒；
迭代数由 74 降至 22，平均每迭代约从 9.85 ms 降至 5.21 ms。两种因素都有贡献。
MOSEK 也从移除中获益，但幅度较小；尚未取得其内部预处理后的模型和搜索方向证据，不能将差异确定归因为某一种 MOSEK 内部技术。

### 解的质量与比较边界

120 次求解均返回 `optimal`。所有移除后解的主动权重 L1 均仍低于原上限 1.8，约为 0.810–1.382。
同日期、预算、后端的保留/移除目标值最大绝对差约为 $7.39\times10^{-6}$；
全部结果已报告的最大约束残差为 $4.29\times10^{-6}$。
因此本批对照不是以明显牺牲原约束或目标值换取速度。

但“这些最优候选不触及该约束”不等于“整个可行域中该约束冗余”，更不保证未来问题可删除。
移除后再验收是本次实验检查，不能直接替代生产环境下的等价性证明。

## 2. 当前表达为何值得优先优化

当前编译器用每资产一个辅助变量表达：

$$
u_i\ge w_i-b_i,\qquad u_i\ge b_i-w_i,\qquad
u_i\ge0,\qquad \sum_i u_i\le L.
$$

08-14 有 5208 只资产，因此增加 5208 个变量、10416 个绝对值不等式、5208 个辅助变量非负边界和一条总和约束。
Clarabel 输入矩阵从移除后的 $15793\times5250$ 增大到 $31418\times10458$，
非零元由 85516 增至 116764。
即使最终持仓离 $L=1.8$ 很远，这些辅助变量和不等式仍会参与内点迭代。

后续值得做的有界实验是**精确稀疏表达**。在 $w_i\ge0$、$b_i\ge0$ 和固定预算 $\sum_i w_i=B$ 下，有恒等式：

$$
\|w-b\|_1=B-\sum_i b_i
  +2\sum_{i:b_i>0}\max(b_i-w_i,0).
$$

只需要对基准正权重资产引入卖出侧辅助变量，而非全部资产。
其中 $B-\sum_i b_i$ 不能硬编码为零；允许卖空等不满足前提的模型必须保留一般表达。
该方案本次**尚未实现或验证性能**，但比直接删除用户约束更适合作为下一步。
也可继续研究可证明的冗余消除；已有编译器会利用换手率的三角不等式上界消除部分冗余 `total_active`，本批无换手率输入不满足该规则。

## 3. 原论文与源码观测：步长实际在哪里受限

阅读 [Clarabel 原论文](https://arxiv.org/pdf/2405.12762) 第 3 节，
并在 [v0.11.1 源码](https://github.com/oxfordcontrol/Clarabel.rs/tree/v0.11.1) 上增加只读观测。
锁定提交为 `25540f559592068d0c8a80e46ded1b21760212a1`。

这些模型只有零锥、非负锥和 SOC。程序先计算仿射预测步，再选中心化参数
$\sigma=(1-\alpha_{\mathrm{aff}})^3$，计算校正方向并求可行步长。
非负锥需保持原始松弛和对偶变量为正；因此它们也会截短整个方向的步长，
并非只有风险 SOC 决定步长。这里没有非对称锥的 barrier 回溯分支。
对应 [solver.rs](https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/core/solver.rs)、
[variables.rs](https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/implementations/default/variables.rs)
和 [nonnegativecone.rs](https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/core/cones/nonnegativecone.rs)。

### 保留 total_active 时的实际观测

以下次数统计最终校正方向的最大可行步长由哪一类坐标限制，而非终止时的活跃约束数。

| 日期/预算 | 总迭代 | 非负锥限制 | SOC 限制 | 其中原始侧股票边界限制 | 平均最终步长 |
|---|---:|---:|---:|---:|---:|
| 08-14 / 2% | 25 | 17 | 7 | 11 | 0.586 |
| 08-14 / 10% | 74 | 66 | 7 | 47 | 0.367 |
| 08-24 / 2% | 25 | 18 | 7 | 12 | 0.575 |
| 08-24 / 10% | 84 | 83 | 1 | 57 | 0.314 |

部分问题有一次完整可行步，因此两类限制次数之和可比总迭代少一。
10% 时，限制原始侧步长的股票上下界分别涉及 45、51 条不同边界，并非某只冻结股票重复造成。
预测步随预算增大也整体变小：两个日期平均中心化参数分别从 0.368 增至 0.545、从 0.372 增至 0.572。

**解释边界：**最终的“步长瓶颈坐标”不等于造成不良搜索方向的唯一建模因素。
`total_active` 辅助系统可以间接改变整个牛顿方向；因此“实际截步发生在股票边界”与“移除 total_active 后迭代大降”并不矛盾。
这是本次消融对单纯源码观测的重要补充。也不能将这里的边界坐标当成不可行诊断或自动放松建议。

### KKT 数值观测

- 10 个保留约束的问题均未触发 QDLDL 的动态枢轴正则化；默认静态正则化仍启用。
- 线性系统迭代改进通常一次完成，各问题平均 1.009–1.051 次，最多三次。
- 接受方向的最大绝对残差约为 $5.04\times10^{-11}$。
- 两个问题最后一次迭代的残差高于所请求的改进目标，但只发生在末次迭代，不能解释此前增加的几十次迭代。

没有观察到反复分解失败或改进失效驱动此次增长的证据。
不过未估计条件数，小后向残差不保证小前向误差，不能据此证明所有线性系统都条件良好。
目前没有足够证据将现象定性为 Clarabel 的实现 bug。

## 4. 观测可信度与复现

源码补丁只输出我们自定义的结构化观测记录，不改变生产求解器，也不解析产品诊断用的原生日志文本。
对同一数值输入分别运行：官方 Python wheel、实验 Rust 二进制关闭观测、同一二进制开启观测。
10 个问题全部 `Solved`，三者迭代数相同，**完整原生解向量的最大差均为 0.0**。
源码观测包含额外计算及日志，不用观测运行的耗时做性能结论；上面的耗时来自官方 wheel 消融实验。

文件：

- [消融脚本](../benchmarks/total_active_ablation.py)：120 次正式运行，输出不包含权重。
- [源码观测驱动](../benchmarks/clarabel_observe.py)：输入导出、开/关观测、解析及对照。
- [v0.11.1 观测补丁](../benchmarks/clarabel_observe_v0111.patch)：已在对应源码执行反向 `git apply --check` 验证。
- 统计结果：`tmp/total_active_ablation_20260909/summary.csv` 和 `runs.jsonl`。
- 源码观测：`tmp/clarabel_observe_v2_20260909/checks.json`、`step_limits.csv`、`refinement.csv`、`pivots.csv`。

消融复现命令（需已安装项目依赖、MOSEK 及有效授权，输出目录需不存在）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/total_active_ablation.py --rounds 3 \
  --output tmp/total_active_ablation_repeat
```

源码观测在干净的 Clarabel.rs v0.11.1 源码目录应用补丁后编译：

```bash
git apply /path/to/optim/benchmarks/clarabel_observe_v0111.patch
cargo build --release --example observe
```

回到 optim 根目录，使用一个新输出路径：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/clarabel_observe.py prepare --output tmp/observe_repeat
python benchmarks/clarabel_observe.py run --output tmp/observe_repeat \
  --binary /path/to/Clarabel.rs/target/release/examples/observe
python benchmarks/clarabel_observe.py analyze --output tmp/observe_repeat
```

本机 Rust 编译器和源码放在隔离的 `/tmp/clarabel-observe-zZ7kpf` 下，未替换 conda 中的 Clarabel wheel。
观测目录的原生输入、行标签及完整解包含业务数据，不能作为普通统计报告公开传输；源码补丁不包含这些数据。

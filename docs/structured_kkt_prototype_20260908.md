# 结构化 KKT 有界原型与原生语言实现的判断

历史实验基于 `0b832f4` 时的运行时实现与 v1 复现包。后续默认路线及复现格式已变更；
重现本文原始统计请使用该版本，或迁移为 v2 输入后重新测试，勿混用两轮结果。

## 结论

本轮验证了结构化线性代数的潜力，但没有得到可以替换 Clarabel 的求解器。
生产后端、路由、API、依赖和构建方式均未改变。

原型通过 CVXOPT 的 `conelp(..., kktsolver=...)` 接口复用内点法，只替换 KKT 求解。
它不是 CVXOPT 默认后端与 Clarabel 的全面横向评测，也不是完整组合优化器实现。

数学接口依据：[CVXOPT 自定义 KKT 文档](https://cvxopt.org/userguide/coneprog.html#exploiting-structure)。

## 实验边界

- 真实输入：`tmp/pp.tar.gz`；分别测试原不可行问题、去掉换手率后的 2% TE、以及 6% TE。
- 仅接受一个固定预算的因子风险 SOC、线性域、线性 alpha 目标。
- 固定变量先代入；识别局部连通块，超过 8 阶则拒绝，不静默退化为稠密大系统。
- 支持当前数据中 2 阶/3 阶股票局部块；全局等式要求满行秩。冲突常数约束直接拒绝。
- 不提供原生证书的业务坐标回映射，不作为正式诊断工具。
- 默认最多 80 次内点迭代、60 秒；本次命令设 30 秒。时间预算在 KKT 构建入口检查，
  不是能中断任意一次底层调用的硬超时；快照对照耗时另计。
- 仅在独立 `/tmp/optim_structured_kkt_env` 安装 CVXOPT 1.3.3 和 threadpoolctl 3.6.0；
  不修改当前 conda 环境。BLAS 限制为单线程。
- 完整求解耗时包含原型内的残差记录和快照复制；离线快照分解对照不计入 solve 时间。

## 结构化分解

对线性锥约束与一个风险 SOC，消去锥乘子后的系统为：

$$
H\Delta x+A^{\mathsf T}\Delta y=r_x,
\qquad A\Delta x=r_y,
\qquad H=G^{\mathsf T}W^{-2}G.
$$

风险锥中 $G_q=[0;-R]$，预算只出现在右端项。使用 CVXOPT 的缩放定义
$W_q=\beta(2vv^{\mathsf T}-J)$，$v^{\mathsf T}Jv=1$，可得到：

$$
G_q^{\mathsf T}W_q^{-2}G_q
=\beta^{-2}\left(R^{\mathsf T}R
+8v_0^2R^{\mathsf T}v_rv_r^{\mathsf T}R\right).
$$

由于 $R^{\mathsf T}R=D^2+EFE^{\mathsf T}$，再合并逐资产局部 L1 约束，有
$H=B+UU^{\mathsf T}$。其中 $B$ 是小块对角矩阵，非局部约束进入 $U$。
实现通过小 Schur 系统求解，不构造资产维度的稠密协方差或全局逆矩阵。

首次实验约 10089 个变量，Schur 系统为 128 阶。风格约束与风险因子共享低秩空间，
合并后低秩基底为 43 阶，最终 Schur 系统为 45 阶；含换手率的原问题为 46 阶。

注意：这里合并基底用的是带主元 QR 的数值秩判定，重构相对残差约 1e-14。
这属于原型中的浮点近似验证，不是任意输入下的精确符号消元承诺。
若未来生产化，应优先依据编译器已有的因子语义做精确共享，避免依赖数值秩阈值改变模型。

## 单步与完整求解必须分开评价

真实迭代前中期快照中，结构化分解比相同增广矩阵的 SciPy SuperLU 分解快数倍，
代表性 45 阶压缩快照约 4 毫秒，而对应通用稀疏分解约 20–55 毫秒。
快照的随机右端残差在约 1e-13 量级。

这不是“比 Clarabel KKT 快数倍”的证据：对照的是同一个原型矩阵的 SuperLU，
并非 Clarabel 的实际矩阵、线性代数实现和迭代轨迹。

完整求解结果（单次探索运行，非生产吞吐统计）：

| 2% TE、无换手率的问题 | 容差 | setup/s | solve/s | 迭代 | 状态 |
|---|---:|---:|---:|---:|---|
| 原始 Schur | 1e-8 | 0.154 | 3.598 | 80 | unknown |
| 小系统对角缩放实验 | 1e-8 | 0.122 | 3.877 | 80 | unknown |
| 带主元稀疏增广分解对照 | 1e-8 | 0.117 | 4.930 | 29 | optimal |
| QR 小系统实验 | 1e-8 | 0.119 | 8.263 | 80 | unknown |
| 合并低秩空间的 Schur | 1e-8 | 0.209 | 2.694 | 80 | unknown |
| Schur | 1e-6 | 0.119 | 1.518 | 28 | optimal |
| 合并低秩空间的 Schur | 1e-6 | 0.211 | 1.095 | 28 | optimal |

最后一行的分解合计 0.307 秒，回代合计 0.497 秒；原始线性约束最大残差约
1.47e-10，TE 约 0.0199999992，原生对偶残差约 1.65e-8。
原生 gap 约 5.61e-8，但其单位是缩放后求解目标，不应直接宣称是原始 alpha 的误差上界。

同一独立环境、单线程设置下，对同一 2% TE 无换手问题，预热后交错三轮的中位数为：
Clarabel 端到端 0.355 秒 / solve 0.226 秒，MOSEK 端到端 0.211 秒 / solve 0.080 秒，
均为 optimal。原型较快的成功尝试 setup+solve 仍为约 1.306 秒，并且还未计入 compiler
耗时；因此这轮没有端到端胜出的证据。原型单次值与基线三轮中位数仅用于判断量级，
不应当成严格匹配精度、完整计时范围的加速倍数。

另外两个边界案例：

- 原不可行问题：合并低秩空间、1e-8 容差，22 次迭代，solve 0.820 秒，原生状态为
  `primal infeasible`。尚未完成业务证书回映射，不把它当作完整的诊断验收。
- 6% TE 无换手问题：合并低秩空间、1e-6 容差，80 次迭代，solve 2.709 秒，仍为 `unknown`。

## 当前阻碍不是纯语言开销

严格精度下，Schur 路径后期原始 KKT 方程残差显著放大，虽然持仓候选可行、primal gap
很小，对偶残差却不满足要求。脚本不会把 `unknown` 包装成成功。

同一锥表述改用带主元的稀疏增广分解能够收敛，支持“目前的消元/法方程和回代路径存在
数值稳定性瓶颈”的判断，但没有证明所有问题都由同一个原因造成。
六个原型测试覆盖缩放关系、右端符号、返回乘子尺度、时间预算、压缩 Hessian 与原始
Hessian 等价，以及真实 compiler 风险锥的基准平移；它们通过并不代表
极端条件数或完整内点轨迹已经验证通过。

QR 小系统和简单对角均衡没有解决这批问题；对角均衡在一个低维精度测试中也产生退化，
因此该均衡没有保留为当前默认。停止继续调参，符合本轮有界原型的范围。

## C++ / nanobind 的适用位置

Clarabel 和 PIQP 的主要数值计算已经分别由 Rust 和 C++ 实现，换一层 Python binding
不会自动减少内点迭代、改善条件数或者显著加速底层分解。

自写 KKT 若要继续推进，C++/Eigen（或适用的 BLAS/LAPACK）＋nanobind 可以用于：

- 批量小块分解/回代，融合循环，复用缓冲区，减少临时矩阵；
- 将一次完整 KKT 操作放在原生调用里，而不是逐股票穿越语言边界；
- 使用只读 NumPy 视图或明确的稀疏数组接口，避免隐式 dtype/layout 转换和复制；
- 在不访问 Python 对象的计算段释放 GIL。

nanobind 支持零复制数组交换，但不合适的类型和内存布局仍可能触发隐式复制。
需要明确所有权和禁用不必要的转换，而不是仅把函数签名换成 C++。
[nanobind 数组接口](https://nanobind.readthedocs.io/en/latest/ndarray.html)

本次还不能估计 C++ 加速倍数：分解/回代计时中同时含 NumPy/SciPy 原生计算、内存操作
和 Python 调度，并未做逐调用 CPU profile。尤其严格精度的稳定性问题不会因换语言自行消失。
若继续投入，应先验证更稳定的结构化增广分解和残差控制，再决定是否移植热点。

## 复测

```bash
python -m venv --system-site-packages /tmp/optim_structured_kkt_env
/tmp/optim_structured_kkt_env/bin/python -m pip install \
  -r benchmarks/structured_kkt_requirements.txt

/tmp/optim_structured_kkt_env/bin/python benchmarks/structured_kkt_prototype.py \
  --case no_turnover --method compressed --tolerance 1e-6 \
  --seconds 30 --output tmp/structured_kkt_compressed_1e6.json

/tmp/optim_structured_kkt_env/bin/python -m pytest \
  tests/test_structured_kkt_prototype.py -q
```

`--method` 支持 `schur`、`augmented`、`qr`、`compressed`。所有方案仅保存在研究脚本，
不是新增公共后端。`tmp/structured_kkt_*.json` 只记录数值和时间，不含持仓或大矩阵。

原型测试与现有后端选择、对偶下界、编译/后端回归合计 78 项通过。

## 2026-09-09 补充探索：压缩基底＋增广系统

本轮只探索，不修改生产默认后端或路由。是否及如何改造，等待用户另行确定。

进一步组合前一轮的两项手段：复用 43 阶低秩基底，同时不使用易丢精度的 Schur
回代，而直接对保留局部块和全局耦合的稀疏增广系统做带主元 LU。
这种路径实际仍分解大增广矩阵，45 阶仅表示保留的全局耦合规模，不是整个 LU 的阶数。
另外比较 COLAMD 和 MMD_AT_PLUS_A 排序，并移除了增广路径中不再使用的 Schur
构建/分解及回代计算。没有加入新求解器、改变可行性定义或放宽本轮容差。

| 实验 | 容差 | setup/s | solve/s | 迭代 | 结果 |
|---|---:|---:|---:|---:|---|
| 2% TE，压缩增广，COLAMD | 1e-8 | 0.218 | 3.197 | 29 | optimal |
| 6% TE，压缩增广，COLAMD | 1e-8 | 0.320 | 9.652 | 80 | unknown |
| 2% TE，压缩增广，MMD_AT_PLUS_A | 1e-8 | 0.268 | 30.052 | 26 次 KKT 构建后停止 | 时间预算耗尽 |

MMD 行是移除冗余 Schur 工作之前的探索值，仍能确认该排序在本例严重退化，但不是
与清理后 COLAMD 代码完全相同工作量的严格排序微基准；不能把全部差额归因于排序。

2% TE 的 COLAMD 结果原始线性约束最大残差约 2.89e-12，原生对偶残差约 3.24e-10。
相较之前未压缩增广系统的单次 4.93 秒，有局部性能改善，但仍显著慢于直接 Clarabel。
6% TE 虽然候选可行，原生对偶残差仍约 0.00549，不能因为 gap 小就验收为最优。

因此目前的明确结论仍是：结构可以利用，但原型尚未同时得到稳定性和端到端速度优势。
尤其不能因为一个 2% 案例收敛，就宣布数值问题已经解决。停止本轮继续扩展参数搜索。

原型代数测试扩展到 12 项，包含压缩增广的两种排序；全部通过。低维代数正确性不能
替代上述大问题的完整收敛检查。

```bash
/tmp/optim_structured_kkt_env/bin/python benchmarks/structured_kkt_prototype.py \
  --case no_turnover --method compressed_augmented --seconds 30 \
  --output tmp/structured_kkt_20260909_colamd_2pct.json
```

`--case te6` 可重测 6% 案例，`--method compressed_augmented_mmd` 为 MMD 排序。
MMD 最初结果保存在 `tmp/structured_kkt_compressed_augmented.json`；当时尚未将两种
排序拆成独立 method 名称，因此补充 ordering 元数据说明，保留原始观测值不变。

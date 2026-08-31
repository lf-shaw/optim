# PIQP 0.6.3 inequality dual-index 缺陷定位与修复

测试日期：2026-08-28；开发机环境：Python 3.11、PIQP 0.6.3、NumPy 2.4.6、
SciPy 1.17.1，x86-64 AVX2 sparse wheel。

## 结论

历史偶发 `PIQP_MAX_ITER_REACHED` 的根因已经从“首次 update 生命周期异常”进一步定位为
PIQP 0.6.3 `include/piqp/kkt_system.tpp` 的确定性源码缺陷：恢复一般不等式 dual 时，
有限上下界索引耗尽后会读取索引向量未初始化的尾元素。未初始化值受堆内存布局影响，
因此相同数值输入会在长进程中偶发改变控制流和迭代行为。

旧代码的关键顺序是：

```cpp
Eigen::Index idx_l = i_l < data.n_h_l ? data.h_l_idx(i_l) : -1;
while (idx_l < i && i_l < data.n_h_l) {
    idx_l = data.h_l_idx(++i_l); // ++i_l 可先到达 n_h_l，再发生读取
}
```

补丁改为每次读取前先推进并检查边界：

```cpp
while (i_l < data.n_h_l && data.h_l_idx(i_l) < i) { i_l++; }
Eigen::Index idx_l = i_l < data.n_h_l ? data.h_l_idx(i_l) : -1;
```

上界索引 `i_u` 同样修复。该改动不改变数学模型、KKT 系统、容差或预条件器，只消除
越界范围内的未初始化读取。

## 公开最小复现与 Valgrind

公开复现只有 1 个变量、3 行一般不等式：

$$
\begin{aligned}
\min_x\quad & \frac12x^2-x \\
\text{s.t.}\quad & -1\le x, \\
& x\le2, \\
& x\le3.
\end{aligned}
$$

第 0 行是仅下界，后两行是仅上界，有限下界索引先于有限上界索引耗尽。脚本为
`benchmarks/solver_evaluation/repro_piqp_063_dual_index.py`，不含任何风险模型或业务数据。

```bash
valgrind --tool=memcheck --track-origins=yes --error-exitcode=99 \
  python repro_piqp_063_dual_index.py
```

| wheel | 求解结果 | Valgrind |
|---|---|---:|
| 官方 `piqp==0.6.3` | `PIQP_SOLVED`, `x=1` | 22 errors / 6 contexts |
| `0.6.3+optim.dualidx1` | `PIQP_SOLVED`, `x=1` | 0 errors |

这个例子说明“结果看起来正确”不能排除 undefined behavior；Valgrind 才是本问题的直接
证据。新增的原生 C++ 参数化回归测试在 4 个可用 KKT 后端通过；BLASFEO 后端因本机未
安装 BLASFEO 而按上游测试逻辑跳过。

## 真实冻结 QP

历史失败案例是 2025-06-24 冷启动 top-560、`active_ub=0.004`、`theta=16384`、
`eps=1e-5`。快照只保留传给 PIQP 的 CSC 数值数组，存于临时目录，不进入测试包。

| 项目 | 值 |
|---|---:|
| 数值输入 SHA-256 | `6379c3a0926382674c90333b6a5d2ed304fba4b60c9dfd15cc7b10766070b557` |
| 资产 / 因子 / 变量 | 5153 / 47 / 5760 |
| 等式 / 一般不等式 | 48 / 6320 |
| 有限下界 / 上界 | 5759 / 5760 |
| `nnz(P/A/G)` | 7362 / 97935 / 12032 |

一般不等式中有 5199 行双边、560 行仅下界、561 行仅上界，最后 561 行正是上界索引仍
存在而下界索引已经耗尽的结构。官方 wheel 在该快照上产生 6 个 Valgrind errors；patched
compact 为 0。

## 隔离压力与性能

官方 0.6.3 的旧 compact 路径在同一进程 500 次中成功 493 次，7 次失败集中于
314–315、323–324、410–412。重新运行可能又全部成功，符合未初始化堆内容对控制流的影响。

| 变体 | workspace / QP | 成功 | 失败 | 成功 solve 均值 |
|---|---:|---:|---:|---:|
| 官方 compact，旧首次 update | 500 | 493 | 7 | 0.0951 s |
| 官方 one-sided 安全展开 | 500 | 500 | 0 | 0.1065 s |
| patched compact，旧首次 update（原故障路径） | 1000 | 1000 | 0 | 0.0886 s |
| patched compact continuation，`theta×[1,4,16,64]` | 800 | 800 | 0 | 0.0858 s/QP（按总 solve 均值折算） |

one-sided 是数学等价且无需自定义 wheel 的安全方案，但冻结问题上约慢 17%。patched
compact 消除了内存错误，并保留紧凑双边表示的性能。首个 theta cost 直接传给 setup
仍然保留，因为它省去无意义的首次 update；但它不是源码缺陷的充分修复，官方 compact
即使直接 setup，Valgrind 仍可进入错误的 dual recovery。

## 完整 v5 本机验证

使用 patched compact 和默认固定 seed，运行链式与每日独立冷启动：

| 场景 | 有效日问题 | PIQP QP 子问题 | direct 成功 | fallback | 最大约束残差 |
|---|---:|---:|---:|---:|---:|
| 风险惩罚 QP + 6% factor-QP SOCP，`active_ub=0.004` | 139 | 413 | 139 | 0 | `6.31e-9` |
| 2% factor-QP SOCP，`active_ub=0.01` | 69 | 350 | 69 | 0 | `7.33e-11` |

后一组更容易触发风险边界：最大独立重算年化 TE 为 `1.999975%`。链式
2025-06-16 仍按既定规则因指数调仓导致共同线性模型不可行而跳过，与 PIQP 无关。

6% 场景开发机均值（求解调用，不含全部端到端数据处理）：

| 模型 / 序列 | 均值/日 | fallback |
|---|---:|---:|
| 风险惩罚 QP / 链式 | 0.0809 s | 0 |
| 风险惩罚 QP / 冷启动 | 0.0719 s | 0 |
| factor-QP SOCP / 链式 | 0.4998 s | 0 |
| factor-QP SOCP / 冷启动 | 0.4712 s | 0 |

这些时间用于验证当前 wheel/表示的相对行为，不替代生产机长区间吞吐测试。

## 生产选择

默认 `piqp_inequality_form=auto`：

```text
0.6.3+optim.dualidx1  -> compact（推荐）
官方 0.6.3            -> one_sided（安全兼容，约慢 17%）
未知 wheel            -> one_sided（保守）
```

已知官方 0.6.3 若显式请求 compact，测试框架直接拒绝。Clarabel/MOSEK fallback 仍保留为
一般数值安全网；补丁只修复这个已定位的 undefined behavior，不意味着 PIQP 对所有未知
问题都不再需要状态、残差和业务约束证书。

## 产物

- 源码补丁：`benchmarks/solver_evaluation/piqp-0.6.3-dual-recovery-index.patch`
- 公开复现：`benchmarks/solver_evaluation/repro_piqp_063_dual_index.py`
- patched wheel：`benchmarks/solver_evaluation/wheels/piqp-0.6.3+optim.dualidx1-cp311-cp311-linux_x86_64.whl`
- wheel 信息：`benchmarks/solver_evaluation/BUILD_INFO_PIQP_PATCHED.md`
- 隔离工具：`benchmarks/solver_evaluation/isolate_direct_piqp.py`
- 本机结果：`tmp/v5/piqp_patched_compact_local/`

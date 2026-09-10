# 组合优化器前端性能 contract

状态：实现 gate。后端 benchmark 已证明 direct PIQP/HiGHS 的求解时间足够低，因此数据
对齐、canonical compilation、验证、fingerprint 和结果转换都属于生产关键路径，不能当作
无成本的包装层。

## 1. 性能目标与计时口径

所有 benchmark 同时报告：

```text
data_fetch_s                 # 外部 I/O，单独统计
schedule_alignment_s         # 全区间一次性对齐
static_validation_s          # prepare 阶段
daily_materialization_s      # 日期引用 -> ndarray views/buffers
canonical_structure_s        # 首次结构模板
canonical_values_s           # 当日数值填充
backend_setup_s
backend_solve_s
solution_validation_s
postprocess_s
frontend_hot_path_s          # materialize + compile/update + validate + postprocess
optimization_total_s         # frontend hot path + backend
end_to_end_s                 # 含 data source
peak_rss / allocated_bytes
```

不能只比较 `backend_solve_s`。性能回归 gate 使用真实 LP/QP/factor-QCQP、链式/独立场景的
p50/p95/P99 和 2500 日投影；第一版阈值以当前生产机重新建立的 accepted baseline 为准，
同硬件和依赖版本下前端 p50/p95 不允许无解释回退超过 10%。另设绝对毫秒预算前必须先在
生产机校准，不能拿开发笔记本数字硬编码。

## 2. 两条输入路径

```text
already-aligned PortfolioData/ndarray
    -> 直接进入 compiler，不经过 pandas/carry

DataFrame/tuda2 schedule
    -> prepare 阶段批量对齐一次
    -> date offsets + integer gather maps + immutable array blocks
    -> 日循环只做 slice/gather，不做 merge/reindex/groupby
```

手动 ndarray 是最低开销核心路径。便利 DataFrame API 不得迫使核心 solver 重新构造
MultiIndex 或 pandas 对象。`pd.Series` 权重只在公共返回边界生成；theta 循环和 backend
内部始终使用 NumPy/SciPy buffer。

## 3. 数据对齐工具的责任边界

`carry/carry2` 属于独立数据对齐工具链，其算法选择、正确性、pandas 兼容性和性能回归由
各自子项目测试维护。`optim` 不实现 pandas/carry 自动路由，也不复制它们的 microbenchmark。

组合优化器只定义对齐后的输入 contract，并记录外部对齐阶段的 `schedule_alignment_s`、
版本和 provenance。无论数据层选择哪一个工具，进入 `PreparedPortfolioRun` 后都必须已经
具有唯一资产坐标、日期 offsets、缺失 mask 和连续/明确布局的数组；carry/carry2 不得进入
solver 日循环或 theta 循环。

本项目的性能工作从“对齐完成”这一边界开始，集中在数组物化、动态持仓处理、canonical
结构/数值编译、workspace 输入、独立验证、fingerprint 和结果输出。

### 3.1 XEngine 对齐后热路径裁剪测试

使用 `/home/shao/workspace/repos/XEngine/xserver/data.h5` 的 `/px` key，只从 849 MiB
fixed-format HDF5 底层裁剪最后 200 日及 `close/amount/volume/pre_close` 四列，没有加载
完整 key，也没有执行跨 key 对齐。样本为 2024-09-03--2025-07-04，1,025,450 行、约
67.9 MiB，每日 5,099--5,153 只证券。

| 对齐后的任务 | 200 日中位数 | 每日均摊 |
|---|---:|---:|
| pandas `xs` 后物化并计算收益/可交易/finite | 40.43 ms | 0.2021 ms |
| 预计算 date offsets 后 NumPy 日切片并计算 | 11.97 ms | 0.0598 ms |
| 整块 NumPy 一次计算 | 19.17 ms | 0.0958 ms |
| 已预计算派生列后，仅生成每日 ndarray views | 0.119 ms | 0.0006 ms |
| 每日四列内容 fingerprint | 31.11 ms | 0.1555 ms |
| 逐日构造 dense weight Series | 5.50 ms | 0.0275 ms |
| 按 `1e-5` 清理后构造 sparse weight Series | 11.09 ms | 0.0554 ms |

结论：date offsets + NumPy 日切片比逐日 pandas `xs` 约快 3.4 倍；如果 prepare 已经生成
静态派生数组，日物化本身几乎可以忽略。整块向量化并不总更快，本例因百万行临时数组和
cache 压力反而慢于 5,000 行日切片，所以实现应以实测决定 batch 边界。fingerprint 必须在
prepare 计算一次并供 fallback 复用。sparse Series 的 CPU 构造略慢，但将历史输出行数从
1,025,450 降至 100,000，适合多期存储；单日即时结果不必为了 CPU 速度强制稀疏化。

可复现入口为 `benchmarks/frontend_hotpath_hdf.py`。HDF 裁剪读取耗时受操作系统 page cache
影响，只单独记录，不纳入上述热路径比较。

## 4. 日循环禁止项与缓存边界

### 2026-09-10：批量静态准备落地复核

多期 `prepare_run` 先将暴露、特异波动率、基准权重按实际调仓 `(dt, sid)` 坐标对齐；
逐日基准覆盖仍针对原始权重检查，审计结果和归一化结果供预检、正式求解复用。
风险坐标缺失时保留原始数据，逐日预检定位错误，不以补零修补。源数据和日程在准备后不得
原地修改。持仓漂移仍逐期检查收益缺失、样本外持仓及显式归一化策略。

日期切片用 MultiIndex 日期 codes 一次建立位置表，连续日期块用 slice，不重复扫描完整
索引；不依赖 carry 的 `_generate_index`。普通手工非连续日期布局可建立位置数组；carry2
批量对齐路径仍要求单调索引，不自动排序、不静默降级。协方差仍以每天的行因子为准，
不能把跨历史时期的因子列并集当成每天的因子集合。

测试入口（固定种子 20260910）：

```bash
PYTHONPATH=. python benchmarks/batch_prepare_flow.py --days 120 --assets 5200 --factors 12 --repeats 3
```

开发机合成数据，每天从 5200 只选取 2600 只，源暴露数值约 59.9 MB，对齐后数值约
30.0 MB。这里是数组大小，不是进程峰值内存；额外存储还包括索引、特异风险和基准等。
不存储全部逐日风险对象副本，不包含 I/O、求解和 carry2 自身算法横评。

| 流程 | prepare 中位数 | 再次逐日物化中位数 | 合计中位数 |
|---|---:|---:|---:|
| 日期缓存＋逐日股票对齐 | 0.661 s | 0.406 s | 1.049 s |
| 日期缓存＋批量股票对齐 | 0.543 s | 0.224 s | 0.759 s |

各列独立取中位数，因此分项之和不必严格等于合计。两条路径执行同样的逐日校验，并核对
数值摘要；结果支持减少重复装配，但不能解释为完整回测的相同比例提速。

回归另覆盖真实后端的链式结果一致性、逐日不同样本、基准阈值内归一化的审计保留、
风险/基准缺日期定位、日期切片顺序及 pandas fallback。

求解热路径禁止重复执行：

- pandas `merge/reindex/concat/groupby/stack/unstack`；
- tuda2 单日网络/数据库调用；
- sid 字符串查找或重复创建 `Index`；
- 相同数组的反复 dtype/layout 转换；
- 每个 theta 重建 sparse matrix、constraint registry 或 turnover auxiliary structure；
- 对同一 dense risk block 重复计算内容 hash；
- 为记录结果而逐日保存完整 dense 权重矩阵。

prepare 阶段生成：

```text
date_offsets
asset_coordinate / sid -> integer position
source gather maps and missing masks
factor coordinate
constraint registry
structure_fingerprint
CSC indptr/indices templates
data/provenance content digests
```

当资产数、因子数、约束集合和稀疏模式的 `structure_fingerprint` 相同时，可以复用
canonical template，只填充当日 `data/q/l/u`。这不改变既定的“PIQP workspace 不跨日复用”
可靠性策略：复用的是 Python/compiler 结构模板，不是 solver 内部数值状态。

## 5. 内存与复制规则

- 输入边界最多进行一次明确的 dtype/layout 规范化；每次 copy 必须能归因到字段和阶段。
- solver 内部统一 float64；float32 数据只在进入数值优化前一次转换，不能在每个 backend
  各转一份。
- exposure 保持一种经过 benchmark 选择的布局，不同时长期保存 C/F 两个全量副本。
- sparse CSC 的 `indices/indptr` 在结构相同时复用；只更新 numeric data。
- `PreparedPortfolioRun` 保存全区间一次取足的数据块、offsets、indexers 和轻量日期引用，不在
  日循环中回源取数，也不为有限可预估的风险模型数据增加 chunk/LRU 分支。prepare 阶段应
  避免把 source-owned 全区间数据再次复制成另一套逐日 dense blocks。
- fingerprint 在 ingest/materialize 时流式计算一次，fallback 复用 digest，不重新扫描大块。
- 多期引擎内部只保留当前真实持仓、必要累计收益状态和 theta；历史权重按显式 output policy
  写成阈值清理后的 sparse `(dt,sid,weight)`、流式 callback 或不保存，避免默认 dense history。

## 6. 验证也必须复用计算

独立 validator 不能省略，但应复用公共数学 kernel：一次计算 active weight、factor exposure、
factor/specific risk 后，同时生成 TE、敞口、certificate 和 metrics。不能让 strategy、backend、
validator、reporter 各做一次相同矩阵乘法。

全区间静态 schema/日期/PSD/覆盖率检查只在 prepare 做一次；通过预检后逐日直接编译当日
模板，不重复调用完整静态 validator。链式期初持仓由受控漂移生成并在生成时检查动态条件。
deep infeasibility diagnostics 已确定为显式单问题调用，不进入正常优化计时路径。

## 7. 必须覆盖的性能测试

1. 已对齐 ndarray 单日：LP、QP、factor-QCQP。
2. 已对齐 DataFrame/array block 的单日与 250/2500 日 materialization；不重复测试对齐工具。
3. 约 5,200 assets × 47 factors，包含 benchmark 缺失、不同 dtype/block layout。
4. 链式持仓漂移与独立冷启动；结构稳定日和指数调整日。
5. canonical template 命中/未命中；fallback 时保证不重新对齐/编译原业务模型。
6. 输出权重 `none/sparse/all` 的时间、文件大小和 peak RSS。
7. 冷进程与长时间 2500 次运行，检查缓存增长、RSS 泄漏和 P99 长尾。

准确性测试必须先于速度比较：对齐后输入 contract、缺失 mask、每日权重和、sid 顺序、
风险矩阵因子顺序完全一致后，非对齐性能数据才有效。

# optim 2.x · MOSEK 维护版

此维护线以 2.0.2 为基线，保留 `optim.linopt`、`optim.opt` 和 MOSEK Fusion。与 optim 3.x 同名，建议在独立 Python 环境安装。

## 运行

在已安装依赖且有有效 MOSEK 授权的环境中设置：

```bash
export MOSEKLM_LICENSE_FILE="$HOME/.mosek/mosek.lic"
export TUDA_FZ_URL="file:///path/to/fizzdb"
```

tuda2 快速接口使用其当前 FizzDB 数据布局，默认 library 为 `basic`。运行依赖在 wheel 元数据中声明，包含 `tuda2[fizzdb]>=2.0.40,<3` 和 `carry2>=2.0.2,<3`。

```python
from optim import linopt, opt

weights = linopt.multioptimize(
    universe=universe,                  # (dt, sid)，包含 alpha、member、tradable
    benchmark="000852.SH",              # 或提供带标签的日度基准权重
    risk_model="cne5",                  # 也可选择 datayes
    constraint_style={"SIZE": (-0.2, 0.2), "ALL": (-0.3, 0.3)},
    constraint_industry={"ALL": (-0.01, 0.01)},
    active_ub=0.004,
    total_active_ub=1.8,
    turnover_limit=0.05,                # 双边换手率
    show_progress=True,
)

risk_weights = opt.multioptimize(
    universe, "000852.SH", risk_model="cne5",
    gamma=0.06,                         # 年化跟踪误差上限 6%；省略则为风险调整目标
    use_pre_portfolio=True, turnover_limit=0.05, show_progress=False,
)
```

LP 单期实盘入口仍是 `linopt.optimize(dt, universe, benchmark, ...)`，风格及行业敞口由调用方传入。多期入口按完整区间取数，收益用收盘到收盘的日收益复合。风险模型使用 tuda2 已年化的小数协方差和特异波动率，并按当日协方差行因子对齐。保留 2.0.2 的权重输出规则，`opt` 的小权重截断会使输出权重和略低于预算。

## 构建与测试

在此工作树运行 `python -m build --wheel`。安装用 wheel 在 `dist/`。在有授权的开发环境可用 `python -m pytest -q` 测试真实 MOSEK 求解。

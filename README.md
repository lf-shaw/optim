# optim

`optim` 是面向 A 股因子风险模型的统一组合优化器。3.0.0 起，公共接口使用不可变的
`PortfolioProblem` 描述数据、目标与约束，由 `PortfolioOptimizer` 统一处理单期、多期、
结果验收、失败路线和显式不可行诊断。

```python
from optim import MaximizeAlpha, PortfolioOptimizer

result = PortfolioOptimizer().optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
)
weights = result.require_weights()
```

主要能力：

- 线性 alpha、风险惩罚 QP 和带因子模型 TE 预算的 Factor-QCQP；
- 换手率、主动权重、风格/行业敞口、基准覆盖及单期交易名单；
- close-to-close 持仓自然漂移和统一多期入口；
- 结构化结果、目标证书、fallback 审计和手动 deep infeasibility diagnosis；
- 手动数据、严格内存数据源及可选 tuda2 批量适配。

当前实现与审阅入口见
[`docs/current_implementation_architecture.md`](docs/current_implementation_architecture.md)，
发布规则见 [`docs/core_wheel_distribution.md`](docs/core_wheel_distribution.md)。


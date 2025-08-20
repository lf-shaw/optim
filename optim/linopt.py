"""
mvo优化封装
"""

import warnings

import pandas as pd
import numpy as np
import bottleneck as bn

from tqdm import notebook

import tuda2
import carry

from .solver import Solver, ProblemInfeasible


def __check_df(df: pd.DataFrame, name: str):
    """
    检查 df 类型及名称

    Parameters
    ----------
    df : pd.DataFrame
        需要检测的 DataFrame
    name : str
        df 对应的名称
    """
    if df.empty:
        raise ValueError(f"{name} 为空")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("{0} 必须是 DataFrame".format(name))
    if df.index.nlevels != 2:
        raise TypeError("{0} 的 index 必须是 2 维 MultiIndex".format(name))
    if not df.index.names == ["dt", "sid"]:
        raise ValueError("{0} 的 index 必须是 [dt, sid]".format(name))


def __validate_universe(universe: pd.DataFrame):
    """验证参数 universe"""
    __check_df(universe, "样本空间")

    if "alpha" not in universe:
        raise ValueError("universe 必须包含 alpha 列")
    if "member" not in universe:
        raise ValueError("universe 必须包含 member 列")
    if "tradable" not in universe:
        raise ValueError("universe 必须包含 tradable 列")

    for c in universe.columns:
        if universe[c].dtype == np.float64 and bn.anynan(universe[c]):
            raise ValueError(f"universe 的列 {c} 不能包含任何 nan")


def __validate_benchmark(benchmark: pd.DataFrame):
    """验证参数 benchmark"""
    __check_df(benchmark, "基准")

    if "weight" not in benchmark:
        raise ValueError("基准必须包含 weight 列")

    if bn.anynan(benchmark["weight"]):
        raise ValueError(f"benchmark 的列 weight 不能包含任何 nan")


def __validate_constraint_extra_attrs(
    is_active: bool,
    constraints: dict[str, tuple[float, float]] | None = None,
    extra_attrs: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    align_to_universe: bool = True,
) -> pd.DataFrame | None:
    """验证额外敞口约束"""
    if constraints is not None:
        if is_active and benchmark is None:
            raise ValueError("必须提供基准才能使用主动额外属性敞口约束")
        if extra_attrs is None:
            raise ValueError("必须提供额外属性才能使用额外属性敞口约束")

        __check_df(extra_attrs, "额外参数")

        for c, w in constraints.items():
            if c not in extra_attrs.columns:
                raise ValueError(f"额外属性 {c} 不存在")
            if len(w) != 2 or not isinstance(w, tuple):
                raise ValueError(f"额外属性 {c} 的约束 {w} 不符合 (lb, ub) 格式")

            if w[0] >= w[1]:
                raise ValueError(f"额外属性 {c} 的约束必须满足下限小于上限 {w}")

        if universe is not None and align_to_universe:
            extra_attrs = carry.utils.reindex(extra_attrs, index=universe.index)

    return extra_attrs


def __validate_init_portfolio(
    init_portfolio: pd.Series | None,
    universe: pd.DataFrame,
    turnover_limit: float | None,
    tolerance,
) -> pd.Series | None:
    """验证初始持仓"""
    if init_portfolio is None:
        return None

    if not isinstance(init_portfolio, pd.Series):
        raise TypeError("init_portfolio 必须是 Series")
    if init_portfolio.index.name != "sid":
        raise ValueError("init_portfolio 的 index 必须是 sid")
    if init_portfolio.name != "weight":
        raise ValueError("init_portfolio 的 name 必须是 weight")

    dt = universe.index.get_level_values("dt")[0]
    init_portfolio = init_portfolio.reindex(universe.loc[dt].index)
    init_portfolio.fillna(0.0, inplace=True)

    diff = init_portfolio.sum() - 1
    if abs(diff) > tolerance:
        raise ValueError(
            f"init_portfolio 的权重和不为 1.0，相差 {diff}，可能是权重不足或者有证券不在 universe 中"
        )

    return init_portfolio


def __validate_asset_ub(asset_ub: float | None):
    """验证资产权重上限"""
    if asset_ub is not None:
        if asset_ub <= 0 or asset_ub > 1:
            raise ValueError("个股权重上限 asset_ub 必须位于 (0, 1] 之间")


def __validate_active_asset_ub(active_asset_ub: float | None):
    """验证资产权重上限"""
    if active_asset_ub is not None:
        if active_asset_ub < 0 or active_asset_ub > 1:
            raise ValueError("个股主动权重上限 asset_ub 必须位于 [0, 1] 之间")


def __align_benchmark_with_universe(
    benchmark: pd.DataFrame,
    universe: pd.DataFrame,
    length,
    tolerance,
) -> pd.DataFrame:
    """验证基准与样本空间的匹配度，如果基准不在样本空间中的日平均权重超过阈值则报错

    Parameters
    ----------
    benchmark : pd.DataFrame
        基准
    universe : pd.DataFrame
        样本空间
    length : int
        期数
    tolerance : float, optional
        可以缺失的权重阈值，默认为 0.0001

    Returns
    -------
    pd.DataFrame
        与 universe 对齐过的 benchmark，weight 为权重，缺失值已填 0
    """
    # 检查 基准 与 universe 的匹配度
    benchmark = carry.utils.reindex(benchmark, index=universe.index)
    avg_weight = benchmark.weight.sum() / length
    if avg_weight < 1 - tolerance:
        raise ValueError(f"基准有效权重为 {avg_weight}，却是比例超过阈值 {tolerance}")
    benchmark.fillna(0.0, inplace=True)
    return benchmark


def __validate_customized_weight_and_extra_lists(
    universe: pd.DataFrame,
    blacklist: list[str] | set[str] | None = None,
    freeze_list: list[str] | set[str] | None = None,
    cap_list: list[str] | set[str] | None = None,
    customized_weight: dict[str, float] | dict[str, tuple[float, float]] | None = None,
):
    # 清单：转换成 set
    blacklist = set(blacklist) if blacklist is not None else set()
    freeze_list = set(freeze_list) if freeze_list is not None else set()
    cap_list = set(cap_list) if cap_list is not None else set()

    if not blacklist.isdisjoint(freeze_list):
        raise ValueError("blacklist 与 freeze_list 不能有重叠")
    if not blacklist.isdisjoint(cap_list):
        raise ValueError("blacklist 与 cap_list 不能有重叠")
    if not freeze_list.isdisjoint(cap_list):
        raise ValueError("freeze_list 与 cap_list 不能有重叠")

    sids = universe.index.get_level_values("sid")

    for sid in blacklist:
        if sid not in sids:
            raise ValueError(f"blacklist 中的 {sid} 不在 universe 中")
    for sid in freeze_list:
        if sid not in sids:
            raise ValueError(f"freeze_list 中的 {sid} 不在 universe 中")
    for sid in cap_list:
        if sid not in sids:
            raise ValueError(f"cap_list 中的 {sid} 不在 universe 中")

    if customized_weight is not None:
        if not isinstance(customized_weight, dict):
            raise TypeError("customized_weight 必须是 dict")
        for sid, w in customized_weight.items():
            if sid not in sids:
                raise ValueError(f"customized_weight 中的 {sid} 不在 universe 中")

            if sid in blacklist:
                raise ValueError(f"customized_weight {sid} 位于 blacklist 中")
            if sid in freeze_list:
                raise ValueError(f"customized_weight {sid} 位于 freeze_list 中")
            if sid in cap_list:
                raise ValueError(f"customized_weight {sid} 位于 cap_list 中")

            if isinstance(w, float):
                if w < 0:
                    raise ValueError(f"customized_weight {sid} 权重不能为负数 {w}")
            elif isinstance(w, tuple):
                if len(w) != 2:
                    raise ValueError(
                        f"customized_weight {sid} 权重上下限必须是 (float, float) 形式 {w}"
                    )

                if not (isinstance(w[0], float) and isinstance(w[1], float)):
                    raise TypeError(
                        f"customized_weight {sid} 权重上下限必须是 float 类型 {w}"
                    )
                if w[0] >= w[1]:
                    raise ValueError(
                        f"customized_weight {sid} 权重上下限必须是下限小于上限 {w}"
                    )
            else:
                raise ValueError(
                    f"customized_weight {sid} 权重类型必须是 float 或者 (float, float) 对"
                )


def __validate_risk_exposure_constraints(
    exposure_type: str,
    constraints: dict[str, tuple[float, float]] | None,
    exposures: pd.DataFrame | None,
    universe: pd.DataFrame,
) -> pd.DataFrame | None:
    """验证风险因子敞口约束"""
    if constraints is None:
        return

    if not isinstance(constraints, dict):
        raise TypeError(f"{exposure_type} 的约束必须是字典类型")

    if exposures is None:
        raise ValueError(f"{exposure_type} 指定了约束条件但没有对应的敞口数据")

    # 遍历查找约束的 key 是否存在，检验 value 是否合法
    for k, v in constraints.items():
        if k not in exposures:
            if k.lower() != "all":
                raise ValueError(f"{exposure_type} 的约束 {k} 对应的敞口不存在")
            if k != "all":
                raise ValueError(f"{exposure_type} 的统一约束请使用 `all` 为 key")

        if not (isinstance(v, tuple) and len(v) == 2):
            raise TypeError(
                f"{exposure_type} 的约束 {k} 的条件不是 (float, float) 的类型"
            )
        if not (isinstance(v[0], float) and isinstance(v[1], float)):
            raise TypeError(f"{exposure_type} 的约束 {k} 的上下限必须是 float 类型")

        if v[0] >= v[1]:
            raise TypeError(f"{exposure_type} 的约束 {k} 的条件必须满足下限小于上限")

    exposures = carry.utils.reindex(exposures, index=universe.index)
    return exposures


def optimize(
    dt: pd.Timestamp,
    universe: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    styles: pd.DataFrame | None = None,
    industries: pd.DataFrame | None = None,
    init_portfolio: pd.Series | None = None,
    blacklist: list[str] | None = None,
    freeze_list: list[str] | None = None,
    cap_list: list[str] | None = None,
    customized_weight: dict[str, float] | dict[str, tuple[float, float]] | None = None,
    constraint_style: dict[str, tuple[float, float]] | None = None,
    constraint_industry: dict[str, tuple[float, float]] | None = None,
    turnover_limit: float | None = None,
    budget_weight: float | None = 1.0,
    asset_ub: float | None = None,
    active_ub: float | None = None,
    total_active_ub: float | None = None,
    weight_in_bench_lb: float | None = None,
    extra_attrs: pd.DataFrame | None = None,
    constraint_extra_attrs_active: dict[str, tuple[float, float]] | None = None,
    constraint_extra_attrs_abs: dict[str, tuple[float, float]] | None = None,
    enable_parameter_validation=True,
    weight_tolerance: float = 1e-6,
):
    """
    执行单期优化

    Notes
    -----
    - universe **必须包含** benchmark 的所有证券
    - init_portfolio 的所有股票必须存在于 universe 中
    - blacklist 禁止买入且会卖出所有已有持仓，持仓权重变为 0
    - freeze_list 禁止买卖任何已有持仓，持仓权重保持不变
    - cap_list 禁止买入但不禁止卖出已有持仓，持仓权重不会增加
    - 清单 blacklist/freeze_list/cap_list 不能重叠，且必须位于 universe 中

    Parameters
    -----------
    dt : pd.Timestamp
        优化组合的日期

    universe : pd.DataFrame
        样本空间

        - index 必须为 [dt, sid] 的 multiindex
        - 必须包含样本空间 member 列和交易空间 tradable 列
        - 必须包含预测 alpha
        - 不能包含 nan

    benchmark : pd.DataFrame
        基准权重，用于设定主动敞口等，

        - 必须包含 weight 列
        - 证券必须是 universe 的子集

    styles : pd.DataFrame, , default is None
        风格因子敞口，每一列为一个风格因子，每一行为一个标的

    industries : pd.DataFrame, default is None
        0-1表示的行业因子敞口，每一列为一个行业，每一行为一个标的

    init_portfolio : pd.Series, default is None
        初始持仓，index 为 sid，value 表示权重，必须是 universe 证券的子集

    blacklist : list of str, default is None
        禁止持有的股票列表

    freeze_list : list of str, default is None
        禁止交易的股票列表

    cap_list : list of str, default is None
        禁止买入但不限制卖出的股票列表

    customized_weight : dict or None
        用于设置指定股票的权重，key 为 sid，value 为 (lb, ub) 或者 float。

    constraint_style : dict, default is None
        风险因子敞口约束，默认为 None 即不设置。当设置时，key 为因子，value 为 (lower, upper)。
        不在因子列表里的会被忽略

        **当要统一设置所有因子的敞口时，请设置 key 为 'all' 的约束。**

        =============  ======================
         因子名           中文名称
        =============  ======================
         resvol          残余波动率
         beta            贝塔
         liquidty        流动性
         momentum        动量
         size            规模
         sizenl          中盘
         earnyild        盈利
         btop            价值
         growth          成长
         leverage        杠杆
        =============  ======================

    constraint_industry : dict, default is None
        申万一级行业敞口约束，默认为 None 即不设置。当设置时，key 为行业名，value 为 (lower, upper)。
        不在行业列表里的会被忽略

        **当要统一设置所有行业因子的敞口时，请设置 key 为 'all' 的约束。**

    turnover_limit : float, default is None
        单次调仓 **双边** 换手率限制，默认为 None 即不设置

    budget_weight : float, default is 1
        组合的权重之和，默认为 1 即满仓

    asset_ub : float, default is None
        个股绝对权重上限，None 表示不设置

    active_ub : float, default is None
        个股主动权重上限，None 表示不设置

    total_active_ub : float, default is None
        个股主动权重绝对值之和的上限，None 表示不设置

    weight_in_bench_lb : float, default is None
        组合中属于 benchmark 成分股的权重合计下限，None 表示不设置

    extra_attrs : pd.DataFrame, default is None
        额外属性，每列一个属性

    constraint_extra_attrs_active : dict, default is None
        额外属性主动敞口约束，默认为 None 即不设置。当设置时，key 为属性，value 为 (lower, upper)。
        所设置的属性必须在 extra_attrs 中存在。只有设置了基准才可以设置此选项。

    constraint_extra_attrs : dict, default is None
        额外属性绝对敞口约束，默认为 None 即不设置。当设置时，key 为属性，value 为 (lower, upper)。
        所设置的属性必须在 extra_attrs 中存在。不管是否设置 benchmark 都只考虑绝对敞口约束。

    enable_parameter_validation : bool, default is True
        是否开启参数检查，多期优化时外层会校验参数，建议关闭，单期优化建议开启

    weight_tolerance : float, default is 1e-4
        在将初始持仓和基准权重对齐到 universe 时允许缺失的权重


    Returns
    -------
    pd.Series
        优化权重

    Raises
    ------
    ProblemInfeasible
        不可行时抛出异常
    ValueError
        参数值不合法时抛出异常
    TypeError
        类型不合法时抛出异常
    """

    # 参数校验
    if enable_parameter_validation:
        __validate_universe(universe)
        __validate_benchmark(benchmark)
        __validate_asset_ub(asset_ub)
        __validate_active_asset_ub(active_ub)
        __validate_customized_weight_and_extra_lists(
            universe, blacklist, freeze_list, cap_list, customized_weight
        )
        styles = __validate_risk_exposure_constraints(
            "风格因子", constraint_style, styles, universe
        )
        industries = __validate_risk_exposure_constraints(
            "行业因子", constraint_industry, industries, universe
        )

        __validate_constraint_extra_attrs(
            True, constraint_extra_attrs_active, extra_attrs, universe, benchmark
        )  # 不对齐额外属性
        extra_attrs = __validate_constraint_extra_attrs(
            False, constraint_extra_attrs_abs, extra_attrs, universe, benchmark
        )  # 对齐额外属性

        init_portfolio = __validate_init_portfolio(
            init_portfolio, universe, turnover_limit, weight_tolerance
        )
        benchmark = __align_benchmark_with_universe(
            benchmark, universe, 1, weight_tolerance
        )

    # 样本空间股票列表
    sids = list(universe.loc[dt].index)

    # 预测 alpha
    mu = universe["alpha"].to_numpy()

    if init_portfolio is not None:
        x0 = init_portfolio.to_numpy()
    else:
        x0 = None

    # 1 初始化求解器
    s = Solver(mu, x0)

    # 2 设置benchmark
    bench_arr = benchmark["weight"].to_numpy()
    s.set_benchmark(bench_arr)

    # 3 风格约束
    if constraint_style:
        constraint_style = constraint_style.copy()  # 不改变
        if "all" in constraint_style:
            cols = list(styles.columns)  # type: ignore
            exposure_style = styles.to_numpy()  # type: ignore
            lb_, ub_ = constraint_style.pop("all")
            lb, ub = np.ones(len(cols)) * lb_, np.ones(len(cols)) * ub_

            # 更新 all 指定的约束
            for k, v in constraint_style.items():
                idx = cols.index(k)
                lb[idx] = v[0]
                ub[idx] = v[1]
        else:
            cols = constraint_style.keys()
            exposure_style = styles[cols].to_numpy()  # type: ignore
            lb, ub = np.ones(len(cols)) * 0.0, np.ones(len(cols)) * 1.0

            for i, k in enumerate(cols):
                lb[i] = constraint_style[k][0]
                ub[i] = constraint_style[k][1]

        s.set_exposure_constraint(exposure_style, lb, ub)

    # 4 行业约束
    if constraint_industry:
        constraint_industry = constraint_industry.copy()  # 不改变
        lb = None
        ub = None
        cols = None
        if "all" in constraint_industry:
            cols = list(industries.columns)  # type: ignore
            exposure_industry = industries[cols].to_numpy()  # type: ignore
            lb_, ub_ = constraint_industry.pop("all")
            lb, ub = np.ones(len(cols)) * lb_, np.ones(len(cols)) * ub_

            # 更新 all 指定的约束
            for k, v in constraint_industry.items():
                idx = cols.index(k)
                lb[idx] = v[0]
                ub[idx] = v[1]
        else:
            cols = constraint_industry.keys()
            exposure_industry = industries[cols].to_numpy()  # type: ignore
            lb, ub = np.ones(len(cols)) * 0.0, np.ones(len(cols)) * 1.0
            for i, k in enumerate(cols):
                lb[i] = constraint_industry[k][0]
                ub[i] = constraint_industry[k][1]

        s.set_exposure_constraint(exposure_industry, lb, ub)

    # 额外属性敞口约束
    if constraint_extra_attrs_active:
        for k, v in constraint_extra_attrs_active.items():
            s.set_extra_attr_constrain(
                v[0], v[1], extra_attrs[k].to_numpy(), is_active=True  # type: ignore
            )

    if constraint_extra_attrs_abs:

        for k, v in constraint_extra_attrs_abs.items():
            s.set_extra_attr_constrain(
                v[0], v[1], extra_attrs[k].to_numpy(), is_active=False  # type: ignore
            )

    # 5 设置全部主动股票之和
    if total_active_ub is not None:
        s.set_total_active_constaint(total_active_ub)

    # 6 设置组合所有权重之和
    if budget_weight is not None:
        s.set_budget_constraint(w=budget_weight)

    # 7 设置换手率约束
    if turnover_limit is not None:
        s.set_turnover_constaint(turnover_limit)

    # 8 设置个股的主动权重（绝对值）上限
    if active_ub is not None:
        _active_ub_ = np.ones(len(mu)) * active_ub

    # 9 设置个股绝对权重上限
    if asset_ub is not None:
        _asset_ub_ = np.ones(len(mu)) * asset_ub
    else:
        _asset_ub_ = np.ones(len(mu)) * 1.0  # 未指定时上限为 1.0

    # 10 设置交易空间限制
    tradable = universe["tradable"].to_numpy()
    _asset_ub_[tradable == 0] = 0  # 不可交易的个股权重上限为 0

    if active_ub is not None:
        # 更新不可交易的主动权重的绝对值上限，不低于基准权重（完全不持有时，主动权重为负的基准权重）
        _active_ub_[tradable == 0] = bench_arr[tradable == 0]

    # 11 如果指定了黑名单
    if blacklist:
        idx = universe.loc[dt].index
        for sid in blacklist:
            try:
                i = idx.get_loc(sid)  # 个股索引
                _asset_ub_[i] = 0  # 个股权重上限为 0
                if active_ub is not None:
                    # 更新不可交易的主动权重绝对值上限，不低于基准权重（完全不持有时）
                    _active_ub_[i] = bench_arr[i]
            except:
                pass

    # 12 如果指定了冻结清单
    if freeze_list:
        for sid in freeze_list:
            # 只有初始持仓有数据的才需要保留
            if init_portfolio is not None and sid in init_portfolio.index:
                # 当前权重
                w: float = init_portfolio.xs(sid)  # type: ignore
            else:
                w = 0.0  # 相当于黑名单

            # 一定存在
            i = sids.index(sid)
            # 锁定权重
            s.set_single_asset_bound(i, w, w)

            # 冻结股票一般都是停牌的，会先被交易状态设置权重上限为 0，此处需要调整
            if w > 0:
                _asset_ub_[i] = w

            if active_ub is not None:
                # 更新主动权重绝对值上限
                _active_ub_[i] = abs(w - bench_arr[i])

    # 13 如果设置了权重上限清单
    if cap_list:
        for sid in cap_list:
            # 没有持仓的，上限为 0
            if init_portfolio is not None and sid in init_portfolio.index:
                # 当前权重
                w = init_portfolio.xs(sid)  # type: ignore
            else:
                w = 0.0  # 相当于黑名单

            # 个股索引
            i = sids.index(sid)
            # 个股权重上限为 w
            _asset_ub_[i] = w
            if active_ub is not None:
                # 更新主动权重绝对值上限
                _active_ub_[i] = abs(w - bench_arr[i])

    # 14 如果有定制权重
    if customized_weight:
        idx = universe.loc[dt].index
        for sid, ww in customized_weight.items():
            # 个股索引
            i = sids.index(sid)

            if isinstance(ww, float):
                ww = (w, w)

            s.set_single_asset_bound(i, ww[0], ww[1])  # type: ignore # 锁定权重
            if active_ub is not None:
                # 更新主动权重绝对值上限
                _active_ub_[i] = max(
                    abs(ww[0] - bench_arr[i]), abs(ww[1] - bench_arr[i])  # type: ignore
                )

    # 14 设置资产上限
    s.set_asset_ub(_asset_ub_)

    # 15 设置主动股票上限
    if active_ub is not None:
        s.set_asset_active_ub(_active_ub_)

    # 16 设置成分股内权重占比
    if weight_in_bench_lb is not None:
        in_bench_flag = (bench_arr > 0) * 1.0
        s.set_weight_in_bench_lb(weight_in_bench_lb, in_bench_flag)

    # 解模型
    x = s.solve()

    if x is not None:
        res = pd.Series(x, index=universe.index.droplevel(0))
        res.name = "weight"
        # 将优化出来结果小于10^5的权重删除
        res = res.loc[abs(res) >= 10e-5]
        return res


def multioptimize(
    universe: pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame | str,
    *,
    init_portfolio: pd.Series | None = None,
    constraint_style: dict[str, tuple[float, float]] | None = None,
    constraint_industry: dict[str, tuple[float, float]] | None = None,
    turnover_limit: float | None = None,
    budget_weight: float = 1,
    asset_ub: float | None = None,
    active_ub: float | None = None,
    total_active_ub: float | None = None,
    weight_in_bench_lb: float | None = None,
    extra_attrs: pd.DataFrame | None = None,
    constraint_extra_attrs_active: dict[str, tuple[float, float]] | None = None,
    constraint_extra_attrs_abs: dict[str, tuple[float, float]] | None = None,
    weight_tolerance: float = 1e-4,
    turnover_limit_relax_range: tuple[float, int] | None = None,
    show_progress=True,
    **kwargs,
):
    """
    调用 mosek 进行多期组合优化，按照时间先后顺序串行执行。

    Notes
    -----
    - universe **必须包含** benchmark 和 init_porfotlio 的所有证券

    Parameters
    ----------
    universe : pd.DataFrame
        整体样本空间。
        - 使用 ``tuda2.get_universe`` 获取，勿指定 type
        - 必须包含样本空间 member 列和交易空间 tradable 列
        - 必须包含预测 alpha
        - 不能包含 nan

    benchmark : pd.DataFrame or string
        基准指数或者基金指数代码，用于设置主动敞口等。如果是 DataFrame 必须满足如下要求
        - 可以使用 ``tuda2.get_index_weight`` 获取
        - 证券必须是 universe 的子集
        - 必须包含 weight 列

    constraint_style : dict, default is None
        风险因子敞口约束，默认为 None 即不设置。当设置时，key 为因子，value 为 (lower, upper)。
        不在因子列表里的会被忽略

        **当要统一设置所有因子的敞口时，请设置 key 为 'all' 的约束。**

        =============  ======================
         因子名           中文名称
        =============  ======================
        resvol           残余波动率
        beta             贝塔
        liquidty         流动性
        momentum         动量
        size             规模
        sizenl           中盘
        earnyild         盈利
        btop             价值
        growth           成长
        leverage         杠杆
        =============  ======================

    constraint_industry : dict, default is None
        申万一级行业敞口约束，默认为 None 即不设置。当设置时，key 为行业名，value 为 (lower, upper)。
        不在行业列表里的会被忽略
        **当要统一设置所有行业因子的敞口时，请设置 key 为 'all'的约束。**

    turnover_limit : float, default is None
        单次调仓 **双边** 换手率限制，默认为 None 即不设置

    budget_weight : float, default is 1
        组合的权重之和，默认为 1 即满仓

    asset_ub : float, default is None
        个股绝对权重上限，None 表示不设置

    active_ub : float, default is None
        个股主动权重上限，None 表示不设置

    total_active_ub : float, default is None
        个股主动权重绝对值之和的上限，None 表示不设置

    weight_in_bench_lb : float, default is None
        组合中属于 benchmark 成分股的权重合计下限，None 表示不设置

    extra_attrs : pd.DataFrame, default is None
        额外属性，每列一个属性

    constraint_extra_attrs_active : dict, default is None
        额外属性主动敞口约束，默认为 None 即不设置。当设置时，key 为属性，value 为 (lower, upper)。
        所设置的属性必须在 extra_attrs 中存在。只有设置了基准才可以设置此选项。

    constraint_extra_attrs : dict, default is None
        额外属性绝对敞口约束，默认为 None 即不设置。当设置时，key 为属性，value 为 (lower, upper)。
        所设置的属性必须在 extra_attrs 中存在。不管是否设置 benchmark 都只考虑绝对敞口约束。

    weight_tolerance : float, default is 1e-4
        在将初始持仓和基准权重对齐到 universe 时允许缺失的权重

    turnover_limit_relax_range : tuple[float|int, int], default is None
        在优化求解不可行时，放松换手率约束的范围，默认为 None 表示求解失败直接跳过当前求解日期；
        第一个参数是单次放松的步长，第二个参数是最大放松的次数，
        如果设如果第一个参数设置为 0，则直接放弃换手率约束进行求解，但此解可能在后续继续触发不可行警告。

    show_progress : bool, default is True
        是否显示进度条，默认为 True，只在 jupyter 环境下有效。

    kwargs : dict
        其他参数
        - exposure 风险模型敞口数据 DataFrame
        - rtn 多期优化时计算期初持仓权重的收益率 DataFrame，需要交易日连续且覆盖所有优化日期

    Returns
    -------
    Series
        优化之后的权重 weight，index 为 [dt, sid]
    """

    if turnover_limit_relax_range is not None:
        if not isinstance(turnover_limit_relax_range, (tuple, list)):
            raise ValueError("turnover_limit_relax_range 必须是一个 list 或者 tuple")

        if len(turnover_limit_relax_range) != 2:
            raise ValueError(
                "turnover_limit_relax_range 必须是一个 2 个元素的 list 或者 tuple"
            )

        if not isinstance(turnover_limit_relax_range[0], (float, int)):
            raise ValueError("turnover_limit_relax_range[0] 必须是 float/int 类型")
        if not isinstance(turnover_limit_relax_range[1], int):
            raise ValueError("turnover_limit_relax_range[1] 必须是 int 类型")

    # 验证样本空间
    __validate_universe(universe)

    # 优化日期列表
    opt_dts: pd.DatetimeIndex = universe.index.get_level_values("dt").unique()  # type: ignore

    # # 交易日
    # dts = tuda2.get_trading_days(opt_dts[0], opt_dts[-1])

    # 如果基准是是字符串，提取基准日度权重
    if isinstance(benchmark, str):
        benchmark = tuda2.get_index_weight(benchmark, dts=opt_dts, type="daily")
    elif isinstance(benchmark, pd.Series):
        benchmark = benchmark.to_frame("weight")

    __validate_benchmark(benchmark)

    # 检查 基准 与 universe 的匹配度
    benchmark = __align_benchmark_with_universe(
        benchmark, universe, len(opt_dts), weight_tolerance
    )

    # 按需读取 barra 数据
    # 因子暴露度, 将行业因子都展开
    if constraint_style:
        constraint_style = {k.lower(): v for k, v in constraint_style.items()}

        # 约定 exposure 中的 行业敞口为 industry 列

        if "exposure" in kwargs:
            # 参数提供了敞口数据
            exposure = kwargs.pop("exposure")

            # TODO 检查日期覆盖度
        else:
            # 提取
            exposure = tuda2.get_risk_model(
                "exposure", since=opt_dts[0], version="cne5"
            )

        styles = carry.utils.reindex(exposure, index=universe.index)
        industry = styles.pop("industry")
        styles.fillna(0.0, inplace=True)

        # 验证风格因子敞口
        styles = __validate_risk_exposure_constraints(
            "风格因子", constraint_style, styles, universe
        )

    else:
        styles = None
        industry = None

    # 行业敞口
    if constraint_industry:
        constraint_industry = {k.lower(): v for k, v in constraint_industry.items()}
        if industry is None:
            if "exposure" in kwargs:
                industry = kwargs.pop("exposure")["industry"]
                # TODO 检查日期覆盖度
            else:
                industry = tuda2.get_risk_model(
                    "exposure", since=opt_dts[0], version="cne5", factor_type="industry"
                )["industry"]
            industry = carry.utils.reindex(industry, index=universe.index)
        industries = pd.get_dummies(industry, dtype=float)

        industries = __validate_risk_exposure_constraints(
            "行业因子", constraint_industry, industries, universe
        )
    else:
        industries = None

    # 额外属性敞口
    __validate_constraint_extra_attrs(
        True,
        constraint_extra_attrs_active,
        extra_attrs,
        universe,
        benchmark,
        align_to_universe=False,
    )

    __validate_constraint_extra_attrs(
        False,
        constraint_extra_attrs_abs,
        extra_attrs,
        universe,
        benchmark,
        align_to_universe=False,
    )

    if (
        constraint_extra_attrs_active is not None
        or constraint_extra_attrs_abs is not None
    ):
        extra_attrs = carry.utils.reindex(extra_attrs, index=universe.index)

    # 初始持仓
    init_portfolio = __validate_init_portfolio(
        init_portfolio, universe, turnover_limit, weight_tolerance
    )

    # 个股上界与主动上限参数检查
    __validate_asset_ub(asset_ub)
    __validate_active_asset_ub(active_ub)

    # 多期优化一定有 rtn
    if len(opt_dts) > 1:
        # 多期调整时计算初始持仓权重用的收益率
        if "rtn" in kwargs:
            rtn = kwargs.pop("rtn")
            # TODO 检查日期覆盖度
        else:
            # 提取整个区间的收盘收益率
            rtn = tuda2.get_return(
                since=opt_dts[0],
                until=opt_dts[-1],
                freq="D",
                window=1,
                shift=False,
                market_side="close",
                price_type="vwap",
                price_window=0,
            )

        rtn.fillna(0.0, inplace=True)
        rtn = rtn.iloc[:, 0]  # 取 第一列
        rtn.name = "rtn"
        dts = rtn.index.get_level_values("dt").unique()

        # 不同日期的块的起始点和长度
        idx = carry.utils._generate_index(rtn.index).check_dt_monotonic()

    else:
        rtn = None
        dts = opt_dts

        # 不同日期的块的起始点和长度
        idx = carry.utils._generate_index(universe).check_dt_monotonic()

    if show_progress:
        idx = notebook.tqdm(idx)  # 显示进度条

    # 保存结果
    opted_weight = {}

    # 遍历所有(收益率中的)交易日
    for i, (start, length) in enumerate(idx):
        dt = dts[i]

        # 多期优化时要更新初始持仓
        if rtn is not None:

            if init_portfolio is not None:
                # 丢弃 index 中的 dt
                rtn_ = rtn.iloc[start : start + length].reset_index("dt", drop=True)
                rtn_ = rtn_.reindex(init_portfolio.index)
                rtn_.fillna(0, inplace=True)
                init_portfolio = init_portfolio * (1 + rtn_)
                init_portfolio.fillna(0.0, inplace=True)

                ss = init_portfolio.sum()
                if ss > 0:
                    # 归一化
                    init_portfolio = init_portfolio / init_portfolio.sum()
                else:
                    raise ValueError("对齐后的 init_portfolio 净值为 0")

                __turnover_limit = turnover_limit
            else:
                # 初始持仓为 None 时也不指定换手率
                # 不给定初始持仓，或者优化失败时，走到此情形
                __turnover_limit = None

        # 单期优化，rtn 为 None
        else:
            # 不提供初始持仓时忽略换手率控制
            if init_portfolio is None:
                # 首次不控制换手率
                __turnover_limit = None

            else:
                # 否则，也只使用第一期的初始持仓
                __turnover_limit = turnover_limit

        # 不是优化日，跳过
        if dt not in opt_dts:
            continue

        __univ = universe.loc[dt:dt]
        __bench = benchmark.loc[dt:dt]

        if init_portfolio is not None:
            init_portfolio = init_portfolio.reindex(__univ.loc[dt].index)
            init_portfolio.fillna(0.0, inplace=True)
        if styles is not None:
            __styles = styles.loc[dt:dt]
        else:
            __styles = None
        if industries is not None:
            __industries = industries.loc[dt:dt]
        else:
            __industries = None

        if extra_attrs is not None:
            __extra_attrs = extra_attrs.loc[dt:dt]
        else:
            __extra_attrs = None

        try:
            opted_daily = optimize(
                dt,
                __univ,
                __bench,
                styles=__styles,
                industries=__industries,
                init_portfolio=init_portfolio,
                constraint_style=constraint_style,
                constraint_industry=constraint_industry,
                turnover_limit=__turnover_limit,
                budget_weight=budget_weight,
                asset_ub=asset_ub,
                active_ub=active_ub,
                total_active_ub=total_active_ub,
                weight_in_bench_lb=weight_in_bench_lb,
                extra_attrs=__extra_attrs,
                constraint_extra_attrs_active=constraint_extra_attrs_active,
                constraint_extra_attrs_abs=constraint_extra_attrs_abs,
                enable_parameter_validation=False,
            )

        except ProblemInfeasible:
            opted_daily = None
            if __turnover_limit is not None and turnover_limit_relax_range is not None:
                # 放弃换手率约束
                if turnover_limit_relax_range[0] == 0:

                    try:
                        opted_daily = optimize(
                            dt,
                            __univ,
                            __bench,
                            styles=__styles,
                            industries=__industries,
                            init_portfolio=None,
                            constraint_style=constraint_style,
                            constraint_industry=constraint_industry,
                            turnover_limit=None,
                            budget_weight=budget_weight,
                            asset_ub=asset_ub,
                            active_ub=active_ub,
                            total_active_ub=total_active_ub,
                            extra_attrs=__extra_attrs,
                            constraint_extra_attrs_active=constraint_extra_attrs_active,
                            constraint_extra_attrs_abs=constraint_extra_attrs_abs,
                            enable_parameter_validation=False,
                        )

                        warnings.warn(f"{dt} 问题不可行，丢弃换手率约束后完成求解")
                    except ProblemInfeasible:
                        warnings.warn(
                            f"{dt} 问题不可行，丢弃换手率约束后仍无法求解，跳过"
                        )
                    except Exception as e:
                        warnings.warn(f"{dt} 糟糕，出错了: {e}")
                        raise e
                else:
                    # 自动增加换手率约束进行再次优化
                    for i in range(turnover_limit_relax_range[1]):
                        __turnover_limit += turnover_limit_relax_range[0]

                        try:
                            opted_daily = optimize(
                                dt,
                                __univ,
                                __bench,
                                styles=__styles,
                                industries=__industries,
                                init_portfolio=init_portfolio,
                                constraint_style=constraint_style,
                                constraint_industry=constraint_industry,
                                turnover_limit=__turnover_limit,
                                budget_weight=budget_weight,
                                asset_ub=asset_ub,
                                active_ub=active_ub,
                                total_active_ub=total_active_ub,
                                extra_attrs=__extra_attrs,
                                constraint_extra_attrs_active=constraint_extra_attrs_active,
                                constraint_extra_attrs_abs=constraint_extra_attrs_abs,
                                enable_parameter_validation=False,
                            )

                            warnings.warn(
                                f"{dt} 问题在放松 turnover_limit 约束到 {__turnover_limit:.2%} 完成求解"
                            )

                            break  # 跳出
                        except:
                            # 不处理异常
                            pass

            if opted_daily is None:
                warnings.warn(
                    f"{dt} 问题不可行, 直接跳过当期，下期优化也将忽略初始权重和换手率。希望尝试求解请设置 turnover_limit_relax_range"
                )
        except Exception as e:
            opted_daily = None
            warnings.warn(f"{dt} 糟糕，出错了: {e}")
            raise e

        # 有结果返回时
        if opted_daily is not None:
            opted_weight[dt] = opted_daily
            init_portfolio = opted_daily
        else:
            init_portfolio = None  # 求解失败，不设定初始持仓

    # 合并结果
    if opted_weight:
        opted_weight = pd.DataFrame(opted_weight)
        opted_weight.columns.name = "dt"
        opted_weight.sort_index(inplace=True)
        opted_weight = opted_weight.T.stack()
        opted_weight.name = "weight"
        opted_weight = carry.utils.scale_by_dt(opted_weight).weight  # type: ignore

        return opted_weight

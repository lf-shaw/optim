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


__FACTORS__ = pd.Series(
    {
        "size": 1,
        "beta": 1,
        "momentum": 1,
        "resvol": 1,
        "sizenl": 1,
        "btop": 1,
        "liquidty": 1,
        "earnyild": 1,
        "growth": 1,
        "leverage": 1,
        "传媒": 2,
        "电力设备": 2,
        "电子": 2,
        "房地产": 2,
        "纺织服饰": 2,
        "非银金融": 2,
        "钢铁": 2,
        "公用事业": 2,
        "国防军工": 2,
        "环保": 2,
        "机械设备": 2,
        "基础化工": 2,
        "计算机": 2,
        "家用电器": 2,
        "建筑材料": 2,
        "建筑装饰": 2,
        "交通运输": 2,
        "煤炭": 2,
        "美容护理": 2,
        "农林牧渔": 2,
        "汽车": 2,
        "轻工制造": 2,
        "商贸零售": 2,
        "社会服务": 2,
        "石油石化": 2,
        "食品饮料": 2,
        "通信": 2,
        "医药生物": 2,
        "银行": 2,
        "有色金属": 2,
        "综合": 2,
        "country": 3,
    }
)

# 因子列表，对于对齐数据使用
STYLE_list = __FACTORS__[__FACTORS__ == 1].sort_index().index.to_list()
INDUSTRY_list = __FACTORS__[__FACTORS__ == 2].sort_index().index.to_list()
COUNTRY_list = ["country"]
ALL_RM_list = COUNTRY_list + STYLE_list + INDUSTRY_list


def __check_df(df, name):
    """
    检查 df 类型及名称

    Parameters
    ----------
    df : pd.DataFrame
        需要检测的 DataFrame
    name : str
        df 对应的名称
    """
    assert isinstance(df, pd.DataFrame), "{0} 必须是 DataFrame".format(name)
    assert isinstance(df.index, pd.MultiIndex), "{0} 的 index 必须是 MultiIndex".format(
        name
    )
    assert df.index.names == ["dt", "sid"], "{0} 的 index 必须是 [dt, sid]".format(name)


def __optimize(
    dt,
    universe,
    benchmark,
    styles,
    industries,
    in_bench_flag,
    init_portfolio=None,
    blacklist=None,
    freeze_list=None,
    cap_list=None,
    customized_weight=None,
    constraint_factor=None,
    constraint_industry=None,
    turnover_limit=None,
    budget_weight=1,
    asset_ub=None,
    active_ub=None,
    total_active_ub=None,
    weight_in_bench_lb=None,
):
    """
    执行单期优化

    Notes
    -----
    - 忽略了大部分数据的类型和属性检查，主要通过 multioptimize 控制
    - 参数含义 与 multioptimize 不一样
    - 不建议单独使用
    """
    # 预测 alpha
    mu = universe["alpha"].values

    if init_portfolio is not None:
        x0 = init_portfolio.values
    else:
        x0 = None

    # 1 初始化求解器
    s = Solver(mu, x0)

    # 2 设置benchmark
    bench_arr = benchmark["weight"].values
    s.set_benchmark(bench_arr)

    # 3 风格约束
    if constraint_factor:
        constraint_factor = constraint_factor.copy()  # 不改变
        lb = None
        ub = None
        cols = None
        if "all" in constraint_factor:
            cols = STYLE_list
            exposure_style = styles[cols].values
            lb_, ub_ = constraint_factor.pop("all")
            lb, ub = np.ones(len(cols)) * lb_, np.ones(len(cols)) * ub_

            # 更新
            for k, v in constraint_factor.items():
                idx = cols.index(k)
                lb[idx] = v[0]
                ub[idx] = v[1]
        else:
            cols = constraint_factor.keys()
            exposure_style = styles[cols].values
            lb, ub = np.ones(len(cols)) * 0.0, np.ones(len(cols)) * 1.0
            for i, k in enumerate(cols):
                lb[i] = constraint_factor[k][0]
                ub[i] = constraint_factor[k][1]

        s.set_exposure_constraint(exposure_style, lb, ub)

    # 4 行业约束
    if constraint_industry:
        constraint_industry = constraint_industry.copy()  # 不改变
        lb = None
        ub = None
        cols = None
        if "all" in constraint_industry:
            cols = INDUSTRY_list
            exposure_industry = industries[cols].values
            lb_, ub_ = constraint_industry.pop("all")
            lb, ub = np.ones(len(cols)) * lb_, np.ones(len(cols)) * ub_

            # 更新
            for k, v in constraint_industry.items():
                idx = cols.index(k)
                lb[idx] = v[0]
                ub[idx] = v[1]
        else:
            cols = constraint_industry.keys()
            exposure_industry = industries[cols].values
            lb, ub = np.ones(len(cols)) * 0.0, np.ones(len(cols)) * 1.0
            for i, k in enumerate(cols):
                lb[i] = constraint_industry[k][0]
                ub[i] = constraint_industry[k][1]

        s.set_exposure_constraint(exposure_industry, lb, ub)

    # 5 设置全部主动股票之和
    if total_active_ub is not None:
        s.set_total_active_constaint(total_active_ub)

    # 6 设置组合所有权重之和
    s.set_budget_constraint(w=budget_weight)

    # 7 设置换手率约束
    if turnover_limit is not None:
        s.set_turnover_constaint(turnover_limit)

    # 8 设置个股的主动权重（绝对值）上限
    if active_ub is not None:
        active_ub = np.ones(len(mu)) * active_ub

    # 9 设置个股绝对权重上限
    if asset_ub is not None:
        asset_ub = np.ones(len(mu)) * asset_ub
    else:
        asset_ub = np.ones(len(mu)) * 1.0  # 未指定时上限为 1.0

    # 10 设置交易空间限制
    for i, k in enumerate(universe["tradable"].values):
        # 不可交易标的
        if k == 0:
            asset_ub[i] = 0  # 个股权重上限为 0

            if active_ub is not None:
                # 更新不可交易的主动权重绝对值上限，不低于基准权重（完全不持有时）
                active_ub[i] = bench_arr[i]

    # 11 如果指定了黑名单
    if blacklist:
        idx = universe.loc[dt].index
        for sid in blacklist:
            try:
                i = idx.get_loc(sid)  # 个股索引
                asset_ub[i] = 0  # 个股权重上限为 0
                if active_ub is not None:
                    # 更新不可交易的主动权重绝对值上限，不低于基准权重（完全不持有时）
                    active_ub[i] = bench_arr[i]
            except:
                pass

    # 12 如果指定了冻结清单
    if freeze_list:
        idx = universe.loc[dt].index
        for sid in freeze_list:
            # 只有初始持仓有数据的才需要保留
            if init_portfolio is not None and sid in init_portfolio.index:
                # 当前权重
                w = init_portfolio.xs(sid)
            else:
                w = 0.0  # 相当于黑名单

            try:
                i = idx.get_loc(sid)  # 个股索引
            except:
                pass

            s.set_single_asset_bound(i, w, w)  # 锁定权重

            # 冻结股票一般都是停牌的，会先被交易状态设置权重上限为 0，此处需要调整
            if w > 0:
                asset_ub[i] = w

            if active_ub is not None:
                # 更新主动权重绝对值上限
                active_ub[i] = abs(w - bench_arr[i])

    # 13 如果设置了权重上限清单
    if cap_list:
        idx = universe.loc[dt].index
        for sid in cap_list:
            # 没有持仓的，上限为 0
            if init_portfolio is not None and sid in init_portfolio.index:
                # 当前权重
                w = init_portfolio.xs(sid)
            else:
                w = 0.0  # 相当于黑名单

            try:
                i = idx.get_loc(sid)  # 个股索引
                asset_ub[i] = w  # 个股权重上限为 w
                if active_ub is not None:
                    # 更新主动权重绝对值上限
                    active_ub[i] = abs(w - bench_arr[i])
            except:
                pass

    # 14 如果有定制权重
    if customized_weight:
        for sid, w in customized_weight.items():
            try:
                # 个股索引
                i = idx.get_loc(sid)  # type: ignore
            except:
                pass

            if isinstance(w, float):
                w = (w, w)

            assert w[0] <= w[1], f"customized_weight {w} is invalid"
            s.set_single_asset_bound(i, w[0], w[1])  # 锁定权重
            if active_ub is not None:
                # 更新主动权重绝对值上限
                active_ub[i] = max(abs(w[0] - bench_arr[i]), abs(w[1] - bench_arr[i]))

    # 14 设置资产上限
    s.set_asset_ub(asset_ub)

    # 15 设置主动股票上限
    if active_ub is not None:
        s.set_asset_active_ub(active_ub)

    # 16 设置成分股内权重占比
    if weight_in_bench_lb is not None:
        s.set_weight_in_bench_lb(weight_in_bench_lb, in_bench_flag.values)

    # 解模型
    x = s.solve()

    if x is not None:
        res = pd.Series(x, index=universe.index.droplevel(0))
        res.name = "weight"
        # 将优化出来结果小于10^5的权重删除
        res = res.loc[abs(res) >= 10e-5]
        return res


def multioptimize(
    universe,
    benchmark,
    *,
    init_portfolio=None,
    blacklist=None,
    freeze_list=None,
    cap_list=None,
    customized_weight=None,
    use_pre_portfolio=True,
    rebalance_type="open",
    rebalance_freq="D",
    constraint_factor=None,
    constraint_industry=None,
    turnover_limit=None,
    budget_weight=1,
    asset_ub=None,
    active_ub=None,
    total_active_ub=None,
    weight_in_bench_lb=None,
    show_progress=True,
    **kwargs,
):
    """
    调用 mosek 进行多期组合优化，按照时间先后顺序串行执行。

    Notes
    -----
    - universe **必须包含** benchmark 的所有证券
    - 设置 **use_pre_portfolio=True** 才可以考虑多期的初始持仓和换手影响
    - init_portfolio/blacklist/freeze_list/cap_list 均只适用于即期优化
    - init_portfolio 的所有股票必须存在于 universe 中
    - blacklist 禁止买入且会卖出所有已有持仓，持仓权重变为 0
    - freeze_list 禁止买卖任何已有持仓，持仓权重保持不变
    - cap_list 禁止买入但不禁止卖出已有持仓，持仓权重不会增加
    - 清单 blacklist/freeze_list/cap_list 不能重叠，所有不在 universe 都会被忽略

    Parameters
    ----------
    universe : pd.DataFrame
        整体样本空间。

        - 使用 ``tuda2.get_universe`` 获取，勿指定 m_type
        - 必须包含样本空间 member 列和交易空间 tradable 列
        - 必须包含预测 alpha
        - 不能包含 nan

    benchmark : pd.DataFrame or string
        基准指数或者基金指数代码，用于设置主动敞口等。如果是 DataFrame 必须满足如下要求

        - 可以使用 ``tuda2.get_index_weight`` 获取
        - 证券必须是 universe 的子集
        - 必须包含 weight 列

    init_portfolio : pd.Series, default is None
        优化时的初始持仓的权重。

        - index 为 sid，不包含 dt，因为只有第一期有效
        - 必须是 universe 第一期股票的子集

    blacklist : list of str, default is None
        禁止持有的股票列表

    freeze_list : list of str, default is None
        禁止交易的股票列表

    cap_list : list of str, default is None
        禁止买入但不限制卖出的股票列表

    customized_weight : dict or None
        用于设置指定股票的权重，key 为 sid，value 为 (lb, ub) 或者 float。

    use_pre_portfolio : bool, defalut is True
        多期优化时，使用上一期优化的结果作为初始持仓，默认为 True；若为 False 则 turnover_limit 无效

    rebalance_type : str, defalult is open
        调仓模式，默认为 open-开盘，可以选择 close-收盘

    rebalance_freq : str, default is D
        调仓频率，默认为 D-日频

    constraint_factor : dict, default is None
        风险因子敞口约束，默认为 None 即不设置。当设置时，key 为因子，value 为 (lower, upper)。
        不在因子列表里的会被忽略

        **当要统一设置所有因子的敞口时，请设置 key 为 'ALL'的约束。**

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

        **当要统一设置所有行业因子的敞口时，请设置 key 为 'ALL'的约束。**

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
        组合中属于基准的持仓的权重和的下限，默认为 None 即不设置

    show_progress : bool, default is True
        是否显示进度条，默认为 True，只在 jupyter 环境下有效。

    kwargs : dict
        其他参数

        - exposure 风险模型敞口数据 DataFrame
        - rtn 多期优化时计算期初持仓权重的收益率 DataFrame

    Returns
    -------
    Series
        优化之后的权重 weight，index 为 [dt, sid]
    """
    __check_df(universe, "样本空间")

    assert "alpha" in universe, "universe 必须包含 alpha 列"
    assert (
        "member" in universe and "tradable" in universe
    ), "universe 必须包含样本空间和交易空间列"
    assert not bn.anynan(universe), "universe 不能包含任何 nan"

    # 移除不必要的 level
    universe.index = universe.index.remove_unused_levels()
    # 优化日期序列，根据 universe 的日期列表来
    dts = universe.index.levels[0]

    assert len(dts) > 0, "优化日期不能为空"

    # 清单：转换成 set
    blacklist = set(blacklist) if blacklist is not None else set()
    freeze_list = set(freeze_list) if freeze_list is not None else set()
    cap_list = set(cap_list) if cap_list is not None else set()

    assert blacklist.isdisjoint(freeze_list), "blacklist 与 freeze_list 不能有重叠"
    assert blacklist.isdisjoint(cap_list), "blacklist 与 cap_list 不能有重叠"
    assert freeze_list.isdisjoint(cap_list), "freeze_list 与 cap_list 不能有重叠"

    if customized_weight is not None:
        assert isinstance(customized_weight, dict), "customized_weight 必须是 dict"
        for k, v in customized_weight.items():
            assert k not in blacklist, "customized_weight 与 blacklist 不能重叠"
            assert k not in freeze_list, "customized_weight 与 freeze_list 不能重叠"
            assert k not in cap_list, "customized_weight 与 cap_list 不能重叠"

            assert isinstance(v, float) or isinstance(
                v, tuple
            ), "customized_weight 权重必须是 float 或者 (float, float) 对"

    # 即期优化检查
    if len(dts) > 1:
        assert init_portfolio is None, "多期优化时不能指定 init_portfolio"
        assert not blacklist, "多期优化时不能指定 blacklist"
        assert not freeze_list, "多期优化时不能指定 freeze_list"
        assert not cap_list, "多期优化时不能指定 cap_list"
        assert not customized_weight, "多期优化时不能指定 customized_weight"

    # 如果基准是是字符串，提取基准日度权重
    if isinstance(benchmark, str):
        benchmark = tuda2.get_index_weight(
            benchmark, since=dts[0], until=dts[-1], w_type="daily"
        )

    __check_df(benchmark, "基准")
    assert "weight" in benchmark, "基准必须包含 weight 列"

    # 检查 基准 与 universe 的匹配度
    benchmark = carry.utils.reindex(benchmark, index=universe.index)
    assert (
        benchmark.weight.sum() / len(dts) > 0.9999
    ), "基准权重包含缺失值或者样本空间未完整包含基准成分"

    in_bench_flag = (
        ~np.isnan(benchmark.weight)
    ) * 1.0  # 如果是基准成分股则为 1，否则为 0

    benchmark.fillna(0.0, inplace=True)

    # 按需读取 barra 数据
    # 因子暴露度, 将行业因子都展开
    if constraint_factor:
        constraint_factor = {k.lower(): v for k, v in constraint_factor.items()}

        if "exposure" in kwargs:
            # 参数提供了敞口数据
            styles = kwargs.pop("exposure")
        else:
            # 提取
            styles = tuda2.get_rm_data(
                "exposure", since=dts[0], rm_type="SJ", as_table=True
            )

        styles = carry.utils.reindex(styles, index=universe.index)
        industry = styles.pop("industry")
        styles.fillna(0.0, inplace=True)
    else:
        styles = None
        industry = None

    # TODO 行业采用稀疏矩阵的表达形式
    if constraint_industry:
        constraint_industry = {k.lower(): v for k, v in constraint_industry.items()}
        if industry is None:
            if "exposure" in kwargs:
                industry = kwargs.pop("exposure")
            else:
                industry = tuda2.get_rm_data("exposure", since=dts[0], rm_type="SJ")[
                    "industry"
                ]
            industry = carry.utils.reindex(industry, index=universe.index)
        industries = pd.get_dummies(industry, dtype=float)
    else:
        industries = None

    # 初始持仓,只有第一期有效
    if init_portfolio is not None:
        assert isinstance(init_portfolio, pd.Series), "init_portfolio 必须是 Series"
        assert init_portfolio.index.name == "sid", "init_portfolio 的 index 必须是 sid"
        assert init_portfolio.name == "weight", "init_portfolio 必须是 weight"

        # 必须对齐到 universe
        init_portfolio = init_portfolio.reindex(universe.loc[dts[0]].index)
        init_portfolio.fillna(0.0, inplace=True)
        assert (
            init_portfolio.sum().round(6) == 1.0
        ), "init_portfolio 的权重和必须为 1.0，可能是权重不足或者有证券不在 universe 中"

    # 个股上界参数检查
    if asset_ub is not None:
        assert 0.0 < asset_ub <= 1.0, "个股权重上限 asset_ub 必须位于 (0, 1] 之间"

    if len(dts) > 1:
        # 多期调整时计算初始持仓规模用的收益率
        if "rtn" in kwargs:
            rtn = kwargs.pop("rtn")
        else:
            # 如果是日频开盘调仓模式，更新初始持仓的收益率需要 shift
            if rebalance_freq == "D" and rebalance_type == "open":
                shift = 1
            else:
                shift = 0  # 收盘调仓模式，更新初始持仓的收益率正好是当日收盘收益率！不能 shift

            rtn = tuda2.get_return(
                since=dts[0],
                until=dts[-1],
                freq=rebalance_freq,
                r_type=rebalance_type,
                shift=shift,
                warning=False,
            )

        # 必须对齐到 universe
        rtn = carry.utils.reindex(rtn, index=universe.index)
        rtn.fillna(0.0, inplace=True)
    else:
        rtn = None

    # 保存结果
    opted_weight = {}

    # 不同日期的块的起始点和长度
    idx = carry.utils._generate_index(universe).check_dt_monotonic()
    if show_progress:
        idx = notebook.tqdm(idx)  # 显示进度条

    # 对每一个日期
    opted_daily = None
    for i, (start, length) in enumerate(idx):
        dt = dts[i]

        # 如果使用上一期的优化结果
        if use_pre_portfolio:
            if opted_daily is not None and rtn is not None:
                # 丢弃 index 中的 dt
                rtn_ = rtn.rtn.iloc[start : start + length].reset_index("dt", drop=True)
                opted_daily = opted_daily.reindex(rtn_.index)
                portfolio = opted_daily * (1 + rtn_)
                portfolio.fillna(0.0, inplace=True)

                ss = portfolio.sum()
                if ss > 0:
                    # 归一化
                    __init_portfolio = portfolio / portfolio.sum()
                else:
                    __init_portfolio = None

                __turnover_limit = turnover_limit

            # 处理第一期, 冻结清单只在第一期有效
            else:
                # 不提供初始持仓时忽略换手率控制
                if init_portfolio is None:
                    # 首次不控制换手率
                    __turnover_limit = None
                    __init_portfolio = None

                else:
                    # 否则，也只使用第一期的初始持仓
                    __init_portfolio = init_portfolio
                    __turnover_limit = turnover_limit

        # 否则一直无换手率约束，无冻结清单约束
        else:
            __turnover_limit = None
            __init_portfolio = None

        try:
            __univ = universe.iloc[start : start + length]
            __bench = benchmark.iloc[start : start + length]
            __in_bench_flag = in_bench_flag.iloc[start : start + length]

            if styles is not None:
                __styles = styles.iloc[start : start + length]
            else:
                __styles = None
            if industries is not None:
                __industries = industries.iloc[start : start + length]
            else:
                __industries = None

            opted_daily = __optimize(
                dt,
                __univ,
                __bench,
                __styles,
                __industries,
                __in_bench_flag,
                init_portfolio=__init_portfolio,
                blacklist=blacklist,
                freeze_list=freeze_list,
                cap_list=cap_list,
                customized_weight=customized_weight,
                constraint_factor=constraint_factor,
                constraint_industry=constraint_industry,
                turnover_limit=__turnover_limit,
                budget_weight=budget_weight,
                asset_ub=asset_ub,
                active_ub=active_ub,
                total_active_ub=total_active_ub,
                weight_in_bench_lb=weight_in_bench_lb,
            )

        except ProblemInfeasible as e:
            opted_daily = None
            warnings.warn(f"{dt} Problem is infeasible")
        except Exception as e:
            opted_daily = None
            warnings.warn(f"{dt} Something is wrong: {e}")
            raise e

        # 有结果返回时
        if opted_daily is not None:
            opted_weight[dt] = opted_daily

    # 合并结果
    if opted_weight:
        opted_weight = pd.DataFrame(opted_weight)
        opted_weight.columns.name = "dt"
        opted_weight.sort_index(inplace=True)
        opted_weight = opted_weight.T.stack()
        opted_weight.name = "weight"
        opted_weight = carry.utils.scale_by_dt(opted_weight).weight

        return opted_weight

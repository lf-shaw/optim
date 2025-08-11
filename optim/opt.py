"""
mvo优化封装
"""

import pandas as pd
import numpy as np
from tqdm import notebook

import tuda2
import carry

from .solver import Solver

import gc

# spec_risk, total_risk: 波动率且年化
# cov : 方差，年化


__FACTORS__ = pd.Series(
    {
        "beta": 1,
        "btop": 1,
        "earnyild": 1,
        "growth": 1,
        "leverage": 1,
        "liquidty": 1,
        "momentum": 1,
        "resvol": 1,
        "size": 1,
        "sizenl": 1,
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


def __gen_style_industry_constraint(constraint_dic, _type="factor"):
    """
    用来设置 风格 或 行业的字典约束
    """

    #  风格约束 或 行业约束
    if _type == "factor":
        factors = STYLE_list
    elif _type == "industry":
        factors = INDUSTRY_list

    d_cons_factor = {}
    if constraint_dic is None:
        lb = None
        ub = None
    else:
        assert isinstance(constraint_dic, dict), "额外约束必须是 dict 形式"

        constraint_dic = {k.lower(): v for k, v in constraint_dic.items()}

        if "all" in constraint_dic:
            all_constraint_factor = constraint_dic.pop("all")
        else:
            all_constraint_factor = (-10, 10)

        # 判断是否是合法的因子名称 或 行业名称
        k_list = list(constraint_dic.keys())
        for k in k_list:
            if k not in factors:
                print(
                    f"{k} 不是合法的 {_type} 名称, 与大小写无关，优化继续，忽略 {k} 的 {_type}约束"
                )
                constraint_dic.pop(k)

        # 生成约束字典
        for r in factors:
            d_cons_factor[r] = all_constraint_factor

        # 更新单独约束
        d_cons_factor.update(constraint_dic)
        lb = []
        ub = []
        for k, v in d_cons_factor.items():
            lb.append(v[0])
            ub.append(v[1])
        lb = np.array(lb) * 1.0
        ub = np.array(ub) * 1.0

    return lb, ub


def __optimize(
    dt,
    universe,
    factor_cov,
    ub=1.0,
    init_portfolio=None,
    constraint_factor=(None, None),
    constraint_industry=(None, None),
    black_list=None,
    freeze_list=None,
    customized_weight=None,
    turnover_limit=None,
    ra_comm=0.75,
    ra_spec=0.75,
    gamma=None,
    budget_weight=1,
    active_ub=None,
    total_active_ub=None,
):
    """
    执行单期优化

    Notes
    -----
    - 忽略了大部分数据的类型和属性检查，主要通过 multioptimize 控制
    - 参数含义 与 multioptimize 不一样
    - 不建议单独使用

    Parameters
    ----------
    dt : TYPE
        DESCRIPTION.
    universe : TYPE
        DESCRIPTION.
    factor_cov : TYPE
        因子协方差矩阵
    """

    # 取当天数据
    universe_dt = universe.xs(dt)

    alpha_dt = universe_dt["alpha"].values
    benchmark_dt = universe_dt["weight"].values
    # 协方差矩阵
    factor_cov_dt = factor_cov.xs(dt)[ALL_RM_list].loc[ALL_RM_list]
    F = factor_cov_dt.values

    spec_risk_dt = universe_dt["spec_risk"]
    D = spec_risk_dt.values

    exposure_dt = universe_dt[ALL_RM_list]
    E = exposure_dt.values

    if init_portfolio is None:
        init_portfolio_dt = None
        x0 = None
    else:
        init_portfolio_dt = init_portfolio.xs(dt)["weight"]
        init_portfolio_dt = init_portfolio_dt.reindex(exposure_dt.index)
        init_portfolio_dt = init_portfolio_dt.fillna(0)
        x0 = init_portfolio_dt.values

    # 1 建立 Solver, 给定alpha和初始值
    s = Solver(mu=alpha_dt, x0=x0, is_linear_obj=False)

    # 2 设置benchmark
    s.set_benchmark(benchmark_dt)

    # 3 设置风险约束
    s.set_risk_constaint(F, D, E, gamma, lambda_F=ra_comm, lambda_D=ra_spec)

    # 4 行业约束
    if constraint_industry[0] is not None:
        exposure_industry = universe_dt[INDUSTRY_list].values
        s.set_exposure_constraint(
            exposure_industry, constraint_industry[0], constraint_industry[1]
        )

    # 5 风格约束
    if constraint_factor[0] is not None:
        exposure_style = universe_dt[STYLE_list].values
        s.set_exposure_constraint(
            exposure_style, constraint_factor[0], constraint_factor[1]
        )

    # 6 换手率约束
    if turnover_limit is not None:
        s.set_turnover_constaint(turnover_limit)

    # 7 设置组合所有权重之和
    s.set_budget_constraint(w=budget_weight)

    num_stock = universe_dt.shape[0]
    # 8 个股绝对权重上限, 先留口子，后面一起设置
    if ub is not None:
        asset_ub = np.ones(num_stock) * ub
    else:
        asset_ub = np.ones(num_stock) * 1.0  # 未指定时上限为 1.0
    # s.set_asset_ub(asset_ub)

    # 9 设置主动股票上限
    if active_ub is not None:
        active_ub = np.ones(num_stock) * active_ub
        # s.set_asset_active_ub(active_ub)

    # 10 设置全部主动股票之和：双边主动偏离,个股偏离基准绝对值之和
    if total_active_ub is not None:
        s.set_total_active_constaint(total_active_ub)

    # 11 个股的权重设置, 涉及到冻结名单（不可交易）、黑名单、自定义权重，优先级上冻结名单>黑名单>自定义权重
    # 11.1 冻结名单
    sids_list = exposure_dt.index.to_list()
    # 冻结名单
    if freeze_list is not None:
        for sid in freeze_list:
            # sid 必须要在 universe 里，否则忽略
            if sid not in sids_list:
                print(f">>> freeze_list 里的 {sid} 不在股票池里，忽略")
            else:
                _w = init_portfolio_dt.loc[sid]
                # 找到代码所在的位置, 并设置上下限
                _idx = sids_list.index(sid)
                s.set_single_asset_bound(_idx, _w, _w)
                # 以冻结名单为准
                asset_ub[_idx] = _w
                if active_ub is not None:
                    # 更新冻结的主动权重绝对值上限，就是冻结与基准之差
                    active_ub[_idx] = max(abs(benchmark_dt[_idx] - _w), active_ub[_idx])
    else:
        freeze_list = []

    # 11.2 黑名单
    # 黑名单 black_list
    # 若是Series, 则每期不同
    if isinstance(black_list, pd.Series):
        if dt in black_list.index.unique():
            __black_list = black_list.loc[dt]
            __black_list = __black_list.to_list()
        else:
            __black_list = []
    # 若是 list, 则直接是其本身
    elif isinstance(black_list, list):
        __black_list = black_list
    elif black_list is None:
        __black_list = []
    else:
        raise ValueError("错误的black_list类型")

    # 设置交易空间限制, 若 tradable 在 样本空间里。不可交易时加入black_list里
    if "tradable" in universe_dt:
        __black_list = (
            __black_list + universe_dt.loc[universe_dt["tradable"] == 0].index.to_list()
        )

    for sid in __black_list:
        # sid 必须要在 universe 里，否则忽略
        if sid not in sids_list:
            print(f">>> {dt} black_list 里的 {sid} 不在股票池里，忽略")
        else:
            if sid in freeze_list:
                _w = init_portfolio_dt.loc[sid]
                # 与 freeze_list 重合的话，则以freeze_list 为准
                print(
                    f">>> 禁投标的 [{sid}] 也在 freeze_list 中，将以后者为准设置权重为{_w}"
                )
            else:
                # 找到代码所在的位置, 并设置上下限
                _idx = sids_list.index(sid)
                # s.set_single_asset_bound(_idx, 0, 0)

                asset_ub[_idx] = 0  # 个股绝对权重上限为 0

                if active_ub is not None:
                    # 更新不可交易的主动权重绝对值上限，不低于基准权重（完全不持有时）
                    active_ub[_idx] = max(benchmark_dt[_idx], active_ub[_idx])

    # 11.3 客户自定义
    # 客户自定义名单, 优先级最后
    if customized_weight is not None:
        for sid in customized_weight.keys():
            if sid not in sids_list:
                print(f">>> {dt} customized 里的 {sid} 不在股票池里，忽略")
            else:
                # 与 freeze_list 重合的话，则以freeze_list 为准
                if sid in freeze_list:
                    _w = init_portfolio_dt.loc[sid]
                    print(
                        f">>> 自定义 [{sid}] 也在 freeze_list 中，将以后者 freeze_list 为准设置权重为{_w}"
                    )
                elif sid in __black_list:
                    print(
                        f">>> 自定义 [{sid}] 也在 black_list 中，以 black_list 为准，上下限设置为0"
                    )
                else:
                    _w = customized_weight[sid]

                    if isinstance(_w, float):
                        _w = max(0, min(_w, 1))
                        _w_lb = _w
                        _w_ub = _w
                    elif isinstance(_w, tuple):
                        _w_lb = max(_w[0], 0)
                        _w_ub = min(_w[1], 1)

                    # 找到代码所在的位置, 并设置上下限
                    _idx = sids_list.index(sid)
                    s.set_single_asset_bound(_idx, _w_lb, _w_ub)

                    asset_ub[_idx] = _w_ub  # 个股绝对权重上限为 _w_ub

                    if active_ub is not None:
                        # 更新自定义的主动权重绝对值上限，不低于上下限与benchmark之差
                        active_ub[_idx] = max(
                            [
                                abs(benchmark_dt[_idx] - _w_ub),
                                abs(benchmark_dt[_idx] - _w_lb),
                                active_ub[_idx],
                            ]
                        )

    # 12 设置资产上限
    s.set_asset_ub(asset_ub)

    # 13 设置主动股票上限
    if active_ub is not None:
        s.set_asset_active_ub(active_ub)

    # 解模型
    x = s.solve()

    if x is not None:
        res = pd.Series(x, index=sids_list)
        res.name = "weight"
        # 将优化出来结果小于10^5的权重删除
        res = res.loc[abs(res) >= 10e-5]
        return res


def multioptimize(
    universe,
    benchmark,
    *,
    init_portfolio=None,
    use_pre_portfolio=False,
    dts=None,
    constraint_factor=None,
    constraint_industry=None,
    black_list=None,
    freeze_list=None,
    customized_weight=None,
    turnover_limit=None,
    ra_spec=0.75,
    ra_comm=0.75,
    gamma=None,
    budget_weight=1,
    ub=None,
    active_ub=None,
    total_active_ub=None,
    show_progress=True,
):
    """
    调用 mosek 进行多期组合优化，按照时间先后顺序串行执行。

    Notes
    -----
    - 设置 **use_pre_portfolio=True** 才可以考虑多期的初始持仓和换手影响
    - 通常 init_portfolio 只适合于 **单期** 赋值，use_pre_portfolio = True, 多期优化时 init_portfolio 仅适用首期，其余期采用上一期优化结果
    - universe 会进行 dropna 操作，因此需要事先尽可能补全缺失值
    - universe **必须包含** benchmark 的所有证券，即 [dt, sid]，不包含的缺失值会被删除
    - 优先级排序：customized_weight < blacklist < freeze_list， 冻结股票清单最优先满足，其次是黑名单，最后才是用户自定义的权重要求

    Parameters
    ----------
    universe : pd.DataFrame
        整体样本空间，包含所有需要考虑的证券。

        - index 必须是 [dt, sid]，其中 sid 形如 600000.SH
        - 只有在 universe 中的股票会被被设置 asset，其余忽略
        - 必须 包含预测 alpha 或 score
        - score 者会通过 IC * spec_risk * score 转化为alpha，IC 默认为 0.05
        - 缺失值为 nan，表示无 alpha 或者 属性缺失
        - 可以使用 ``tuda2.get_universe`` 获取

    benchmark : pd.DataFrame
        基准指数，用于设置主动敞口等。

        - index 必须是 [dt, sid]，其中 sid 形如 600000.SH
        - 证券必须是 universe 的子集
        - 必须包含 weight 列
        - 可以使用 ``tuda2.get_index_weight`` 获取

    init_portfolio : pd.DataFrame, default is None
        优化时的初始持仓组合，必须包含 weight 列，可以包含 quantity 列

        - 形同 benchmark，index [dt, sid]，sid 形如 600000.SH，必须是 universe 的子集

    use_pre_portfolio : bool
        使用上一期优化的结果作为初始持仓，默认为 False，此时 turnover_limit 无效

    black_list : list or Series or None
        排除列表，即 universe 中存在但是在优化组合中不会被持有的股票，通常是因为投资限制。

        - 一般请勿与 freeze_list 重复
        - 当为 list 时，指需要排除的股票清单，且每一期都会被排除
        - 当为 Series 时，指指定日期需要排除的股票清单，index 为 dt，值为 sid

    freeze_list : list or None
        冻结股票清单，一般指初始持仓中不能交易的部分，默认为 None，优先级高于 blacklist。

        - 只是保持权重固定（加减仓 组合价值 变化时有区别）
        - 多期指定暂时只有 **第一期有效**

    customized_weight : dict or None
        用于设置指定股票的权重，key 为 sid，value 为 (lb, ub) 或者 float。

    dts : list of date or pd.DatetimeIndex or None
        优化日期序列，必须存在于 benchmark 的日期中。如果为 None, 则使用 benchmark 的时间序列。

    constraint_factor : dict
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

    constraint_ind : dict
        申万一级行业敞口约束，默认为 None 即不设置。当设置时，key 为行业名，value 为 (lower, upper)。
        不在行业列表里的会被忽略

        **当要统一设置所有行业因子的敞口时，请设置 key 为 'ALL'的约束。**

    turnover_limit : float
        最大换手率，默认为 None 即不设置

    ra_spec : float
        个股残差风险厌恶系数，默认为 0.75

    ra_comm : float
        共同风险厌恶系数，默认为 0.75

    gamma : None or float
        最大跟踪误差

    budget_weight : float
        组合的权重之和，默认为1

    ub : float or None
        个股权重上限，None表示不设置,则上限为1

    active_ub : float or None
        个股主动权重上限，None表示不设置

    total_active_ub : float or None
        个股总主动权重，None表示不设置

    show_progress : bool
        是否显示进度条，默认为 True，只在 jupyter 环境下有效。

    Returns
    -------
    Series
        优化之后的权重 weight，index 为 [dt, sid]
    """
    if dts is not None:
        dts = pd.to_datetime(dts)
        dts = dts.sort_values()
        dts = dts.drop_duplicates()
        dts_bench = benchmark.index.get_level_values("dt").unique()

        assert set(dts).issubset(set(dts_bench)), "优化日期必须是基准日期的子集"
        benchmark = benchmark.loc[dts]
    else:
        dts = benchmark.index.get_level_values("dt").unique()
        dts = dts.sort_values()
    dts = list(dts)

    universe = universe.loc[dts]

    # universe 需要的列
    to_concats = []

    # 读取 barra 数据
    # 因子暴露度, 将行业与国家因子都展开
    exposure = tuda2.get_rm_data("exposure", dts=dts, rm_type="SJ")
    industries = pd.get_dummies(exposure["industry"])
    exposure = carry.utils.reindex(exposure, industries)
    exposure["country"] = 1

    # 因子收益率协方差矩阵
    factor_cov = tuda2.get_rm_data("cov", dts=dts, rm_type="SJ")

    # 特异波动率对角阵
    spec_risk = tuda2.get_rm_data("spec_risk", dts=dts, rm_type="SJ")
    spec_risk.dropna(inplace=True)

    if "alpha" in universe:
        alpha1 = universe["alpha"]
    # score to alpha
    elif "score" in universe:
        alpha1 = 0.05 * spec_risk["spec_risk"] * universe["score"]
        alpha1.name = "alpha"
    to_concats.append(alpha1)

    to_concats.extend([exposure, benchmark, spec_risk])

    if "tradable" in universe:
        to_concats.extend([universe["tradable"]])

    # 1. 检查 样本空间 和 基准
    __check_df(universe, "样本空间")
    __check_df(benchmark, "基准")
    assert "weight" in benchmark, "基准必须包含 weight 列"

    # 合并数据
    universe = carry.utils.reindex(*to_concats, index=universe.index)
    # 基准权重用0补齐
    universe["weight"] = universe["weight"].fillna(0)

    universe.dropna(inplace=True)
    universe.index = universe.index.remove_unused_levels()

    # 3、初始持仓,只有第一期
    if init_portfolio is not None:
        init_portfolio = init_portfolio.loc[dts[0] : dts[0]]

    if ub is not None:
        assert 0.0 < ub <= 1.0, "个股权重上限必须位于 (0, 1] 之间"

    begt = dts[0]
    prices = tuda2.get_prices(begt, dts[-1], fields=["close", "adj_factor"])

    # 多期调整时计算初始持仓规模用
    rtn = prices.loc[dts].close.unstack().pct_change()

    # 保存结果
    opted_weight = []

    # 显示进度条
    if show_progress:
        dts = notebook.tqdm(dts)

    # 行业与风格约束每一期都一样
    # 行业约束
    lb_industry, up_industry = __gen_style_industry_constraint(
        constraint_industry, "industry"
    )

    # 风格约束
    lb_style, up_style = __gen_style_industry_constraint(constraint_factor, "factor")

    # 对每一个日期
    opted_daily = None
    for dt in dts:
        # print(dt)
        # 如果使用上一期的优化结果
        if use_pre_portfolio:
            if opted_daily is not None:
                # 为了保留 index 中的 dt，使用字符串时间 loc
                dt_str = dt.strftime("%Y-%m-%d")

                portfolio = (
                    opted_daily.reset_index("dt", drop=True)["weight"]
                    * (1 + rtn.loc[dt_str])
                ).dropna()

                # 归一化
                portfolio = portfolio / portfolio.sum()
                __init_portfolio = pd.DataFrame(portfolio, columns=["weight"])

                # 更改日期为当前
                __init_portfolio["dt"] = dt
                __init_portfolio.set_index("dt", append=True, inplace=True)
                __init_portfolio.index = __init_portfolio.index.swaplevel()

                __turnover_limit = turnover_limit
                __freeze_list = None  # 冻结清单只在第一期有效

            # 处理第一期, 冻结清单只在第一期有效
            else:
                # 不提供初始持仓时忽略换手率控制
                if init_portfolio is None:
                    # 首次不控制换手率
                    __turnover_limit = None
                    __init_portfolio = None
                    __freeze_list = None

                else:
                    # 否则，也只使用第一期的初始持仓
                    __init_portfolio = init_portfolio[dt:dt].copy()

                    __turnover_limit = turnover_limit
                    __freeze_list = freeze_list

        # 否则一直无换手率约束，无冻结清单约束
        else:
            __turnover_limit = None
            __init_portfolio = None
            __freeze_list = None

        try:
            out = __optimize(
                dt,
                universe,
                factor_cov,
                init_portfolio=__init_portfolio,
                constraint_factor=(lb_style, up_style),
                constraint_industry=(lb_industry, up_industry),
                black_list=black_list,
                freeze_list=__freeze_list,
                customized_weight=customized_weight,
                turnover_limit=__turnover_limit,
                ra_comm=ra_comm,
                ra_spec=ra_spec,
                gamma=gamma,
                budget_weight=budget_weight,
                ub=ub,
                active_ub=active_ub,
                total_active_ub=total_active_ub,
            )

        except Exception as e:
            out = None
            print(f"{dt} sth is wrong: {e}")

        # 有结果返回时
        if out is not None:
            opted_daily = out.reset_index()
            opted_daily.columns = ["sid", "weight"]
            opted_daily["dt"] = dt
            opted_daily.set_index(["dt", "sid"], inplace=True)
            opted_weight.append(opted_daily)
        else:
            opted_daily = None

    # 合并结果
    if opted_weight:
        opted_weight = pd.concat(opted_weight, sort=True)
        opted_weight.sort_index(inplace=True)

        return opted_weight["weight"]

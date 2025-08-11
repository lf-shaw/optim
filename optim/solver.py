"""
组合优化求解器
=============

支持求解 `预期收益最大化问题` 与 `风险调整后收益最大化问题`。
"""

# import libs
import os
import datetime
import numbers
import warnings

import numpy as np

from mosek import fusion


def check_license():
    """
    检查 mosek license

    - 如果 MOSEKLM_LICENSE_FILE 环境变量未设置，则抛出异常
    - 如果 MOSEKLM_LICENSE_FILE 对应的文件不存在，则抛出异常
    - 如果解析授权文件中的日期失败，则抛出警告
    - 如果授权文件中的日期已过期，则抛出异常
    - 如果授权文件有效日期不足 30 天，则抛出警告

    """
    if "MOSEKLM_LICENSE_FILE" not in os.environ:
        raise RuntimeError("mosek: 未设置 MOSEKLM_LICENSE_FILE 环境变量，请检查！")

    if not os.path.exists(os.environ["MOSEKLM_LICENSE_FILE"]):
        raise RuntimeError("mosek: MOSEKLM_LICENSE_FILE 指定的文件不存在，请检查！")

    with open(os.environ["MOSEKLM_LICENSE_FILE"], "r", encoding="utf-8") as f:
        feature_found = False  # 是否有 feature

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # 跳过空行和注释
            if line.startswith("FEATURE"):
                feature_found = True
                # FEATURE PTS MOSEKLM 10 07-jul-2023 ...
                parts = line.split()
                if len(parts) >= 5:
                    date_str = parts[4]  # 取第5个字段
                    try:
                        expiry_date = datetime.datetime.strptime(
                            date_str, "%d-%b-%Y"
                        ).date()

                    except ValueError:
                        warnings.warn(
                            f"无法解析授权文件日期: {date_str}", RuntimeWarning
                        )
                        return

                    # 剩余有效日期
                    days_left = (expiry_date - datetime.date.today()).days

                    # 过期
                    if days_left < 0:
                        raise RuntimeError(f"mosek: 授权文件日期 {expiry_date} 过期")

                    # 检查是否不足 30 天
                    if days_left < 30:
                        warnings.warn(
                            f"mosek: 授权文件即将过期，仅剩 {days_left} 天（到期日: {expiry_date}）",
                            RuntimeWarning,
                        )

        if not feature_found:
            warnings.warn("mosek: 授权文件不存在 feature", RuntimeWarning)


class ProblemInfeasible(Exception):
    pass


class Solver:
    """组合优化求解器

    Notes
    ------
    约定在因子协方差矩阵和因子敞口矩阵中，第一列为 country 因子，接下来 nS 列均为风险因子，剩余的为行业因子
    """

    def __init__(
        self, mu: np.ndarray, x0: np.ndarray | None = None, is_linear_obj: bool = True
    ):
        """初始化求解器

        Parameters
        -----------
        mu : np.ndarray
            预期收益率向量, shape=(n,)
        x0 : np.ndarray, default None
            初始权重向量, shape=(n,)
        is_linear_obj : bool, default True
            是否是线性目标函数，如果要最大化风险调整后收益，应该设置为 False
        """
        # 个股的预期收益率
        if not isinstance(mu, np.ndarray):
            raise ValueError(f"mu must be a ndarray not {type(mu)}")
        if not mu.ndim == 1:
            raise ValueError(f"mu must be an 1-d array not {mu.ndim}-d array")
        self.__mu = mu
        self.__n = mu.size

        # 初始权重
        if x0 is not None:
            if not isinstance(x0, np.ndarray):
                raise ValueError(f"x0 must be a ndarray not {type(x0)} if not None")
            if not mu.ndim == 1:
                raise ValueError(
                    f"x0 must be 1-d array not {x0.ndim}-d array if not None"
                )
            if x0.size != self.__n:
                raise ValueError(
                    f"x0 must be 1-d array with size equals to mu's size {self.__n}"
                )
            self.__x0 = x0
        else:
            self.__x0 = None

        # 主动权重
        self.__active_weight = None

        # mosek 模型
        self.__model = fusion.Model()

        # 添加待求的目标权重变量，上下限由参数设置
        self.__xs = self.__model.variable(self.__n, fusion.Domain.greaterThan(0.0))

        # 最大化预期收益率
        if is_linear_obj:
            # assert self.__gamma is not None, '目标风险优化必须指定 gamma'
            # 目标
            self.__model.objective(
                fusion.ObjectiveSense.Maximize, fusion.Expr.dot(self.__mu, self.__xs)
            )
        # 最大化风险调整后收益
        else:
            # 辅助变量（标量）
            self.__s = self.__model.variable(1, fusion.Domain.unbounded())
            # 目标
            self.__model.objective(
                fusion.ObjectiveSense.Maximize,
                fusion.Expr.sub(fusion.Expr.dot(self.__mu, self.__xs), self.__s),
            )

    def set_asset_ub(self, ub: float | np.ndarray | None):
        """设置个股绝对权重上限

        Parameters
        ----------
        lb : np.ndarray or float or None
            个股权重上限，
            - 如果是一个数，则表示统一的上限，
            - 如果是一个 shape=(n,) 的 array 则是每个个股对应的上限，
            - 如果是 None 或其他类型则不设置
        """
        if isinstance(ub, numbers.Number):
            ub = np.full(self.__n, ub, dtype=np.float64)
        elif isinstance(ub, np.ndarray):
            if not (ub.ndim == 1 and ub.size == self.__n):
                raise ValueError(f"ub must be 1-d array with size {self.__n}")
            self.__model.constraint(
                fusion.Expr.sub(self.__xs, ub), fusion.Domain.lessThan(0.0)
            )
        else:
            if ub is not None:
                raise ValueError("ub must be None or a number or an 1-d array")

    def set_single_asset_bound(self, idx: int, lb: float, ub: float):
        """设定指定序号的证券的权重上下限

        Notes
        ------
        频繁调用此函数将严重影响优化速度，应尽量使用 *set_asset_ub* 统一设置

        Parameters
        -----------
        idx : int
            证券序号，从 0 开始
        lb : float
            权重下界
        ub : float
            权重上界
        """
        if not 0 <= idx < self.__n:
            raise ValueError(f"asset [{idx}] out of bounds [0, {self.__n})")
        xi = self.__xs.index(idx)
        # 下界
        self.__model.constraint(fusion.Expr.sub(xi, lb), fusion.Domain.greaterThan(0.0))
        # 上界
        self.__model.constraint(fusion.Expr.sub(xi, ub), fusion.Domain.lessThan(0.0))

    def set_benchmark(self, weight: np.ndarray | None):
        """设置基准权重

        Parameters
        -----------
        weight : np.ndarray
            基准权重, shape=(n,)，如果为None则不设置
        """
        if weight is None:
            return

        if isinstance(weight, np.ndarray):

            if not weight.ndim == 1:
                raise ValueError(f"weight must be 1-d array not {weight.ndim}-d array")
            if not (weight.size == self.__n):
                raise ValueError(
                    f"length of benchmark weight must match length of mu {self.__n}"
                )
            self.__active_weight = fusion.Expr.sub(self.__xs, weight)
        else:
            if weight is not None:
                raise ValueError("benchmark weight must be None or an 1-d array")

    def set_asset_active_ub(self, ub):
        """设置个股主动权重上限

        Paramters
        ----------
        lb : np.ndarray or float
            个股权重上限，shape=(n,)
        """
        if self.__active_weight is None:
            return

        if isinstance(ub, numbers.Number):
            if ub < 0:  # type: ignore
                raise ValueError(f"active ub must be positive but {ub} < 0")
            ub = np.full(self.__n, ub, dtype=np.float64)
        else:
            if not (ub.ndim == 1 and ub.size == self.__n):
                raise ValueError(f"ub must be an 1-d array with size {self.__n}")

        self.__model.constraint(
            fusion.Expr.sub(self.__active_weight, ub), fusion.Domain.lessThan(0.0)
        )
        self.__model.constraint(
            fusion.Expr.add(self.__active_weight, ub), fusion.Domain.greaterThan(0.0)
        )

    def set_total_active_constaint(self, ub: float | None):
        r"""双边主动偏离,个股偏离基准绝对值之和

        Notes
        ------
        $\sum |x_i| \leq c$ 等价于 $-z \leq x \leq z; \sum z_i = c$

        Parameters
        -----------
        ub : float
            累计主动权重上限
        """
        if ub is None:
            return

        # 辅助变量
        z = self.__model.variable(self.__n, fusion.Domain.unbounded())

        # x + z >=0
        self.__model.constraint(
            fusion.Expr.add(self.__active_weight, z), fusion.Domain.greaterThan(0.0)
        )
        # x - z <=0
        self.__model.constraint(
            fusion.Expr.sub(self.__active_weight, z), fusion.Domain.lessThan(0.0)
        )
        # sum(z) = c
        self.__model.constraint(fusion.Expr.sum(z), fusion.Domain.equalsTo(ub))

    def set_weight_in_bench_lb(self, lb: float, in_bench_flag: np.ndarray):
        """设置成分股内权重 **下限** 约束

        Parameters
        ----------
        lb : float
            成分股内权重的下限
        in_bench_flag : 1d np.ndarray
            是否是成分股标志，是为 1 不是为 0
        """
        if not isinstance(in_bench_flag, np.ndarray):
            raise TypeError(
                f"in_bench_flag must be np.ndarray not {type(in_bench_flag)}"
            )

        if not (in_bench_flag.ndim == 1 and in_bench_flag.size == self.__n):
            raise ValueError(f"in_bench_flag must be an 1-d array with size {self.__n}")

        self.__model.constraint(
            fusion.Expr.mul(
                np.expand_dims(in_bench_flag, 0),  # shape (1, n)
                self.__xs,  # shape (n, 1)
            ),
            fusion.Domain.greaterThan(lb),
        )

    def set_budget_constraint(self, w=1.0):
        """仓位约束

        Parameters
        -----------
        w : float
            可投资权重上限，默认为 1.0
        """
        self.__model.constraint(fusion.Expr.sum(self.__xs), fusion.Domain.equalsTo(w))

    def set_exposure_constraint(
        self, exposure: np.ndarray, lb: float | np.ndarray, ub: float | np.ndarray
    ):
        """设置指定因子(行业或风格)敞口上下限约束

        Parameters
        -----------
        exposure : np.ndarray
            因子向量，每一列为一个因子，shape=(n, m)
        lb : float or np.ndarray
            加权敞口下限
            - 如果是float，则所有因子的敞口下限都设为 lb
            - 如果是np.ndarray，则对应因子的敞口下限都设为 lb[i]
        ub : float or np.ndarray
            加权权重上限
            - 如果是float，则所有因子的敞口下限都设为 ub
            - 如果是np.ndarray，则对应因子的敞口下限都设为 ub[i]
        """
        if not isinstance(exposure, np.ndarray):
            raise ValueError(f"exposure must be a ndarray not {type(exposure)}")
        if not (exposure.ndim == 2 and exposure.shape[0] == self.__n):
            raise ValueError(f"exposure must be 2-d array with length {self.__n}")

        # 因子数量
        m = exposure.shape[1]

        if isinstance(lb, numbers.Number):
            lb = np.full(m, lb, dtype=np.float64)
        elif isinstance(lb, np.ndarray):
            if not (lb.ndim == 1 and lb.size == m):
                raise ValueError(
                    "lb array size dose not match the column size of exposure"
                )
        else:
            raise ValueError("lb must be a number or a numpy array")

        if isinstance(ub, numbers.Number):
            ub = np.full(m, ub, dtype=np.float64)
        elif isinstance(ub, np.ndarray):
            if not (ub.ndim == 1 and ub.size == m):
                raise ValueError(
                    "ub array size dose not match the column size of exposure"
                )
        else:
            raise ValueError("ub must be a number or a numpy array")

        if self.__active_weight is not None:
            ax = self.__active_weight
        else:
            ax = self.__xs

        # 下限
        self.__model.constraint(
            fusion.Expr.mul(exposure.T, ax), fusion.Domain.greaterThan(lb)
        )
        # 上限
        self.__model.constraint(
            fusion.Expr.mul(exposure.T, ax), fusion.Domain.lessThan(ub)
        )

    def set_turnover_constaint(self, ub: float | None):
        r"""双边换手率约束 **上限**，如果初始持仓为 None 则不设置此约束

        Notes
        ------
        $\sum |x_i| \leq c$ 等价于 $-z \leq x \leq z 且 \sum z_i = c$

        Parameters
        -----------
        ub : float or None
            双边换手率上限，如果为 None 则不设置
        """
        # 没有初始持仓不设置换手率约束
        if self.__x0 is None:
            return

        if ub is None:
            return

        # 相对初始权重的变化
        dx = fusion.Expr.sub(self.__xs, self.__x0)

        # 辅助变量
        z = self.__model.variable(self.__n, fusion.Domain.unbounded())

        # x + z >=0
        self.__model.constraint(fusion.Expr.add(dx, z), fusion.Domain.greaterThan(0.0))
        # x - z <=0
        self.__model.constraint(fusion.Expr.sub(dx, z), fusion.Domain.lessThan(0.0))
        # sum(z) = c
        self.__model.constraint(fusion.Expr.sum(z), fusion.Domain.equalsTo(ub))

    def set_risk_constaint(
        self,
        F: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        gamma: float | None,
        lambda_F: float,
        lambda_D: float,
    ):
        """设置风险约束

        Parameters
        -----------
        F : np.ndarray
            因子协方差矩阵, shape=(k, k)
        D : np.ndarray
            个股特异风险(标准差)向量, shape=(n,)
        E : np.ndarray
            敞口矩阵, shape=(n, k)
        gamma : float or None
            最大组合风险，默认为 None，即优化最大化风险调整后收益，非 None 时为最大化在给定风险约束下的期望收益
        lambda_F : float
            共同风险厌恶系数，gamma 为 None 时有效，默认为 0.75
        lambda_D : float
            个股特异风险厌恶系数，gamma 为 None 时有效，默认为 0.75
        """

        if not F.ndim == 2:
            raise ValueError("F must be 2-d array")
        self.__k = F.shape[0]

        if not D.ndim == 1:
            raise ValueError("D must be 1-d array")
        self.__n = D.size

        if not E.ndim == 2:
            raise ValueError("E must be 2-d array")

        if not (E.shape[0] == self.__n and E.shape[1] == self.__k):
            raise ValueError("exposure shape dose not match mu and F")

        # 协方差矩阵分解
        V = np.linalg.cholesky(F)  # shape is (k, k)

        # G 矩阵
        G = E @ V  # shape is (n, k)

        if self.__active_weight is not None:
            ax = self.__active_weight
        else:
            ax = self.__xs

        if gamma is not None:
            # 风险约束 ||sqrt(D)x|| + ||Gx|| <= 2 * (0.5 * gamma^2)
            self.__model.constraint(
                fusion.Expr.vstack(
                    0.5,
                    gamma**2,
                    fusion.Expr.vstack(
                        fusion.Expr.mul(G.T, ax),  # shape is (k,)
                        fusion.Expr.mulElm(D, ax),  # shape is (n,)
                    ),
                ),
                fusion.Domain.inRotatedQCone(),
            )
        else:
            if not isinstance(lambda_F, numbers.Number):
                raise ValueError("need lambda_F when gamma is None")
            if not isinstance(lambda_D, numbers.Number):
                raise ValueError("need lambda_D when gamma is None")

            ld = np.sqrt(lambda_D)
            lf = np.sqrt(lambda_F)
            # 风险约束 ||ld * sqrt(D)x|| + || lf*Gx|| <= 2 * (0.5 * S)
            self.__model.constraint(
                fusion.Expr.vstack(
                    0.5,
                    self.__s,
                    fusion.Expr.vstack(
                        fusion.Expr.mul(lf * G.T, ax),  # shape is (k,)
                        fusion.Expr.mulElm(ld * D, ax),  # shape is (n,)
                    ),
                ),
                fusion.Domain.inRotatedQCone(),
            )

    def solve(self, dump=False):
        """求解模型

        Notes
        -----
        求解完成后会自动释放模型，无需手动调用 dispose 方法

        Parameters
        -----------
        dump : bool, default False
            是否保存 Problem 到 dump.ptf 文件（用于调试）
        """
        if dump:
            self.__model.writeTask("dump.ptf")
        self.__model.solve()

        if (
            self.__model.getProblemStatus() == fusion.ProblemStatus.PrimalFeasible
            or self.__model.getProblemStatus()
            == fusion.ProblemStatus.PrimalAndDualFeasible
        ):
            self.__model.acceptedSolutionStatus(fusion.AccSolutionStatus.Optimal)

            res = self.__xs.level()
            self.__model.dispose()
            return res
        else:
            self.__model.dispose()
            raise ProblemInfeasible("model is infeasible")

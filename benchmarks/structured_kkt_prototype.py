r"""有界研究原型：单风险锥的分块对角＋低秩 KKT，不用于生产或诊断导出。

独立环境依赖：cvxopt、numpy、scipy、threadpoolctl 及项目运行依赖。
执行：python benchmarks/structured_kkt_prototype.py --case no_turnover
输出仅含残差、目标和耗时，不含权重。默认 60 秒、80 次内点迭代上限。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim import load_repro, TrackingErrorLimit
from optim.model.compiler import compile_problem
from optim._core.canonical import FactorQCQP


@dataclass
class ConeProblem:
    """原型锥模型：坐标变换仅用于恢复候选，不提供证书回映射。

    G/h 是非负锥与一个 SOC 的堆叠；A/b 是等式。R 的前 nf 行为因子风险，
    后续行为特异风险。free/fixed/full_values 对应原始 canonical 变量坐标。
    """

    G: sp.csc_matrix
    h: np.ndarray
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    R: sp.csc_matrix
    nf: int
    nl: int
    free: np.ndarray
    full_values: np.ndarray
    model: FactorQCQP


def compile_cone(model: FactorQCQP) -> ConeProblem:
    """精确消去固定变量，并拒绝不可支持的常数冲突及冗余等式。"""
    d, risk = model.domain, model.risk_operator
    equal = np.isfinite(d.lower) & (d.lower == d.upper)
    upper, lower = ~equal & np.isfinite(d.upper), ~equal & np.isfinite(d.lower)
    vu, vl = np.isfinite(d.variable_upper), np.isfinite(d.variable_lower)
    identity = sp.eye(d.n_variables, format="csc")
    linear = sp.vstack(
        [d.A[upper], -d.A[lower], identity[vu], -identity[vl]], format="csc"
    )
    rhs = np.r_[
        d.upper[upper], -d.lower[lower], d.variable_upper[vu], -d.variable_lower[vl]
    ]
    vals, vecs = la.eigh((risk.covariance + risk.covariance.T) * 0.5)
    if vals.min() < -1e-10:
        raise ValueError("风险协方差不是半正定矩阵")
    factor = (np.sqrt(np.maximum(vals, 0))[:, None] * vecs.T) @ risk.exposure.T
    nf = len(vals)
    # 风险矩阵按真实 weight_indices 嵌入，不假定权重在前 N 列。
    asset_map = sp.csc_matrix(
        (
            np.ones(len(d.weight_indices)),
            (np.arange(len(d.weight_indices)), d.weight_indices),
        ),
        shape=(len(d.weight_indices), d.n_variables),
    )
    R = (
        sp.vstack(
            [sp.csc_matrix(factor), sp.diags(risk.specific_volatility)], format="csc"
        )
        @ asset_map
    )
    benchmark_full = np.zeros(d.n_variables)
    benchmark_full[d.weight_indices] = risk.benchmark
    fixed = vu & vl & (d.variable_upper == d.variable_lower)
    free = ~fixed
    full_values = np.zeros(d.n_variables)
    full_values[fixed] = d.variable_lower[fixed]
    rhs = rhs - linear @ full_values
    linear = linear[:, free].tocsc()
    row_nonzero = np.asarray(linear.getnnz(axis=1)).ravel() != 0
    if np.any(rhs[~row_nonzero] < -1e-12):
        raise ValueError("固定变量代入后出现常数不等式冲突")
    linear, rhs = linear[row_nonzero], rhs[row_nonzero]
    A = d.A[equal][:, free].toarray()
    b = d.lower[equal] - d.A[equal] @ full_values
    active = np.any(A != 0, axis=1)
    if np.any(np.abs(b[~active]) > 1e-12):
        raise ValueError("固定变量代入后出现常数等式冲突")
    A, b = A[active], b[active]
    if A.shape[0] and np.linalg.matrix_rank(A) != A.shape[0]:
        raise ValueError("原型不处理秩亏等式")
    h = np.r_[rhs, model.risk_limit, R @ (full_values - benchmark_full)]
    R = R[:, free].tocsc()
    G = sp.vstack([linear, sp.csc_matrix((1, int(free.sum()))), -R], format="csc")
    alpha = model.alpha - np.mean(model.alpha)
    scale = 0.2 / max(np.max(np.abs(alpha)), 1e-15)
    c = np.zeros(d.n_variables)
    c[d.weight_indices] = -alpha * scale
    return ConeProblem(G, h, A, b, c[free], R, nf, len(rhs), free, full_values, model)


class LocalBlocks:
    """从至多两个非零元的局部行发现小连通块；拒绝宽度超过 8 的块。"""

    def __init__(self, G: sp.csc_matrix):
        csr = G.tocsr()
        local = np.diff(csr.indptr) <= 2
        self.local_rows = np.flatnonzero(local)
        self.global_rows = np.flatnonzero(~local)
        gl = csr[local]
        # 即使符号抵消也不能遗漏连接，所以邻接以绝对系数构建。
        graph = (abs(gl).T @ abs(gl)).tocsc()
        self.count, self.group = connected_components(graph, directed=False)
        sizes = np.bincount(self.group)
        self.width = int(sizes.max())
        if self.width > 8:
            raise ValueError(f"局部块宽 {self.width} 超出原型范围")
        self.position = np.empty(G.shape[1], dtype=int)
        order = np.argsort(self.group, kind="stable")
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        self.position[order] = np.arange(len(order)) - np.repeat(starts, sizes)
        self.indices = np.full((self.count, self.width), -1, dtype=int)
        self.indices[self.group, self.position] = np.arange(G.shape[1])
        self.valid = self.indices >= 0
        self.n = G.shape[1]
        rr, cc, vv = [], [], []
        # 预计算每个局部行的外积坐标，后续迭代仅更新标量权重。
        for row in self.local_rows:
            lo, hi = csr.indptr[row : row + 2]
            cols, values = csr.indices[lo:hi], csr.data[lo:hi]
            for i, ci in enumerate(cols):
                for j, cj in enumerate(cols):
                    rr.append(
                        (self.group[ci] * self.width + self.position[ci]) * self.width
                        + self.position[cj]
                    )
                    cc.append(row)
                    vv.append(values[i] * values[j])
        self.flat = np.asarray(rr)
        self.source = np.asarray(cc)
        self.values = np.asarray(vv)

    def build(self, weights: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
        """构建逐资产小 Hessian 块；补齐位置设单位对角，不正则化真实变量。"""
        B = np.zeros((self.count, self.width, self.width))
        np.add.at(B.ravel(), self.flat, self.values * weights[self.source])
        B[self.group, self.position, self.position] += diagonal
        for j in range(self.width):
            B[~self.valid[:, j], j, j] = 1.0
        return B

    def apply(self, blocks: np.ndarray, value: np.ndarray) -> np.ndarray:
        """按块作用于一个向量或多列矩阵，不构造全局逆矩阵。"""
        vector = value.ndim == 1
        array = value[:, None] if vector else value
        packed = np.zeros((self.count, self.width, array.shape[1]))
        packed[self.group, self.position] = array
        out = (blocks @ packed)[self.group, self.position]
        return out[:, 0] if vector else out

    def sparse(self, B: np.ndarray) -> sp.csc_matrix:
        """仅用于与通用稀疏分解做同矩阵对照。"""
        rows = np.broadcast_to(self.indices[:, :, None], B.shape)
        cols = np.broadcast_to(self.indices[:, None, :], B.shape)
        mask = (rows >= 0) & (cols >= 0)
        return sp.csc_matrix(
            (B[mask], (rows[mask], cols[mask])), shape=(self.n, self.n)
        )


class StructuredKKT:
    r"""CVXOPT KKT 回调：$H=B+UU^{\mathsf T}$，对小 Schur 系统分解。

    对 SOC 的 $W=\beta(2vv^{\mathsf T}-J)$，且 $G_q=[0;-R]$，有
    $G_q^{\mathsf T}W^{-2}G_q=\beta^{-2}(R^{\mathsf T}R+8v_0^2R^{\mathsf T}v_rv_r^{\mathsf T}R)$。
    """

    def __init__(
        self, problem: ConeProblem, seconds: float = 60, method: str = "schur"
    ):
        self.p = problem
        self.local = LocalBlocks(problem.G[: problem.nl])
        self.global_G = problem.G[: problem.nl][self.local.global_rows].toarray()
        self.factor = problem.R[: problem.nf].toarray()
        self.specific_diag = np.asarray(
            problem.R[problem.nf :].power(2).sum(axis=0)
        ).ravel()
        self.started = time.perf_counter()
        self.seconds = seconds
        self.method = method
        self.factor_s = 0.0
        self.backsolve_s = 0.0
        self.factors = 0
        self.backsolves = 0
        self.snapshots = []
        self.residuals = []
        self.basis = None
        self.coefficients = None
        self.basis_residual = None
        if method in {"compressed", "compressed_augmented", "compressed_augmented_mmd"}:
            base = np.column_stack([self.global_G.T, self.factor.T])
            Q, R, pivot = la.qr(base, pivoting=True, mode="economic")
            rank = int(np.sum(abs(np.diag(R)) > 1e-12 * np.max(abs(np.diag(R)))))
            self.basis = Q[:, :rank]
            coefficients = np.zeros((rank, base.shape[1]))
            coefficients[:, pivot] = R[:rank]
            self.coefficients = coefficients
            self.basis_residual = float(
                np.max(abs(base - self.basis @ coefficients)) / (1 + np.max(abs(base)))
            )
            if self.basis_residual > 1e-12:
                raise ValueError("低秩基底重构残差超限")

    def __call__(self, W):
        """每个内点迭代按当前锥缩放构建小系统；超时则停止实验。"""
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError("结构化 KKT 原型超过时间预算")
        from cvxopt import matrix

        started = time.perf_counter()
        di = np.asarray(W["di"]).ravel()
        beta = float(W["beta"][0])
        v = np.asarray(W["v"][0]).ravel()
        B = self.local.build(di**2, self.specific_diag / beta**2)
        # 小块求逆只发生在 <=8 阶；全局矩阵不求逆。
        augmented = self.method in {
            "augmented",
            "compressed_augmented",
            "compressed_augmented_mmd",
        }
        inverse = None if augmented else np.linalg.inv(B)
        U = np.column_stack(
            [
                self.global_G.T * di[self.local.global_rows],
                self.factor.T / beta,
                (np.sqrt(8.0) * v[0] / beta) * (self.p.R.T @ v[1:]),
            ]
        )
        if self.basis is not None:
            weights = np.r_[di[self.local.global_rows], np.full(self.p.nf, 1 / beta)]
            _, smallR = la.qr((self.coefficients * weights).T, mode="economic")
            U = np.column_stack([self.basis @ smallR.T, U[:, -1]])
        T = np.column_stack([U, self.p.A.T])
        rank = U.shape[1]
        BT, factor = None, None
        if not augmented:
            BT = self.local.apply(inverse, T)
            S = T.T @ BT
            S[np.arange(rank), np.arange(rank)] += 1.0
            factor = la.cho_factor(S, lower=True, check_finite=False)
        qr_factor = None
        if self.method == "qr":
            whitened = self.local.apply(np.linalg.inv(np.linalg.cholesky(B)), T)
            regularizer = np.zeros((rank, T.shape[1]))
            regularizer[np.arange(rank), np.arange(rank)] = 1
            qr_factor = la.qr(
                np.vstack([whitened, regularizer]), mode="r", check_finite=False
            )[0][: T.shape[1]]
        generic = None
        if augmented:
            ne = self.p.A.shape[0]
            K = sp.bmat(
                [
                    [self.local.sparse(B), sp.csc_matrix(U), sp.csc_matrix(self.p.A.T)],
                    [sp.csc_matrix(U.T), -sp.eye(U.shape[1], format="csc"), None],
                    [sp.csc_matrix(self.p.A), None, sp.csc_matrix((ne, ne))],
                ],
                format="csc",
            )
            generic = splu(
                K,
                permc_spec="MMD_AT_PLUS_A"
                if self.method == "compressed_augmented_mmd"
                else "COLAMD",
            )
        self.factor_s += time.perf_counter() - started
        if self.factors in (0, 5, 10, 20):
            self.snapshots.append((B.copy(), U.copy(), self.p.A.copy()))
        self.factors += 1

        def winv(z):
            out = z.copy()
            out[: self.p.nl] *= di
            q = out[self.p.nl :]
            Jq = q.copy()
            Jq[1:] *= -1
            Jv = v.copy()
            Jv[1:] *= -1
            out[self.p.nl :] = (2 * Jv * (Jv @ q) - Jq) / beta
            return out

        def solve(x, y, z):
            started = time.perf_counter()
            bx, by, bz = (np.asarray(item).ravel().copy() for item in (x, y, z))
            rhs = bx + self.p.G.T @ winv(winv(bz))
            if generic is not None:
                actual = generic.solve(np.r_[rhs, np.zeros(rank), by])
                ux, uy = actual[: len(rhs)], actual[len(rhs) + rank :]
            else:
                Br = self.local.apply(inverse, rhs)
                small_rhs = T.T @ Br
                small_rhs[rank:] -= by
                small = la.cho_solve(factor, small_rhs, check_finite=False)
                if qr_factor is not None:
                    small = la.solve_triangular(
                        qr_factor,
                        la.solve_triangular(qr_factor.T, small_rhs, lower=True),
                        lower=False,
                    )
                ux = Br - BT @ small
                uy = small[rank:]
            uz = winv(self.p.G @ ux - bz)
            # 直接验原始缩放 KKT，而非仅检验缩小后的法方程。
            primal_residual = self.p.A @ ux - by
            dual_residual = self.p.A.T @ uy + self.p.G.T @ winv(uz) - bx
            self.residuals.append(
                float(
                    max(
                        np.max(abs(primal_residual), initial=0),
                        np.max(abs(dual_residual), initial=0),
                    )
                    / (
                        1
                        + max(
                            np.max(abs(bx), initial=0),
                            np.max(abs(by), initial=0),
                            np.max(abs(bz), initial=0),
                        )
                    )
                )
            )
            x[:] = matrix(ux)
            y[:] = matrix(uy)
            z[:] = matrix(uz)
            self.backsolve_s += time.perf_counter() - started
            self.backsolves += 1

        return solve


def snapshot_comparison(kkt: StructuredKKT) -> list[dict]:
    """在真实内点缩放快照上比较等价 KKT，并复算线性方程残差。"""
    rng = np.random.default_rng(20260908)
    rows = []
    for index, (B, U, A) in enumerate(kkt.snapshots):
        n, rank, ne = len(U), U.shape[1], len(A)
        rhs = rng.normal(size=n + rank + ne)
        K = sp.bmat(
            [
                [kkt.local.sparse(B), sp.csc_matrix(U), sp.csc_matrix(A.T)],
                [sp.csc_matrix(U.T), -sp.eye(rank, format="csc"), None],
                [sp.csc_matrix(A), None, sp.csc_matrix((ne, ne))],
            ],
            format="csc",
        )
        t = time.perf_counter()
        generic = splu(K)
        generic_factor_s = time.perf_counter() - t
        t = time.perf_counter()
        reference = generic.solve(rhs)
        generic_solve_s = time.perf_counter() - t
        t = time.perf_counter()
        inverse = np.linalg.inv(B)
        T = np.column_stack([U, A.T])
        BT = kkt.local.apply(inverse, T)
        S = T.T @ BT
        S[np.arange(rank), np.arange(rank)] += 1
        factor = la.cho_factor(S, lower=True)
        structured_factor_s = time.perf_counter() - t
        t = time.perf_counter()
        Br = kkt.local.apply(inverse, rhs[:n])
        q = T.T @ Br - rhs[n:]
        small = la.cho_solve(factor, q)
        actual = np.r_[Br - BT @ small, small]
        structured_solve_s = time.perf_counter() - t
        # 下块为 U'x-y=rhs_y，因此消去时小系统右端需要 -rhs_y，已包含于 q。
        rows.append(
            dict(
                snapshot=index,
                n=n,
                schur=rank + ne,
                generic_factor_s=generic_factor_s,
                generic_solve_s=generic_solve_s,
                structured_factor_s=structured_factor_s,
                structured_solve_s=structured_solve_s,
                generic_residual=float(
                    np.max(abs(K @ reference - rhs)) / (1 + np.max(abs(rhs)))
                ),
                structured_residual=float(
                    np.max(abs(K @ actual - rhs)) / (1 + np.max(abs(rhs)))
                ),
            )
        )
    return rows


def run(problem, seconds=60, maxiters=80, method="schur", tolerance=1e-8):
    """返回完整求解和独立验收统计；失败不包装成成功，不触发自动回退。"""
    from cvxopt import matrix, spmatrix, solvers

    model = compile_problem(problem).model
    if not isinstance(model, FactorQCQP):
        raise ValueError("原型仅接受 Factor-QCQP")
    t = time.perf_counter()
    p = compile_cone(model)
    kkt = StructuredKKT(p, seconds, method)
    G = p.G.tocoo()
    nativeG = spmatrix(G.data, G.row.astype(int), G.col.astype(int), size=G.shape)
    setup_s = time.perf_counter() - t
    t = time.perf_counter()
    result = {}
    try:
        native = solvers.conelp(
            matrix(p.c),
            nativeG,
            matrix(p.h),
            dims={"l": p.nl, "q": [p.R.shape[0] + 1], "s": []},
            A=matrix(p.A),
            b=matrix(p.b),
            kktsolver=kkt,
            options={
                "show_progress": False,
                "maxiters": maxiters,
                "abstol": tolerance,
                "reltol": tolerance,
                "feastol": tolerance,
                "refinement": 2,
            },
        )
        result.update(
            status=native["status"],
            iterations=native["iterations"],
            gap=native.get("gap"),
            primal_infeasibility=native.get("primal infeasibility"),
            dual_infeasibility=native.get("dual infeasibility"),
        )
        if native["x"] is not None:
            full = p.full_values.copy()
            full[p.free] = np.asarray(native["x"]).ravel()
            d = model.domain
            activity = d.A @ full
            violation = max(
                0.0,
                np.max(d.lower - activity),
                np.max(activity - d.upper),
                np.max(d.variable_lower - full),
                np.max(full - d.variable_upper),
            )
            w = full[d.weight_indices]
            a = w - model.risk_operator.benchmark
            f = model.risk_operator.exposure.T @ a
            te = float(
                np.sqrt(
                    f @ model.risk_operator.covariance @ f
                    + np.sum((a * model.risk_operator.specific_volatility) ** 2)
                )
            )
            result.update(
                objective=float(model.alpha @ w),
                te=te,
                max_linear_violation=float(violation),
                risk_violation=max(0.0, te - model.risk_limit),
                candidate_feasible=bool(
                    violation <= 1e-5 and te <= model.risk_limit + 1e-5
                ),
            )
    except Exception as exc:
        result.update(status="prototype_error", error=f"{type(exc).__name__}: {exc}")
    result.update(
        setup_s=setup_s,
        solve_s=time.perf_counter() - t,
        factor_s=kkt.factor_s,
        backsolve_s=kkt.backsolve_s,
        factorizations=kkt.factors,
        backsolves=kkt.backsolves,
        variables=len(p.c),
        local_width=kkt.local.width,
        global_rows=len(kkt.local.global_rows),
        basis_rank=None if kkt.basis is None else kkt.basis.shape[1],
        basis_residual=kkt.basis_residual,
        max_kkt_residual=max(kkt.residuals, default=0),
        kkt_residuals=kkt.residuals,
    )
    result["accepted"] = result.get("status") == "optimal" and result.get(
        "candidate_feasible", False
    )
    result["gap_units"] = "scaled_solver_objective"
    try:
        result["snapshots"] = snapshot_comparison(kkt)
    except Exception as exc:
        result["snapshot_error"] = str(exc)
    return result


def main():
    """读取一个真实复现包，串行运行受限原型并输出不含权重的 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro", default="tmp/pp.tar.gz")
    parser.add_argument(
        "--case", choices=["original", "no_turnover", "te6"], default="no_turnover"
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--maxiters", type=int, default=80)
    parser.add_argument(
        "--method",
        choices=[
            "schur",
            "augmented",
            "qr",
            "compressed",
            "compressed_augmented",
            "compressed_augmented_mmd",
        ],
        default="schur",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", default="tmp/structured_kkt.json")
    args = parser.parse_args()
    if args.seconds <= 0 or args.maxiters <= 0 or args.tolerance <= 0:
        parser.error("时间、迭代数和容差必须为正")
    p = load_repro(args.repro).problem
    if args.case != "original":
        p = p.with_constraints(turnover=None)
    if args.case == "te6":
        p = p.with_constraints(tracking_error=TrackingErrorLimit(0.06))
    with threadpool_limits(limits=1):
        result = run(p, args.seconds, args.maxiters, args.method, args.tolerance)
    result["configuration"] = vars(args)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "kkt_residuals"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

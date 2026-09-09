r"""总主动权重的精确单边表达原型；只在实验进程替换编译器 finish。

$\|w-b\|_1=B-\sum b+2\sum(b-w)_+$。利用有效边界对正部函数分类，
无须假定预算为 1 或禁止卖空。诊断域必须禁用此变换。
本原型先建立完整域再裁剪，仅用来评估数学表达，非生产性能实现。
"""

from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from optim.model.canonical import ConstraintRecord, VariableRecord
from optim._impl.solution import lift_weights
from optim._impl.compiler import _DomainBuilder


class FullTurnoverBuilder(_DomainBuilder):
    """实验专用：仅关闭换手稀疏化；新版本仍可先消除已证明冗余的换手约束。"""

    def _turnover_uses_sparse_form(self, initial):
        """保留原问题全部业务约束，以完整换手 epigraph 进行对照。"""
        return False


def lift_active_weights(problem, compiled, weight):
    """同步实验辅助变量的清理后重建语义，不改变公共权重清理规则。"""
    vector = lift_weights(problem, compiled, weight)
    sparse = "exact_sparse_total_active" in compiled.compiler_optimizations
    split = "split_total_active" in compiled.compiler_optimizations
    if not (sparse or split):
        return vector
    positions = {str(asset): i for i, asset in enumerate(problem.data.assets)}
    active = np.asarray(weight) - np.asarray(problem.data.benchmark)
    for record in compiled.model.domain.variables:
        if record.group == "active_aux":
            difference = active[positions[record.key]]
            vector[record.index] = max(-difference if sparse else difference, 0.0)
        elif record.group == "active_negative":
            vector[record.index] = max(-active[positions[record.key]], 0.0)
    return vector


def sparse_active_domain(builder, domain):
    """保持持仓投影可行域不变，删去可消除的辅助变量并重建约束坐标。"""
    aux = builder.layout.active_aux
    if not builder.simplify or aux is None or builder.layout.sparse_active:
        return domain
    b = np.asarray(builder.data.benchmark)
    weight = domain.weight_indices
    lower, upper = domain.variable_lower[weight], domain.variable_upper[weight]
    # 完全确定符号的项无需辅助变量；等号按零低配优先归类，不使用近似阈值。
    uncertain = np.flatnonzero((lower < b) & (upper > b))
    negative = np.flatnonzero((lower < b) & (upper <= b))
    discard = np.setdiff1d(aux, aux[uncertain])
    columns = np.setdiff1d(np.arange(domain.n_variables), discard)
    colmap = np.full(domain.n_variables, -1, dtype=int)
    colmap[columns] = np.arange(len(columns))
    active_rows = {
        r.index
        for r in domain.constraints
        if r.location == "row"
        and r.group
        in (
            "total_active_epigraph_positive",
            "total_active_epigraph_negative",
            "total_active",
        )
    }
    rows = np.array(
        [i for i in range(domain.A.shape[0]) if i not in active_rows], dtype=int
    )
    rowmap = {old: new for new, old in enumerate(rows)}
    records = [
        replace(
            r,
            index=int(colmap[r.index]) if r.location == "variable" else rowmap[r.index],
        )
        for r in domain.constraints
        if (r.location == "variable" and colmap[r.index] >= 0)
        or (r.location == "row" and r.index in rowmap)
    ]
    size = len(uncertain)
    local = np.arange(size)
    epigraph = sp.csc_matrix(
        (
            -np.ones(2 * size),
            (
                np.r_[local, local],
                np.r_[colmap[weight[uncertain]], colmap[aux[uncertain]]],
            ),
        ),
        shape=(size, len(columns)),
    )
    total = sp.csc_matrix(
        (
            np.r_[np.full(size, 2.0), np.full(len(negative), -2.0)],
            (
                np.zeros(size + len(negative), dtype=int),
                np.r_[colmap[aux[uncertain]], colmap[weight[negative]]],
            ),
        ),
        shape=(1, len(columns)),
    )
    offset = float(builder.constraints_config.budget - b.sum() + 2 * b[negative].sum())
    for i, asset in enumerate(uncertain):
        key = str(builder.data.assets[asset])
        records.append(
            ConstraintRecord(
                constraint_id=f"total_active_epigraph_negative:{key}",
                group="total_active_epigraph_negative",
                location="row",
                index=len(rows) + i,
                key=key,
                source="compiler",
                relaxable=False,
            )
        )
    records.append(
        ConstraintRecord(
            constraint_id="total_active:l1",
            group="total_active",
            location="row",
            index=len(rows) + size,
            key="l1",
            metadata={
                "representation": "exact_sparse_total_active",
                "expression_offset": offset,
                "original_bound": builder.constraints_config.total_active,
            },
        )
    )
    if builder.layout.factor is not None:
        builder.layout.factor = colmap[builder.layout.factor]
    builder.optimizations.append("exact_sparse_total_active")
    return replace(
        domain,
        A=sp.vstack([domain.A[rows][:, columns], epigraph, total], format="csc"),
        lower=np.r_[domain.lower[rows], np.full(size + 1, -np.inf)],
        upper=np.r_[
            domain.upper[rows],
            -b[uncertain],
            builder.constraints_config.total_active - offset,
        ],
        variable_lower=domain.variable_lower[columns],
        variable_upper=domain.variable_upper[columns],
        variables=tuple(
            replace(v, index=int(colmap[v.index]))
            for v in domain.variables
            if colmap[v.index] >= 0
        ),
        constraints=tuple(records),
        weight_indices=colmap[weight],
    )


def split_active_domain(builder, domain):
    """完整正负分解对照：新增负部变量，原绝对值辅助变量改为正部变量。"""
    aux = builder.layout.active_aux
    if not builder.simplify or aux is None or builder.layout.sparse_active:
        return domain
    b = np.asarray(builder.data.benchmark)
    n = len(b)
    negative = np.arange(domain.n_variables, domain.n_variables + n)
    width = domain.n_variables + n
    excluded = {
        r.index
        for r in domain.constraints
        if r.location == "row" and r.group.startswith("total_active")
    }
    rows = np.array(
        [i for i in range(domain.A.shape[0]) if i not in excluded], dtype=int
    )
    rowmap = {old: new for new, old in enumerate(rows)}
    records = [
        replace(r, index=rowmap[r.index]) if r.location == "row" else r
        for r in domain.constraints
        if r.location == "variable" or r.index in rowmap
    ]
    local = np.arange(n)
    linking = sp.csc_matrix(
        (
            np.r_[np.ones(n), -np.ones(n), np.ones(n)],
            (np.tile(local, 3), np.r_[domain.weight_indices, aux, negative]),
        ),
        shape=(n, width),
    )
    total = sp.csc_matrix(
        (np.ones(2 * n), (np.zeros(2 * n, dtype=int), np.r_[aux, negative])),
        shape=(1, width),
    )
    records.extend(
        ConstraintRecord(
            constraint_id=f"total_active_split:{asset}",
            group="total_active_split",
            location="row",
            index=len(rows) + i,
            key=str(asset),
            source="compiler",
            relaxable=False,
        )
        for i, asset in enumerate(builder.data.assets)
    )
    records.append(
        ConstraintRecord(
            constraint_id="total_active:l1",
            group="total_active",
            location="row",
            index=len(rows) + n,
            key="l1",
        )
    )
    builder.optimizations.append("split_total_active")
    return replace(
        domain,
        A=sp.vstack(
            [
                sp.hstack([domain.A[rows], sp.csc_matrix((len(rows), n))]),
                linking,
                total,
            ],
            format="csc",
        ),
        lower=np.r_[domain.lower[rows], b, -np.inf],
        upper=np.r_[domain.upper[rows], b, builder.constraints_config.total_active],
        variable_lower=np.r_[domain.variable_lower, np.zeros(n)],
        variable_upper=np.r_[domain.variable_upper, np.full(n, np.inf)],
        variables=domain.variables
        + tuple(
            VariableRecord(
                index=int(index),
                variable_id=f"active_negative:{i}",
                group="active_negative",
                key=str(builder.data.assets[i]),
            )
            for i, index in enumerate(negative)
        ),
        constraints=tuple(records),
    )

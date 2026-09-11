"""因子风险模型的共享锥建模：仅扩展线性域，不构造参数 QP。"""

import numpy as np
import scipy.sparse as sp

from .canonical import FactorQCQP, LinearDomain, ConstraintRecord, VariableRecord


def extend_factor_domain(model: FactorQCQP) -> LinearDomain:
    r"""追加主动因子暴露 $f=E^{\mathsf T}(w-b)$ 并合并风格/行业行。

    供 Clarabel 和 MOSEK 共享；保留变量和约束来源，不分配无用的风险 Hessian。
    """
    base = model.domain
    risk = model.risk_operator
    n_base = base.n_variables
    n_factors = risk.exposure.shape[1]
    factor_indices = np.arange(n_base, n_base + n_factors, dtype=np.int32)
    n_variables = n_base + n_factors

    original_rows = sp.hstack(
        [base.A, sp.csc_matrix((base.n_constraints, n_factors))], format="csc"
    )
    extended_lower = base.lower.copy()
    extended_upper = base.upper.copy()
    factor_lookup = {
        name.lower(): index for index, name in enumerate(risk.factor_names)
    }
    active_bound_records = tuple(
        record
        for record in base.constraints
        if record.location == "row"
        and record.group in {"style", "industry"}
        and record.key is not None
    )
    if active_bound_records:
        bound_rows = np.fromiter(
            (record.index for record in active_bound_records), dtype=np.int32
        )
        bound_factors = np.fromiter(
            (factor_lookup[str(record.key).lower()] for record in active_bound_records),
            dtype=np.int32,
        )
        # 这些 canonical 行为 $E_j^{\mathsf T}x$，其边界已经按基准平移。引入
        # $f=E^{\mathsf T}(x-b)$ 后，将每个稠密行替换为 $f_j$ 上对应的单列边界。
        keep = np.ones(base.n_constraints, dtype=float)
        keep[bound_rows] = 0.0
        original_rows = sp.diags(keep, format="csc") @ original_rows
        direct_factor_bounds = sp.csc_matrix(
            (
                np.ones(len(bound_rows)),
                (bound_rows, factor_indices[bound_factors]),
            ),
            shape=(base.n_constraints, n_variables),
        )
        original_rows = (original_rows + direct_factor_bounds).tocsc()
        benchmark_shift = np.einsum(
            "ij,i->j",
            risk.exposure[:, bound_factors],
            risk.benchmark,
            optimize=False,
        )
        extended_lower[bound_rows] -= benchmark_shift
        extended_upper[bound_rows] -= benchmark_shift
    factor_rows = np.arange(n_factors, dtype=np.int32)
    factor_identity = sp.csc_matrix(
        (-np.ones(n_factors), (factor_rows, factor_indices)),
        shape=(n_factors, n_variables),
    )
    exposure_rows = sp.hstack(
        [
            sp.csc_matrix(risk.exposure.T),
            sp.csc_matrix((n_factors, n_base - len(risk.weight_indices))),
            sp.csc_matrix((n_factors, n_factors)),
        ],
        format="csc",
    )
    factor_definition = exposure_rows + factor_identity
    factor_rhs = np.einsum(
        "ij,i->j", risk.exposure, risk.benchmark, optimize=False
    )
    A = sp.vstack([original_rows, factor_definition], format="csc")
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    variables = list(base.variables)
    constraints = list(base.constraints)
    for local_index, (name, index) in enumerate(zip(risk.factor_names, factor_indices)):
        variables.append(
            VariableRecord(
                index=int(index),
                variable_id=f"factor_active:{name}",
                group="factor_active",
                key=str(name),
                unit="exposure",
            )
        )
        constraints.append(
            ConstraintRecord(
                constraint_id=f"factor_definition:{name}",
                group="factor_definition",
                location="row",
                index=base.n_constraints + local_index,
                key=str(name),
                unit="exposure",
                source="factor_conic",
                relaxable=False,
            )
        )
    extended_domain = LinearDomain(
        A=A,
        lower=np.concatenate([extended_lower, factor_rhs]),
        upper=np.concatenate([extended_upper, factor_rhs]),
        variable_lower=np.concatenate(
            [base.variable_lower, np.full(n_factors, -np.inf)]
        ),
        variable_upper=np.concatenate(
            [base.variable_upper, np.full(n_factors, np.inf)]
        ),
        variables=tuple(variables),
        constraints=tuple(constraints),
        weight_indices=base.weight_indices,
        assets=base.assets,
    )

    return extended_domain

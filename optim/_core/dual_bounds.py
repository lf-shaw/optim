"""与后端无关的 LP 数值下界复算；仅显式诊断使用，不负责原生 API 或业务诊断编排。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import LinearProgram, QuadraticProgram


def lp_dual_bound_diagnostics(
    model: LinearProgram | QuadraticProgram,
    row_dual: Any,
    primal: Any = None,
) -> dict[str, Any]:
    r"""对各后端统一复算 LP 数值拉格朗日下界，不读取日志或使用 primal 目标代替。

    约定 $y$ 为下界正、上界负的行乘子，$r=c-A^{\mathsf T}y$。
    在变量盒域上求 $r^{\mathsf T}x$ 的下确界，并加回行边界项与目标常数。
    无穷变量侧有不利残差时，尝试从原约束及可行候选目标子水平集推导盒界；仍无界则拒绝。
    不通过截断小残差制造证书。
    仅显式 collect_dual_bound 请求此计算，普通求解不增加开销。

    Parameters
    ----------
    model : LinearProgram | QuadraticProgram
        原坐标和原目标单位的 LP，或零 Hessian 的等价 QP。
    row_dual : Any
        已撤销后端缩放并统一符号的行乘子，shape 为 (n_constraints,)。
    primal : Any
        可选候选，shape 为 (n_variables,)；仅在基础盒界无法处理残差时使用。
        子水平集计算前独立检查候选残差，不将其目标值视为下界。

    Returns
    -------
    dict[str, Any]
        dual_lower_bound 为数值估计或 None；dual_bound_status、dual_bound_method、
        dual_bound_reason 说明可用性、方法和失败原因。均为内部证据，不表示严格区间证明。
    """
    output: dict[str, Any] = {
        "dual_lower_bound": None,
        "dual_bound_status": "unavailable",
        "dual_bound_method": "box_lagrangian",
        "dual_bound_reason": None,
    }
    if isinstance(model, QuadraticProgram) and model.P.nnz:
        output["dual_bound_reason"] = "nonlinear_objective"
        return output
    d = model.domain
    y = np.asarray(row_dual, dtype=float)
    if y.shape != (d.n_constraints,) or not np.all(np.isfinite(y)):
        output["dual_bound_reason"] = "invalid_row_dual"
        return output
    c = model.c if isinstance(model, LinearProgram) else model.q
    # 将行乘子投影到原行允许的符号域，重新计算 reduced cost；不是截断平稳性残差。
    y = y.copy()
    y[~np.isfinite(d.lower) & (y > 0)] = 0.0
    y[~np.isfinite(d.upper) & (y < 0)] = 0.0
    reduced = np.asarray(c - d.A.T @ y)
    row_bound = np.where(y > 0, d.lower, d.upper)
    col_bound = np.where(reduced > 0, d.variable_lower, d.variable_upper)
    row_mask, col_mask = y != 0, reduced != 0
    if (
        np.all(np.isfinite(reduced))
        and not np.all(np.isfinite(col_bound[col_mask]))
        and primal is not None
    ):
        inferred = _sublevel_box(model, c, primal)
        if inferred is not None:
            lower, upper, cutoff = inferred
            col_bound = np.where(reduced > 0, lower, upper)
            output.update(
                dual_bound_method="sublevel_box_lagrangian",
                dual_bound_objective_cutoff=cutoff,
            )
    if not np.all(np.isfinite(reduced)):
        output["dual_bound_reason"] = "invalid_reduced_cost"
    elif not np.all(np.isfinite(row_bound[row_mask])):
        output["dual_bound_reason"] = "unsupported_row_multiplier_sign"
    elif not np.all(np.isfinite(col_bound[col_mask])):
        output["dual_bound_reason"] = "unbounded_residual_direction"
    else:
        value = float(
            y[row_mask] @ row_bound[row_mask]
            + reduced[col_mask] @ col_bound[col_mask]
            + model.objective_offset
        )
        if np.isfinite(value):
            output.update(
                dual_lower_bound=value, dual_bound_status="numerical_estimate"
            )
        else:
            output["dual_bound_reason"] = "nonfinite_bound"
    return output


def _sublevel_box(
    model: LinearProgram | QuadraticProgram,
    c: np.ndarray,
    primal: Any,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """在目标不劣于已验收候选的子水平集传播变量界，仅用于显式诊断。

    该集合保留最优值，不修改原求解问题。通用稀疏区间传播不依赖股票、预算或松弛变量类型。
    所有新边界向外留数值裕量；无法推导有限界则由调用者拒绝证书。属于浮点数值估计。
    """
    import scipy.sparse as sp

    d = model.domain
    x = np.asarray(primal, dtype=float)
    if x.shape != (d.n_variables,) or not np.all(np.isfinite(x)):
        return None
    activity = d.A @ x
    violation = max(
        float(np.max(d.lower - activity, initial=0)),
        float(np.max(activity - d.upper, initial=0)),
        float(np.max(d.variable_lower - x, initial=0)),
        float(np.max(x - d.variable_upper, initial=0)),
    )
    if violation > 1e-7:
        return None
    # 候选仅用于有效的子水平集上界，不将其目标当作最优值下界。
    margin = 1e-7 * (1 + float(np.abs(c) @ np.abs(x)))
    cutoff = float(c @ x) + margin
    matrix = sp.vstack([d.A, sp.csr_matrix(c.reshape(1, -1))], format="csr")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_lower = np.append(d.lower, -np.inf)
    row_upper = np.append(d.upper, cutoff)
    lower, upper = d.variable_lower.copy(), d.variable_upper.copy()
    for _ in range(8):
        changed = False
        for i in range(matrix.shape[0]):
            start, stop = matrix.indptr[i : i + 2]
            indices, a = matrix.indices[start:stop], matrix.data[start:stop]
            if not len(indices):
                continue
            for bound, is_upper in ((row_upper[i], True), (row_lower[i], False)):
                if not np.isfinite(bound):
                    continue
                selected = (
                    np.where(a > 0, lower[indices], upper[indices])
                    if is_upper
                    else np.where(a > 0, upper[indices], lower[indices])
                )
                terms = a * selected
                finite = np.isfinite(terms)
                total = float(terms[finite].sum())
                infinite = int((~finite).sum())
                eligible = (infinite - (~finite).astype(int)) == 0
                others = total - np.where(finite, terms, 0.0)
                values = (bound - others) / a
                # 外扩包含求和误差，避免减法消去误差使推导界过紧。
                padding = (
                    1e-10
                    * (1 + abs(bound) + float(np.abs(terms[finite]).sum()))
                    / np.abs(a)
                )
                to_upper = (a > 0) if is_upper else (a < 0)
                for mask, target, values_out in (
                    (eligible & to_upper, upper, values + padding),
                    (eligible & ~to_upper, lower, values - padding),
                ):
                    ids = indices[mask]
                    proposed = values_out[mask]
                    improved = (
                        proposed < target[ids]
                        if target is upper
                        else proposed > target[ids]
                    )
                    if np.any(improved):
                        target[ids[improved]] = proposed[improved]
                        changed = True
        if np.any(lower > upper):
            return None
        if not changed:
            break
    return lower, upper, cutoff + model.objective_offset

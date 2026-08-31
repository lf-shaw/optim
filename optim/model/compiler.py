r"""Compile portfolio semantics into solver-independent sparse models.

The compiler is the only layer allowed to translate business constraints into
matrix rows and auxiliary variables.  Every problem shares a polyhedral domain
$\underline z\le z\le\overline z$ and $l\le Az\le u$;
LP/QP/factor-QCQP objects add only their objective and risk representation.

The first ``n_assets`` columns of ``z`` are always weights.  Optional columns
represent turnover epigraphs, total-active epigraphs and factor active exposure.
Each row/column receives an audit record used later by fingerprints, independent
solution validation and infeasibility diagnostics.  This module never imports
or initializes a numerical solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

from ..asset_bounds import AssetBoundsError, resolve_asset_bounds
from ..fingerprint import fingerprint
from ..portfolio_types import (
    FactorRiskModel,
    FullCovarianceRiskModel,
    MaximizeAlpha,
    MinimizeTrackingError,
    PortfolioProblem,
    ProblemKind,
    RiskAdjustedAlpha,
)
from ..validation import validate_problem
from .canonical import (
    CompiledProblem,
    CanonicalModel,
    ConstraintRecord,
    FactorQCQP,
    FactorRiskOperator,
    LinearDomain,
    LinearProgram,
    QuadraticProgram,
    VariableRecord,
)


class CanonicalCompilationError(ValueError):
    """A valid semantic request cannot be represented by this compiler."""


def classify_problem(problem: PortfolioProblem) -> ProblemKind:
    """Classify by mathematical structure, not by the historical API name.

    A linear alpha problem remains an LP regardless of turnover or exposure
    constraints because those constraints compile linearly.  Adding one factor
    tracking-error budget to ``MaximizeAlpha`` selects the specialized
    factor-QCQP representation.  ``RiskAdjustedAlpha`` plus another risk budget
    is classified as generic conic, which the current v1 compiler reports as
    unsupported rather than silently changing the requested objective.
    """

    objective = problem.objective
    has_risk_limit = problem.constraints.tracking_error is not None
    if isinstance(objective, MaximizeAlpha):
        return ProblemKind.FACTOR_QCQP if has_risk_limit else ProblemKind.LP
    if isinstance(objective, RiskAdjustedAlpha):
        return ProblemKind.CONIC if has_risk_limit else ProblemKind.QP
    if isinstance(objective, MinimizeTrackingError):
        return ProblemKind.QP
    raise CanonicalCompilationError(f"unsupported objective type: {type(objective).__name__}")


@dataclass
class _Layout:
    """Contiguous column ranges selected for one canonical model."""

    n_variables: int
    weight: np.ndarray
    turnover_aux: np.ndarray | None
    turnover_support: np.ndarray | None
    turnover_new: np.ndarray | None
    sparse_turnover: bool
    active_aux: np.ndarray | None
    factor: np.ndarray | None


class _DomainBuilder:
    """Incrementally build one auditable sparse linear domain.

    Layout decisions are made once in ``__init__`` so later constraint blocks
    can be assembled directly in final-width CSC form.  The builder also records
    exact structural simplifications, such as sparse turnover and provably
    redundant aggregate active constraints, on ``optimizations``.
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        *,
        include_factor_variables: bool,
    ) -> None:
        self.problem = problem
        self.data = problem.data
        self.constraints_config = problem.constraints
        self.n_assets = len(self.data.assets)
        self.risk_model = self.data.risk_model
        self.optimizations: list[str] = []
        n_factors = (
            len(self.risk_model.factor_names)
            if include_factor_variables and isinstance(self.risk_model, FactorRiskModel)
            else 0
        )

        cursor = self.n_assets
        turnover_aux = None
        turnover_support = None
        turnover_new = None
        sparse_turnover = False
        if self.constraints_config.turnover is not None:
            initial = np.asarray(self.data.initial_weight, dtype=float)
            sparse_turnover = bool(
                self.constraints_config.long_only
                and np.all(initial >= 0.0)
                and np.isclose(
                    initial.sum(),
                    self.constraints_config.budget,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if sparse_turnover:
                turnover_support = np.flatnonzero(initial > 0.0).astype(np.int32)
                turnover_new = np.flatnonzero(initial <= 0.0).astype(np.int32)
                turnover_aux = np.arange(
                    cursor,
                    cursor + len(turnover_support),
                    dtype=np.int32,
                )
                cursor += len(turnover_support)
                self.optimizations.append("exact_sparse_turnover")
            else:
                turnover_aux = np.arange(cursor, cursor + self.n_assets, dtype=np.int32)
                cursor += self.n_assets
        active_aux = None
        if self.constraints_config.total_active is not None:
            turnover = self.constraints_config.turnover
            benchmark = self.data.benchmark
            current_weight = self.data.initial_weight
            redundant = bool(
                turnover is not None
                and benchmark is not None
                and current_weight is not None
                and float(np.abs(np.asarray(current_weight) - np.asarray(benchmark)).sum())
                + turnover.l1_limit
                <= self.constraints_config.total_active + 1e-12
            )
            if redundant:
                self.optimizations.append("omit_redundant_total_active")
            else:
                active_aux = np.arange(cursor, cursor + self.n_assets, dtype=np.int32)
                cursor += self.n_assets
        factor = None
        if n_factors:
            factor = np.arange(cursor, cursor + n_factors, dtype=np.int32)
            cursor += n_factors
        self.layout = _Layout(
            n_variables=cursor,
            weight=np.arange(self.n_assets, dtype=np.int32),
            turnover_aux=turnover_aux,
            turnover_support=turnover_support,
            turnover_new=turnover_new,
            sparse_turnover=sparse_turnover,
            active_aux=active_aux,
            factor=factor,
        )
        self.blocks: list[sp.csc_matrix] = []
        self.lowers: list[np.ndarray] = []
        self.uppers: list[np.ndarray] = []
        self.row_records: list[ConstraintRecord] = []

        self.variable_lower = np.full(cursor, -np.inf, dtype=float)
        self.variable_upper = np.full(cursor, np.inf, dtype=float)
        self.variables: list[VariableRecord] = []
        self.variable_records: list[ConstraintRecord] = []
        self._configure_variables()

    def _configure_variables(self) -> None:
        """Resolve effective asset bounds and register every canonical column.

        Operational instructions are already merged by ``resolve_asset_bounds``.
        Common immutable metadata is interned because an ordinary 5,000-asset
        problem has only a handful of distinct bound-source combinations.
        """

        assets = self.data.assets
        try:
            resolved = resolve_asset_bounds(self.problem)
        except AssetBoundsError as exc:
            raise CanonicalCompilationError(str(exc)) from exc
        lower = resolved.lower
        upper = resolved.upper
        self.variable_lower[self.layout.weight] = lower
        self.variable_upper[self.layout.weight] = upper
        # In an ordinary large universe only a handful of source combinations
        # exist.  Intern their immutable mappings instead of allocating and
        # copying an equivalent dictionary for every asset.
        metadata_flyweights: dict[
            tuple[tuple[str, ...], tuple[str, ...]], MappingProxyType[str, Any]
        ] = {}
        for index, asset in enumerate(assets):
            key = str(asset)
            lower_sources = resolved.lower_sources[index]
            upper_sources = resolved.upper_sources[index]
            extra_metadata = resolved.metadata[index]
            if extra_metadata is None:
                metadata_key = (lower_sources, upper_sources)
                record_metadata = metadata_flyweights.get(metadata_key)
                if record_metadata is None:
                    record_metadata = MappingProxyType(
                        {
                            "lower_sources": lower_sources,
                            "upper_sources": upper_sources,
                        }
                    )
                    metadata_flyweights[metadata_key] = record_metadata
            else:
                record_metadata = MappingProxyType(
                    {
                        "lower_sources": lower_sources,
                        "upper_sources": upper_sources,
                        **extra_metadata,
                    }
                )
            self.variables.append(VariableRecord(index, f"weight:{key}", "weight", key))
            self.variable_records.append(
                ConstraintRecord(
                    constraint_id=f"asset_bound:{key}",
                    group="asset_bound",
                    location="variable",
                    index=index,
                    key=key,
                    metadata=record_metadata,
                )
            )

        for group, indices in (
            ("turnover_aux", self.layout.turnover_aux),
            ("active_aux", self.layout.active_aux),
        ):
            if indices is None:
                continue
            self.variable_lower[indices] = 0.0
            for local_index, variable_index in enumerate(indices):
                asset_position = (
                    int(self.layout.turnover_support[local_index])  # type: ignore[index]
                    if group == "turnover_aux" and self.layout.sparse_turnover
                    else local_index
                )
                self.variables.append(
                    VariableRecord(
                        int(variable_index),
                        f"{group}:{local_index}",
                        group,
                        str(self.data.assets[asset_position]),
                    )
                )
        if self.layout.factor is not None:
            assert isinstance(self.risk_model, FactorRiskModel)
            for factor_name, variable_index in zip(
                self.risk_model.factor_names, self.layout.factor
            ):
                self.variables.append(
                    VariableRecord(
                        int(variable_index),
                        f"factor_active:{factor_name}",
                        "factor_active",
                        factor_name,
                        "exposure",
                    )
                )
        self.variables.sort(key=lambda item: item.index)

    def _pad(self, matrix: sp.spmatrix) -> sp.csc_matrix:
        matrix = matrix.tocsc()
        if matrix.shape[1] == self.layout.n_variables:
            return matrix
        if matrix.shape[1] > self.layout.n_variables:
            raise AssertionError("constraint matrix is wider than the variable layout")
        return sp.hstack(
            [matrix, sp.csc_matrix((matrix.shape[0], self.layout.n_variables - matrix.shape[1]))],
            format="csc",
        )

    def add_block(
        self,
        matrix: sp.spmatrix,
        lower: np.ndarray | float,
        upper: np.ndarray | float,
        *,
        group: str,
        keys: Iterable[str | None] | None = None,
        unit: str = "weight",
        source: str = "user",
        relaxable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a bounded row block and its business-level audit records."""

        block = self._pad(matrix)
        n_rows = block.shape[0]
        lower_array = np.broadcast_to(np.asarray(lower, dtype=float), (n_rows,)).copy()
        upper_array = np.broadcast_to(np.asarray(upper, dtype=float), (n_rows,)).copy()
        key_items: list[str | None] = [None] * n_rows if keys is None else list(keys)
        if len(key_items) != n_rows:
            raise AssertionError("constraint keys do not match row count")
        row_start = sum(item.shape[0] for item in self.blocks)
        self.blocks.append(block)
        self.lowers.append(lower_array)
        self.uppers.append(upper_array)
        for local_index, key in enumerate(key_items):
            suffix = str(key) if key is not None else str(local_index)
            self.row_records.append(
                ConstraintRecord(
                    constraint_id=f"{group}:{suffix}",
                    group=group,
                    location="row",
                    index=row_start + local_index,
                    key=key,
                    unit=unit,
                    source=source,
                    relaxable=relaxable,
                    metadata={} if metadata is None else metadata,
                )
            )

    def _weight_block(self, values: np.ndarray) -> sp.csc_matrix:
        values = np.asarray(values, dtype=float).reshape(1, -1)
        return self._pad(sp.csc_matrix(values))

    def add_common_constraints(self) -> None:
        """Compile the linear feasible domain shared by all objective types.

        This includes budget, factor definitions, turnover, total active weight,
        benchmark-member coverage, style/industry bounds, extra attributes and
        an optional alpha floor.  Absolute and per-name active bounds are column
        bounds configured earlier rather than matrix rows.

        The sparse turnover formulation is exact for a long-only fully invested
        initial portfolio.  Because total buys equal total sells, L1 turnover is
        twice the buys: positive increases on the original support plus weights
        opened outside that support.  No relationship with tracking error is
        assumed.
        """

        n = self.n_assets
        x = self.layout.weight
        config = self.constraints_config
        benchmark = None if self.data.benchmark is None else np.asarray(self.data.benchmark, dtype=float)

        self.add_block(
            self._weight_block(np.ones(n)),
            config.budget,
            config.budget,
            group="budget",
            keys=("total",),
            source="system",
            relaxable=False,
        )

        if self.layout.factor is not None:
            assert isinstance(self.risk_model, FactorRiskModel)
            assert benchmark is not None
            exposure = np.asarray(self.risk_model.exposure, dtype=float)
            rows = np.arange(len(self.layout.factor), dtype=np.int32)
            factor_identity = sp.csc_matrix(
                (-np.ones(len(rows)), (rows, self.layout.factor)),
                shape=(len(rows), self.layout.n_variables),
            )
            factor_matrix = self._pad(sp.csc_matrix(exposure.T)) + factor_identity
            rhs = exposure.T @ benchmark
            self.add_block(
                factor_matrix,
                rhs,
                rhs,
                group="factor_definition",
                keys=self.risk_model.factor_names,
                unit="exposure",
                source="compiler",
                relaxable=False,
            )

        if self.layout.turnover_aux is not None:
            assert config.turnover is not None
            assert self.data.initial_weight is not None
            initial = np.asarray(self.data.initial_weight, dtype=float)
            asset_keys = tuple(str(item) for item in self.data.assets)
            aux = self.layout.turnover_aux
            if self.layout.sparse_turnover:
                assert self.layout.turnover_support is not None
                assert self.layout.turnover_new is not None
                support = self.layout.turnover_support
                rows = np.arange(len(support), dtype=np.int32)
                positive = sp.csc_matrix(
                    (
                        np.concatenate([np.ones(len(support)), -np.ones(len(support))]),
                        (
                            np.concatenate([rows, rows]),
                            np.concatenate([support, aux]),
                        ),
                    ),
                    shape=(len(support), self.layout.n_variables),
                )
                self.add_block(
                    positive,
                    -np.inf,
                    initial[support],
                    group="turnover_epigraph_positive",
                    keys=(asset_keys[int(index)] for index in support),
                    source="compiler",
                    relaxable=False,
                )
                turnover_columns = np.concatenate([aux, self.layout.turnover_new])
                turnover_row = sp.csc_matrix(
                    (
                        np.full(len(turnover_columns), 2.0),
                        (np.zeros(len(turnover_columns), dtype=np.int32), turnover_columns),
                    ),
                    shape=(1, self.layout.n_variables),
                )
            else:
                rows = np.arange(n, dtype=np.int32)
                positive = sp.csc_matrix(
                    (
                        np.concatenate([np.ones(n), -np.ones(n)]),
                        (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                    ),
                    shape=(n, self.layout.n_variables),
                )
                negative = sp.csc_matrix(
                    (
                        np.concatenate([-np.ones(n), -np.ones(n)]),
                        (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                    ),
                    shape=(n, self.layout.n_variables),
                )
                self.add_block(
                    positive,
                    -np.inf,
                    initial,
                    group="turnover_epigraph_positive",
                    keys=asset_keys,
                    source="compiler",
                    relaxable=False,
                )
                self.add_block(
                    negative,
                    -np.inf,
                    -initial,
                    group="turnover_epigraph_negative",
                    keys=asset_keys,
                    source="compiler",
                    relaxable=False,
                )
                turnover_row = sp.csc_matrix(
                    (np.ones(n), (np.zeros(n, dtype=np.int32), aux)),
                    shape=(1, self.layout.n_variables),
                )
            self.add_block(
                turnover_row,
                -np.inf,
                config.turnover.l1_limit,
                group="turnover",
                keys=("l1",),
            )

        if self.layout.active_aux is not None:
            assert config.total_active is not None
            assert benchmark is not None
            aux = self.layout.active_aux
            rows = np.arange(n, dtype=np.int32)
            positive = sp.csc_matrix(
                (
                    np.concatenate([np.ones(n), -np.ones(n)]),
                    (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                ),
                shape=(n, self.layout.n_variables),
            )
            negative = sp.csc_matrix(
                (
                    np.concatenate([-np.ones(n), -np.ones(n)]),
                    (np.concatenate([rows, rows]), np.concatenate([x, aux])),
                ),
                shape=(n, self.layout.n_variables),
            )
            asset_keys = tuple(str(item) for item in self.data.assets)
            self.add_block(
                positive,
                -np.inf,
                benchmark,
                group="total_active_epigraph_positive",
                keys=asset_keys,
                source="compiler",
                relaxable=False,
            )
            self.add_block(
                negative,
                -np.inf,
                -benchmark,
                group="total_active_epigraph_negative",
                keys=asset_keys,
                source="compiler",
                relaxable=False,
            )
            total_active_row = sp.csc_matrix(
                (np.ones(n), (np.zeros(n, dtype=np.int32), aux)),
                shape=(1, self.layout.n_variables),
            )
            self.add_block(
                total_active_row,
                -np.inf,
                float(config.total_active),
                group="total_active",
                keys=("l1",),
            )

        if config.benchmark_member_weight is not None:
            assert benchmark is not None
            turnover = config.turnover
            current_weight = self.data.initial_weight
            member = benchmark > 0.0
            redundant = bool(
                turnover is not None
                and current_weight is not None
                and float(np.asarray(current_weight)[member].sum())
                - turnover.l1_limit / 2.0
                >= config.benchmark_member_weight.value - 1e-12
            )
            if redundant:
                self.optimizations.append("omit_redundant_benchmark_member_weight")
            else:
                self.add_block(
                    self._weight_block(member.astype(float)),
                    config.benchmark_member_weight.value,
                    np.inf,
                    group="benchmark_member_weight",
                    keys=("members",),
                )

        self._add_factor_bounds("style", config.style)
        self._add_factor_bounds("industry", config.industry)

        if benchmark is not None:
            for key, pair in config.extra_active.items():
                values = np.asarray(self.data.extra_attributes[key], dtype=float)
                shift = float(values @ benchmark)
                self.add_block(
                    self._weight_block(values),
                    pair[0] + shift,
                    pair[1] + shift,
                    group="extra_active",
                    keys=(key,),
                    unit="exposure",
                )
        for key, pair in config.extra_absolute.items():
            values = np.asarray(self.data.extra_attributes[key], dtype=float)
            self.add_block(
                self._weight_block(values),
                pair[0],
                pair[1],
                group="extra_absolute",
                keys=(key,),
                unit="exposure",
            )

        objective = self.problem.objective
        if isinstance(objective, MinimizeTrackingError) and objective.alpha_floor is not None:
            assert self.data.alpha is not None
            self.add_block(
                self._weight_block(np.asarray(self.data.alpha, dtype=float)),
                objective.alpha_floor,
                np.inf,
                group="alpha_floor",
                keys=("alpha",),
                unit=self.data.alpha_spec.units if self.data.alpha_spec is not None else "alpha",
            )

    def _add_factor_bounds(self, expected_type: str, bounds: Any) -> None:
        r"""Compile style/industry active exposure bounds.

        QPs already contain $f=E^{\mathsf T}(x-b)$ variables, so a factor bound
        is one sparse coefficient on $f_j$. LP/factor-QCQP base domains do not
        yet contain $f$ and therefore use $E_j^{\mathsf T}x$ with
        benchmark-shifted bounds. The factor-QCQP strategy later replaces those
        dense rows when it introduces factor variables for its parameterized QP.
        """

        if bounds is None:
            return
        assert isinstance(self.risk_model, FactorRiskModel)
        assert self.data.benchmark is not None
        names = self.risk_model.factor_names
        factor_types = self.risk_model.factor_types
        selected: dict[int, tuple[float, float]] = {}
        if bounds.default is not None:
            selected.update(
                {
                    index: bounds.default
                    for index, factor_type in enumerate(factor_types)
                    if factor_type.lower() == expected_type
                }
            )
        lookup = {name.lower(): index for index, name in enumerate(names)}
        for name, pair in bounds.overrides.items():
            selected[lookup[name.lower()]] = pair
        if not selected:
            return
        indices = np.fromiter(sorted(selected), dtype=np.int32)
        lower = np.asarray([selected[int(i)][0] for i in indices])
        upper = np.asarray([selected[int(i)][1] for i in indices])
        if self.layout.factor is not None:
            # Factor variables already equal $E^{\mathsf T}(x-b)$, so their
            # bounds are sparse single-column rows. Repeating $E^{\mathsf T}$
            # here would duplicate the dense exposure block in PIQP's KKT matrix.
            rows = np.arange(len(indices), dtype=np.int32)
            matrix = sp.csc_matrix(
                (
                    np.ones(len(indices)),
                    (rows, self.layout.factor[indices]),
                ),
                shape=(len(indices), self.layout.n_variables),
            )
        else:
            exposure = np.asarray(self.risk_model.exposure, dtype=float)[:, indices]
            benchmark_shift = exposure.T @ np.asarray(self.data.benchmark, dtype=float)
            lower = lower + benchmark_shift
            upper = upper + benchmark_shift
            matrix = self._pad(sp.csc_matrix(exposure.T))
        self.add_block(
            matrix,
            lower,
            upper,
            group=expected_type,
            keys=(names[int(i)] for i in indices),
            unit="active_exposure",
        )

    def finish(self) -> LinearDomain:
        """Freeze accumulated blocks as sorted canonical CSC arrays."""

        if self.blocks:
            matrix = sp.vstack(self.blocks, format="csc")
            lower = np.concatenate(self.lowers)
            upper = np.concatenate(self.uppers)
        else:
            matrix = sp.csc_matrix((0, self.layout.n_variables), dtype=float)
            lower = np.empty(0, dtype=float)
            upper = np.empty(0, dtype=float)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return LinearDomain(
            A=matrix,
            lower=lower,
            upper=upper,
            variable_lower=self.variable_lower,
            variable_upper=self.variable_upper,
            variables=tuple(self.variables),
            constraints=tuple(self.variable_records + self.row_records),
            weight_indices=self.layout.weight,
            assets=tuple(self.data.assets),
        )


def _risk_operator(problem: PortfolioProblem, domain: LinearDomain) -> FactorRiskOperator:
    """Create the annualized factor-plus-specific tracking-risk operator."""

    risk_model = problem.data.risk_model
    if isinstance(risk_model, FullCovarianceRiskModel):
        raise CanonicalCompilationError("full-covariance risk objectives are not implemented in v1")
    if not isinstance(risk_model, FactorRiskModel):
        raise CanonicalCompilationError("a factor risk model is required")
    if problem.data.benchmark is None:
        raise CanonicalCompilationError("tracking risk requires a benchmark")
    covariance = np.asarray(risk_model.covariance, dtype=float)
    covariance = (covariance + covariance.T) * 0.5
    return FactorRiskOperator(
        exposure=np.asarray(risk_model.exposure, dtype=float),
        covariance=covariance,
        specific_volatility=np.asarray(risk_model.specific_volatility, dtype=float),
        benchmark=np.asarray(problem.data.benchmark, dtype=float),
        factor_names=risk_model.factor_names,
        weight_indices=domain.weight_indices,
    )


def _compile_lp(problem: PortfolioProblem) -> tuple[LinearProgram, tuple[str, ...]]:
    r"""Compile $\max_x\alpha^{\mathsf T}x$ as minimization vector $c$."""

    builder = _DomainBuilder(problem, include_factor_variables=False)
    builder.add_common_constraints()
    domain = builder.finish()
    assert problem.data.alpha is not None
    c = np.zeros(domain.n_variables, dtype=float)
    c[domain.weight_indices] = -np.asarray(problem.data.alpha, dtype=float)
    return LinearProgram(ProblemKind.LP, domain, c), tuple(builder.optimizations)


def _compile_qp(problem: PortfolioProblem) -> tuple[QuadraticProgram, tuple[str, ...]]:
    """Compile a low-rank factor plus diagonal-specific convex QP.

    Factor exposure is represented by explicit ``f`` variables, keeping the
    quadratic matrix at an asset diagonal plus a small factor block.  For a
    risk-adjusted alpha objective, alpha is translated by ``alpha @ benchmark``;
    the budget equality makes that translation constant, while PIQP objective
    scaling becomes insensitive to an arbitrary common shift of alpha.
    """

    if not isinstance(problem.data.risk_model, FactorRiskModel):
        raise CanonicalCompilationError("v1 QP compiler supports FactorRiskModel only")
    if problem.data.benchmark is None:
        raise CanonicalCompilationError("tracking-risk QP requires a benchmark")
    builder = _DomainBuilder(problem, include_factor_variables=True)
    builder.add_common_constraints()
    domain = builder.finish()
    assert builder.layout.factor is not None
    risk = _risk_operator(problem, domain)
    objective = problem.objective
    if isinstance(objective, RiskAdjustedAlpha):
        factor_aversion = objective.factor_aversion
        specific_aversion = objective.specific_aversion
    elif isinstance(objective, MinimizeTrackingError):
        factor_aversion = 1.0
        specific_aversion = 1.0
    else:
        raise CanonicalCompilationError(f"unsupported QP objective: {type(objective).__name__}")

    n_variables = domain.n_variables
    specific_variance = np.square(risk.specific_volatility)
    diagonal_rows = domain.weight_indices
    diagonal_values = 2.0 * specific_aversion * specific_variance
    factor_matrix = 2.0 * factor_aversion * risk.covariance
    factor_rows, factor_columns = np.nonzero(factor_matrix)
    P = sp.csc_matrix(
        (
            np.concatenate([diagonal_values, factor_matrix[factor_rows, factor_columns]]),
            (
                np.concatenate([diagonal_rows, builder.layout.factor[factor_rows]]),
                np.concatenate([diagonal_rows, builder.layout.factor[factor_columns]]),
            ),
        ),
        shape=(n_variables, n_variables),
    )
    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    q = np.zeros(n_variables, dtype=float)
    q[domain.weight_indices] = -2.0 * specific_aversion * specific_variance * risk.benchmark
    objective_scale_reference = None
    alpha_shift = 0.0
    if isinstance(objective, RiskAdjustedAlpha):
        assert problem.data.alpha is not None
        alpha = np.asarray(problem.data.alpha, dtype=float)
        # The budget equality makes an alpha translation mathematically
        # constant. Center at the benchmark so PIQP scaling depends on alpha
        # dispersion rather than on the risk linear term or arbitrary level.
        alpha_shift = float(alpha @ risk.benchmark)
        centered_alpha = alpha - alpha_shift
        q[domain.weight_indices] -= centered_alpha
        objective_scale_reference = float(np.max(np.abs(centered_alpha)))
    offset = float(
        specific_aversion * np.sum(specific_variance * np.square(risk.benchmark))
        - alpha_shift * problem.constraints.budget
    )
    risk_limit = (
        problem.constraints.tracking_error.annualized
        if problem.constraints.tracking_error is not None
        else None
    )
    return (
        QuadraticProgram(
            kind=ProblemKind.QP,
            domain=domain,
            P=P,
            q=q,
            objective_offset=offset,
            objective_scale_reference=objective_scale_reference,
            risk_operator=risk,
            risk_limit=risk_limit,
        ),
        tuple(builder.optimizations),
    )


def _compile_factor_qcqp(problem: PortfolioProblem) -> tuple[FactorQCQP, tuple[str, ...]]:
    """Compile the linear domain and factor risk operator for one TE budget."""

    if not isinstance(problem.data.risk_model, FactorRiskModel):
        raise CanonicalCompilationError("factor-QCQP requires FactorRiskModel")
    assert problem.constraints.tracking_error is not None
    assert problem.data.alpha is not None
    builder = _DomainBuilder(problem, include_factor_variables=False)
    builder.add_common_constraints()
    domain = builder.finish()
    model = FactorQCQP(
        kind=ProblemKind.FACTOR_QCQP,
        domain=domain,
        alpha=np.asarray(problem.data.alpha, dtype=float),
        risk_operator=_risk_operator(problem, domain),
        risk_limit=problem.constraints.tracking_error.annualized,
    )
    return model, tuple(builder.optimizations)


def compile_problem(problem: PortfolioProblem, *, validate: bool = True) -> CompiledProblem:
    """Validate and compile one problem without importing a solver backend.

    ``validate=False`` is reserved for callers, such as
    :meth:`PortfolioOptimizer.prepare`, that have just produced an equivalent
    validation report.  The returned fingerprint covers both semantic inputs
    and the final numerical payload so all fallback attempts can be audited as
    the same mathematical problem.
    """

    if validate:
        report = validate_problem(problem)
        report.raise_for_errors()
    kind = classify_problem(problem)
    model: CanonicalModel
    if kind is ProblemKind.LP:
        model, optimizations = _compile_lp(problem)
    elif kind is ProblemKind.QP:
        model, optimizations = _compile_qp(problem)
    elif kind is ProblemKind.FACTOR_QCQP:
        model, optimizations = _compile_factor_qcqp(problem)
    else:
        raise CanonicalCompilationError(
            "risk-adjusted objectives with an additional TE constraint require the conic fallback, "
            "which is not implemented in the first compiler milestone"
        )
    return CompiledProblem(
        model=model,
        fingerprint=fingerprint(problem, model),
        compiler_optimizations=optimizations,
    )

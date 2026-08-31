#!/usr/bin/env python3
"""Reproducible LP/QP/SOCP portfolio-solver benchmark.

The three model families are deliberately independent.  Every chained
(model, solver, repeat) case carries only its own previous solution.  Initial
portfolios can use benchmark top-N names, a date-stable random sample of
benchmark constituents, or a date-stable random top-N size.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


def _set_mosek_license_before_solver_import() -> None:
    """MOSEK may cache its license search path while CVXPY imports solvers."""
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == "--mosek-license" and index + 1 < len(sys.argv):
            os.environ["MOSEKLM_LICENSE_FILE"] = str(Path(sys.argv[index + 1]).resolve())
            return
        if argument.startswith("--mosek-license="):
            os.environ["MOSEKLM_LICENSE_FILE"] = str(
                Path(argument.split("=", 1)[1]).resolve()
            )
            return


_set_mosek_license_before_solver_import()

import cvxpy as cp
import numpy as np
import pandas as pd


STYLE_FACTORS = [
    "beta",
    "liquidty",
    "ltrevrsl",
    "midcap",
    "momentum",
    "resvol",
    "size",
    "btop",
    "divyild",
    "earnqlty",
    "earnyild",
    "leverage",
    "profit",
    "growth",
    "invsqlty",
    "earnvar",
]
RISK_MODEL_SHEETS = [
    "asset_exposure",
    "asset_data",
    "covariance",
]
LEGACY_ALPHA_SHEETS = [
    "dlyfacret",
    "specific_ret",
]
REQUIRED_SHEETS = RISK_MODEL_SHEETS + LEGACY_ALPHA_SHEETS
SUCCESS_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
PACKAGE_VERSION = "5.6.1-piqp-patched-acceptance"


def process_rss_mb() -> float | None:
    """Read current Linux RSS without adding a production dependency."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(frozen=True)
class Config:
    turnover_limit: float = 0.05
    active_ub: float = 0.004
    total_active_ub: float = 1.8
    benchmark_weight_lb: float = 0.81
    style_default_lb: float = -0.5
    style_default_ub: float = 0.5
    size_lb: float = -0.3
    size_ub: float = 0.3
    industry_lb: float = -0.03
    industry_ub: float = 0.03
    risk_budget_pct: float = 6.0
    common_risk_aversion: float = 0.75
    specific_risk_aversion: float = 0.75
    asset_ub: float = 1.0
    initial_top_n: int = 500
    weight_zero_tol: float = 1e-12
    missing_holding_tol: float = 1e-8


@dataclass
class DayData:
    date: pd.Timestamp
    sids: np.ndarray
    factors: list[str]
    style_idx: list[int]
    industry_idx: list[int]
    exposure: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    alpha: np.ndarray
    benchmark: np.ndarray
    initial: np.ndarray
    spec_risk: np.ndarray
    benchmark_missing_fraction: float
    initial_missing_mass: float
    initial_renormalization_factor: float
    benchmark_source_date: pd.Timestamp | None
    alpha_source_date: pd.Timestamp
    benchmark_stale_days: int
    alpha_stale_days: int
    alpha_missing_count: int
    alpha_missing_fraction: float


class DataStore:
    CACHE_SCHEMA = 1

    def __init__(
        self,
        xlsx: Path,
        benchmark_csv: Path,
        alpha_csv: Path | None = None,
        cache_dir: Path | None = None,
    ):
        started = time.perf_counter()
        required_sheets = list(RISK_MODEL_SHEETS)
        if alpha_csv is None:
            required_sheets.extend(LEGACY_ALPHA_SHEETS)
        cache_path = self._cache_path(
            cache_dir, xlsx, benchmark_csv, alpha_csv, required_sheets
        )
        cached = False
        if cache_path is not None and cache_path.is_file():
            print(f"[data] loading parsed-data cache: {cache_path}", flush=True)
            try:
                payload = pd.read_pickle(cache_path)
                if payload.get("cache_schema") != self.CACHE_SCHEMA:
                    raise ValueError("cache schema mismatch")
                self.raw = payload["raw"]
                self.benchmark_raw = payload["benchmark_raw"]
                self.benchmark_dates = payload["benchmark_dates"]
                self.alpha_raw = payload["alpha_raw"]
                self.alpha_dates = payload["alpha_dates"]
                cached = True
            except Exception as exc:
                print(
                    f"[data] ignoring unreadable cache ({type(exc).__name__}: {exc})",
                    flush=True,
                )
        if not cached:
            print(f"[data] inspecting workbook: {xlsx}", flush=True)
            with pd.ExcelFile(xlsx) as workbook:
                workbook_sheet_names = set(workbook.sheet_names)
            missing_sheets = sorted(set(required_sheets) - workbook_sheet_names)
            if missing_sheets:
                raise ValueError(f"workbook is missing sheets: {missing_sheets}")
            self.raw = {}
            for index, name in enumerate(required_sheets, start=1):
                print(
                    f"[data] reading Excel sheet {index}/{len(required_sheets)}: {name}",
                    flush=True,
                )
                self.raw[name] = pd.read_excel(xlsx, sheet_name=name)
            print(f"[data] reading benchmark CSV: {benchmark_csv}", flush=True)
            self.benchmark_raw, self.benchmark_dates = self._read_panel_csv(
                benchmark_csv, "benchmark", "weight"
            )
            self.alpha_raw = None
            self.alpha_dates = None
            if alpha_csv is not None:
                print(f"[data] reading alpha CSV: {alpha_csv}", flush=True)
                self.alpha_raw, self.alpha_dates = self._read_panel_csv(
                    alpha_csv, "alpha", "fv"
                )
        self._validate_columns()
        if not cached:
            for frame in self.raw.values():
                frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
                if "sid" in frame.columns:
                    frame["sid"] = frame["sid"].astype(str)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_cache = cache_path.with_name(
                    f"{cache_path.name}.{os.getpid()}.tmp"
                )
                print(f"[data] writing parsed-data cache: {cache_path}", flush=True)
                pd.to_pickle(
                    {
                        "cache_schema": self.CACHE_SCHEMA,
                        "raw": self.raw,
                        "benchmark_raw": self.benchmark_raw,
                        "benchmark_dates": self.benchmark_dates,
                        "alpha_raw": self.alpha_raw,
                        "alpha_dates": self.alpha_dates,
                    },
                    temporary_cache,
                    protocol=5,
                )
                os.replace(temporary_cache, cache_path)

        self.risk_date_sheets = required_sheets

        risk_date_sets = [
            set(self.raw[name]["date"].dropna().unique()) for name in required_sheets
        ]
        self.risk_dates = [
            pd.Timestamp(value) for value in sorted(set.intersection(*risk_date_sets))
        ]
        self._benchmark_groups = self._group_panel(self.benchmark_raw, "weight")
        self._alpha_groups = (
            self._group_panel(self.alpha_raw, "fv") if self.alpha_raw is not None else {}
        )
        self._alignment: dict[
            pd.Timestamp, tuple[pd.Timestamp | None, pd.Timestamp]
        ] = {}
        self.alignment_records: list[dict[str, Any]] = []
        self.dropped_leading_dates: list[pd.Timestamp] = []
        aligned_dates: list[pd.Timestamp] = []
        for date in self.risk_dates:
            benchmark_source = self._asof_source_date(self.benchmark_dates, date)
            alpha_source = (
                self._asof_source_date(self.alpha_dates, date)
                if self.alpha_raw is not None
                else date
            )
            benchmark_available = self.benchmark_dates is None or benchmark_source is not None
            alpha_available = self.alpha_raw is None or alpha_source is not None
            if not benchmark_available or not alpha_available:
                self.dropped_leading_dates.append(date)
                continue
            assert alpha_source is not None
            self._alignment[date] = (benchmark_source, alpha_source)
            aligned_dates.append(date)
            self.alignment_records.append(
                {
                    "risk_date": date.strftime("%Y-%m-%d"),
                    "benchmark_source_date": (
                        None
                        if self.benchmark_dates is None
                        else benchmark_source.strftime("%Y-%m-%d")
                    ),
                    "alpha_source_date": alpha_source.strftime("%Y-%m-%d"),
                    "benchmark_stale_days": (
                        0 if self.benchmark_dates is None else int((date - benchmark_source).days)
                    ),
                    "alpha_stale_days": int((date - alpha_source).days),
                }
            )
        self._dates = aligned_dates
        self._risk_universe_cache: dict[pd.Timestamp, set[str]] = {}
        self.load_elapsed_s = time.perf_counter() - started
        source = "cache" if cached else "source files"
        print(
            f"[data] ready from {source}: {len(self._dates)} aligned risk dates "
            f"in {self.load_elapsed_s:.3f}s",
            flush=True,
        )

    @classmethod
    def _cache_path(
        cls,
        cache_dir: Path | None,
        xlsx: Path,
        benchmark_csv: Path,
        alpha_csv: Path | None,
        required_sheets: list[str],
    ) -> Path | None:
        if cache_dir is None:
            return None

        def signature(path: Path | None) -> dict[str, Any] | None:
            if path is None:
                return None
            resolved = path.resolve()
            stat = resolved.stat()
            return {
                "path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        key = json.dumps(
            {
                "cache_schema": cls.CACHE_SCHEMA,
                "xlsx": signature(xlsx),
                "benchmark_csv": signature(benchmark_csv),
                "alpha_csv": signature(alpha_csv),
                "required_sheets": required_sheets,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:24]
        return cache_dir.resolve() / f"datastore_{digest}.pkl"

    @staticmethod
    def _read_panel_csv(
        path: Path, name: str, value_column: str
    ) -> tuple[pd.DataFrame, list[pd.Timestamp] | None]:
        frame = pd.read_csv(path, dtype={"sid": str})
        missing = {"sid", value_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
        date_column = "dt" if "dt" in frame.columns else "date" if "date" in frame.columns else None
        keep = ["sid", value_column] + ([date_column] if date_column is not None else [])
        frame = frame[keep].copy()
        if date_column is not None:
            frame = frame.rename(columns={date_column: "source_date"})
            frame["source_date"] = pd.to_datetime(frame["source_date"]).dt.normalize()
            if frame["source_date"].isna().any():
                raise ValueError(f"{name} CSV contains invalid dates")
            duplicate_columns = ["source_date", "sid"]
            dates = [
                pd.Timestamp(value)
                for value in sorted(frame["source_date"].drop_duplicates().tolist())
            ]
        else:
            duplicate_columns = ["sid"]
            dates = None
        frame["sid"] = frame["sid"].astype(str)
        frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
        if not np.isfinite(frame[value_column].to_numpy(float)).all():
            raise ValueError(f"{name} CSV contains non-finite {value_column} values")
        if frame.duplicated(duplicate_columns).any():
            raise ValueError(f"{name} CSV contains duplicate keys {duplicate_columns}")
        if name == "benchmark":
            if (frame[value_column] < 0.0).any():
                raise ValueError("benchmark weights must be nonnegative")
            sums = (
                frame.groupby("source_date")[value_column].sum()
                if dates is not None
                else pd.Series([frame[value_column].sum()])
            )
            if (sums <= 0.0).any():
                raise ValueError("every benchmark cross-section must have positive weight")
        return frame, dates

    @staticmethod
    def _group_panel(
        frame: pd.DataFrame | None, value_column: str
    ) -> dict[pd.Timestamp | None, pd.Series]:
        if frame is None:
            return {}
        if "source_date" not in frame.columns:
            return {None: frame.set_index("sid")[value_column].astype(float)}
        return {
            pd.Timestamp(date): group.set_index("sid")[value_column].astype(float)
            for date, group in frame.groupby("source_date", sort=True)
        }

    @staticmethod
    def _asof_source_date(
        available: list[pd.Timestamp] | None, target: pd.Timestamp
    ) -> pd.Timestamp | None:
        if available is None:
            return None
        position = int(pd.DatetimeIndex(available).searchsorted(target, side="right")) - 1
        return None if position < 0 else available[position]

    def _validate_columns(self) -> None:
        required = {
            "asset_exposure": {"date", "sid", "INDUSTRY"},
            "dlyfacret": {"date"},
            "asset_data": {"date", "sid", "spec_risk"},
            "covariance": {"date", "factor_name"},
            "specific_ret": {"date", "sid", "spret"},
        }
        for sheet, columns in required.items():
            if sheet not in self.raw:
                continue
            missing = columns - set(self.raw[sheet].columns)
            if missing:
                raise ValueError(f"sheet {sheet!r} is missing columns: {sorted(missing)}")

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(self._dates)

    def benchmark_for_date(self, date: pd.Timestamp) -> pd.Series:
        benchmark_source, _ = self._alignment[date]
        result = self._benchmark_groups[benchmark_source].copy()
        result = result[result > 0.0]
        total = float(result.sum())
        if total <= 0.0:
            raise ValueError(f"empty benchmark on {date:%Y-%m-%d}")
        return result / total

    def benchmark_turnover_l1(self, date: pd.Timestamp) -> float:
        position = self._dates.index(date)
        if position == 0:
            return 0.0
        previous = self.benchmark_for_date(self._dates[position - 1]).rename("previous")
        current = self.benchmark_for_date(date).rename("current")
        aligned = pd.concat([previous, current], axis=1).fillna(0.0)
        return float((aligned["current"] - aligned["previous"]).abs().sum())

    def alpha_for_date(self, date: pd.Timestamp) -> pd.Series | None:
        if self.alpha_raw is None:
            return None
        _, alpha_source = self._alignment[date]
        return self._alpha_groups[alpha_source].copy()

    def risk_universe_sids(self, date: pd.Timestamp) -> set[str]:
        cached = self._risk_universe_cache.get(date)
        if cached is not None:
            return cached
        exposure = set(
            self.raw["asset_exposure"].loc[
                self.raw["asset_exposure"]["date"] == date, "sid"
            ]
        )
        asset_data = set(
            self.raw["asset_data"].loc[
                self.raw["asset_data"]["date"] == date, "sid"
            ]
        )
        result = exposure & asset_data
        if self.alpha_raw is None:
            specific = set(
                self.raw["specific_ret"].loc[
                    self.raw["specific_ret"]["date"] == date, "sid"
                ]
            )
            result &= specific
        self._risk_universe_cache[date] = result
        return result

    def top_n_initial(self, n: int, date: pd.Timestamp | None = None) -> pd.Series:
        date = self.dates[0] if date is None else date
        benchmark = self.benchmark_for_date(date)
        benchmark = benchmark[benchmark.index.isin(self.risk_universe_sids(date))]
        result = (
            benchmark.rename("weight")
            .rename_axis("sid")
            .reset_index(name="weight")
            .sort_values(["weight", "sid"], ascending=[False, True], kind="mergesort")
            .head(n)
            .set_index("sid")["weight"]
        )
        if len(result) < n:
            raise ValueError(
                f"{date:%Y-%m-%d}: only {len(result)} benchmark members have risk data; "
                f"cannot form top-{n} initial portfolio"
            )
        return result / result.sum()

    def random_n_initial(self, date: pd.Timestamp, n: int, seed: int) -> pd.Series:
        benchmark = self.benchmark_for_date(date)
        benchmark = benchmark[benchmark.index.isin(self.risk_universe_sids(date))]
        benchmark = benchmark.sort_index(kind="mergesort")
        if len(benchmark) < n:
            raise ValueError(
                f"{date:%Y-%m-%d}: only {len(benchmark)} benchmark members have risk data; "
                f"cannot sample {n} names"
            )
        date_key = int(date.strftime("%Y%m%d"))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), date_key]))
        selected_positions = np.sort(rng.choice(len(benchmark), size=n, replace=False))
        result = benchmark.iloc[selected_positions].copy()
        return result / result.sum()

    def random_top_range_initial(
        self,
        date: pd.Timestamp,
        n_min: int,
        n_max: int,
        seed: int,
    ) -> tuple[pd.Series, int]:
        """Take the benchmark's largest N names for date-stable random N.

        For a fixed N this choice retains the greatest possible benchmark mass
        among all N-name subsets, so truncation and renormalization introduce
        the smallest benchmark L1 displacement.  Only N is random; membership
        is determined by benchmark weights with sid as a stable tie-breaker.
        """
        if n_min <= 0 or n_max < n_min:
            raise ValueError(f"invalid random top-N range [{n_min}, {n_max}]")
        date_key = int(date.strftime("%Y%m%d"))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), date_key]))
        n = int(rng.integers(n_min, n_max + 1))
        return self.top_n_initial(n, date), n

    def prepare_day(
        self,
        date: pd.Timestamp,
        initial: pd.Series,
        config: Config,
        missing_holding_policy: str,
    ) -> DayData:
        exposure_raw = self.raw["asset_exposure"]
        exposure = exposure_raw[exposure_raw["date"] == date].copy()
        risk_raw = self.raw["asset_data"]
        risk = risk_raw[risk_raw["date"] == date][["sid", "spec_risk"]]
        universe = exposure.merge(risk, on="sid", how="inner")
        if self.alpha_raw is None:
            specific_raw = self.raw["specific_ret"]
            specific = specific_raw[specific_raw["date"] == date][["sid", "spret"]]
            universe = universe.merge(specific, on="sid", how="inner")
        universe = universe.sort_values("sid", kind="mergesort").reset_index(drop=True)
        if universe.empty:
            raise ValueError(f"empty risk universe on {date:%Y-%m-%d}")

        cov_raw = self.raw["covariance"]
        cov = cov_raw[cov_raw["date"] == date].copy()
        factors = cov["factor_name"].astype(str).tolist()
        if len(factors) != len(set(factors)):
            raise ValueError(f"duplicate covariance factors on {date:%Y-%m-%d}")
        covariance = cov.set_index("factor_name").loc[factors, factors].to_numpy(float)
        covariance = (covariance + covariance.T) / 2.0
        chol = np.linalg.cholesky(covariance)

        exposure_columns = {
            column.removeprefix("CNLTS_").lower(): column
            for column in exposure.columns
            if column.startswith("CNLTS_")
        }
        industry = (
            universe["INDUSTRY"].astype(str).str.removeprefix("CNLTS_").str.lower()
        )
        columns: list[np.ndarray] = []
        for factor in factors:
            if factor in exposure_columns:
                columns.append(universe[exposure_columns[factor]].to_numpy(float))
            elif factor == "country":
                columns.append(np.ones(len(universe)))
            else:
                columns.append((industry == factor).to_numpy(float))
        exposure_matrix = np.column_stack(columns)

        absent_styles = sorted(set(STYLE_FACTORS) - set(factors))
        if absent_styles:
            raise ValueError(f"missing style factors on {date:%Y-%m-%d}: {absent_styles}")
        style_idx = [factors.index(factor) for factor in STYLE_FACTORS]
        industry_idx = [
            idx for idx, factor in enumerate(factors) if factor not in STYLE_FACTORS and factor != "country"
        ]

        alpha_map = self.alpha_for_date(date)
        if alpha_map is None:
            factor_return_raw = self.raw["dlyfacret"]
            factor_return_rows = factor_return_raw[factor_return_raw["date"] == date]
            if len(factor_return_rows) != 1:
                raise ValueError(f"expected one factor-return row on {date:%Y-%m-%d}")
            factor_return = factor_return_rows.iloc[0][factors].to_numpy(float)
            alpha = 100.0 * (exposure_matrix @ factor_return) + universe["spret"].to_numpy(float)
            alpha_missing_count = 0
        else:
            aligned_alpha = universe["sid"].map(alpha_map)
            alpha_missing_count = int(aligned_alpha.isna().sum())
            # A missing asset-level alpha means "no signal", not a missing risk row.
            # Date-level gaps are handled separately by strict backward as-of lookup.
            alpha = aligned_alpha.fillna(0.0).to_numpy(float)

        benchmark_map = self.benchmark_for_date(date)
        benchmark = universe["sid"].map(benchmark_map).fillna(0.0).to_numpy(float)
        covered_benchmark_mass = float(benchmark.sum())
        if covered_benchmark_mass <= 0:
            raise ValueError(f"no benchmark members have risk data on {date:%Y-%m-%d}")
        benchmark /= covered_benchmark_mass

        initial = initial.astype(float)
        initial = initial[initial > config.weight_zero_tol]
        source_mass = float(initial.sum())
        if source_mass <= 0:
            raise ValueError("initial holdings have no positive mass")
        initial = initial / source_mass
        initial_weights = universe["sid"].map(initial).fillna(0.0).to_numpy(float)
        retained_mass = float(initial_weights.sum())
        missing_mass = max(0.0, 1.0 - retained_mass)
        if missing_mass > config.missing_holding_tol and missing_holding_policy == "error":
            raise ValueError(
                f"{date:%Y-%m-%d}: {missing_mass:.6%} of previous holdings lack risk data; "
                "use --missing-holding-policy renormalize only if that treatment is intended"
            )
        if retained_mass <= 0:
            raise ValueError(f"no initial holdings survive on {date:%Y-%m-%d}")
        initial_weights /= retained_mass
        initial_weights[np.abs(initial_weights) <= config.weight_zero_tol] = 0.0
        initial_weights /= initial_weights.sum()

        benchmark_source, alpha_source = self._alignment[date]
        return DayData(
            date=date,
            sids=universe["sid"].astype(str).to_numpy(),
            factors=factors,
            style_idx=style_idx,
            industry_idx=industry_idx,
            exposure=exposure_matrix,
            covariance=covariance,
            chol=chol,
            alpha=alpha,
            benchmark=benchmark,
            initial=initial_weights,
            spec_risk=universe["spec_risk"].to_numpy(float),
            benchmark_missing_fraction=max(0.0, 1.0 - covered_benchmark_mass),
            initial_missing_mass=missing_mass,
            initial_renormalization_factor=1.0 / retained_mass,
            benchmark_source_date=benchmark_source,
            alpha_source_date=alpha_source,
            benchmark_stale_days=(
                0 if benchmark_source is None else int((date - benchmark_source).days)
            ),
            alpha_stale_days=int((date - alpha_source).days),
            alpha_missing_count=alpha_missing_count,
            alpha_missing_fraction=alpha_missing_count / len(universe),
        )


@dataclass
class BuiltProblem:
    problem: cp.Problem
    x: cp.Variable
    risk: cp.Expression
    factor_risk: cp.Expression
    specific_risk: cp.Expression
    omitted_total_active: bool
    omitted_benchmark_weight: bool
    sparse_turnover: bool
    objective_scale: float
    alpha_shift: float


def build_problem(
    day: DayData,
    model: str,
    config: Config,
    formulation: str,
    objective_target: float | None = None,
) -> BuiltProblem:
    n = len(day.sids)
    x = cp.Variable(n)
    active = x - day.benchmark
    # CVXPY 1.9.x infers Sum shapes by reducing an uninitialized np.empty
    # array. NumPy 2.4 can therefore emit a harmless "invalid value encountered
    # in reduce" warning. No numerical values are evaluated at this stage.
    with np.errstate(invalid="ignore"):
        budget_sum = cp.sum(x)
    constraints: list[cp.Constraint] = [x >= 0.0, budget_sum == 1.0]
    if formulation == "canonical":
        constraints.append(x <= config.asset_ub)
    elif config.asset_ub < 1.0:
        constraints.append(x <= config.asset_ub)
    constraints += [active >= -config.active_ub, active <= config.active_ub]

    style_exposure = day.exposure[:, day.style_idx].T @ active
    style_lb = np.full(len(STYLE_FACTORS), config.style_default_lb)
    style_ub = np.full(len(STYLE_FACTORS), config.style_default_ub)
    size_idx = STYLE_FACTORS.index("size")
    style_lb[size_idx], style_ub[size_idx] = config.size_lb, config.size_ub
    constraints += [style_exposure >= style_lb, style_exposure <= style_ub]
    industry_exposure = day.exposure[:, day.industry_idx].T @ active
    constraints += [
        industry_exposure >= config.industry_lb,
        industry_exposure <= config.industry_ub,
    ]

    sparse_turnover = formulation == "optimized"
    if sparse_turnover:
        support = day.initial > config.weight_zero_tol
        positive_trade = cp.Variable(int(support.sum()), nonneg=True)
        constraints.append(positive_trade >= x[support] - day.initial[support])
        # $\mathbf1^{\mathsf T}(x-x_0)=0$, hence L1 turnover equals twice the
        # sum of positive trades.
        with np.errstate(invalid="ignore"):
            positive_trade_sum = cp.sum(positive_trade)
            new_name_sum = cp.sum(x[~support])
        constraints.append(
            2.0 * (positive_trade_sum + new_name_sum) <= config.turnover_limit
        )
    else:
        constraints.append(cp.norm1(x - day.initial) <= config.turnover_limit)

    x0_active_l1 = float(np.abs(day.initial - day.benchmark).sum())
    omitted_total_active = (
        formulation == "optimized"
        and x0_active_l1 + config.turnover_limit <= config.total_active_ub + 1e-12
    )
    if not omitted_total_active:
        constraints.append(cp.norm1(active) <= config.total_active_ub)

    benchmark_mask = day.benchmark > 0
    x0_benchmark_weight = float(day.initial[benchmark_mask].sum())
    omitted_benchmark_weight = (
        formulation == "optimized"
        and x0_benchmark_weight - config.turnover_limit / 2.0
        >= config.benchmark_weight_lb - 1e-12
    )
    if not omitted_benchmark_weight:
        with np.errstate(invalid="ignore"):
            benchmark_weight_sum = cp.sum(x[benchmark_mask])
        constraints.append(benchmark_weight_sum >= config.benchmark_weight_lb)

    factor_risk = day.chol.T @ day.exposure.T @ active
    specific_risk = cp.multiply(day.spec_risk, active)
    risk = cp.norm(cp.hstack([factor_risk, specific_risk]), 2)
    if model == "socp":
        constraints.append(risk <= config.risk_budget_pct)

    alpha_shift = 0.0
    objective_scale = 1.0
    solver_alpha = day.alpha
    if objective_target is not None:
        # $\mathbf1^{\mathsf T}x=1$ means subtracting a constant from every alpha coefficient
        # only changes the objective by a constant.  Multiplying the complete
        # objective by a positive scalar also leaves the optimizer unchanged.
        alpha_shift = float(day.alpha @ day.benchmark)
        solver_alpha = day.alpha - alpha_shift
        max_abs_alpha = float(np.max(np.abs(solver_alpha)))
        if max_abs_alpha > 0.0:
            objective_scale = objective_target / max_abs_alpha

    if model == "qp":
        raw_objective = (
            solver_alpha @ x
            - config.common_risk_aversion * cp.sum_squares(factor_risk)
            - config.specific_risk_aversion * cp.sum_squares(specific_risk)
        )
        objective = cp.Maximize(objective_scale * raw_objective)
    else:
        objective = cp.Maximize(objective_scale * (solver_alpha @ x))
    return BuiltProblem(
        problem=cp.Problem(objective, constraints),
        x=x,
        risk=risk,
        factor_risk=factor_risk,
        specific_risk=specific_risk,
        omitted_total_active=omitted_total_active,
        omitted_benchmark_weight=omitted_benchmark_weight,
        sparse_turnover=sparse_turnover,
        objective_scale=objective_scale,
        alpha_shift=alpha_shift,
    )


@dataclass(frozen=True)
class CaseSpec:
    name: str
    solver: str
    objective_target: float | None = None
    lp_screen: bool = False
    direct_method: str | None = None
    max_threads: int | None = None
    factor_qcqp: bool = False
    factor_qcqp_backend: str | None = None
    factor_penalty_qp: bool = False


def resolve_case(model: str, name: str, alpha_target: float) -> CaseSpec:
    name = name.upper()
    variants = {
        "FACTOR_PENALTY_QP_PIQP": CaseSpec(
            name="FACTOR_PENALTY_QP_PIQP",
            solver="PIQP",
            objective_target=alpha_target,
            factor_penalty_qp=True,
        ),
        "FACTOR_QP_OSQP": CaseSpec(
            name="FACTOR_QP_OSQP",
            solver="OSQP",
            objective_target=alpha_target,
            factor_qcqp=True,
            factor_qcqp_backend="OSQP",
        ),
        "FACTOR_QP_OSQP_SCREENED": CaseSpec(
            name="FACTOR_QP_OSQP_SCREENED",
            solver="OSQP",
            objective_target=alpha_target,
            lp_screen=True,
            factor_qcqp=True,
            factor_qcqp_backend="OSQP",
        ),
        "FACTOR_QP_PIQP": CaseSpec(
            name="FACTOR_QP_PIQP",
            solver="PIQP",
            objective_target=alpha_target,
            factor_qcqp=True,
            factor_qcqp_backend="PIQP",
        ),
        "FACTOR_QP_PIQP_SCREENED": CaseSpec(
            name="FACTOR_QP_PIQP_SCREENED",
            solver="PIQP",
            objective_target=alpha_target,
            lp_screen=True,
            factor_qcqp=True,
            factor_qcqp_backend="PIQP",
        ),
        "CLARABEL_SCALED": CaseSpec(
            name="CLARABEL_SCALED", solver="CLARABEL", objective_target=alpha_target
        ),
        "CLARABEL_SCREENED": CaseSpec(
            name="CLARABEL_SCREENED", solver="CLARABEL", lp_screen=True
        ),
        "CLARABEL_SCREENED_SCALED": CaseSpec(
            name="CLARABEL_SCREENED_SCALED",
            solver="CLARABEL",
            objective_target=alpha_target,
            lp_screen=True,
        ),
        "CLARABEL_QDLDL": CaseSpec(
            name="CLARABEL_QDLDL", solver="CLARABEL", direct_method="qdldl"
        ),
        "CLARABEL_SCALED_QDLDL": CaseSpec(
            name="CLARABEL_SCALED_QDLDL",
            solver="CLARABEL",
            objective_target=alpha_target,
            direct_method="qdldl",
        ),
        "CLARABEL_SCALED_FAER": CaseSpec(
            name="CLARABEL_SCALED_FAER",
            solver="CLARABEL",
            objective_target=alpha_target,
            direct_method="faer",
        ),
    }
    for threads in (1, 4, 8, 16, 32):
        variants[f"CLARABEL_MKL_T{threads}"] = CaseSpec(
            name=f"CLARABEL_MKL_T{threads}",
            solver="CLARABEL",
            direct_method="mkl",
            max_threads=threads,
        )
        variants[f"CLARABEL_SCALED_MKL_T{threads}"] = CaseSpec(
            name=f"CLARABEL_SCALED_MKL_T{threads}",
            solver="CLARABEL",
            objective_target=alpha_target,
            direct_method="mkl",
            max_threads=threads,
        )
    if name in variants:
        # LP screening and the parametric factor-QCQP path are SOCP-specific.
        # Objective scaling and Clarabel direct linear algebra backends are
        # equally valid for LP/QP because the scaling is positive and the alpha
        # The shift is constant under $\mathbf1^{\mathsf T}x=1$.
        if model != "socp" and (
            variants[name].lp_screen or variants[name].factor_qcqp
        ):
            raise ValueError(f"case {name} is only defined for SOCP")
        if variants[name].factor_penalty_qp and model != "qp":
            raise ValueError(f"case {name} is only defined for QP")
        return variants[name]
    return CaseSpec(name=name, solver=name)


def solver_kwargs(
    solver: str,
    time_limit_s: float | None,
    direct_method: str | None = None,
    clarabel_max_threads: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"solver": solver, "verbose": verbose}
    if solver == "ECOS":
        kwargs.update(abstol=1e-8, reltol=1e-8, feastol=1e-8, max_iters=200)
    elif solver == "SCS":
        kwargs.update(eps=1e-5, max_iters=100_000)
        if time_limit_s is not None:
            kwargs["time_limit_secs"] = time_limit_s
    elif solver == "OSQP":
        kwargs.update(eps_abs=1e-6, eps_rel=1e-6, max_iter=100_000, polish=True)
        if time_limit_s is not None:
            kwargs["time_limit"] = time_limit_s
    elif solver == "CLARABEL":
        if time_limit_s is not None:
            kwargs["time_limit"] = time_limit_s
        if direct_method is not None:
            kwargs["direct_solve_method"] = direct_method
        if clarabel_max_threads is not None:
            kwargs["max_threads"] = clarabel_max_threads
        if direct_method == "mkl" and verbose:
            kwargs["pardiso_verbose"] = True
    elif solver == "MOSEK" and time_limit_s is not None:
        kwargs["mosek_params"] = {"MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit_s}
    elif solver == "HIGHS" and time_limit_s is not None:
        kwargs["highs_options"] = {"time_limit": time_limit_s}
    return kwargs


def max_positive(value: float) -> float:
    return max(0.0, float(value))


def evaluate_weight(
    day: DayData,
    weight: np.ndarray,
    model: str,
    config: Config,
    objective: float,
) -> dict[str, Any]:
    weight = np.asarray(weight, dtype=float).reshape(-1)
    active = weight - day.benchmark
    style = day.exposure[:, day.style_idx].T @ active
    industry = day.exposure[:, day.industry_idx].T @ active
    style_lb = np.full(len(STYLE_FACTORS), config.style_default_lb)
    style_ub = np.full(len(STYLE_FACTORS), config.style_default_ub)
    size_idx = STYLE_FACTORS.index("size")
    style_lb[size_idx], style_ub[size_idx] = config.size_lb, config.size_ub
    tracking_error = float(
        np.linalg.norm(
            np.concatenate(
                [day.chol.T @ day.exposure.T @ active, day.spec_risk * active]
            )
        )
    )
    turnover = float(np.abs(weight - day.initial).sum())
    total_active = float(np.abs(active).sum())
    benchmark_weight = float(weight[day.benchmark > 0].sum())
    factor_variance = float(np.sum((day.chol.T @ day.exposure.T @ active) ** 2))
    specific_variance = float(np.sum((day.spec_risk * active) ** 2))
    raw_objective = float(day.alpha @ weight)
    if model == "qp":
        raw_objective -= config.common_risk_aversion * factor_variance
        raw_objective -= config.specific_risk_aversion * specific_variance
    violations = {
        "budget": abs(float(weight.sum()) - 1.0),
        "nonnegative": max_positive(-float(weight.min())),
        "asset_ub": max_positive(float(weight.max()) - config.asset_ub),
        "active_ub": max_positive(float(np.abs(active).max()) - config.active_ub),
        "style": max(
            max_positive(float(np.max(style_lb - style))),
            max_positive(float(np.max(style - style_ub))),
        ),
        "industry": max(
            max_positive(float(np.max(config.industry_lb - industry))),
            max_positive(float(np.max(industry - config.industry_ub))),
        ),
        "turnover": max_positive(turnover - config.turnover_limit),
        "total_active": max_positive(total_active - config.total_active_ub),
        "benchmark_weight": max_positive(config.benchmark_weight_lb - benchmark_weight),
        "risk": max_positive(tracking_error - config.risk_budget_pct) if model == "socp" else 0.0,
    }
    return {
        "weight": weight,
        "objective": float(objective),
        "raw_objective": raw_objective,
        "alpha_value": float(day.alpha @ weight),
        "tracking_error_pct": tracking_error,
        "factor_variance_pct2": factor_variance,
        "specific_variance_pct2": specific_variance,
        "factor_risk_pct": float(np.sqrt(max(0.0, factor_variance))),
        "specific_risk_pct": float(np.sqrt(max(0.0, specific_variance))),
        "max_abs_style_exposure": float(np.max(np.abs(style))),
        "max_abs_industry_exposure": float(np.max(np.abs(industry))),
        "turnover_l1": turnover,
        "total_active_l1": total_active,
        "benchmark_weight": benchmark_weight,
        "max_weight": float(weight.max()),
        "min_weight": float(weight.min()),
        "max_active_abs": float(np.abs(active).max()),
        "max_violation": max(violations.values()),
        **{f"violation_{key}": value for key, value in violations.items()},
    }


def evaluate(day: DayData, built: BuiltProblem, model: str, config: Config) -> dict[str, Any]:
    return evaluate_weight(
        day,
        np.asarray(built.x.value, dtype=float),
        model,
        config,
        float(built.problem.value),
    )


def portfolio_fingerprint(sids: np.ndarray, weights: np.ndarray, tolerance: float) -> str:
    selected = [
        f"{sid}:{weight:.17g}"
        for sid, weight in zip(sids, weights, strict=True)
        if weight > tolerance
    ]
    return hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()[:16]


def chained_state_from_weight(
    sids: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Convert a numerical solver result into the next tradable state.

    Interior-point solutions commonly contain thousands of tiny positive values.
    Treating those values as real holdings makes the next day's sparse turnover
    formulation dense. The discarded mass and renormalization are returned so
    this execution-level cleanup is fully auditable in ``runs.csv``.
    """
    raw = np.asarray(weights, dtype=float).reshape(-1)
    if raw.shape != sids.shape:
        raise ValueError("weight and sid shapes differ while building chained state")
    if not np.isfinite(raw).all():
        raise ValueError("non-finite weight while building chained state")
    positive = np.clip(raw, 0.0, None)
    keep = positive > 0.0 if tolerance == 0.0 else positive >= tolerance
    kept_mass = float(positive[keep].sum())
    if kept_mass <= 0.0:
        raise ValueError(
            "chained-state tolerance removed the entire portfolio; "
            f"tolerance={tolerance:g}"
        )
    discarded = positive[~keep]
    state = pd.Series(positive[keep] / kept_mass, index=sids[keep])
    return state, {
        "next_state_nonzero_count": int(keep.sum()),
        "next_state_cleanup_mass": float(discarded.sum()),
        "next_state_cleanup_max_weight": (
            float(discarded.max()) if discarded.size else 0.0
        ),
        "next_state_negative_mass_clipped": float(np.maximum(-raw, 0.0).sum()),
        "next_state_pre_normalization_mass": kept_mass,
        "next_state_renormalization_factor": 1.0 / kept_mass,
    }


def minimum_turnover_to_asset_bounds(day: DayData, config: Config) -> float:
    """Lower bound on L1 turnover needed to satisfy long-only active bounds.

    This is an exact lower bound for the budget-plus-box relaxation.  If it is
    above the turnover limit, the full portfolio model is provably infeasible,
    independently of the numerical solver used.
    """
    lower = np.maximum(0.0, day.benchmark - config.active_ub)
    upper = np.minimum(config.asset_ub, day.benchmark + config.active_ub)
    if float(lower.sum()) > 1.0 + 1e-12 or float(upper.sum()) < 1.0 - 1e-12:
        return float("inf")
    mandatory_sales = float(np.maximum(day.initial - upper, 0.0).sum())
    mandatory_buys = float(np.maximum(lower - day.initial, 0.0).sum())
    return 2.0 * max(mandatory_sales, mandatory_buys)


def solve_chain(
    store: DataStore,
    dates: list[pd.Timestamp],
    model: str,
    case: CaseSpec,
    repeat: int,
    config: Config,
    formulation: str,
    initial_mode: str,
    missing_holding_policy: str,
    time_limit_s: float | None,
    screen_time_limit_s: float | None,
    save_weights: bool,
    screen_solver: str,
    screen_margin_pct: float,
    screen_feasibility_tol: float,
    clarabel_max_threads: int | None,
    solver_verbose: bool,
    factor_qcqp_settings: Any,
    factor_carry_theta: bool,
    initial_random_seed: int,
    initial_top_n_max: int,
    skip_infeasible_rebalance_dates: bool,
    chained_state_weight_tol: float,
    screen_max_initial_n: int | None,
    factor_diagnostics: bool,
    direct_qp_fallback: str | None,
    factor_fallback_solver: str,
    factor_fallback_clarabel_method: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    previous: pd.Series | None = None
    previous_factor_theta: float | None = None
    for date in dates:
        process_rss_before_mb = process_rss_mb()
        generated_initial_n: int | None = None
        if initial_mode == "independent_top500":
            initial = store.top_n_initial(config.initial_top_n, date)
            generated_initial_n = config.initial_top_n
        elif initial_mode == "independent_random500":
            initial = store.random_n_initial(
                date, config.initial_top_n, initial_random_seed
            )
            generated_initial_n = config.initial_top_n
        elif initial_mode == "independent_topn_random":
            initial, generated_initial_n = store.random_top_range_initial(
                date,
                config.initial_top_n,
                initial_top_n_max,
                initial_random_seed,
            )
        elif initial_mode == "chained":
            initial = (
                store.top_n_initial(config.initial_top_n, date)
                if previous is None
                else previous
            )
            if previous is None:
                generated_initial_n = config.initial_top_n
        elif initial_mode == "chained_random500":
            initial = (
                store.random_n_initial(date, config.initial_top_n, initial_random_seed)
                if previous is None
                else previous
            )
            if previous is None:
                generated_initial_n = config.initial_top_n
        elif initial_mode == "chained_topn_random":
            if previous is None:
                initial, generated_initial_n = store.random_top_range_initial(
                    date,
                    config.initial_top_n,
                    initial_top_n_max,
                    initial_random_seed,
                )
            else:
                initial = previous
        else:
            raise ValueError(f"unsupported initial mode {initial_mode!r}")
        theta_seed = (
            previous_factor_theta
            if initial_mode.startswith("chained") and factor_carry_theta
            else None
        )
        prepare_started = time.perf_counter()
        day = store.prepare_day(date, initial, config, missing_holding_policy)
        prepare_elapsed = time.perf_counter() - prepare_started
        minimum_required_turnover = minimum_turnover_to_asset_bounds(day, config)
        benchmark_turnover = store.benchmark_turnover_l1(date)
        infeasibility_precheck_build_s = 0.0
        infeasibility_precheck_s = 0.0
        infeasibility_precheck_solver_s = 0.0
        infeasibility_precheck_status = "not_run"
        infeasibility_precheck_iters = 0
        infeasibility_precheck_setup_s = 0.0
        skip_reason = ""
        if minimum_required_turnover > config.turnover_limit + 1e-10:
            skip_reason = "asset_active_bounds_require_turnover_above_limit"
        elif (
            skip_infeasible_rebalance_dates
            and initial_mode.startswith("chained")
            and benchmark_turnover > config.turnover_limit + 1e-10
        ):
            precheck_build_started = time.perf_counter()
            precheck_problem = build_problem(day, "lp", config, formulation)
            infeasibility_precheck_build_s = (
                time.perf_counter() - precheck_build_started
            )
            precheck_started = time.perf_counter()
            precheck_problem.problem.solve(
                **solver_kwargs(screen_solver, None, verbose=solver_verbose)
            )
            infeasibility_precheck_s = time.perf_counter() - precheck_started
            infeasibility_precheck_status = str(
                precheck_problem.problem.status or "unknown"
            )
            precheck_stats = precheck_problem.problem.solver_stats
            if precheck_stats is not None:
                infeasibility_precheck_solver_s = float(
                    precheck_stats.solve_time or 0.0
                )
                infeasibility_precheck_setup_s = float(
                    precheck_stats.setup_time or 0.0
                )
                infeasibility_precheck_iters = int(precheck_stats.num_iters or 0)
            if precheck_problem.problem.status in {
                cp.INFEASIBLE,
                cp.INFEASIBLE_INACCURATE,
            }:
                skip_reason = "rebalance_date_common_linear_model_infeasible"
            elif precheck_problem.problem.status not in SUCCESS_STATUSES:
                raise RuntimeError(
                    f"{date:%Y-%m-%d}: infeasibility precheck ended with "
                    f"status {precheck_problem.problem.status}"
                )

        if skip_reason:
            state_reset = initial_mode.startswith("chained")
            print(
                f"[{initial_mode}/{model}/{case.name}] skip {date:%Y-%m-%d}: "
                f"{skip_reason}; benchmark turnover={benchmark_turnover:.6f}, "
                f"box lower bound={minimum_required_turnover:.6f}; "
                f"state reset={'benchmark' if state_reset else 'none'}",
                flush=True,
            )
            rows.append(
                {
                    "model": model,
                    "case": case.name,
                    "solver": case.solver,
                    "initial_mode": initial_mode,
                    "executed_solver": "NONE",
                    "solver_path": "PROVABLY_INFEASIBLE_SKIP",
                    "repeat": repeat,
                    "date": date.strftime("%Y-%m-%d"),
                    "month": date.strftime("%Y-%m"),
                    "status": "skipped_infeasible",
                    "skip_reason": skip_reason,
                    "state_reset": state_reset,
                    "state_reset_to": "daily_benchmark" if state_reset else "",
                    "minimum_required_turnover_l1": minimum_required_turnover,
                    "benchmark_turnover_l1": benchmark_turnover,
                    "infeasibility_precheck_build_s": infeasibility_precheck_build_s,
                    "infeasibility_precheck_s": infeasibility_precheck_s,
                    "infeasibility_precheck_solver_s": infeasibility_precheck_solver_s,
                    "infeasibility_precheck_status": infeasibility_precheck_status,
                    "prepare_s": prepare_elapsed,
                    "model_build_s": infeasibility_precheck_build_s,
                    "solve_total_s": infeasibility_precheck_s,
                    "optimization_total_s": (
                        infeasibility_precheck_build_s + infeasibility_precheck_s
                    ),
                    "end_to_end_s": (
                        prepare_elapsed
                        + infeasibility_precheck_build_s
                        + infeasibility_precheck_s
                    ),
                    "solver_s": infeasibility_precheck_solver_s,
                    "setup_s": infeasibility_precheck_setup_s,
                    "num_iters": infeasibility_precheck_iters,
                    "screen_status": "not_run",
                    "screen_accepted": False,
                    "fallback_used": False,
                    "benchmark_source_date": (
                        "static"
                        if day.benchmark_source_date is None
                        else day.benchmark_source_date.strftime("%Y-%m-%d")
                    ),
                    "alpha_source_date": day.alpha_source_date.strftime("%Y-%m-%d"),
                    "benchmark_stale_days": day.benchmark_stale_days,
                    "alpha_stale_days": day.alpha_stale_days,
                    "alpha_missing_count": day.alpha_missing_count,
                    "alpha_missing_fraction": day.alpha_missing_fraction,
                    "initial_nonzero_count": int(
                        np.count_nonzero(day.initial > config.weight_zero_tol)
                    ),
                    "initial_fingerprint": portfolio_fingerprint(
                        day.sids, day.initial, config.weight_zero_tol
                    ),
                    "generated_initial_n": generated_initial_n,
                }
            )
            # A benchmark reconstitution can make the old portfolio unable to
            # cross into the new active-weight box within the turnover limit.
            # Merely retaining that portfolio would therefore make every later
            # chained date infeasible as well.  Treat the common-infeasible day
            # as a sequence break and use the same deterministic, zero-active
            # benchmark state for every solver case.  The skipped day and this
            # reset are not counted as an optimization result.
            if state_reset:
                previous = pd.Series(day.benchmark, index=day.sids)
                previous = previous[previous > config.weight_zero_tol]
                previous /= previous.sum()
                previous_factor_theta = None
            continue
        screen_build_s = 0.0
        screen_s = 0.0
        screen_solver_s = 0.0
        main_build_s = 0.0
        main_s = 0.0
        main_solver_s = 0.0
        setup_s = infeasibility_precheck_setup_s
        num_iters = infeasibility_precheck_iters
        screen_accepted = False
        screen_status = "not_run"
        screen_failure = ""
        screen_warning_messages: list[str] = []
        main_warning_messages: list[str] = []
        factor_fallback_used = False
        screen_problem: BuiltProblem | None = None
        initial_nonzero_count = int(
            np.count_nonzero(day.initial > config.weight_zero_tol)
        )

        factor_fields: dict[str, Any] = {
            "factor_workspace_setup_s": 0.0,
            "factor_workspace_setup_solver_s": 0.0,
            "factor_search_s": 0.0,
            "factor_final_s": 0.0,
            "factor_update_s": 0.0,
            "factor_solve_wall_s": 0.0,
            "factor_polish_s": 0.0,
            "factor_qp_solves": 0,
            "factor_qp_iters": 0,
            "factor_outer_iters": 0,
            "factor_theta": None,
            # ``None`` tells the solver to use its configured initial theta.
            # Persist the effective value so that a failed first subproblem is
            # still reproducible even though no FactorSolveResult exists yet.
            "factor_theta_seed": (
                factor_qcqp_settings.theta_initial
                if theta_seed is None
                else theta_seed
            ),
            "factor_dual_gap_abs": None,
            "factor_min_risk_pct": None,
            "factor_risk_active": None,
            "factor_linear_violation": None,
            "factor_osqp_status": "not_run",
            "factor_qp_backend": case.factor_qcqp_backend,
            "factor_qp_status": "not_run",
            "factor_piqp_inequality_form": "not_run",
            "factor_message": "",
            "factor_failed_attempt_s": 0.0,
            "factor_failure": "",
            "factor_fallback_used": False,
            "factor_fallback_solver": "",
            "factor_qp_trace_json": "[]",
        }

        screen_skipped_large_support = bool(
            case.lp_screen
            and screen_max_initial_n is not None
            and initial_nonzero_count > screen_max_initial_n
        )
        if screen_skipped_large_support:
            screen_status = "skipped_large_support"
            screen_failure = (
                f"initial_nonzero_count={initial_nonzero_count} exceeds "
                f"screen_max_initial_n={screen_max_initial_n}"
            )
        elif case.lp_screen:
            screen_build_started = time.perf_counter()
            screen_problem = build_problem(
                day, "lp", config, formulation, objective_target=case.objective_target
            )
            screen_build_s = time.perf_counter() - screen_build_started
            screen_started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    screen_problem.problem.solve(
                        **solver_kwargs(
                            screen_solver,
                            screen_time_limit_s,
                            verbose=solver_verbose,
                        )
                    )
                screen_warning_messages = [
                    f"{item.category.__name__}: {item.message}"
                    for item in caught_warnings
                ]
            except Exception as exc:
                screen_failure = f"{type(exc).__name__}: {exc}"
            screen_s = time.perf_counter() - screen_started
            screen_stats = screen_problem.problem.solver_stats
            if screen_stats is not None:
                screen_solver_s = float(screen_stats.solve_time or 0.0)
                setup_s += float(screen_stats.setup_time or 0.0)
                num_iters += int(screen_stats.num_iters or 0)
            screen_status = str(screen_problem.problem.status or "exception")
            if (
                not screen_failure
                and screen_problem.problem.status in SUCCESS_STATUSES
                and screen_problem.x.value is not None
            ):
                screen_result = evaluate(day, screen_problem, "socp", config)
                screen_accepted = (
                    screen_result["tracking_error_pct"]
                    <= config.risk_budget_pct - screen_margin_pct
                    and screen_result["max_violation"] <= screen_feasibility_tol
                )
            elif not screen_failure:
                screen_failure = f"status={screen_status}"

        if screen_accepted:
            built = screen_problem
            executed_solver = screen_solver
            solver_path = f"{screen_solver}_ACCEPT"
            status = built.problem.status
            result = evaluate(day, built, model, config)
            objective_scale = built.objective_scale
            alpha_shift = built.alpha_shift
            omitted_total_active = built.omitted_total_active
            omitted_benchmark_weight = built.omitted_benchmark_weight
            sparse_turnover = built.sparse_turnover
        elif case.factor_penalty_qp:
            try:
                from .factor_qcqp import solve_factor_penalty_qp
            except ImportError:
                from factor_qcqp import solve_factor_penalty_qp

            direct_trace: list[dict[str, Any]] = []
            main_started = time.perf_counter()
            try:
                direct_result = solve_factor_penalty_qp(
                    day,
                    config,
                    factor_qcqp_settings,
                    diagnostics=direct_trace if factor_diagnostics else None,
                )
            except Exception as exc:
                factor_failed_attempt_s = time.perf_counter() - main_started
                factor_failure = f"{type(exc).__name__}: {exc}"
                if direct_qp_fallback is None:
                    raise RuntimeError(
                        f"{model}/{case.name}/{date:%Y-%m-%d} direct PIQP failed"
                    ) from exc
                factor_fallback_used = True
                fallback_build_started = time.perf_counter()
                built = build_problem(
                    day,
                    model,
                    config,
                    formulation,
                    objective_target=case.objective_target,
                )
                main_build_s = time.perf_counter() - fallback_build_started
                fallback_started = time.perf_counter()
                fallback_method = (
                    factor_fallback_clarabel_method
                    if direct_qp_fallback == "CLARABEL"
                    else None
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    built.problem.solve(
                        **solver_kwargs(
                            direct_qp_fallback,
                            time_limit_s,
                            direct_method=fallback_method,
                            clarabel_max_threads=clarabel_max_threads,
                            verbose=solver_verbose,
                        )
                    )
                main_warning_messages.extend(
                    f"{item.category.__name__}: {item.message}"
                    for item in caught_warnings
                )
                fallback_s = time.perf_counter() - fallback_started
                main_s = factor_failed_attempt_s + fallback_s
                if built.problem.status not in SUCCESS_STATUSES or built.x.value is None:
                    raise RuntimeError(
                        f"{model}/{case.name}/{date:%Y-%m-%d} direct PIQP failed "
                        f"and {direct_qp_fallback} fallback ended with status "
                        f"{built.problem.status}"
                    ) from exc
                fallback_stats = built.problem.solver_stats
                main_solver_s = float(fallback_stats.solve_time or 0.0)
                setup_s += float(fallback_stats.setup_time or 0.0)
                num_iters += int(fallback_stats.num_iters or 0)
                executed_solver = direct_qp_fallback
                solver_path = (
                    f"PIQP_DIRECT_FACTOR_PENALTY_QP_FAIL->{direct_qp_fallback}"
                )
                status = built.problem.status
                result = evaluate(day, built, model, config)
                objective_scale = built.objective_scale
                alpha_shift = built.alpha_shift
                omitted_total_active = built.omitted_total_active
                omitted_benchmark_weight = built.omitted_benchmark_weight
                sparse_turnover = built.sparse_turnover
                factor_fields.update(
                    {
                        "factor_failed_attempt_s": factor_failed_attempt_s,
                        "factor_failure": factor_failure,
                        "factor_fallback_used": True,
                        "factor_fallback_solver": direct_qp_fallback,
                        "factor_qp_status": "failed_before_result",
                        "factor_message": "direct penalty-QP failed; fallback accepted",
                        "factor_qp_trace_json": json.dumps(
                            direct_trace, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
            else:
                direct_total_s = time.perf_counter() - main_started
                main_build_s = direct_result.matrix_build_s
                main_s = max(0.0, direct_total_s - main_build_s)
                main_solver_s = direct_result.solver_s
                setup_s += direct_result.workspace_setup_solver_s
                num_iters += direct_result.qp_iters
                executed_solver = "PIQP_DIRECT_FACTOR_PENALTY_QP"
                solver_path = executed_solver
                status = direct_result.status
                result = evaluate_weight(
                    day,
                    direct_result.weight,
                    model,
                    config,
                    direct_result.objective,
                )
                objective_scale = direct_result.alpha_scale
                alpha_shift = direct_result.alpha_shift
                omitted_total_active = direct_result.omitted_total_active
                omitted_benchmark_weight = direct_result.omitted_benchmark_weight
                sparse_turnover = direct_result.sparse_turnover
                factor_fields.update(
                    {
                        "factor_workspace_setup_s": direct_result.workspace_setup_s,
                        "factor_workspace_setup_solver_s": (
                            direct_result.workspace_setup_solver_s
                        ),
                        "factor_final_s": direct_result.solve_wall_s,
                        "factor_solve_wall_s": direct_result.solve_wall_s,
                        "factor_qp_solves": 1,
                        "factor_qp_iters": direct_result.qp_iters,
                        "factor_outer_iters": 1,
                        "factor_linear_violation": direct_result.max_linear_violation,
                        "factor_osqp_status": "not_applicable",
                        "factor_qp_backend": "PIQP_DIRECT",
                        "factor_qp_status": direct_result.qp_status,
                        "factor_piqp_inequality_form": (
                            direct_result.piqp_inequality_form
                        ),
                        "factor_message": "direct fixed factor-penalty QP",
                        "factor_qp_trace_json": json.dumps(
                            direct_result.qp_trace,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
        elif case.factor_qcqp:
            try:
                from .factor_qcqp import solve_factor_qcqp
            except ImportError:
                from factor_qcqp import solve_factor_qcqp

            factor_trace: list[dict[str, Any]] = []
            main_started = time.perf_counter()
            try:
                factor_result = solve_factor_qcqp(
                    day,
                    config,
                    replace(
                        factor_qcqp_settings,
                        backend=case.factor_qcqp_backend or factor_qcqp_settings.backend,
                    ),
                    theta_seed=theta_seed,
                    linear_warm_start=(
                        np.asarray(screen_problem.x.value, dtype=float).reshape(-1)
                        if case.lp_screen
                        and screen_problem is not None
                        and screen_problem.x.value is not None
                        else None
                    ),
                    diagnostics=factor_trace if factor_diagnostics else None,
                )
            except Exception as exc:
                factor_failed_attempt_s = time.perf_counter() - main_started
                factor_fallback_used = True
                factor_failure = f"{type(exc).__name__}: {exc}"
                fallback_build_started = time.perf_counter()
                built = build_problem(
                    day,
                    model,
                    config,
                    formulation,
                    objective_target=case.objective_target,
                )
                main_build_s = time.perf_counter() - fallback_build_started
                fallback_started = time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        built.problem.solve(
                            **solver_kwargs(
                                factor_fallback_solver,
                                time_limit_s,
                                direct_method=(
                                    factor_fallback_clarabel_method
                                    if factor_fallback_solver == "CLARABEL"
                                    else None
                                ),
                                clarabel_max_threads=clarabel_max_threads,
                                verbose=solver_verbose,
                            )
                        )
                    main_warning_messages.extend(
                        f"{item.category.__name__}: {item.message}"
                        for item in caught_warnings
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"{model}/{case.name}/{date:%Y-%m-%d} factor-QCQP and "
                        f"{factor_fallback_solver} fallback both failed"
                    ) from fallback_exc
                fallback_s = time.perf_counter() - fallback_started
                main_s = factor_failed_attempt_s + fallback_s
                if built.problem.status not in SUCCESS_STATUSES or built.x.value is None:
                    raise RuntimeError(
                        f"{model}/{case.name}/{date:%Y-%m-%d} factor-QCQP failed "
                        f"and {factor_fallback_solver} fallback ended with status "
                        f"{built.problem.status}"
                    ) from exc
                fallback_stats = built.problem.solver_stats
                main_solver_s = float(fallback_stats.solve_time or 0.0)
                setup_s += float(fallback_stats.setup_time or 0.0)
                num_iters += int(fallback_stats.num_iters or 0)
                executed_solver = factor_fallback_solver
                prefix = ""
                if case.lp_screen:
                    prefix = (
                        f"{screen_solver}_SKIP_LARGE_SUPPORT->"
                        if screen_skipped_large_support
                        else f"{screen_solver}->"
                    )
                fallback_label = factor_fallback_solver
                if factor_fallback_solver == "CLARABEL":
                    fallback_label += f"_{factor_fallback_clarabel_method.upper()}"
                solver_path = f"{prefix}PIQP_FACTOR_QP_FAIL->{fallback_label}"
                status = built.problem.status
                result = evaluate(day, built, model, config)
                objective_scale = built.objective_scale
                alpha_shift = built.alpha_shift
                omitted_total_active = built.omitted_total_active
                omitted_benchmark_weight = built.omitted_benchmark_weight
                sparse_turnover = built.sparse_turnover
                previous_factor_theta = None
                factor_fields.update(
                    {
                        "factor_failed_attempt_s": factor_failed_attempt_s,
                        "factor_failure": factor_failure,
                        "factor_fallback_used": True,
                        "factor_fallback_solver": fallback_label,
                        "factor_qp_status": "failed_before_result",
                        "factor_message": (
                            f"factor-QP failed; {factor_fallback_solver} fallback accepted"
                        ),
                        "factor_qp_trace_json": json.dumps(
                            factor_trace, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
            else:
                factor_total_s = time.perf_counter() - main_started
                main_build_s = factor_result.matrix_build_s
                main_s = max(0.0, factor_total_s - main_build_s)
                main_solver_s = factor_result.solver_s
                num_iters += factor_result.qp_iters
                factor_backend = factor_result.qp_backend
                executed_solver = f"{factor_backend}_FACTOR_QP"
                if case.lp_screen:
                    screen_path = (
                        f"{screen_solver}_SKIP_LARGE_SUPPORT"
                        if screen_skipped_large_support
                        else screen_solver
                    )
                    solver_path = f"{screen_path}->{factor_backend}_FACTOR_QP"
                else:
                    solver_path = f"{factor_backend}_FACTOR_QP"
                status = factor_result.status
                result = evaluate_weight(
                    day,
                    factor_result.weight,
                    model,
                    config,
                    factor_result.objective,
                )
                objective_scale = factor_result.alpha_scale
                alpha_shift = factor_result.alpha_shift
                omitted_total_active = factor_result.omitted_total_active
                omitted_benchmark_weight = factor_result.omitted_benchmark_weight
                sparse_turnover = factor_result.sparse_turnover
                previous_factor_theta = factor_result.theta
                factor_fields.update(
                    {
                        "factor_workspace_setup_s": factor_result.workspace_setup_s,
                        "factor_workspace_setup_solver_s": factor_result.workspace_setup_solver_s,
                        "factor_search_s": factor_result.search_s,
                        "factor_final_s": factor_result.final_s,
                        "factor_update_s": factor_result.update_s,
                        "factor_solve_wall_s": factor_result.solve_wall_s,
                        "factor_polish_s": factor_result.polish_s,
                        "factor_qp_solves": factor_result.qp_solves,
                        "factor_qp_iters": factor_result.qp_iters,
                        "factor_outer_iters": factor_result.outer_iters,
                        "factor_theta": factor_result.theta,
                        "factor_theta_seed": factor_result.theta_seed,
                        "factor_dual_gap_abs": factor_result.dual_gap_abs,
                        "factor_min_risk_pct": factor_result.min_risk_pct,
                        "factor_risk_active": factor_result.risk_active,
                        "factor_linear_violation": factor_result.max_linear_violation,
                        "factor_osqp_status": (
                            factor_result.qp_status
                            if factor_result.qp_backend == "OSQP"
                            else "not_applicable"
                        ),
                        "factor_qp_backend": factor_result.qp_backend,
                        "factor_qp_status": factor_result.qp_status,
                        "factor_piqp_inequality_form": (
                            factor_result.piqp_inequality_form
                        ),
                        "factor_message": factor_result.message,
                        "factor_qp_trace_json": json.dumps(
                            factor_result.qp_trace,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
        else:
            main_build_started = time.perf_counter()
            built = build_problem(
                day, model, config, formulation, objective_target=case.objective_target
            )
            main_build_s = time.perf_counter() - main_build_started
            main_started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    built.problem.solve(
                        **solver_kwargs(
                            case.solver,
                            time_limit_s,
                            direct_method=case.direct_method,
                            clarabel_max_threads=(
                                case.max_threads
                                if case.max_threads is not None
                                else clarabel_max_threads
                            ),
                            verbose=solver_verbose,
                        )
                    )
                main_warning_messages.extend(
                    f"{item.category.__name__}: {item.message}"
                    for item in caught_warnings
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{model}/{case.name}/{date:%Y-%m-%d} solver call failed"
                ) from exc
            main_s = time.perf_counter() - main_started
            if built.problem.status not in SUCCESS_STATUSES or built.x.value is None:
                raise RuntimeError(
                    f"{model}/{case.name}/{date:%Y-%m-%d} ended with "
                    f"status {built.problem.status}"
                )
            main_stats = built.problem.solver_stats
            main_solver_s = float(main_stats.solve_time or 0.0)
            setup_s += float(main_stats.setup_time or 0.0)
            num_iters += int(main_stats.num_iters or 0)
            executed_solver = case.solver
            if case.lp_screen:
                screen_path = (
                    f"{screen_solver}_SKIP_LARGE_SUPPORT"
                    if screen_skipped_large_support
                    else screen_solver
                )
                solver_path = f"{screen_path}->{case.solver}"
            else:
                solver_path = case.solver
            status = built.problem.status
            result = evaluate(day, built, model, config)
            objective_scale = built.objective_scale
            alpha_shift = built.alpha_shift
            omitted_total_active = built.omitted_total_active
            omitted_benchmark_weight = built.omitted_benchmark_weight
            sparse_turnover = built.sparse_turnover

        build_elapsed = (
            infeasibility_precheck_build_s + screen_build_s + main_build_s
        )
        solve_elapsed = infeasibility_precheck_s + screen_s + main_s
        weight = result.pop("weight")
        optimization_total_s = build_elapsed + solve_elapsed
        end_to_end_s = prepare_elapsed + optimization_total_s
        process_rss_after_mb = process_rss_mb()
        row = {
            "model": model,
            "case": case.name,
            "solver": case.solver,
            "initial_mode": initial_mode,
            "executed_solver": executed_solver,
            "solver_path": solver_path,
            "repeat": repeat,
            "date": date.strftime("%Y-%m-%d"),
            "month": date.strftime("%Y-%m"),
            "status": status,
            "n_assets": len(day.sids),
            "n_factors": len(day.factors),
            "prepare_s": prepare_elapsed,
            "model_build_s": build_elapsed,
            "solve_total_s": solve_elapsed,
            "optimization_total_s": optimization_total_s,
            "end_to_end_s": end_to_end_s,
            "process_rss_before_mb": process_rss_before_mb,
            "process_rss_after_mb": process_rss_after_mb,
            "process_rss_delta_mb": (
                None
                if process_rss_before_mb is None or process_rss_after_mb is None
                else process_rss_after_mb - process_rss_before_mb
            ),
            "solver_s": (
                infeasibility_precheck_solver_s + screen_solver_s + main_solver_s
            ),
            "setup_s": setup_s,
            "num_iters": num_iters,
            "infeasibility_precheck_build_s": infeasibility_precheck_build_s,
            "infeasibility_precheck_s": infeasibility_precheck_s,
            "infeasibility_precheck_solver_s": infeasibility_precheck_solver_s,
            "infeasibility_precheck_status": infeasibility_precheck_status,
            "screen_build_s": screen_build_s,
            "screen_s": screen_s,
            "screen_solver_s": screen_solver_s,
            "screen_accepted": screen_accepted,
            "fallback_used": bool(
                (case.lp_screen and not screen_accepted) or factor_fallback_used
            ),
            "screen_status": screen_status,
            "screen_failure": screen_failure,
            "screen_warning_count": len(screen_warning_messages),
            "screen_warnings": " | ".join(screen_warning_messages),
            "screen_time_limit_s": screen_time_limit_s,
            "screen_max_initial_n": screen_max_initial_n,
            "screen_skipped_large_support": screen_skipped_large_support,
            "main_build_s": main_build_s,
            "main_s": main_s,
            "main_solver_s": main_solver_s,
            "main_warning_count": len(main_warning_messages),
            "main_warnings": " | ".join(main_warning_messages),
            "objective_target": case.objective_target,
            "objective_scale": objective_scale,
            "alpha_shift": alpha_shift,
            "clarabel_direct_method": case.direct_method,
            "clarabel_max_threads": (
                case.max_threads
                if case.max_threads is not None
                else clarabel_max_threads
            ),
            "initial_missing_mass": day.initial_missing_mass,
            "initial_renormalization_factor": day.initial_renormalization_factor,
            "benchmark_missing_fraction": day.benchmark_missing_fraction,
            "benchmark_source_date": (
                "static"
                if day.benchmark_source_date is None
                else day.benchmark_source_date.strftime("%Y-%m-%d")
            ),
            "alpha_source_date": day.alpha_source_date.strftime("%Y-%m-%d"),
            "benchmark_stale_days": day.benchmark_stale_days,
            "alpha_stale_days": day.alpha_stale_days,
            "alpha_missing_count": day.alpha_missing_count,
            "alpha_missing_fraction": day.alpha_missing_fraction,
            "initial_nonzero_count": initial_nonzero_count,
            "initial_fingerprint": portfolio_fingerprint(
                day.sids, day.initial, config.weight_zero_tol
            ),
            "generated_initial_n": generated_initial_n,
            "skip_reason": "",
            "state_reset": False,
            "state_reset_to": "",
            "minimum_required_turnover_l1": minimum_required_turnover,
            "benchmark_turnover_l1": benchmark_turnover,
            "x0_active_l1": float(np.abs(day.initial - day.benchmark).sum()),
            "x0_benchmark_weight": float(day.initial[day.benchmark > 0].sum()),
            "omitted_total_active": omitted_total_active,
            "omitted_benchmark_weight": omitted_benchmark_weight,
            "sparse_turnover": sparse_turnover,
            **factor_fields,
            **result,
        }
        if initial_mode.startswith("chained"):
            previous, state_fields = chained_state_from_weight(
                day.sids, weight, chained_state_weight_tol
            )
            row.update(state_fields)
            row["chained_state_weight_tol"] = chained_state_weight_tol
        rows.append(row)
        if save_weights:
            weight_frames.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "case": case.name,
                        "solver": case.solver,
                        "initial_mode": initial_mode,
                        "repeat": repeat,
                        "date": date.strftime("%Y-%m-%d"),
                        "sid": day.sids,
                        "weight": weight,
                    }
                )
            )
    return rows, weight_frames


def parse_csv_list(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def parse_initial_modes(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def package_versions() -> dict[str, str | None]:
    names = [
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "cvxpy",
        "clarabel",
        "ecos",
        "highspy",
        "Mosek",
        "osqp",
        "piqp",
        "qoco",
        "scs",
    ]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def summarize_skip_consistency(runs: pd.DataFrame) -> list[dict[str, Any]]:
    if runs.empty or "skip_reason" not in runs.columns:
        return []
    records: list[dict[str, Any]] = []
    for (initial_mode, model, repeat), group in runs.groupby(
        ["initial_mode", "model", "repeat"], sort=True
    ):
        case_skip_dates = {
            str(case): sorted(
                case_group.loc[
                    case_group["status"] == "skipped_infeasible", "date"
                ].astype(str)
            )
            for case, case_group in group.groupby("case", sort=True)
        }
        signatures = {tuple(dates) for dates in case_skip_dates.values()}
        records.append(
            {
                "initial_mode": initial_mode,
                "model": model,
                "repeat": int(repeat),
                "consistent": len(signatures) <= 1,
                "case_skip_dates": case_skip_dates,
            }
        )
    return records


def summarize(runs: pd.DataFrame, *, by_month: bool = False) -> pd.DataFrame:
    successful = runs[runs["status"].isin(SUCCESS_STATUSES)].copy()
    if successful.empty:
        return pd.DataFrame()
    group_columns = ["initial_mode", "model", "case", "solver"]
    if by_month:
        group_columns.insert(1, "month")
    summary = (
        successful.groupby(group_columns, as_index=False)
        .agg(
            solves=("date", "count"),
            optimal=("status", lambda values: int((values == cp.OPTIMAL).sum())),
            optimal_inaccurate=(
                "status",
                lambda values: int((values == cp.OPTIMAL_INACCURATE).sum()),
            ),
            solve_median_s=("solve_total_s", "median"),
            solve_mean_s=("solve_total_s", "mean"),
            solve_min_s=("solve_total_s", "min"),
            solve_max_s=("solve_total_s", "max"),
            optimization_median_s=("optimization_total_s", "median"),
            optimization_mean_s=("optimization_total_s", "mean"),
            end_to_end_median_s=("end_to_end_s", "median"),
            end_to_end_mean_s=("end_to_end_s", "mean"),
            solver_median_s=("solver_s", "median"),
            screen_median_s=("screen_s", "median"),
            screen_mean_s=("screen_s", "mean"),
            screen_max_s=("screen_s", "max"),
            main_median_s=("main_s", "median"),
            main_mean_s=("main_s", "mean"),
            factor_workspace_setup_mean_s=("factor_workspace_setup_s", "mean"),
            factor_search_mean_s=("factor_search_s", "mean"),
            factor_final_mean_s=("factor_final_s", "mean"),
            factor_qp_solves_mean=("factor_qp_solves", "mean"),
            factor_qp_iters_mean=("factor_qp_iters", "mean"),
            factor_dual_gap_max=("factor_dual_gap_abs", "max"),
            screen_accepts=("screen_accepted", "sum"),
            fallbacks=("fallback_used", "sum"),
            objective_scale_median=("objective_scale", "median"),
            max_violation=("max_violation", "max"),
            max_turnover_violation=("violation_turnover", "max"),
            max_risk_violation=("violation_risk", "max"),
            te_min_pct=("tracking_error_pct", "min"),
            te_median_pct=("tracking_error_pct", "median"),
            te_max_pct=("tracking_error_pct", "max"),
        )
        .sort_values(
            ["initial_mode"] + (["month"] if by_month else []) + ["model", "solve_median_s"],
            kind="mergesort",
        )
    )
    # Keep the original column for compatibility, but expose both projections.
    # Mean-based projection is the meaningful one for mixed screen/fallback cases.
    summary["projected_2500_min"] = summary["solve_median_s"] * 2500.0 / 60.0
    summary["projected_2500_median_min"] = (
        summary["solve_median_s"] * 2500.0 / 60.0
    )
    summary["projected_2500_mean_min"] = summary["solve_mean_s"] * 2500.0 / 60.0
    summary["projected_2500_optimization_mean_min"] = (
        summary["optimization_mean_s"] * 2500.0 / 60.0
    )
    summary["projected_2500_end_to_end_mean_min"] = (
        summary["end_to_end_mean_s"] * 2500.0 / 60.0
    )
    return summary


def summarize_screens(runs: pd.DataFrame) -> pd.DataFrame:
    screened = runs[runs["screen_status"] != "not_run"].copy()
    if screened.empty:
        return pd.DataFrame()
    screened["screen_outcome"] = np.where(
        screened["screen_accepted"], "accepted", "fallback"
    )
    return (
        screened.groupby(
            ["initial_mode", "month", "model", "case", "screen_outcome"],
            as_index=False,
        )
        .agg(
            dates=("date", "count"),
            screen_mean_s=("screen_s", "mean"),
            screen_median_s=("screen_s", "median"),
            screen_max_s=("screen_s", "max"),
            main_mean_s=("main_s", "mean"),
            optimization_mean_s=("optimization_total_s", "mean"),
        )
        .sort_values(
            ["initial_mode", "month", "model", "case", "screen_outcome"],
            kind="mergesort",
        )
    )


def compare_weights(weights: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Compare each case with a same-model and same-initial-mode reference."""
    references = {"lp": "HIGHS", "qp": "MOSEK", "socp": "MOSEK"}
    rows: list[dict[str, Any]] = []
    for (initial_mode, model, repeat, date), group in weights.groupby(
        ["initial_mode", "model", "repeat", "date"], sort=False
    ):
        cases = group["case"].drop_duplicates().tolist()
        reference = references.get(model)
        if reference not in cases:
            reference = "CLARABEL" if "CLARABEL" in cases else cases[0]
        reference_weights = (
            group[group["case"] == reference].set_index("sid")["weight"]
        )
        reference_run = runs[
            (runs["initial_mode"] == initial_mode)
            & (runs["model"] == model)
            & (runs["case"] == reference)
            & (runs["repeat"] == repeat)
            & (runs["date"] == date)
        ].iloc[0]
        for case in cases:
            candidate = group[group["case"] == case].set_index("sid")["weight"]
            aligned = pd.concat(
                [reference_weights.rename("reference"), candidate.rename("candidate")],
                axis=1,
            ).fillna(0.0)
            delta = aligned["candidate"] - aligned["reference"]
            candidate_run = runs[
                (runs["initial_mode"] == initial_mode)
                & (runs["model"] == model)
                & (runs["case"] == case)
                & (runs["repeat"] == repeat)
                & (runs["date"] == date)
            ].iloc[0]
            rows.append(
                {
                    "initial_mode": initial_mode,
                    "model": model,
                    "reference_case": reference,
                    "candidate_case": case,
                    "repeat": repeat,
                    "date": date,
                    "weight_l1_diff": float(np.abs(delta).sum()),
                    "weight_max_abs_diff": float(np.abs(delta).max()),
                    "raw_objective_diff": float(
                        candidate_run["raw_objective"]
                        - reference_run["raw_objective"]
                    ),
                    "alpha_value_diff": float(
                        candidate_run["alpha_value"] - reference_run["alpha_value"]
                    ),
                    "tracking_error_diff_pct": float(
                        candidate_run["tracking_error_pct"]
                        - reference_run["tracking_error_pct"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def cpu_model_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or None


def clarabel_build_info() -> str | None:
    try:
        import clarabel

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = clarabel.buildinfo()
        text = output.getvalue().strip()
        return text or (str(result) if result is not None else None)
    except (ImportError, AttributeError):
        return None


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-model-xlsx", type=Path, required=True)
    ap.add_argument("--benchmark-csv", type=Path, required=True)
    ap.add_argument(
        "--alpha-csv",
        type=Path,
        help="optional dated alpha CSV with dt/date,sid,fv; missing dates use prior data",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    ap.add_argument(
        "--data-cache-dir",
        type=Path,
        help=(
            "optional shared cache for parsed Excel/CSV inputs; the key includes "
            "absolute paths, sizes and mtimes"
        ),
    )
    ap.add_argument("--models", type=parse_csv_list, default=parse_csv_list("LP,QP,SOCP"))
    ap.add_argument(
        "--lp-solvers", "--lp-cases", dest="lp_cases", type=parse_csv_list,
        default=parse_csv_list("HIGHS,MOSEK,CLARABEL"),
    )
    ap.add_argument(
        "--qp-solvers", "--qp-cases", dest="qp_cases", type=parse_csv_list,
        default=parse_csv_list("HIGHS,CLARABEL,MOSEK"),
    )
    ap.add_argument(
        "--socp-solvers", "--socp-cases", dest="socp_cases", type=parse_csv_list,
        default=parse_csv_list("MOSEK,CLARABEL,ECOS"),
    )
    ap.add_argument("--formulation", choices=["canonical", "optimized"], default="optimized")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-dates", type=int)
    ap.add_argument("--initial-top-n", type=int, default=500)
    ap.add_argument(
        "--initial-top-n-max",
        type=int,
        default=600,
        help=(
            "inclusive upper bound for *_topn_random modes; --initial-top-n "
            "is the lower bound"
        ),
    )
    ap.add_argument(
        "--initial-random-seed",
        type=int,
        default=20260826,
        help="base seed for deterministic date-specific initial portfolios",
    )
    ap.add_argument(
        "--initial-modes",
        type=parse_initial_modes,
        default=parse_initial_modes("chained"),
        help=(
            "comma-separated: chained,independent_top500,"
            "chained_random500,independent_random500,"
            "chained_topn_random,independent_topn_random"
        ),
    )
    ap.add_argument("--time-limit-s", type=float)
    ap.add_argument(
        "--active-ub",
        type=float,
        default=0.004,
        help="per-asset absolute active-weight upper bound (default: 0.004)",
    )
    ap.add_argument(
        "--style-bound",
        type=float,
        default=0.6,
        help="symmetric active-exposure bound for every style factor, including SIZE",
    )
    ap.add_argument(
        "--industry-bound",
        type=float,
        default=0.05,
        help="symmetric active-exposure bound for every industry",
    )
    ap.add_argument(
        "--risk-budget-pct",
        type=float,
        default=6.0,
        help="annualized tracking-error budget in percent units (default: 6.0)",
    )
    ap.add_argument(
        "--common-risk-aversion",
        type=float,
        default=0.75,
        help="QP penalty coefficient for common/factor variance (default: 0.75)",
    )
    ap.add_argument(
        "--specific-risk-aversion",
        type=float,
        default=0.75,
        help="QP penalty coefficient for specific variance (default: 0.75)",
    )
    ap.add_argument(
        "--alpha-target", type=float, default=0.2,
        help="max absolute centered alpha coefficient for *_SCALED cases",
    )
    ap.add_argument("--screen-solver", default="HIGHS", type=str.upper)
    ap.add_argument(
        "--screen-time-limit-s",
        type=float,
        help=(
            "LP solver time limit; timeout/non-optimal status safely falls back "
            "(some HiGHS versions may exceed it in wall time)"
        ),
    )
    ap.add_argument(
        "--screen-max-initial-n",
        type=int,
        help=(
            "skip LP screening and go directly to the main solver when the "
            "initial portfolio has more than this many nonzero names"
        ),
    )
    ap.add_argument(
        "--chained-state-weight-tol",
        type=float,
        default=1e-12,
        help=(
            "drop numerical weights below this threshold before carrying "
            "a chained solution to the next date (default preserves legacy 1e-12)"
        ),
    )
    ap.add_argument(
        "--screen-margin-pct", type=float, default=1e-6,
        help="LP screen is accepted only below risk budget minus this margin",
    )
    ap.add_argument(
        "--screen-feasibility-tol", type=float, default=1e-7,
        help="maximum independently recomputed violation for accepting LP screen",
    )
    ap.add_argument(
        "--clarabel-max-threads", type=int,
        help="optional Clarabel max_threads setting; 0 or unset uses automatic",
    )
    ap.add_argument("--factor-objective-gap-abs", type=float, default=1e-4)
    ap.add_argument("--factor-risk-margin-pct", type=float, default=1e-5)
    ap.add_argument("--factor-intermediate-eps", type=float, default=1e-5)
    ap.add_argument("--factor-final-eps", type=float, default=1e-8)
    ap.add_argument(
        "--factor-piqp-max-iter",
        type=int,
        default=1_000,
        help="per-subproblem PIQP iteration limit (default: 1000)",
    )
    ap.add_argument(
        "--factor-piqp-inequality-form",
        choices=["auto", "one_sided", "compact"],
        default="auto",
        help=(
            "PIQP inequality representation; auto selects compact for the "
            "required PIQP 0.6.4+ runtime; one_sided is diagnostic only"
        ),
    )
    ap.add_argument("--factor-max-outer-iters", type=int, default=30)
    ap.add_argument("--factor-theta-initial", type=float, default=65_536.0)
    ap.add_argument("--factor-theta-growth", type=float, default=4.0)
    ap.add_argument(
        "--factor-carry-theta",
        action="store_true",
        help="reuse the previous date theta in chained mode; fixed cold theta is the default",
    )
    ap.add_argument(
        "--factor-diagnostics",
        action="store_true",
        help=(
            "record every direct QP theta/status/iteration/timing diagnostic as "
            "factor_qp_trace_json; intended for reliability audits"
        ),
    )
    ap.add_argument(
        "--direct-qp-fallback",
        type=str.upper,
        choices=["MOSEK", "CLARABEL"],
        help=(
            "optional same-day fallback for FACTOR_PENALTY_QP_PIQP so one "
            "direct PIQP failure does not discard the remaining sequence"
        ),
    )
    ap.add_argument(
        "--factor-fallback-solver",
        type=str.upper,
        choices=["CLARABEL", "MOSEK"],
        default="CLARABEL",
        help="same-day fallback when a factor-QP frontier subproblem fails",
    )
    ap.add_argument(
        "--factor-fallback-clarabel-method",
        choices=["qdldl", "faer", "mkl"],
        default="qdldl",
        help="Clarabel direct linear solver used only by factor-QP fallback",
    )
    ap.add_argument(
        "--solver-verbose", action="store_true",
        help="enable solver diagnostic output",
    )
    ap.add_argument("--missing-holding-policy", choices=["error", "renormalize"], default="error")
    ap.add_argument("--mosek-license", type=Path)
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument(
        "--require-consistent-skips",
        action="store_true",
        help="return a failure if solver cases skip different infeasible dates",
    )
    ap.add_argument(
        "--skip-infeasible-rebalance-dates",
        action="store_true",
        help=(
            "on chained dates where benchmark L1 turnover exceeds the portfolio "
            "limit, use an exact LP precheck and skip only a proven infeasible date"
        ),
    )
    ap.add_argument("--fail-fast", action="store_true")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.alpha_target <= 0.0:
        raise ValueError("--alpha-target must be positive")
    if args.screen_margin_pct < 0.0:
        raise ValueError("--screen-margin-pct must be nonnegative")
    if args.screen_feasibility_tol < 0.0:
        raise ValueError("--screen-feasibility-tol must be nonnegative")
    supported_initial_modes = {
        "chained",
        "independent_top500",
        "chained_random500",
        "independent_random500",
        "chained_topn_random",
        "independent_topn_random",
    }
    invalid_initial_modes = sorted(set(args.initial_modes) - supported_initial_modes)
    if invalid_initial_modes:
        raise ValueError(f"unsupported initial modes: {invalid_initial_modes}")
    if not args.initial_modes:
        raise ValueError("--initial-modes must not be empty")
    if args.initial_top_n <= 0 or args.initial_top_n_max < args.initial_top_n:
        raise ValueError(
            "initial top-N range must satisfy 0 < --initial-top-n <= "
            "--initial-top-n-max"
        )
    if args.screen_time_limit_s is not None and args.screen_time_limit_s <= 0.0:
        raise ValueError("--screen-time-limit-s must be positive")
    if args.screen_max_initial_n is not None and args.screen_max_initial_n <= 0:
        raise ValueError("--screen-max-initial-n must be positive")
    if args.chained_state_weight_tol < 0.0:
        raise ValueError("--chained-state-weight-tol must be nonnegative")
    if args.factor_objective_gap_abs <= 0.0:
        raise ValueError("--factor-objective-gap-abs must be positive")
    if args.factor_risk_margin_pct < 0.0:
        raise ValueError("--factor-risk-margin-pct must be nonnegative")
    if args.factor_intermediate_eps <= 0.0 or args.factor_final_eps <= 0.0:
        raise ValueError("factor QP tolerances must be positive")
    if args.factor_piqp_max_iter <= 0:
        raise ValueError("--factor-piqp-max-iter must be positive")
    if args.factor_max_outer_iters < 2:
        raise ValueError("--factor-max-outer-iters must be at least 2")
    if args.factor_theta_initial <= 0.0 or args.factor_theta_growth <= 1.0:
        raise ValueError("factor theta initial/growth settings are invalid")
    if args.active_ub <= 0.0:
        raise ValueError("--active-ub must be positive")
    if args.style_bound <= 0.0 or args.industry_bound <= 0.0:
        raise ValueError("--style-bound and --industry-bound must be positive")
    if args.risk_budget_pct <= 0.0:
        raise ValueError("--risk-budget-pct must be positive")
    if args.common_risk_aversion < 0.0 or args.specific_risk_aversion < 0.0:
        raise ValueError("QP risk-aversion coefficients must be nonnegative")
    if args.clarabel_max_threads is not None and args.clarabel_max_threads < 0:
        raise ValueError("--clarabel-max-threads must be nonnegative when supplied")
    if args.mosek_license is not None:
        os.environ["MOSEKLM_LICENSE_FILE"] = str(args.mosek_license.resolve())
    config = Config(
        initial_top_n=args.initial_top_n,
        active_ub=args.active_ub,
        style_default_lb=-args.style_bound,
        style_default_ub=args.style_bound,
        size_lb=-args.style_bound,
        size_ub=args.style_bound,
        industry_lb=-args.industry_bound,
        industry_ub=args.industry_bound,
        risk_budget_pct=args.risk_budget_pct,
        common_risk_aversion=args.common_risk_aversion,
        specific_risk_aversion=args.specific_risk_aversion,
    )
    try:
        from .factor_qcqp import (
            FactorQCQPSettings,
            resolve_piqp_inequality_form,
        )
    except ImportError:
        from factor_qcqp import FactorQCQPSettings, resolve_piqp_inequality_form
    factor_qcqp_settings = FactorQCQPSettings(
        alpha_target=args.alpha_target,
        objective_gap_abs=args.factor_objective_gap_abs,
        risk_margin_pct=args.factor_risk_margin_pct,
        intermediate_eps=args.factor_intermediate_eps,
        final_eps=args.factor_final_eps,
        piqp_max_iter=args.factor_piqp_max_iter,
        piqp_inequality_form=args.factor_piqp_inequality_form,
        max_outer_iters=args.factor_max_outer_iters,
        theta_initial=args.factor_theta_initial,
        theta_growth=args.factor_theta_growth,
        verbose=args.solver_verbose,
        time_limit_s=args.time_limit_s,
    )
    factor_piqp_inequality_form_resolved = resolve_piqp_inequality_form(
        args.factor_piqp_inequality_form
    )
    store = DataStore(
        args.risk_model_xlsx,
        args.benchmark_csv,
        args.alpha_csv,
        cache_dir=args.data_cache_dir,
    )
    dates = store.dates[: args.max_dates]
    if not dates:
        raise ValueError("no common dates found")
    case_map = {
        "LP": args.lp_cases,
        "QP": args.qp_cases,
        "SOCP": args.socp_cases,
    }
    installed = set(cp.installed_solvers())
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    started_epoch_s = time.time()
    started = time.perf_counter()
    for model_upper in args.models:
        if model_upper not in case_map:
            raise ValueError(f"unsupported model {model_upper!r}; choose LP, QP, SOCP")
        model = model_upper.lower()
        for case_name in case_map[model_upper]:
            case = resolve_case(model, case_name, args.alpha_target)
            direct_piqp = case.factor_qcqp or case.factor_penalty_qp
            required_solvers = set() if direct_piqp else {case.solver}
            if case.factor_qcqp:
                # Factor-QP is the fast path; the configured conic solver is its
                # numerical safety net.
                required_solvers.add(args.factor_fallback_solver)
            if case.factor_penalty_qp and args.direct_qp_fallback is not None:
                required_solvers.add(args.direct_qp_fallback)
            if case.factor_qcqp and case.factor_qcqp_backend == "OSQP":
                required_solvers.add("OSQP")
            if (
                case.factor_penalty_qp
                or (
                    case.factor_qcqp
                    and case.factor_qcqp_backend == "PIQP"
                )
            ):
                try:
                    import piqp  # noqa: F401
                except ImportError:
                    errors.append(
                        {
                            "model": model,
                            "case": case.name,
                            "solver": case.solver,
                            "repeat": None,
                            "error": "not installed: ['PIQP Python package']",
                        }
                    )
                    continue
            if case.lp_screen:
                required_solvers.add(args.screen_solver)
            if args.skip_infeasible_rebalance_dates:
                required_solvers.add(args.screen_solver)
            missing_solvers = sorted(required_solvers - installed)
            if missing_solvers:
                errors.append(
                    {
                        "model": model,
                        "case": case.name,
                        "solver": case.solver,
                        "repeat": None,
                        "error": f"not installed: {missing_solvers}",
                    }
                )
                continue
            for initial_mode in args.initial_modes:
                for repeat in range(args.repeats):
                    print(
                        f"[{initial_mode}/{model}/{case.name}] "
                        f"repeat {repeat + 1}/{args.repeats}",
                        flush=True,
                    )
                    try:
                        case_rows, case_weights = solve_chain(
                            store=store,
                            dates=dates,
                            model=model,
                            case=case,
                            repeat=repeat,
                            config=config,
                            formulation=args.formulation,
                            initial_mode=initial_mode,
                            missing_holding_policy=args.missing_holding_policy,
                            time_limit_s=args.time_limit_s,
                            screen_time_limit_s=args.screen_time_limit_s,
                            save_weights=args.save_weights,
                            screen_solver=args.screen_solver,
                            screen_margin_pct=args.screen_margin_pct,
                            screen_feasibility_tol=args.screen_feasibility_tol,
                            clarabel_max_threads=args.clarabel_max_threads,
                            solver_verbose=args.solver_verbose,
                            factor_qcqp_settings=factor_qcqp_settings,
                            factor_carry_theta=args.factor_carry_theta,
                            initial_random_seed=args.initial_random_seed,
                            initial_top_n_max=args.initial_top_n_max,
                            skip_infeasible_rebalance_dates=(
                                args.skip_infeasible_rebalance_dates
                            ),
                            chained_state_weight_tol=args.chained_state_weight_tol,
                            screen_max_initial_n=args.screen_max_initial_n,
                            factor_diagnostics=args.factor_diagnostics,
                            direct_qp_fallback=args.direct_qp_fallback,
                            factor_fallback_solver=args.factor_fallback_solver,
                            factor_fallback_clarabel_method=(
                                args.factor_fallback_clarabel_method
                            ),
                        )
                        rows.extend(case_rows)
                        weight_frames.extend(case_weights)
                    except Exception as exc:
                        error = {
                            "initial_mode": initial_mode,
                            "model": model,
                            "case": case.name,
                            "solver": case.solver,
                            "repeat": repeat,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        errors.append(error)
                        print(error["traceback"], file=sys.stderr, flush=True)
                        if args.fail_fast:
                            raise

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    skip_consistency = summarize_skip_consistency(runs)
    inconsistent_skips = [record for record in skip_consistency if not record["consistent"]]
    if inconsistent_skips:
        errors.append(
            {
                "error": "inconsistent infeasible-date skips across solver cases",
                "details": inconsistent_skips,
            }
        )
    summary = summarize(runs) if not runs.empty else pd.DataFrame()
    monthly_summary = summarize(runs, by_month=True) if not runs.empty else pd.DataFrame()
    screen_summary = summarize_screens(runs) if not runs.empty else pd.DataFrame()
    runs.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    monthly_summary.to_csv(output_dir / "summary_by_month.csv", index=False)
    screen_summary.to_csv(output_dir / "screen_summary.csv", index=False)
    if args.save_weights and weight_frames:
        weights = pd.concat(weight_frames, ignore_index=True)
        weights.to_csv(
            output_dir / "weights.csv.gz", index=False, compression="gzip"
        )
        compare_weights(weights, runs).to_csv(
            output_dir / "solution_comparisons.csv", index=False
        )
    metadata = {
        "package_version": PACKAGE_VERSION,
        "command": sys.argv,
        "started_at_epoch_s": started_epoch_s,
        "finished_at_epoch_s": time.time(),
        "total_elapsed_s": time.perf_counter() - started,
        "data_load_elapsed_s": store.load_elapsed_s,
        "data_cache_dir": (
            None if args.data_cache_dir is None else str(args.data_cache_dir.resolve())
        ),
        "risk_model_date_count": len(store.risk_dates),
        "dropped_leading_risk_dates": [
            date.strftime("%Y-%m-%d") for date in store.dropped_leading_dates
        ],
        "date_alignment": store.alignment_records,
        "skip_consistency": skip_consistency,
        "dates": [date.strftime("%Y-%m-%d") for date in dates],
        "config": asdict(config),
        "risk_units": {
            "covariance_input": "annualized_percent_squared",
            "specific_risk_input": "annualized_percent_volatility",
            "tracking_error_output": "annualized_percent",
            "annualization_applied_by_benchmark": False,
            "identity": (
                r"\operatorname{TE}_{\%}(a)^2="
                r"f^{\mathsf T}\Sigma_{\%}f+\lVert d_{\%}\odot a\rVert_2^2"
            ),
        },
        "formulation": args.formulation,
        "initial_modes": args.initial_modes,
        "initial_random_seed": args.initial_random_seed,
        "initial_top_n_range": [args.initial_top_n, args.initial_top_n_max],
        "skip_infeasible_rebalance_dates": args.skip_infeasible_rebalance_dates,
        "factor_carry_theta": args.factor_carry_theta,
        "factor_diagnostics": args.factor_diagnostics,
        "direct_qp_fallback": args.direct_qp_fallback,
        "factor_fallback_solver": args.factor_fallback_solver,
        "factor_fallback_clarabel_method": args.factor_fallback_clarabel_method,
        "missing_holding_policy": args.missing_holding_policy,
        "alpha_target": args.alpha_target,
        "screen_solver": args.screen_solver,
        "screen_time_limit_s": args.screen_time_limit_s,
        "screen_max_initial_n": args.screen_max_initial_n,
        "chained_state_weight_tol": args.chained_state_weight_tol,
        "screen_margin_pct": args.screen_margin_pct,
        "screen_feasibility_tol": args.screen_feasibility_tol,
        "clarabel_max_threads": args.clarabel_max_threads,
        "factor_qcqp_settings": asdict(factor_qcqp_settings),
        "factor_piqp_inequality_form_resolved": (
            factor_piqp_inequality_form_resolved
        ),
        "solver_verbose": args.solver_verbose,
        "cases": {
            model: [
                asdict(resolve_case(model.lower(), name, args.alpha_target))
                for name in names
            ]
            for model, names in case_map.items()
        },
        "installed_cvxpy_solvers": sorted(installed),
        "versions": package_versions(),
        "clarabel_build_info": clarabel_build_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": cpu_model_name(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_DYNAMIC",
                "MKL_THREADING_LAYER",
                "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
        "errors": errors,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults: {output_dir}")
    if not summary.empty:
        print(summary.to_string(index=False))
    if errors:
        print(f"\nCompleted with {len(errors)} skipped/failed case(s); see metadata.json")
    if not rows:
        return 2
    if args.require_consistent_skips and inconsistent_skips:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

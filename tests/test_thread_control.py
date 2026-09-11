from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from optim import PortfolioOptimizer, SolverPolicy, SolverTuning
from optim._core.thread_control import (
    NumericalThreadScope,
    effective_cpu_count,
    resolve_thread_setting,
)


def test_solver_tuning_validates_thread_policy():
    assert SolverTuning(threads="auto").threads == "auto"
    assert SolverTuning(threads="max").threads == "max"
    assert SolverTuning(threads=3).threads == 3
    for invalid in (0, -1, True, "all", 1.5):
        with pytest.raises(ValueError, match="threads"):
            SolverTuning(threads=invalid)


def test_effective_cpu_count_uses_strictest_v2_quota(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/workload/child\n", encoding="utf-8")
    (cgroup / "workload/child").mkdir(parents=True)
    (cgroup / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (cgroup / "workload/cpu.max").write_text("1200000 100000\n", encoding="utf-8")
    (cgroup / "workload/child/cpu.max").write_text("max 100000\n", encoding="utf-8")
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(64)))

    count, source = effective_cpu_count(proc_root=proc, cgroup_root=cgroup)

    assert count == 12
    assert source == "affinity+cgroup_v2"


def test_fixed_thread_setting_cannot_exceed_effective_cpu_limit(monkeypatch):
    monkeypatch.setattr(
        "optim._core.thread_control.effective_cpu_count",
        lambda: (12, "test"),
    )
    assert resolve_thread_setting("auto").limit == 1
    assert resolve_thread_setting("max").limit == 12
    fixed = resolve_thread_setting(12)
    assert fixed.policy == "fixed"
    assert fixed.limit == 12
    assert fixed.effective_cpus == 12
    with pytest.raises(ValueError, match="exceeds the effective CPU limit 12"):
        resolve_thread_setting(13)


def test_optimizer_rejects_fixed_threads_above_runtime_limit(monkeypatch):
    monkeypatch.setattr(
        "optim._core.thread_control.effective_cpu_count",
        lambda: (2, "test"),
    )
    policy = SolverPolicy(tuning=SolverTuning(threads=3))

    with pytest.raises(ValueError, match="threads=3.*effective CPU limit 2"):
        PortfolioOptimizer(policy)


def test_thread_scope_is_nested_once_and_restores_after_exception():
    events = []

    class FakeController:
        @contextmanager
        def limit(self, *, limits, user_api):
            events.append(("enter", limits, user_api))
            try:
                yield
            finally:
                events.append(("exit", limits, user_api))

    scope = NumericalThreadScope(1)
    scope._controller = FakeController()
    with pytest.raises(RuntimeError, match="stop"):
        with scope.activate():
            with scope.activate():
                np.ones(1)
                raise RuntimeError("stop")

    assert events == [("enter", 1, "blas"), ("exit", 1, "blas")]


def test_real_threadpool_scope_restores_blas_state():
    from threadpoolctl import threadpool_info

    before = [
        item["num_threads"] for item in threadpool_info() if item["user_api"] == "blas"
    ]
    assert before
    scope = NumericalThreadScope(1)
    with scope.activate():
        during = [
            item["num_threads"]
            for item in threadpool_info()
            if item["user_api"] == "blas"
        ]
        assert during == [1] * len(during)
    after = [
        item["num_threads"] for item in threadpool_info() if item["user_api"] == "blas"
    ]
    assert after == before


def test_optimizer_reports_resolved_thread_policy(sample_lp_problem):
    optimizer = PortfolioOptimizer(
        SolverPolicy(tuning=SolverTuning(threads="auto"))
    )

    result = optimizer.solve(sample_lp_problem)

    assert result.status.has_solution
    metadata = result.route[0].metadata
    assert metadata["thread_policy"] == "auto"
    assert metadata["thread_limit"] == 1
    assert metadata["effective_cpu_count"] >= 1
    assert metadata["native_thread_limit"] == 0

"""Tests for the kill-switch planning + execution logic.

`build_kill_plan` is tested with fabricated AgentProcess lists (no OS
interaction). `execute_kill` is tested against real, disposable `sleep`
subprocesses spawned just for the test, so the kill semantics (success,
"already exited", permission denied) are genuinely exercised rather than
mocked — while never touching anything outside the test's own subprocesses.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil

from agentop.control import KillScope, build_kill_plan, execute_kill
from agentop.models import AgentProcess, Category, Risk


def _sleep_process() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )


def _agent(
    pid: int,
    ppid: int,
    category: Category,
    risk: Risk,
    name: str = "proc",
    create_time: float | None = None,
) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        ppid=ppid,
        name=name,
        category=category,
        subtype="x",
        cmdline=name,
        cpu_percent=0.0,
        mem_mb=0.0,
        create_time=time.time() if create_time is None else create_time,
        risk=risk,
        session_key=ppid,
    )


def _fleet() -> list[AgentProcess]:
    return [
        _agent(100, 1, Category.MCP_TOOL, Risk.LOW, "mcp-a"),
        _agent(101, 1, Category.MCP_TOOL, Risk.LOW, "mcp-b"),
        _agent(102, 2, Category.WORKIQ, Risk.LOW, "workiq"),
        _agent(103, 3, Category.CLAUDE_DESKTOP, Risk.HIGH, "claude"),
    ]


def test_single_scope_targets_exactly_one_pid():
    plan = build_kill_plan(KillScope.SINGLE, _fleet(), pid=101)
    assert plan.count == 1
    assert plan.targets[0].pid == 101
    assert plan.highest_risk is Risk.LOW


def test_session_scope_groups_by_session_key():
    plan = build_kill_plan(KillScope.SESSION, _fleet(), session_key=1)
    assert {t.pid for t in plan.targets} == {100, 101}


def test_category_scope_filters_by_category():
    plan = build_kill_plan(KillScope.CATEGORY, _fleet(), category=Category.MCP_TOOL)
    assert {t.pid for t in plan.targets} == {100, 101}


def test_all_scope_targets_everything():
    plan = build_kill_plan(KillScope.ALL, _fleet())
    assert plan.count == 4


def test_highest_risk_is_reported_for_bulk_scopes():
    plan = build_kill_plan(KillScope.ALL, _fleet())
    assert plan.highest_risk is Risk.HIGH  # claude (HIGH) is in the mix


def test_empty_fleet_produces_empty_plan():
    plan = build_kill_plan(KillScope.ALL, [])
    assert plan.count == 0
    assert plan.highest_risk is Risk.LOW


def test_execute_kill_terminates_a_real_process():
    proc = _sleep_process()
    try:
        agent = _agent(
            proc.pid,
            os.getpid(),
            Category.OTHER_AGENT,
            Risk.LOW,
            "sleep",
            psutil.Process(proc.pid).create_time(),
        )
        plan = build_kill_plan(KillScope.SINGLE, [agent], pid=proc.pid)
        results = execute_kill(plan)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].signal_used == "SIGTERM"
        proc.wait(timeout=5)
        assert proc.poll() is not None  # process has actually exited
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_execute_kill_reports_already_exited():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=5)  # already dead by the time we try to signal it
    agent = _agent(proc.pid, os.getpid(), Category.OTHER_AGENT, Risk.LOW, "true")
    plan = build_kill_plan(KillScope.SINGLE, [agent], pid=proc.pid)
    results = execute_kill(plan)
    assert results[0].success is False
    assert results[0].error == "already exited"


def test_execute_kill_bulk_reports_one_result_per_target():
    procs = [_sleep_process() for _ in range(3)]
    try:
        agents = [
            _agent(
                p.pid,
                os.getpid(),
                Category.MCP_TOOL,
                Risk.LOW,
                f"sleep{i}",
                psutil.Process(p.pid).create_time(),
            )
            for i, p in enumerate(procs)
        ]
        plan = build_kill_plan(KillScope.ALL, agents)
        results = execute_kill(plan)
        assert len(results) == 3
        assert all(r.success for r in results)
        for p in procs:
            p.wait(timeout=5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)


def test_execute_kill_force_uses_sigkill():
    proc = _sleep_process()
    try:
        agent = _agent(
            proc.pid,
            os.getpid(),
            Category.OTHER_AGENT,
            Risk.LOW,
            "sleep",
            psutil.Process(proc.pid).create_time(),
        )
        plan = build_kill_plan(KillScope.SINGLE, [agent], pid=proc.pid)
        results = execute_kill(plan, force=True)
        assert results[0].signal_used == "SIGKILL"
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_execute_kill_refuses_reused_pid(monkeypatch):
    class ReusedProcess:
        def create_time(self):
            return 200.0

        def kill(self):
            kill_calls.append("kill")

        def terminate(self):
            kill_calls.append("terminate")

    kill_calls: list[str] = []
    monkeypatch.setattr("agentop.control.psutil.Process", lambda pid: ReusedProcess())

    agent = _agent(123, 1, Category.MCP_TOOL, Risk.LOW, create_time=100.0)
    plan = build_kill_plan(KillScope.SINGLE, [agent], pid=123)
    results = execute_kill(plan, force=True)

    assert results[0].success is False
    assert results[0].error == "PID reused; identity mismatch"
    assert kill_calls == []

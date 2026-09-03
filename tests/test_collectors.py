"""Collector tests. These touch the real OS/network (read-only, no mutation),
so assertions are structural/type-based rather than asserting exact values
that would make the suite flaky across machines or time.
"""
from __future__ import annotations

from agentop.collectors.network import collect_listening_ports
from agentop.collectors.ollama import collect_ollama_status
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import collect_system_stats
from agentop.models import AgentProcess


def test_system_stats_returns_sane_ranges():
    stats = collect_system_stats()
    assert 0.0 <= stats.cpu_percent <= 100.0
    assert 0.0 <= stats.mem_percent <= 100.0
    assert stats.mem_total_gb > 0
    assert len(stats.cpu_history) >= 1


def test_ollama_offline_is_reported_gracefully_not_raised():
    # Port 1 is a privileged, essentially-never-listening port — connection
    # should fail fast and be reported as offline rather than raising.
    status = collect_ollama_status(base_url="http://localhost:1")
    assert status.online is False
    assert status.error != ""


def test_ollama_online_reports_version_when_running():
    status = collect_ollama_status()
    if not status.online:
        return  # environment-dependent: skip assertions if not running here
    assert status.version != ""


def test_collect_agent_processes_returns_list_without_raising():
    procs = collect_agent_processes()
    assert isinstance(procs, list)
    for p in procs:
        assert isinstance(p, AgentProcess)
        assert p.pid > 0
        assert p.cpu_percent >= 0.0
        assert p.mem_mb >= 0.0


def test_collect_listening_ports_handles_empty_agent_list():
    assert collect_listening_ports([]) == []


def test_collect_listening_ports_does_not_raise_with_real_agents():
    agents = collect_agent_processes()
    ports = collect_listening_ports(agents)
    assert isinstance(ports, list)
    for p in ports:
        assert p.port > 0

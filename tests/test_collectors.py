"""Collector tests. These touch the real OS/network (read-only, no mutation),
so assertions are structural/type-based rather than asserting exact values
that would make the suite flaky across machines or time.
"""
from __future__ import annotations

import subprocess

from agentop.collectors.network import collect_listening_ports
from agentop.collectors.ollama import collect_ollama_status
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import (
    collect_gpu_percent,
    collect_system_stats,
    parse_apple_gpu_metrics,
    parse_rocm_smi_metrics,
)
from agentop.models import AgentProcess


def test_system_stats_returns_sane_ranges():
    stats = collect_system_stats()
    assert 0.0 <= stats.cpu_percent <= 100.0
    assert stats.gpu_percent is None or 0.0 <= stats.gpu_percent <= 100.0
    assert 0.0 <= stats.mem_percent <= 100.0
    assert stats.mem_total_gb > 0
    assert len(stats.cpu_history) >= 1


def test_gpu_collector_parses_apple_device_utilization(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=["ioreg"],
        returncode=0,
        stdout=b'\"PerformanceStatistics\" = {\"Device Utilization %\"=27}',
        stderr=b"",
    )
    monkeypatch.setattr("agentop.collectors.system.sys.platform", "darwin")
    monkeypatch.setattr("agentop.collectors.system.subprocess.run", lambda *args, **kwargs: completed)
    assert collect_gpu_percent() == 27.0


def test_gpu_collector_returns_none_when_telemetry_is_unavailable(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=["ioreg"],
        returncode=0,
        stdout=b"no utilization field",
        stderr=b"",
    )
    monkeypatch.setattr("agentop.collectors.system.sys.platform", "darwin")
    monkeypatch.setattr("agentop.collectors.system.subprocess.run", lambda *args, **kwargs: completed)
    assert collect_gpu_percent() is None


def test_apple_gpu_parser_captures_unified_memory():
    metrics = parse_apple_gpu_metrics(
        b'\"Device Utilization %\"=42,\"In use system memory\"=2147483648',
        36.0,
    )
    assert metrics.vendor == "Apple"
    assert metrics.utilization_percent == 42
    assert metrics.memory_used_gb == 2.0
    assert metrics.memory_total_gb is None


def test_rocm_parser_captures_util_memory_temperature_and_power():
    metrics = parse_rocm_smi_metrics(
        """
        {"card0": {
          "GPU use (%)": "55",
          "VRAM Total Memory (B)": "17179869184",
          "VRAM Total Used Memory (B)": "4294967296",
          "Temperature (Sensor edge) (C)": "64.0",
          "Average Graphics Package Power (W)": "118.0"
        }}
        """
    )
    assert metrics is not None
    assert metrics.vendor == "AMD"
    assert metrics.utilization_percent == 55
    assert metrics.memory_used_gb == 4.0
    assert metrics.memory_total_gb == 16.0
    assert metrics.temperature_c == 64
    assert metrics.power_w == 118


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


def test_ollama_loaded_model_captures_accelerator_memory_in_gb(monkeypatch):
    gib = 1024**3

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, timeout, trust_env):
        if url.endswith("/api/version"):
            return Response({"version": "test"})
        if url.endswith("/api/ps"):
            return Response(
                {
                    "models": [
                        {
                            "name": "test-model",
                            "size": 10 * gib,
                            "size_vram": 8 * gib,
                            "context_length": 4096,
                        }
                    ]
                }
            )
        if url.endswith("/api/tags"):
            return Response(
                {
                    "models": [
                        {
                            "name": "test-model",
                            "size": 10 * gib,
                            "modified_at": "2026-09-03T12:00:00Z",
                            "details": {
                                "family": "test-family",
                                "parameter_size": "7B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("agentop.collectors.ollama.httpx.get", fake_get)
    status = collect_ollama_status("http://test")

    assert status.loaded_models[0].processor == "80% GPU"
    assert status.loaded_models[0].memory_gb == 10.0
    assert status.loaded_models[0].gpu_memory_gb == 8.0
    assert status.available_models[0].name == "test-model"
    assert status.available_models[0].size_gb == 10.0
    assert status.available_models[0].family == "test-family"
    assert status.available_models[0].parameter_size == "7B"
    assert status.available_models[0].quantization == "Q4_K_M"


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

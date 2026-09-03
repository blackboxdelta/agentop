"""System-wide CPU/memory/swap stats with bounded rolling history for sparklines."""
from __future__ import annotations

from collections import deque
import re
import subprocess
import sys

import psutil

from agentop.models import SystemStats

_HISTORY_LEN = 60  # ~2 minutes of history at a 2s poll interval

_cpu_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_gpu_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_mem_history: deque[float] = deque(maxlen=_HISTORY_LEN)

# Prime psutil's internal CPU-percent baseline once at import time so the
# first real poll returns a meaningful value instead of 0.0.
psutil.cpu_percent(None)


def collect_gpu_percent() -> float | None:
    """Read Apple Silicon GPU utilization without requiring elevated access.

    macOS exposes the aggregate device utilization in AGXAccelerator's
    PerformanceStatistics dictionary. Other platforms return None so the UI
    can render GPU as unavailable instead of presenting a false zero.
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True,
            check=False,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(rb'"Device Utilization %"=(\d+)', result.stdout)
    if match is None:
        return None
    return min(100.0, max(0.0, float(match.group(1))))


def collect_system_stats() -> SystemStats:
    cpu = psutil.cpu_percent(None)
    gpu = collect_gpu_percent()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    _cpu_history.append(cpu)
    if gpu is not None:
        _gpu_history.append(gpu)
    _mem_history.append(vm.percent)

    gb = 1024**3
    return SystemStats(
        cpu_percent=cpu,
        gpu_percent=gpu,
        mem_percent=vm.percent,
        mem_used_gb=vm.used / gb,
        mem_total_gb=vm.total / gb,
        swap_percent=swap.percent,
        swap_used_gb=swap.used / gb,
        swap_total_gb=swap.total / gb,
        cpu_history=list(_cpu_history),
        gpu_history=list(_gpu_history),
        mem_history=list(_mem_history),
    )

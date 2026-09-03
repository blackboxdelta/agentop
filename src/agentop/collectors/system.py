"""System-wide CPU/memory/swap stats with bounded rolling history for sparklines."""
from __future__ import annotations

from collections import deque

import psutil

from agentop.models import SystemStats

_HISTORY_LEN = 60  # ~2 minutes of history at a 2s poll interval

_cpu_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_mem_history: deque[float] = deque(maxlen=_HISTORY_LEN)

# Prime psutil's internal CPU-percent baseline once at import time so the
# first real poll returns a meaningful value instead of 0.0.
psutil.cpu_percent(None)


def collect_system_stats() -> SystemStats:
    cpu = psutil.cpu_percent(None)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    _cpu_history.append(cpu)
    _mem_history.append(vm.percent)

    gb = 1024**3
    return SystemStats(
        cpu_percent=cpu,
        mem_percent=vm.percent,
        mem_used_gb=vm.used / gb,
        mem_total_gb=vm.total / gb,
        swap_percent=swap.percent,
        swap_used_gb=swap.used / gb,
        swap_total_gb=swap.total / gb,
        cpu_history=list(_cpu_history),
        mem_history=list(_mem_history),
    )

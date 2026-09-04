"""Cross-platform host and GPU metrics."""
from __future__ import annotations

from collections import deque
import json
import re
import shutil
import subprocess
import sys
import time

import psutil

from agentop.models import GpuStats, SystemStats

_HISTORY_LEN = 60
_cpu_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_gpu_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_mem_history: deque[float] = deque(maxlen=_HISTORY_LEN)
_last_page_in: tuple[float, int] | None = None

psutil.cpu_percent(None)


def parse_apple_gpu_metrics(data: bytes, total_memory_gb: float) -> GpuStats:
    utilization = re.search(rb'"Device Utilization %"=(\d+)', data)
    used_memory = re.search(rb'"In use system memory"=(\d+)', data)
    return GpuStats(
        vendor="Apple",
        utilization_percent=(
            min(100.0, max(0.0, float(utilization.group(1))))
            if utilization
            else None
        ),
        memory_used_gb=(
            float(used_memory.group(1)) / (1024**3) if used_memory else None
        ),
        # Apple Silicon uses unified memory; pairing this allocation counter
        # with total system RAM as if it were dedicated VRAM is misleading.
        memory_total_gb=None,
    )


def collect_apple_gpu_metrics(total_memory_gb: float) -> GpuStats | None:
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
    metrics = parse_apple_gpu_metrics(result.stdout, total_memory_gb)
    return metrics if metrics.utilization_percent is not None else None


def collect_nvidia_gpu_metrics() -> GpuStats | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        return GpuStats(
            vendor="NVIDIA",
            utilization_percent=float(utilization.gpu),
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            temperature_c=float(temperature),
            power_w=float(power),
        )
    except pynvml.NVMLError:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def parse_rocm_smi_metrics(payload: str) -> GpuStats | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    first = next(iter(data.values()))
    if not isinstance(first, dict):
        return None

    def number_for(*parts: str) -> float | None:
        for key, value in first.items():
            normalized = str(key).casefold()
            if all(part in normalized for part in parts):
                match = re.search(r"-?\d+(?:\.\d+)?", str(value))
                if match:
                    return float(match.group())
        return None

    used_bytes = number_for("vram", "used")
    total_bytes = number_for("vram", "total")
    return GpuStats(
        vendor="AMD",
        utilization_percent=number_for("gpu", "use"),
        memory_used_gb=used_bytes / (1024**3) if used_bytes else None,
        memory_total_gb=total_bytes / (1024**3) if total_bytes else None,
        temperature_c=number_for("temperature"),
        power_w=number_for("power"),
    )


def collect_amd_gpu_metrics() -> GpuStats | None:
    executable = shutil.which("rocm-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--showuse",
                "--showmeminfo",
                "vram",
                "--showtemp",
                "--showpower",
                "--json",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return parse_rocm_smi_metrics(result.stdout)


def collect_gpu_metrics(total_memory_gb: float) -> GpuStats:
    if sys.platform == "darwin":
        return collect_apple_gpu_metrics(total_memory_gb) or GpuStats()
    return (
        collect_nvidia_gpu_metrics()
        or collect_amd_gpu_metrics()
        or GpuStats()
    )


def collect_gpu_percent() -> float | None:
    """Compatibility wrapper retained for integrations and focused tests."""
    return collect_gpu_metrics(psutil.virtual_memory().total / (1024**3)).utilization_percent


def _page_in_rate(swap_in_bytes: int, sampled_at: float) -> float | None:
    global _last_page_in
    previous = _last_page_in
    _last_page_in = (sampled_at, swap_in_bytes)
    if previous is None:
        return None
    elapsed = sampled_at - previous[0]
    if elapsed <= 0 or swap_in_bytes < previous[1]:
        return None
    return (swap_in_bytes - previous[1]) / elapsed / (1024**2)


def collect_system_stats() -> SystemStats:
    sampled_at = time.monotonic()
    cpu = psutil.cpu_percent(None)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = collect_gpu_metrics(vm.total / (1024**3))
    page_in = _page_in_rate(int(getattr(swap, "sin", 0)), sampled_at)

    _cpu_history.append(cpu)
    if gpu.utilization_percent is not None:
        _gpu_history.append(gpu.utilization_percent)
    _mem_history.append(vm.percent)

    gb = 1024**3
    load_1m = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0
    return SystemStats(
        cpu_percent=cpu,
        cpu_count=psutil.cpu_count(logical=True) or 0,
        load_1m=load_1m,
        gpu_percent=gpu.utilization_percent,
        gpu_memory_used_gb=gpu.memory_used_gb,
        gpu_memory_total_gb=gpu.memory_total_gb,
        gpu_temperature_c=gpu.temperature_c,
        gpu_power_w=gpu.power_w,
        gpu_vendor=gpu.vendor,
        mem_percent=vm.percent,
        mem_used_gb=vm.used / gb,
        mem_total_gb=vm.total / gb,
        swap_percent=swap.percent,
        swap_used_gb=swap.used / gb,
        swap_total_gb=swap.total / gb,
        page_in_mb_s=page_in,
        cpu_history=list(_cpu_history),
        gpu_history=list(_gpu_history),
        mem_history=list(_mem_history),
    )

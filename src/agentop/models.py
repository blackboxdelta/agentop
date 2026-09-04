"""Typed data structures shared across collectors and the UI layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    """Groupings of agent-related processes, by originating app/runtime."""

    COPILOT_SESSION = "Copilot CLI Session"
    COPILOT_EXTENSION = "Copilot Extension (VS Code)"
    MCP_TOOL = "MCP Tool Server"
    WORKIQ = "WorkIQ MCP"
    MODEL_SERVER = "Model Server"
    CLAUDE_DESKTOP = "Claude Desktop"
    CHATGPT_CODEX = "ChatGPT / Codex"
    CURSOR = "Cursor"
    OTHER_AGENT = "Other Agent"


class Risk(str, Enum):
    """Blast radius if a process in this bucket is terminated."""

    LOW = "low"        # auto-respawns on next tool call; safe to kill
    MEDIUM = "medium"   # kills a live server/session; recoverable but disruptive
    HIGH = "high"       # quits a desktop app or a live coding session


# Ordering used for sorting / picking the "worst" risk in a bulk selection.
RISK_ORDER = {Risk.LOW: 0, Risk.MEDIUM: 1, Risk.HIGH: 2}


@dataclass
class AgentProcess:
    """A single OS process identified as belonging to an agent/tool runtime."""

    pid: int
    ppid: int
    name: str
    category: Category
    subtype: str
    cmdline: str
    cpu_percent: float
    mem_mb: float
    create_time: float
    risk: Risk
    session_key: int  # grouping key: usually the parent pid that spawned this process
    port: int | None = None

    @property
    def uptime_seconds(self) -> float:
        import time

        return max(0.0, time.time() - self.create_time)


@dataclass
class SystemStats:
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_1m: float = 0.0
    gpu_percent: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_total_gb: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    gpu_vendor: str = ""
    mem_percent: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    swap_percent: float = 0.0
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0
    page_in_mb_s: float | None = None
    cpu_history: list[float] = field(default_factory=list)
    gpu_history: list[float] = field(default_factory=list)
    mem_history: list[float] = field(default_factory=list)


@dataclass
class GpuStats:
    vendor: str = ""
    utilization_percent: float | None = None
    memory_used_gb: float | None = None
    memory_total_gb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None


@dataclass
class OllamaModel:
    name: str
    size_gb: float
    processor: str
    memory_gb: float
    gpu_memory_gb: float
    context: int
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    expires_at: str = ""
    context_used: int = 0
    total_layers: int = 0
    gpu_layers: int = 0
    state: str = "idle"


@dataclass
class OllamaAvailableModel:
    name: str
    size_gb: float
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    context: int = 0
    total_layers: int = 0
    digest: str = ""
    description: str = ""


@dataclass
class OllamaStatus:
    online: bool
    version: str = ""
    loaded_models: list[OllamaModel] = field(default_factory=list)
    available_models: list[OllamaAvailableModel] = field(default_factory=list)
    loaded_models_error: str = ""
    available_models_error: str = ""
    error: str = ""
    stale: bool = False
    stale_seconds: float = 0.0
    sampled_at: float = 0.0
    in_flight: int | None = None
    queue_depth: int | None = None


@dataclass
class PortInfo:
    port: int
    pid: int
    process_name: str
    category: Category | None = None


@dataclass
class CompletionRecord:
    model: str
    client: str
    started_at: float
    completed_at: float
    success: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    prompt_eval_ns: int = 0
    eval_ns: int = 0
    total_ns: int = 0
    ttft_ms: float | None = None
    error_type: str = ""
    context_window: int = 0
    batch_size: int = 0


@dataclass
class SessionMetric:
    client: str
    request_count: int
    held_seconds: float
    idle_seconds: float


@dataclass
class ModelMetrics:
    model: str
    run_count: int = 0
    last_run_at: float | None = None
    generation_tps: float | None = None
    prompt_tps: float | None = None
    ttft_p50_ms: float | None = None
    tokens_in_60s: int = 0
    tokens_out_60s: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    success_rate: float | None = None
    oom_count: int = 0
    timeout_count: int = 0
    context_overflow_count: int = 0
    last_context_used: int = 0
    last_context_window: int = 0
    batch_size: int = 0
    sessions: list[SessionMetric] = field(default_factory=list)


@dataclass
class EventRecord:
    timestamp: float
    level: str
    kind: str
    model: str
    message: str

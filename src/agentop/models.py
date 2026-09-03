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
    gpu_percent: float | None = None
    mem_percent: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    swap_percent: float = 0.0
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0
    cpu_history: list[float] = field(default_factory=list)
    gpu_history: list[float] = field(default_factory=list)
    mem_history: list[float] = field(default_factory=list)


@dataclass
class OllamaModel:
    name: str
    size_gb: float
    processor: str
    context: int


@dataclass
class OllamaStatus:
    online: bool
    version: str = ""
    loaded_models: list[OllamaModel] = field(default_factory=list)
    available_models: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class PortInfo:
    port: int
    pid: int
    process_name: str
    category: Category | None = None

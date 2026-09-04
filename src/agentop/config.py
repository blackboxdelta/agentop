"""Configuration loading with file < environment < CLI precedence."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from urllib.parse import urlparse


if os.name == "nt":
    _config_root = Path(
        os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    )
    _state_root = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    DEFAULT_CONFIG_PATH = _config_root / "agentop" / "config.toml"
    DEFAULT_STATE_PATH = _state_root / "agentop" / "events.db"
else:
    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "agentop" / "config.toml"
    DEFAULT_STATE_PATH = (
        Path.home() / ".local" / "state" / "agentop" / "events.db"
    )


@dataclass(frozen=True)
class AgentopConfig:
    host: str = "http://localhost:11434"
    refresh_interval: float = 1.0
    tags_interval: float = 30.0
    request_timeout: float = 5.0
    model_load_timeout: float = 600.0
    proxy_max_body_bytes: int = 16 * 1024 * 1024
    memory_headroom_gb: float = 2.0
    model_memory_overhead: float = 1.15
    state_path: Path = DEFAULT_STATE_PATH
    no_color: bool = False


def normalize_host(value: str) -> str:
    host = value.strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid Ollama host: {value!r}")
    return host


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(
    *,
    config_path: str | Path | None = None,
    host: str | None = None,
    refresh_interval: float | None = None,
    state_path: str | Path | None = None,
    no_color: bool = False,
) -> AgentopConfig:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    raw = _read_config(path)
    ollama = raw.get("ollama", {})
    polling = raw.get("polling", {})
    state = raw.get("state", {})
    proxy = raw.get("proxy", {})
    safety = raw.get("safety", {})

    resolved_host = (
        host
        or os.environ.get("OLLAMA_HOST")
        or ollama.get("host")
        or AgentopConfig.host
    )
    resolved_refresh = (
        refresh_interval
        if refresh_interval is not None
        else float(polling.get("refresh_interval", AgentopConfig.refresh_interval))
    )
    tags_interval = float(
        polling.get("tags_interval", AgentopConfig.tags_interval)
    )
    request_timeout = float(
        polling.get("request_timeout", AgentopConfig.request_timeout)
    )
    model_load_timeout = float(
        polling.get("model_load_timeout", AgentopConfig.model_load_timeout)
    )
    proxy_max_body_bytes = int(
        proxy.get("max_body_bytes", AgentopConfig.proxy_max_body_bytes)
    )
    memory_headroom_gb = float(
        safety.get("memory_headroom_gb", AgentopConfig.memory_headroom_gb)
    )
    model_memory_overhead = float(
        safety.get("model_memory_overhead", AgentopConfig.model_memory_overhead)
    )
    resolved_state = Path(
        state_path
        or state.get("path")
        or DEFAULT_STATE_PATH
    ).expanduser()

    if (
        resolved_refresh <= 0
        or tags_interval <= 0
        or request_timeout <= 0
        or model_load_timeout <= 0
        or proxy_max_body_bytes <= 0
        or memory_headroom_gb < 0
        or model_memory_overhead < 1
    ):
        raise ValueError("Polling, proxy, and safety limits are invalid")

    return AgentopConfig(
        host=normalize_host(str(resolved_host)),
        refresh_interval=resolved_refresh,
        tags_interval=tags_interval,
        request_timeout=request_timeout,
        model_load_timeout=model_load_timeout,
        proxy_max_body_bytes=proxy_max_body_bytes,
        memory_headroom_gb=memory_headroom_gb,
        model_memory_overhead=model_memory_overhead,
        state_path=resolved_state,
        no_color=no_color or "NO_COLOR" in os.environ,
    )

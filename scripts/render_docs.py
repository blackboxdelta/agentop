"""Generate deterministic, privacy-safe documentation screenshots."""
from __future__ import annotations

import asyncio
from pathlib import Path
import re
import tempfile
import time

from textual.widgets import Input, Label, Select

from agentop.conversation import ConversationEntry
from agentop.events import EventStore
from agentop.models import (
    AgentProcess,
    Category,
    EventRecord,
    ModelMetrics,
    OllamaAvailableModel,
    OllamaModel,
    OllamaStatus,
    PortInfo,
    Risk,
    SessionMetric,
    SystemStats,
)
from agentop.ui.app import AgentopApp, TopBar

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "images"


class StaticClient:
    async def poll(self, force_tags=False):
        return STATUS


def process(
    pid: int,
    category: Category,
    subtype: str,
    cpu: float,
    memory: float,
    risk: Risk,
) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        ppid=100,
        name=subtype,
        category=category,
        subtype=subtype,
        cmdline=f"/usr/local/bin/{subtype}",
        cpu_percent=cpu,
        mem_mb=memory,
        create_time=time.time() - 3600,
        risk=risk,
        session_key=100,
    )


SYSTEM = SystemStats(
    cpu_percent=38.2,
    cpu_count=12,
    load_1m=3.4,
    gpu_percent=71.0,
    gpu_memory_used_gb=16,
    gpu_memory_total_gb=24,
    gpu_temperature_c=64,
    gpu_power_w=118,
    gpu_vendor="NVIDIA",
    mem_percent=59.9,
    mem_used_gb=18.3,
    mem_total_gb=30.5,
    swap_percent=86.1,
    swap_used_gb=7.7,
    swap_total_gb=9,
    page_in_mb_s=2.1,
)

AVAILABLE = [
    OllamaAvailableModel(
        name="devstral:latest",
        size_gb=14.3,
        family="llama",
        parameter_size="23.6B",
        quantization="Q4_K_M",
        context=131072,
        total_layers=48,
        description="Agentic coding model",
    ),
    OllamaAvailableModel(
        name="phi4:latest",
        size_gb=9.1,
        family="phi3",
        parameter_size="14.7B",
        quantization="Q4_K_M",
        context=16384,
        total_layers=40,
    ),
    OllamaAvailableModel(
        name="qwen2.5-coder:32b",
        size_gb=19.9,
        family="qwen2",
        parameter_size="32.8B",
        quantization="Q4_K_M",
        context=32768,
        total_layers=64,
    ),
    OllamaAvailableModel(
        name="llava:latest",
        size_gb=4.7,
        family="llava",
        parameter_size="7B",
        quantization="Q4_0",
        context=8192,
    ),
]

LOADED = [
    OllamaModel(
        name="devstral:latest",
        size_gb=14.3,
        processor="100% GPU",
        memory_gb=11.6,
        gpu_memory_gb=11.2,
        context=131072,
        family="llama",
        parameter_size="23.6B",
        quantization="Q4_K_M",
        total_layers=48,
        gpu_layers=48,
        expires_at="2099-01-01T00:00:00Z",
        state="generating",
    ),
    OllamaModel(
        name="phi4:latest",
        size_gb=9.1,
        processor="55% GPU",
        memory_gb=8.1,
        gpu_memory_gb=4.5,
        context=16384,
        family="phi3",
        parameter_size="14.7B",
        quantization="Q4_K_M",
        total_layers=40,
        gpu_layers=22,
        expires_at="2099-01-01T00:00:00Z",
    ),
]

STATUS = OllamaStatus(
    online=True,
    version="0.33.2",
    loaded_models=LOADED,
    available_models=AVAILABLE,
    in_flight=3,
    queue_depth=1,
)

AGENTS = [
    process(2101, Category.MCP_TOOL, "github", 2.1, 90, Risk.LOW),
    process(2102, Category.MCP_TOOL, "filesystem", 1.3, 64, Risk.LOW),
    process(2201, Category.COPILOT_SESSION, "copilot", 7.8, 520, Risk.HIGH),
    process(2301, Category.CLAUDE_DESKTOP, "claude", 3.2, 410, Risk.HIGH),
    process(2401, Category.MODEL_SERVER, "ollama", 18.4, 11800, Risk.MEDIUM),
]

METRICS = {
    "devstral:latest": ModelMetrics(
        model="devstral:latest",
        run_count=196,
        last_run_at=time.time() - 120,
        generation_tps=29.4,
        prompt_tps=842,
        ttft_p50_ms=310,
        tokens_in_60s=12600,
        tokens_out_60s=3900,
        in_flight=2,
        queue_depth=1,
        latency_p50_ms=1900,
        latency_p95_ms=6400,
        success_rate=99.1,
        timeout_count=1,
        context_overflow_count=3,
        last_context_used=98300,
        last_context_window=131072,
        batch_size=512,
        sessions=[
            SessionMetric("claude-code · repo/agentop", 2, 2460, 10),
            SessionMetric("aider · notes", 5, 1200, 360),
        ],
    ),
    "phi4:latest": ModelMetrics(
        model="phi4:latest",
        run_count=87,
        generation_tps=11.8,
        ttft_p50_ms=140,
    ),
}

DOC_TIMESTAMP = 1_735_689_600.0

EVENTS = [
    EventRecord(DOC_TIMESTAMP - 5, "ok", "load", "devstral:latest", "devstral warmed on GPU"),
    EventRecord(DOC_TIMESTAMP - 45, "warn", "poll", "", "swap pressure high during load"),
    EventRecord(DOC_TIMESTAMP - 90, "ok", "proxy", "devstral:latest", "completion finished in 41.2 s"),
]

PLAYGROUND_MODELS = (
    "devstral:latest",
    "phi4:latest",
    "qwen2.5-coder:32b",
)


async def render(size: tuple[int, int], tab: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = EventStore(Path(directory) / "events.db")
        app = AgentopApp(refresh_interval=100, client=StaticClient(), store=store)
        app._trigger_refresh = lambda: None
        async with app.run_test(size=size) as pilot:
            app._agents = AGENTS
            app._system_stats = SYSTEM
            app._ollama_status = STATUS
            app._model_metrics = METRICS
            app._selected_model_name = "devstral:latest"
            app.query_one(TopBar).update_stats(SYSTEM, STATUS, AGENTS)
            app._update_overview_table(AGENTS)
            app._update_process_table(AGENTS)
            app._update_models_tables(STATUS)
            app._update_playground_model_options(STATUS)
            app._update_network_table(
                [PortInfo(11434, 2401, "ollama", Category.MODEL_SERVER)]
            )
            app._render_model_events(EVENTS)
            app.action_show_tab(tab)
            if tab == "tab-playground":
                populate_playground(app, compact=size[0] < 100)
            await pilot.pause()
            app.set_focus(None)
            await pilot.pause()
            screenshot = Path(
                app.save_screenshot(filename=filename, path=str(OUTPUT))
            )
            normalize_svg(screenshot)


def normalize_svg(path: Path) -> None:
    content = re.sub(r"terminal-\d+", "terminal-agentop", path.read_text())
    lines = content.splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def populate_playground(app: AgentopApp, *, compact: bool) -> None:
    mode = "duo" if compact else "trio"
    models = PLAYGROUND_MODELS[: 2 if compact else 3]
    app.query_one("#playground-mode", Select).value = mode
    for index, model in enumerate(models, start=1):
        app.query_one(f"#playground-model-{index}", Select).value = model
    app.query_one("#playground-rounds", Input).value = "25"

    app._playground_session_key = (mode, *models)
    app._write_playground_entry(
        ConversationEntry(
            "User",
            "What is the strongest reason to run AI models locally?",
        )
    )
    app._write_playground_round_header(1, 25)
    app._write_playground_entry(
        ConversationEntry(
            models[0],
            "Privacy: prompts and sensitive context can remain on your machine.",
        )
    )
    app._write_playground_entry(
        ConversationEntry(
            models[1],
            "Reliability matters too: local inference works without a network.",
        )
    )
    if not compact:
        app._write_playground_entry(
            ConversationEntry(
                models[2],
                "Control is the key benefit: you choose the model, version, and data path.",
            )
        )
    app.query_one("#playground-status", Label).update(
        f"Complete · {len(models)} models · round 1 of 25 shown."
    )


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    await render((160, 52), "tab-overview", "overview.svg")
    await render((160, 52), "tab-models", "models.svg")
    await render((80, 24), "tab-models", "models-compact.svg")
    await render((160, 52), "tab-playground", "playground.svg")
    await render((80, 24), "tab-playground", "playground-compact.svg")


if __name__ == "__main__":
    asyncio.run(main())

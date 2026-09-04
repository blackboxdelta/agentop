"""agentop's Textual application."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import sqlite3
import time

from rich.markup import escape
from rich.text import Text
import httpx
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Label, ProgressBar, Static, TabbedContent, TabPane

from agentop.collectors.network import collect_listening_ports
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import collect_system_stats
from agentop.config import AgentopConfig, normalize_host
from agentop.control import KillScope, build_kill_plan, execute_kill
from agentop.events import EventStore
from agentop.fit import FitPlan, calculate_fit_plan
from agentop.models import (
    AgentProcess,
    Category,
    OllamaAvailableModel,
    OllamaModel,
    OllamaStatus,
    PortInfo,
    Risk,
    RISK_ORDER,
    EventRecord,
    ModelMetrics,
    SystemStats,
)
from agentop.ollama_client import OllamaClient
from agentop.ui.screens import ConfirmKillScreen, ConfirmTextScreen, HelpScreen, PreflightScreen

REFRESH_INTERVAL = 2.0

_RISK_STYLE = {
    Risk.LOW: "#5EE6A8",
    Risk.MEDIUM: "#E0A65C",
    Risk.HIGH: "bold #F0706E",
}

_EVENT_STYLE = {
    "ok": "#5EE6A8",
    "load": "#8FA9FF",
    "unload": "#E0A65C",
    "warn": "#E0A65C",
}

_MODEL_KIND_STYLE = {
    "GGUF": "bold #8FA9FF on #171F33",
    "CODE": "bold #5EE6A8 on #0F2620",
    "CHAT": "bold #C9A6FF on #1E1830",
    "VLM": "bold #E0A65C on #2A1F12",
    "LLM": "#8A98A0 on #232A2E",
}


def _risk_text(risk: Risk) -> Text:
    return Text(risk.value, style=_RISK_STYLE[risk])


def _fmt_mem(mem_mb: float) -> str:
    return f"{mem_mb:.0f} MB" if mem_mb < 1024 else f"{mem_mb / 1024:.2f} GB"


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _fmt_rate(value: float | None, *, estimated: bool = False) -> str:
    if value is None:
        return "-"
    prefix = "~" if estimated else ""
    return f"{prefix}{value:.1f} t/s"


def _fmt_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _fmt_seconds(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.1f}m"
    return f"{value:.0f}s"


def _fmt_age(value: str) -> str:
    if not value:
        return "-"
    try:
        modified = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "-"
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - modified).total_seconds()))
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _fmt_remaining(value: str) -> str:
    if not value:
        return "-"
    try:
        expires = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "-"
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    seconds = int((expires - datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return "expired"
    if seconds < 60:
        return "<1m"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 86400}d"


def _remaining_keep_alive(value: str) -> str:
    if not value:
        return "5m"
    try:
        expires = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "5m"
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    seconds = int((expires - datetime.now(timezone.utc)).total_seconds())
    return f"{max(1, seconds)}s" if seconds > 0 else "5m"


def _model_kind(model: OllamaAvailableModel) -> str:
    name = model.name.casefold()
    family = model.family.casefold()
    if "gguf" in name:
        return "GGUF"
    if "llava" in name or "moondream" in name or "clip" in family:
        return "VLM"
    if "code" in name or "coder" in name:
        return "CODE"
    if "phi" in name or "chat" in name:
        return "CHAT"
    return "LLM"


def _model_quantization(model: OllamaAvailableModel | OllamaModel) -> str:
    if model.quantization and model.quantization.casefold() != "unknown":
        return model.quantization
    tag = model.name.rpartition(":")[2]
    return tag if tag.casefold().startswith(("q", "f")) else "-"


def _model_kind_text(model: OllamaAvailableModel) -> Text:
    kind = _model_kind(model)
    return Text(f" {kind} ", style=_MODEL_KIND_STYLE[kind])


class TopBar(Static):
    """Compact resource header matching the supplied agentop design."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-bar"):
            with Vertical(id="brand"):
                yield Label("agentop", id="brand-name")
                yield Label("local agent control plane", id="brand-subtitle")
            with Vertical(classes="metric-block"):
                with Horizontal(classes="metric-heading"):
                    yield Label("CPU", classes="metric-name")
                    yield Label("--%", id="cpu-value", classes="metric-value cpu")
                yield ProgressBar(
                    total=100,
                    show_percentage=False,
                    show_eta=False,
                    id="cpu-bar",
                )
                yield Label("agents --", id="cpu-detail", classes="metric-detail")
            with Vertical(classes="metric-block"):
                with Horizontal(classes="metric-heading"):
                    yield Label("GPU", classes="metric-name")
                    yield Label("--%", id="gpu-value", classes="metric-value gpu")
                yield ProgressBar(
                    total=100,
                    show_percentage=False,
                    show_eta=False,
                    id="gpu-bar",
                )
                yield Label("GPU telemetry", id="gpu-detail", classes="metric-detail")
            with Vertical(classes="metric-block vram-block"):
                with Horizontal(classes="metric-heading"):
                    yield Label("VRAM", id="gpu-memory-name", classes="metric-name")
                    yield Label("N/A", id="vram-percent", classes="metric-value gpu")
                yield ProgressBar(
                    total=100,
                    show_percentage=False,
                    show_eta=False,
                    id="vram-bar",
                )
                yield Label("N/A", id="vram-detail", classes="metric-detail")
            with Vertical(classes="metric-block"):
                with Horizontal(classes="metric-heading"):
                    yield Label("MEM", classes="metric-name")
                    yield Label("--%", id="memory-percent", classes="metric-value memory")
                yield ProgressBar(total=100, show_percentage=False, show_eta=False, id="memory-bar")
                yield Label("-- / -- GB", id="memory-detail", classes="metric-detail")
            with Vertical(classes="metric-block"):
                with Horizontal(classes="metric-heading"):
                    yield Label("SWAP", classes="metric-name")
                    yield Label("--%", id="swap-percent", classes="metric-value swap")
                yield ProgressBar(total=100, show_percentage=False, show_eta=False, id="swap-bar")
                yield Label("-- / -- GB", id="swap-detail", classes="metric-detail")
            with Vertical(id="ollama-status"):
                yield Label("● ollama checking", id="ollama-title")
                yield Label("-- loaded · -- available", id="ollama-detail")

    def update_stats(
        self,
        stats: SystemStats,
        ollama: OllamaStatus,
        agents: list[AgentProcess],
    ) -> None:
        self.query_one("#cpu-bar", ProgressBar).update(progress=stats.cpu_percent)
        self.query_one("#cpu-value", Label).update(f"{stats.cpu_percent:.1f}%")
        self.query_one("#cpu-detail", Label).update(
            f"{stats.cpu_count} cores · load {stats.load_1m:.1f}"
        )

        self.query_one("#gpu-bar", ProgressBar).update(
            progress=stats.gpu_percent or 0.0
        )
        self.query_one("#gpu-value", Label).update(
            f"{stats.gpu_percent:.1f}%" if stats.gpu_percent is not None else "N/A"
        )
        gpu_details = []
        if stats.gpu_temperature_c is not None:
            gpu_details.append(f"{stats.gpu_temperature_c:.0f} °C")
        if stats.gpu_power_w is not None:
            gpu_details.append(f"{stats.gpu_power_w:.0f} W")
        self.query_one("#gpu-detail", Label).update(
            " · ".join(gpu_details) or stats.gpu_vendor or "unavailable"
        )

        vram_used = stats.gpu_memory_used_gb
        vram_total = stats.gpu_memory_total_gb
        apple_unified = stats.gpu_vendor == "Apple"
        self.query_one("#gpu-memory-name", Label).update(
            "UNIFIED" if apple_unified else "VRAM"
        )
        vram_percent = (
            vram_used / vram_total * 100
            if vram_used is not None and vram_total
            else None
        )
        self.query_one("#vram-percent", Label).update(
            f"{vram_used:.1f} GB"
            if apple_unified and vram_used is not None
            else f"{vram_percent:.1f}%"
            if vram_percent is not None
            else "N/A"
        )
        self.query_one("#vram-bar", ProgressBar).update(progress=vram_percent or 0)
        self.query_one("#vram-detail", Label).update(
            f"{vram_used:.1f} / {vram_total:.1f} GB"
            if vram_used is not None and vram_total is not None
            else "GPU allocation"
            if apple_unified and vram_used is not None
            else "not exposed"
        )

        self.query_one("#memory-percent", Label).update(f"{stats.mem_percent:.1f}%")
        self.query_one("#memory-bar", ProgressBar).update(progress=stats.mem_percent)
        self.query_one("#memory-detail", Label).update(
            f"{stats.mem_used_gb:.1f} / {stats.mem_total_gb:.1f} GB"
        )

        self.query_one("#swap-percent", Label).update(f"{stats.swap_percent:.1f}%")
        self.query_one("#swap-bar", ProgressBar).update(progress=stats.swap_percent)
        self.query_one("#swap-detail", Label).update(
            f"{stats.page_in_mb_s:.1f} MB/s page-in"
            if stats.page_in_mb_s is not None
            else f"{stats.swap_used_gb:.1f} / {stats.swap_total_gb:.1f} GB"
        )

        title = self.query_one("#ollama-title", Label)
        if ollama.online and not ollama.stale:
            title.update(f"● ollama v{ollama.version}")
            title.set_classes("online")
        elif ollama.stale:
            title.update(f"● ollama stale {ollama.stale_seconds:.0f} s")
            title.set_classes("stale")
        else:
            title.update("● ollama offline")
            title.set_classes("offline")
        detail = (
            "model status unavailable"
            if ollama.error or ollama.loaded_models_error
            else (
                f"{len(ollama.loaded_models)} resident · "
                f"{ollama.in_flight or 0} in flight · q {ollama.queue_depth or 0}"
            )
        )
        self.query_one("#ollama-detail", Label).update(detail)


class AgentopApp(App):
    """Local terminal control plane for agent processes and models."""

    TITLE = "agentop"
    CSS_PATH = "agentop.tcss"

    BINDINGS = [
        Binding("1", "show_tab('tab-overview')", "Overview"),
        Binding("2", "show_tab('tab-processes')", "Processes"),
        Binding("3", "show_tab('tab-models')", "Models"),
        Binding("4", "show_tab('tab-network')", "Network"),
        Binding("/", "focus_model_filter", "Filter Models"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("enter", "model_warm", "Warm"),
        Binding("k", "context_kill", "Unload / Kill"),
        Binding("shift+s", "kill_switch_session", "Kill Session"),
        Binding("shift+k", "context_kill_all", "Unload All / Kill Category"),
        Binding("shift+a", "kill_switch_all", "KILL ALL"),
        Binding("t", "trim_context", "Reload Context"),
        Binding("p", "toggle_pin", "Pin"),
        Binding("u", "undo_unload", "Undo Unload"),
        Binding("?", "show_help", "Help"),
        Binding("escape", "close_detail", "Back", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        ollama_url: str | None = None,
        refresh_interval: float | None = None,
        *,
        config: AgentopConfig | None = None,
        store: EventStore | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AgentopConfig(
            host=normalize_host(ollama_url or "http://localhost:11434"),
            refresh_interval=refresh_interval or REFRESH_INTERVAL,
        )
        self.refresh_interval = self.config.refresh_interval
        self.store = store or EventStore(self.config.state_path)
        self.client = client or OllamaClient(self.config)
        self._owns_client = client is None
        self._agents: list[AgentProcess] = []
        self._system_stats = SystemStats()
        self._ollama_status = OllamaStatus(online=False)
        self._selected_agent: AgentProcess | None = None
        self._selected_category: Category | None = None
        self._selected_model_name: str | None = None
        self._model_filter = ""
        self._model_metrics: dict[str, ModelMetrics] = {}
        self._pinned_models: set[str] = set()
        self._pending_force_targets: list[AgentProcess] = []
        self._previous_loaded_names: set[str] | None = None
        self._model_poll_failed = False
        self._rebuilding_model_tables = False
        self._undo_model: tuple[str, float] | None = None
        self._model_mutation_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield TopBar()
        with TabbedContent(initial="tab-overview", id="main-tabs"):
            with TabPane("1  Overview", id="tab-overview"):
                yield DataTable(id="overview-table", cursor_type="row", zebra_stripes=True)
            with TabPane("2  Processes", id="tab-processes"):
                yield DataTable(id="process-table", cursor_type="row", zebra_stripes=True)
                with HorizontalScroll(id="process-toolbar"):
                    yield Button("Refresh", id="btn-refresh")
                    yield Button("Kill Selected", id="btn-kill-selected", variant="warning")
                    yield Button("Kill Session", id="btn-kill-session", variant="warning")
                    yield Button("Kill Category", id="btn-kill-category", variant="error")
                    yield Button("KILL ALL", id="btn-kill-all", variant="error")
                    yield Button(
                        "Force Kill Remaining (0)",
                        id="btn-force-kill",
                        variant="error",
                        disabled=True,
                    )
            with TabPane("3  Models", id="tab-models"):
                with Horizontal(id="models-layout"):
                    with Vertical(id="models-main"):
                        with Horizontal(id="model-filter-row"):
                            yield Label("MODELS", classes="pane-title")
                            yield Input(
                                placeholder="/  filter models",
                                id="model-filter",
                                compact=True,
                            )
                        yield Label("RESIDENT  0 models · 0.0 GB held", id="resident-heading", classes="section-heading")
                        yield DataTable(id="resident-models-table", cursor_type="row", zebra_stripes=True)
                        yield Static(
                            "[#252E33]Ø[/#252E33]  [bold #8A98A0]No models resident in memory.[/bold #8A98A0]\n"
                            "[#5C6A72]Models appear here while Ollama keeps them loaded.[/#5C6A72]",
                            id="resident-empty",
                        )
                        yield Label("AVAILABLE  0 models · 0.0 GB on disk", id="available-heading", classes="section-heading")
                        yield DataTable(id="available-models-table", cursor_type="row", zebra_stripes=True)
                    with VerticalScroll(id="model-sidebar"):
                        yield Label("SELECTED", classes="sidebar-heading")
                        yield Static("Select a model to inspect it.", id="model-details")
                        with Horizontal(id="model-actions"):
                            yield Button("Warm", id="btn-model-warm", variant="primary")
                            yield Button("Unload", id="btn-model-unload", variant="error")
                            yield Button("Pin", id="btn-model-pin")
                            yield Button("Reload ctx", id="btn-model-trim")
                        yield Label("THROUGHPUT · 60 s", id="throughput-heading", classes="sidebar-heading")
                        yield Static("", id="throughput-detail")
                        yield Label("CONTEXT & KV CACHE", id="context-heading", classes="sidebar-heading")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="context-bar",
                        )
                        yield Static("", id="context-detail")
                        yield Static("", id="context-warning")
                        yield Label("PLACEMENT", classes="sidebar-heading")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="placement-bar",
                        )
                        yield Static("", id="placement-detail")
                        yield Label("RELIABILITY · 24 h", id="reliability-heading", classes="sidebar-heading")
                        yield Static("", id="reliability-detail")
                        yield Label("SESSIONS HOLDING THIS MODEL", id="sessions-heading", classes="sidebar-heading")
                        yield Static("", id="sessions-detail")
                        yield Label("RECENT EVENTS", classes="sidebar-heading")
                        yield Static("Waiting for model activity.", id="model-events")
            with TabPane("4  Network", id="tab-network"):
                yield DataTable(id="network-table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self.store.initialize()
        self.set_class(self.config.no_color, "no-color")
        self._setup_tables()
        self.set_interval(self.refresh_interval, self._trigger_refresh)
        self.set_interval(3600, self._trigger_store_maintenance)
        self._trigger_store_maintenance()
        self._update_responsive_class(self.size.width)
        self.call_after_refresh(self._trigger_refresh)

    async def on_unmount(self) -> None:
        if self._owns_client:
            await self.client.close()

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_class(event.size.width)

    def _update_responsive_class(self, width: int) -> None:
        self.set_class(width <= 120, "compact")
        self.set_class(width < 90, "narrow")

    def _setup_tables(self) -> None:
        self.query_one("#overview-table", DataTable).add_columns(
            "Category", "Count", "CPU %", "Memory", "Risk"
        )
        self.query_one("#process-table", DataTable).add_columns(
            "PID", "Category", "Tool", "CPU %", "Memory", "Uptime", "Risk"
        )
        self.query_one("#resident-models-table", DataTable).add_columns(
            "State", "Model", "Tok/s", "TTFT", "Context", "Layers", "Evicts in"
        )
        self.query_one("#available-models-table", DataTable).add_columns(
            "Type", "Model", "Disk", "Max ctx", "Est tok/s", "Last run", "Runs"
        )
        self.query_one("#network-table", DataTable).add_columns(
            "Port", "PID", "Process", "Category"
        )

    def _trigger_refresh(self) -> None:
        self.refresh_data()

    def _trigger_store_maintenance(self) -> None:
        self.maintain_store()

    @work(exclusive=True, group="store-maintenance")
    async def maintain_store(self) -> None:
        await asyncio.to_thread(self.store.recover_stale_requests)
        await asyncio.to_thread(self.store.cleanup, 30)

    @work(exclusive=True)
    async def refresh_data(self) -> None:
        await self._do_refresh()

    async def _do_refresh(self) -> None:
        system_stats, agents, ollama_status = await asyncio.gather(
            asyncio.to_thread(collect_system_stats),
            asyncio.to_thread(collect_agent_processes),
            self.client.poll(),
        )
        model_names = [model.name for model in ollama_status.available_models]
        ports, metrics, pins = await asyncio.gather(
            asyncio.to_thread(collect_listening_ports, agents),
            asyncio.to_thread(self.store.all_model_metrics, model_names),
            asyncio.to_thread(self.store.pinned_models),
        )
        self._agents = agents
        self._system_stats = system_stats
        self._ollama_status = ollama_status
        self._model_metrics = metrics
        self._pinned_models = pins
        ollama_status.in_flight = sum(metric.in_flight for metric in metrics.values())
        ollama_status.queue_depth = sum(metric.queue_depth for metric in metrics.values())
        self.query_one(TopBar).update_stats(system_stats, ollama_status, agents)
        await self._update_model_events(ollama_status)
        self._update_overview_table(agents)
        self._update_process_table(agents)
        self._update_models_tables(ollama_status)
        self._update_network_table(ports)
        self._refresh_force_kill_button(agents)

    def _update_overview_table(self, agents: list[AgentProcess]) -> None:
        table = self.query_one("#overview-table", DataTable)
        table.clear()
        by_category: dict[Category, list[AgentProcess]] = defaultdict(list)
        for agent in agents:
            by_category[agent.category].append(agent)
        for category, processes in sorted(by_category.items(), key=lambda item: -len(item[1])):
            highest_risk = max(processes, key=lambda process: RISK_ORDER[process.risk]).risk
            table.add_row(
                category.value,
                str(len(processes)),
                f"{sum(process.cpu_percent for process in processes):.1f}",
                _fmt_mem(sum(process.mem_mb for process in processes)),
                _risk_text(highest_risk),
                key=category.value,
            )

    def _update_process_table(self, agents: list[AgentProcess]) -> None:
        table = self.query_one("#process-table", DataTable)
        table.clear()
        for agent in sorted(agents, key=lambda process: -process.cpu_percent):
            table.add_row(
                str(agent.pid),
                agent.category.value,
                agent.subtype,
                f"{agent.cpu_percent:.1f}",
                _fmt_mem(agent.mem_mb),
                _fmt_uptime(agent.uptime_seconds),
                _risk_text(agent.risk),
                key=str(agent.pid),
            )

    def _filtered_available_models(self) -> list[OllamaAvailableModel]:
        query = self._model_filter.casefold()
        loaded_names = {model.name for model in self._ollama_status.loaded_models}
        models = [
            model
            for model in self._ollama_status.available_models
            if model.name not in loaded_names
        ]
        if not query:
            return models
        return [
            model
            for model in models
            if query
            in " ".join(
                (
                    model.name,
                    model.family,
                    model.parameter_size,
                    model.quantization,
                )
            ).casefold()
        ]

    def _update_models_tables(self, ollama: OllamaStatus) -> None:
        self._rebuilding_model_tables = True
        try:
            resident = self.query_one("#resident-models-table", DataTable)
            resident.clear()
            for model in ollama.loaded_models:
                metrics = self._model_metrics.get(
                    model.name, ModelMetrics(model=model.name)
                )
                generating = metrics.in_flight > 0
                state = Text(
                    "● generating" if generating else "○ idle",
                    style="#5EE6A8" if generating else "#5C6A72",
                )
                context_max = model.context or metrics.last_context_window
                context_used = metrics.last_context_used
                context = (
                    f"{_fmt_tokens(context_used)}/{_fmt_tokens(context_max)}"
                    if context_max
                    else "-"
                )
                if model.total_layers:
                    layer_style = (
                        "#5EE6A8"
                        if model.gpu_layers >= model.total_layers
                        else "#E0A65C"
                    )
                    layers: str | Text = Text(
                        f"{model.gpu_layers}/{model.total_layers}",
                        style=layer_style,
                    )
                else:
                    layers = model.processor
                resident.add_row(
                    state,
                    model.name,
                    _fmt_rate(metrics.generation_tps),
                    (
                        f"{metrics.ttft_p50_ms:.0f} ms"
                        if metrics.ttft_p50_ms is not None
                        else "-"
                    ),
                    context,
                    layers,
                    _fmt_remaining(model.expires_at),
                    key=model.name,
                )
            total_memory = sum(model.memory_gb for model in ollama.loaded_models)
            self.query_one("#resident-heading", Label).update(
                f"RESIDENT  {len(ollama.loaded_models)} models · {total_memory:.1f} GB held"
            )
            empty = self.query_one("#resident-empty", Static)
            empty.display = not ollama.loaded_models
            resident.display = bool(ollama.loaded_models)

            available = self.query_one("#available-models-table", DataTable)
            available.clear()
            filtered = self._filtered_available_models()
            for model in filtered:
                metrics = self._model_metrics.get(
                    model.name, ModelMetrics(model=model.name)
                )
                available.add_row(
                    _model_kind_text(model),
                    model.name,
                    f"{model.size_gb:.1f} GB",
                    _fmt_tokens(model.context) if model.context else "-",
                    _fmt_rate(metrics.generation_tps, estimated=True),
                    (
                        _fmt_age(
                            datetime.fromtimestamp(
                                metrics.last_run_at, timezone.utc
                            ).isoformat()
                        )
                        if metrics.last_run_at
                        else "-"
                    ),
                    str(metrics.run_count),
                    key=model.name,
                )
        finally:
            self._rebuilding_model_tables = False
        loaded_names = {model.name for model in ollama.loaded_models}
        cold_models = [
            model
            for model in ollama.available_models
            if model.name not in loaded_names
        ]
        total_disk = sum(model.size_gb for model in cold_models)
        filter_suffix = f" · {len(filtered)} shown" if self._model_filter else ""
        self.query_one("#available-heading", Label).update(
            f"AVAILABLE  {len(cold_models)} cold · {total_disk:.1f} GB on disk{filter_suffix}"
        )

        valid_names = {model.name for model in ollama.available_models}
        valid_names.update(model.name for model in ollama.loaded_models)
        if self._selected_model_name not in valid_names:
            self._selected_model_name = (
                filtered[0].name
                if filtered
                else ollama.loaded_models[0].name
                if ollama.loaded_models
                else None
            )
        if self._selected_model_name:
            resident_keys = {key.value for key in resident.rows}
            available_keys = {key.value for key in available.rows}
            if self._selected_model_name in resident_keys:
                resident.move_cursor(
                    row=resident.get_row_index(self._selected_model_name),
                    scroll=False,
                )
            if self._selected_model_name in available_keys:
                available.move_cursor(
                    row=available.get_row_index(self._selected_model_name),
                    scroll=False,
                )
        self._update_model_details()

    def _selected_model(self) -> tuple[OllamaAvailableModel | OllamaModel | None, bool]:
        if self._selected_model_name is None:
            return None, False
        for model in self._ollama_status.loaded_models:
            if model.name == self._selected_model_name:
                return model, True
        for model in self._ollama_status.available_models:
            if model.name == self._selected_model_name:
                return model, False
        return None, False

    def _update_model_details(self) -> None:
        model, resident = self._selected_model()
        details = self.query_one("#model-details", Static)
        throughput = self.query_one("#throughput-detail", Static)
        context_bar = self.query_one("#context-bar", ProgressBar)
        context_detail = self.query_one("#context-detail", Static)
        context_warning = self.query_one("#context-warning", Static)
        placement_bar = self.query_one("#placement-bar", ProgressBar)
        placement_detail = self.query_one("#placement-detail", Static)
        reliability_heading = self.query_one("#reliability-heading", Label)
        reliability = self.query_one("#reliability-detail", Static)
        sessions_heading = self.query_one("#sessions-heading", Label)
        sessions = self.query_one("#sessions-detail", Static)
        warm_button = self.query_one("#btn-model-warm", Button)
        unload_button = self.query_one("#btn-model-unload", Button)
        pin_button = self.query_one("#btn-model-pin", Button)
        trim_button = self.query_one("#btn-model-trim", Button)
        if model is None:
            details.update("Select a model to inspect it.")
            throughput.update("")
            context_bar.update(progress=0)
            context_detail.update("")
            context_warning.update("")
            placement_bar.update(progress=0)
            placement_detail.update("")
            reliability.display = reliability_heading.display = False
            sessions.display = sessions_heading.display = False
            for button in (warm_button, unload_button, pin_button, trim_button):
                button.disabled = True
            return

        metadata = next(
            (
                available
                for available in self._ollama_status.available_models
                if available.name == model.name
            ),
            None,
        )
        metrics = self._model_metrics.get(
            model.name, ModelMetrics(model=model.name)
        )
        size_gb = model.size_gb
        state = (
            "GENERATING"
            if metrics.in_flight
            else "IDLE"
            if resident
            else "COLD"
        )
        state_color = "#5EE6A8" if resident else "#8FA9FF"
        description = (
            metadata.description
            if metadata and metadata.description
            else f"{model.parameter_size} {model.family} model".strip()
        )
        details.update(
            f"[bold #E8EEF1]{escape(model.name)}[/bold #E8EEF1]\n"
            f"[#8A98A0]{escape(description or 'Local Ollama model')}[/#8A98A0]\n"
            f"[bold {state_color}]{state}[/bold {state_color}]"
            f"{'  PINNED' if model.name in self._pinned_models else ''}\n\n"
            f"[#46525A]PARAMS[/#46525A] [bold #E8EEF1]{escape(model.parameter_size or '-')}[/bold #E8EEF1]   "
            f"[#46525A]QUANT[/#46525A] [bold #E8EEF1]{escape(_model_quantization(model))}[/bold #E8EEF1]\n\n"
            f"[#46525A]FAMILY[/#46525A] [bold #E8EEF1]{escape(model.family or '-')}[/bold #E8EEF1]\n"
            f"[#46525A]DISK[/#46525A] [bold #E8EEF1]{size_gb:.1f} GB[/bold #E8EEF1]"
        )

        has_history = metrics.run_count > 0
        throughput.update(
            (
                f"[#46525A]GEN[/#46525A] [bold #5EE6A8]{_fmt_rate(metrics.generation_tps)}[/bold #5EE6A8]   "
                f"[#46525A]PROMPT[/#46525A] {_fmt_rate(metrics.prompt_tps)}   "
                f"[#46525A]TTFT[/#46525A] "
                f"{metrics.ttft_p50_ms:.0f} ms p50\n"
                f"[#46525A]IN[/#46525A] {_fmt_tokens(metrics.tokens_in_60s)}   "
                f"[#46525A]OUT[/#46525A] {_fmt_tokens(metrics.tokens_out_60s)}   "
                f"[#46525A]IN FLIGHT[/#46525A] {metrics.in_flight}   "
                f"[#46525A]QUEUE[/#46525A] {metrics.queue_depth}"
                if has_history and metrics.ttft_p50_ms is not None
                else "[#5C6A72]Observed completion data unavailable.\n"
                "Run clients through `agentop proxy` to populate this section.[/#5C6A72]"
            )
        )

        context_max = (
            model.context
            if isinstance(model, OllamaModel)
            else model.context
            or metrics.last_context_window
        )
        context_used = metrics.last_context_used if resident else 0
        context_percent = (
            min(100.0, context_used / context_max * 100) if context_max else 0.0
        )
        context_bar.update(progress=context_percent)
        context_detail.update(
            (
                f"{'est ' if not resident else ''}{_fmt_tokens(context_used)} tokens of "
                f"{_fmt_tokens(context_max)} window   "
                "KV cache not exposed by Ollama"
                if context_max
                else "Context window unavailable"
            )
        )
        if context_percent > 70:
            context_warning.update(
                f"[#E0A65C]{context_percent:.0f}% full — oldest turns may be dropped soon.[/#E0A65C]"
            )
        else:
            context_warning.update("")

        if isinstance(model, OllamaModel):
            gpu_percent = (
                model.gpu_memory_gb / model.memory_gb * 100
                if model.memory_gb
                else 0
            )
            placement_bar.update(progress=gpu_percent)
            ram_gb = max(0.0, model.memory_gb - model.gpu_memory_gb)
            layers = (
                f"GPU {model.gpu_layers} layers · CPU "
                f"{max(0, model.total_layers - model.gpu_layers)} layers\n"
                if model.total_layers
                else f"{model.processor}\n"
            )
            placement_detail.update(
                f"{layers}VRAM {model.gpu_memory_gb:.1f} GB   RAM {ram_gb:.1f} GB\n"
                f"QUANT {_model_quantization(model)}   "
                f"BATCH {metrics.batch_size or '-'}"
            )
        else:
            placement_bar.update(progress=0)
            placement_detail.update(
                f"[#5C6A72]est placement determined on load\n"
                f"QUANT {_model_quantization(model)}[/#5C6A72]"
            )

        reliability_heading.display = reliability.display = has_history
        sessions_heading.display = sessions.display = bool(metrics.sessions)
        if has_history:
            oom = (
                f"[#E0A65C]{metrics.oom_count}[/#E0A65C]"
                if metrics.oom_count
                else "0"
            )
            timeouts = (
                f"[#E0A65C]{metrics.timeout_count}[/#E0A65C]"
                if metrics.timeout_count
                else "0"
            )
            overflows = (
                f"[#E0A65C]{metrics.context_overflow_count}[/#E0A65C]"
                if metrics.context_overflow_count
                else "0"
            )
            reliability.update(
                f"p50 latency {metrics.latency_p50_ms or 0:.0f} ms   "
                f"p95 latency {metrics.latency_p95_ms or 0:.0f} ms\n"
                f"success {metrics.success_rate or 0:.1f}%   "
                f"OOM {oom}   timeouts {timeouts}   "
                f"ctx overflow {overflows}"
            )
        if metrics.sessions:
            sessions.update(
                "\n".join(
                    f"{escape(session.client)}   {session.request_count} req · "
                    f"{_fmt_seconds(session.held_seconds)} held · "
                    f"idle {_fmt_seconds(session.idle_seconds)}"
                    for session in metrics.sessions
                )
            )

        warm_button.disabled = resident
        unload_button.disabled = not resident
        trim_button.disabled = not resident
        pin_button.disabled = False
        pin_button.label = "Unpin" if model.name in self._pinned_models else "Pin"

    async def _update_model_events(self, ollama: OllamaStatus) -> None:
        now = time.time()
        poll_error = ollama.error or ollama.loaded_models_error
        pending: list[tuple[str, str, str, str]] = []
        if poll_error:
            if not self._model_poll_failed:
                pending.append(
                    ("warn", "poll", "", "resident model status unavailable")
                )
            self._model_poll_failed = True
        else:
            if self._model_poll_failed:
                pending.append(("ok", "poll", "", "resident model status restored"))
            self._model_poll_failed = False
            loaded_names = {model.name for model in ollama.loaded_models}
            if self._previous_loaded_names is None:
                pending.append(("ok", "monitor", "", "model monitor connected"))
            else:
                for name in sorted(loaded_names - self._previous_loaded_names):
                    pending.append(("ok", "load", name, f"{name} became resident"))
                for name in sorted(self._previous_loaded_names - loaded_names):
                    pending.append(
                        ("warn", "unload", name, f"{name} left memory")
                    )
            self._previous_loaded_names = loaded_names
        for level, kind, model_name, message in pending:
            await asyncio.to_thread(
                self.store.record_event,
                level,
                kind,
                message,
                model=model_name,
                timestamp=now,
            )
        events = await asyncio.to_thread(self.store.recent_events, 8)
        self._render_model_events(events)

    def _render_model_events(self, events: list[EventRecord]) -> None:
        lines = [
            (
                f"[#3D474D]{datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')}[/#3D474D]  "
                f"[{_EVENT_STYLE.get(event.kind, '#8FA9FF')}]{event.kind:<7}"
                f"[/{_EVENT_STYLE.get(event.kind, '#8FA9FF')}] {escape(event.message)}"
            )
            for event in events
        ]
        self.query_one("#model-events", Static).update(
            "\n".join(lines) if lines else "Waiting for model activity."
        )

    def _update_network_table(self, ports: list[PortInfo]) -> None:
        table = self.query_one("#network-table", DataTable)
        table.clear()
        for port in ports:
            category = port.category.value if port.category else "-"
            table.add_row(
                str(port.port),
                str(port.pid),
                port.process_name,
                category,
                key=f"{port.port}:{port.pid}",
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key.value is None:
            return
        if event.data_table.id == "process-table":
            pid = int(event.row_key.value)
            self._selected_agent = next(
                (agent for agent in self._agents if agent.pid == pid),
                None,
            )
        elif event.data_table.id == "overview-table":
            self._selected_category = Category(event.row_key.value)
        elif event.data_table.id in {"resident-models-table", "available-models-table"}:
            if self._rebuilding_model_tables:
                return
            self._selected_model_name = str(event.row_key.value)
            self._update_model_details()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        event.data_table.sort(event.column_key)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "model-filter":
            return
        self._model_filter = event.value.strip()
        self._update_models_tables(self._ollama_status)

    def action_show_tab(self, tab_id: str) -> None:
        self.remove_class("detail-open")
        self.query_one("#main-tabs", TabbedContent).active = tab_id

    def action_focus_model_filter(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.query_one("#model-filter", Input).focus()

    def action_refresh_now(self) -> None:
        self._trigger_refresh()

    def action_model_warm(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active != "tab-models":
            return
        if self.has_class("compact") and not self.has_class("detail-open"):
            self.add_class("detail-open")
            return
        self.run_worker(self._warm_selected_model())

    def action_close_detail(self) -> None:
        self.remove_class("detail-open")

    def action_context_kill(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.run_worker(self._unload_selected_model())
        else:
            self.action_kill_selected()

    def action_context_kill_all(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.run_worker(self._unload_all_models())
        else:
            self.action_kill_switch_category()

    def action_trim_context(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.run_worker(self._trim_selected_context())

    def action_toggle_pin(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.run_worker(self._toggle_selected_pin())

    def action_undo_unload(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.run_worker(self._undo_last_unload())

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def _selected_model_size(self) -> float:
        model, _ = self._selected_model()
        return model.size_gb if model else 0.0

    def _fit_plan_for_selected(self) -> FitPlan | None:
        model, resident = self._selected_model()
        if model is None or resident:
            return None
        metrics = self._model_metrics.get(
            model.name, ModelMetrics(model=model.name)
        )
        estimated_load = (
            metrics.latency_p50_ms / 1000
            if metrics.latency_p50_ms is not None
            else max(1.0, model.size_gb / 1.5)
        )
        return calculate_fit_plan(
            model=model.name,
            required_gb=model.size_gb * self.config.model_memory_overhead,
            used_gb=self._system_stats.mem_used_gb,
            total_gb=self._system_stats.mem_total_gb,
            loaded_models=self._ollama_status.loaded_models,
            pinned_models=self._pinned_models,
            headroom_gb=self.config.memory_headroom_gb,
            estimated_load_seconds=estimated_load,
        )

    async def _record_action(
        self,
        level: str,
        kind: str,
        message: str,
        model: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.store.record_event,
                level,
                kind,
                message,
                model=model,
            )
        except (sqlite3.Error, OSError):
            return

    async def _warm_selected_model(self) -> None:
        model, resident = self._selected_model()
        if model is None:
            self.notify("Select a model first.", severity="warning")
            return
        if resident:
            self.notify(f"{model.name} is already resident.", severity="information")
            return
        plan = self._fit_plan_for_selected()
        if plan is None:
            return
        def signature(value: FitPlan) -> tuple:
            return (
                tuple(value.evictions),
                value.will_page,
                round(value.free_gb, 1),
                round(value.required_gb, 1),
            )

        approved_signature = None
        if plan.requires_confirmation:
            confirmed = await self.push_screen_wait(PreflightScreen(plan))
            if not confirmed:
                return
            approved_signature = signature(plan)
        async with self._model_mutation_lock:
            execution_plan = None
            for _ in range(3):
                fresh, fresh_system = await asyncio.gather(
                    self.client.poll(),
                    asyncio.to_thread(collect_system_stats),
                )
                if not fresh.online or fresh.stale:
                    self.notify(
                        "Warm cancelled: a fresh Ollama state snapshot is required.",
                        severity="warning",
                    )
                    return
                self._ollama_status = fresh
                self._system_stats = fresh_system
                candidate = self._fit_plan_for_selected()
                if candidate is None:
                    self.notify(f"{model.name} is already resident.")
                    return
                candidate_signature = signature(candidate)
                if (
                    not candidate.requires_confirmation
                    or candidate_signature == approved_signature
                ):
                    execution_plan = candidate
                    break
                confirmed = await self.push_screen_wait(
                    PreflightScreen(candidate)
                )
                if not confirmed:
                    return
                approved_signature = candidate_signature

            if execution_plan is None:
                self.notify(
                    "Warm cancelled: model residency changed repeatedly during confirmation.",
                    severity="warning",
                )
                return

            loaded_by_name = {
                loaded.name: loaded
                for loaded in self._ollama_status.loaded_models
            }
            evicted: list[OllamaModel] = []
            target_loaded = False
            try:
                for eviction in execution_plan.evictions:
                    await self.client.unload_model(eviction)
                    if eviction in loaded_by_name:
                        evicted.append(loaded_by_name[eviction])
                    await self._record_action(
                        "warn",
                        "unload",
                        f"{eviction} evicted by confirmed pre-flight plan",
                        eviction,
                    )
                keep_alive: str | int = (
                    -1 if model.name in self._pinned_models else "5m"
                )
                started = time.monotonic()
                await self.client.warm_model(model.name, keep_alive=keep_alive)
                target_loaded = True
                elapsed = time.monotonic() - started
                await self._record_action(
                    "ok",
                    "load",
                    f"{model.name} warmed in {elapsed:.1f} s",
                    model.name,
                )
            except asyncio.CancelledError:
                if not target_loaded:
                    await asyncio.shield(self._restore_evicted_models(evicted))
                raise
            except (httpx.HTTPError, OSError, ValueError) as exc:
                await self._restore_evicted_models(evicted)
                await self._record_action(
                    "error",
                    "load",
                    f"{model.name} warm failed: {exc}",
                    model.name,
                )
                self.notify(f"Warm failed; prior models restored where possible: {exc}", severity="error")
                return
        self.notify(f"{model.name} is resident.")
        self.refresh_data()

    async def _restore_evicted_models(
        self,
        evicted: list[OllamaModel],
    ) -> None:
        for model in evicted:
            try:
                await self.client.warm_model(
                    model.name,
                    keep_alive=(
                        -1
                        if model.name in self._pinned_models
                        else _remaining_keep_alive(model.expires_at)
                    ),
                    context=model.context or None,
                )
            except (httpx.HTTPError, OSError, ValueError):
                await self._record_action(
                    "error",
                    "rollback",
                    f"failed to restore {model.name} after warm failure",
                    model.name,
                )

    async def _unload_selected_model(self) -> None:
        model, resident = self._selected_model()
        if model is None:
            self.notify("Select a model first.", severity="warning")
            return
        if not resident:
            self.notify(f"{model.name} is already cold.", severity="information")
            return
        async with self._model_mutation_lock:
            try:
                await self.client.unload_model(model.name)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                self.notify(f"Unload failed: {exc}", severity="error")
                return
        self._undo_model = (model.name, time.monotonic() + 5.0)
        await self._record_action(
            "warn",
            "unload",
            f"{model.name} unloaded by user; undo available for 5 s",
            model.name,
        )
        self.notify(f"Unloaded {model.name}. Press u within 5 s to undo.")
        self.refresh_data()

    async def _undo_last_unload(self) -> None:
        if self._undo_model is None or time.monotonic() > self._undo_model[1]:
            self._undo_model = None
            self.notify("No unload is available to undo.", severity="information")
            return
        model = self._undo_model[0]
        self._undo_model = None
        async with self._model_mutation_lock:
            try:
                await self.client.warm_model(
                    model,
                    keep_alive=-1 if model in self._pinned_models else "5m",
                )
            except (httpx.HTTPError, OSError, ValueError) as exc:
                self.notify(f"Undo failed: {exc}", severity="error")
                return
        await self._record_action("ok", "load", f"{model} unload undone", model)
        self.notify(f"Restored {model}.")
        self.refresh_data()

    async def _unload_all_models(self) -> None:
        loaded = list(self._ollama_status.loaded_models)
        if not loaded:
            self.notify("No resident models.", severity="information")
            return
        confirmed = await self.push_screen_wait(
            ConfirmTextScreen(
                "UNLOAD ALL MODELS",
                f"This will unload {len(loaded)} resident model(s), including pinned models.",
                "yes",
            )
        )
        if not confirmed:
            return
        failures = []
        async with self._model_mutation_lock:
            fresh = await self.client.poll()
            if fresh.online:
                loaded = list(fresh.loaded_models)
                self._ollama_status = fresh
            for model in loaded:
                try:
                    await self.client.unload_model(model.name)
                    await self._record_action(
                        "warn",
                        "unload",
                        f"{model.name} unloaded by unload-all",
                        model.name,
                    )
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    failures.append(f"{model.name}: {exc}")
        if failures:
            self.notify("; ".join(failures[:3]), severity="error")
        else:
            self.notify(f"Unloaded {len(loaded)} models.")
        self.refresh_data()

    async def _toggle_selected_pin(self) -> None:
        model, resident = self._selected_model()
        if model is None:
            self.notify("Select a model first.", severity="warning")
            return
        pinned = model.name not in self._pinned_models
        async with self._model_mutation_lock:
            runtime_changed = False
            if resident:
                try:
                    await self.client.pin_model(model.name, pinned)
                    runtime_changed = True
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    self.notify(f"Pin update failed: {exc}", severity="error")
                    return
            try:
                await asyncio.to_thread(
                    self.store.set_pinned, model.name, pinned
                )
            except (sqlite3.Error, OSError) as exc:
                if runtime_changed:
                    try:
                        await self.client.pin_model(model.name, not pinned)
                    except (httpx.HTTPError, OSError, ValueError):
                        self.notify(
                            "Pin persistence failed and runtime rollback also failed.",
                            severity="error",
                        )
                        return
                self.notify(f"Pin persistence failed: {exc}", severity="error")
                return
        await self._record_action(
            "ok",
            "pin",
            f"{model.name} {'pinned' if pinned else 'unpinned'}",
            model.name,
        )
        self._pinned_models = await asyncio.to_thread(self.store.pinned_models)
        self._update_model_details()

    async def _trim_selected_context(self) -> None:
        model, resident = self._selected_model()
        if not isinstance(model, OllamaModel) or not resident:
            self.notify("Select a resident model to trim.", severity="warning")
            return
        if model.context <= 2048:
            self.notify("Context is already at the minimum trim target.")
            return
        metrics = self._model_metrics.get(
            model.name, ModelMetrics(model=model.name)
        )
        if metrics.in_flight:
            confirmed = await self.push_screen_wait(
                ConfirmTextScreen(
                    "RELOAD WITH SMALLER DEFAULT CONTEXT",
                    f"{metrics.in_flight} request(s) are active. This reload changes "
                    "the runner default only; existing conversations keep their own "
                    "request context.",
                    "yes",
                )
            )
            if not confirmed:
                return
        target = max(2048, model.context // 2)
        async with self._model_mutation_lock:
            try:
                await self.client.warm_model(
                    model.name,
                    keep_alive=-1 if model.name in self._pinned_models else "5m",
                    context=target,
                )
            except (httpx.HTTPError, OSError, ValueError) as exc:
                self.notify(f"Context reload failed: {exc}", severity="error")
                return
        await self._record_action(
            "ok",
            "reload",
            f"{model.name} reloaded with default context {target} tokens",
            model.name,
        )
        self.notify(
            f"Reloaded {model.name} with default context {target} tokens."
        )
        self.refresh_data()

    def action_kill_selected(self) -> None:
        self.run_worker(self._run_kill_flow(KillScope.SINGLE))

    def action_kill_switch_session(self) -> None:
        self.run_worker(self._run_kill_flow(KillScope.SESSION))

    def action_kill_switch_category(self) -> None:
        self.run_worker(self._run_kill_flow(KillScope.CATEGORY))

    def action_kill_switch_all(self) -> None:
        self.run_worker(self._run_kill_flow(KillScope.ALL))

    async def _run_kill_flow(self, scope: KillScope, *, force: bool = False) -> None:
        pid = (
            self._selected_agent.pid
            if scope is KillScope.SINGLE and self._selected_agent
            else None
        )
        session_key = (
            self._selected_agent.session_key
            if scope is KillScope.SESSION and self._selected_agent
            else None
        )
        category = None
        if scope is KillScope.CATEGORY:
            category = self._selected_category or (
                self._selected_agent.category if self._selected_agent else None
            )

        if scope is KillScope.SINGLE and pid is None:
            self.notify("No process selected in the Processes tab.", severity="warning")
            return
        if scope is KillScope.SESSION and session_key is None:
            self.notify("No process selected to determine a session.", severity="warning")
            return
        if scope is KillScope.CATEGORY and category is None:
            self.notify("Select a row in Overview or Processes first.", severity="warning")
            return

        plan = build_kill_plan(
            scope,
            self._agents,
            pid=pid,
            session_key=session_key,
            category=category,
        )
        if plan.count == 0:
            self.notify("Nothing to kill (already gone).", severity="information")
            return

        confirmed = await self.push_screen_wait(
            ConfirmKillScreen(plan, force=force)
        )
        if not confirmed:
            return

        results = await asyncio.to_thread(execute_kill, plan, force=force)
        succeeded = sum(1 for result in results if result.success)
        failed = [result for result in results if not result.success]
        message = f"Killed {succeeded}/{len(results)} process(es)."
        if failed:
            message += (
                " Failed: "
                + ", ".join(
                    f"{result.pid} ({result.error})" for result in failed[:5]
                )
            )
        self.notify(
            message,
            severity="information" if not failed else "warning",
        )

        if not force:
            self._pending_force_targets = list(plan.targets)
        self._trigger_refresh()

    def _refresh_force_kill_button(self, agents: list[AgentProcess]) -> None:
        if not self._pending_force_targets:
            return
        live_by_pid = {agent.pid: agent for agent in agents}
        still_alive = []
        for target in self._pending_force_targets:
            live = live_by_pid.get(target.pid)
            if live is not None and live.create_time == target.create_time:
                still_alive.append(live)
        self._pending_force_targets = still_alive
        button = self.query_one("#btn-force-kill", Button)
        button.label = f"Force Kill Remaining ({len(still_alive)})"
        button.disabled = not still_alive

    async def _force_kill_remaining(self) -> None:
        if not self._pending_force_targets:
            self.notify("Nothing pending.", severity="information")
            return
        plan = build_kill_plan(KillScope.ALL, self._pending_force_targets)
        confirmed = await self.push_screen_wait(ConfirmKillScreen(plan, force=True))
        if not confirmed:
            return
        results = await asyncio.to_thread(execute_kill, plan, force=True)
        succeeded = sum(1 for result in results if result.success)
        self.notify(f"Force-killed {succeeded}/{len(results)} process(es).")
        self._pending_force_targets = []
        button = self.query_one("#btn-force-kill", Button)
        button.label = "Force Kill Remaining (0)"
        button.disabled = True
        self._trigger_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-refresh":
            self._trigger_refresh()
        elif button_id == "btn-kill-selected":
            self.run_worker(self._run_kill_flow(KillScope.SINGLE))
        elif button_id == "btn-kill-session":
            self.run_worker(self._run_kill_flow(KillScope.SESSION))
        elif button_id == "btn-kill-category":
            self.run_worker(self._run_kill_flow(KillScope.CATEGORY))
        elif button_id == "btn-kill-all":
            self.run_worker(self._run_kill_flow(KillScope.ALL))
        elif button_id == "btn-force-kill":
            self.run_worker(self._force_kill_remaining())
        elif button_id == "btn-model-warm":
            self.run_worker(self._warm_selected_model())
        elif button_id == "btn-model-unload":
            self.run_worker(self._unload_selected_model())
        elif button_id == "btn-model-pin":
            self.run_worker(self._toggle_selected_pin())
        elif button_id == "btn-model-trim":
            self.run_worker(self._trim_selected_context())

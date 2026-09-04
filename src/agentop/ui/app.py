"""agentop's Textual application."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone

from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, HorizontalScroll, Vertical
from textual.widgets import Button, DataTable, Footer, Input, Label, ProgressBar, Static, TabbedContent, TabPane

from agentop.collectors.network import collect_listening_ports
from agentop.collectors.ollama import collect_ollama_status
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import collect_system_stats
from agentop.control import KillScope, build_kill_plan, execute_kill
from agentop.models import (
    AgentProcess,
    Category,
    OllamaAvailableModel,
    OllamaModel,
    OllamaStatus,
    PortInfo,
    Risk,
    RISK_ORDER,
    SystemStats,
)
from agentop.ui.screens import ConfirmKillScreen

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
                yield Label("Apple Metal", classes="metric-detail")
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
        self.query_one("#cpu-detail", Label).update(f"{len(agents)} agent processes")

        self.query_one("#gpu-bar", ProgressBar).update(
            progress=stats.gpu_percent or 0.0
        )
        self.query_one("#gpu-value", Label).update(
            f"{stats.gpu_percent:.1f}%" if stats.gpu_percent is not None else "N/A"
        )

        self.query_one("#memory-percent", Label).update(f"{stats.mem_percent:.1f}%")
        self.query_one("#memory-bar", ProgressBar).update(progress=stats.mem_percent)
        self.query_one("#memory-detail", Label).update(
            f"{stats.mem_used_gb:.1f} / {stats.mem_total_gb:.1f} GB"
        )

        self.query_one("#swap-percent", Label).update(f"{stats.swap_percent:.1f}%")
        self.query_one("#swap-bar", ProgressBar).update(progress=stats.swap_percent)
        self.query_one("#swap-detail", Label).update(
            f"{stats.swap_used_gb:.1f} / {stats.swap_total_gb:.1f} GB"
        )

        title = self.query_one("#ollama-title", Label)
        if ollama.online:
            title.update(f"● ollama v{ollama.version}")
            title.set_classes("online")
        else:
            title.update("● ollama offline")
            title.set_classes("offline")
        detail = (
            "model status unavailable"
            if ollama.error or ollama.loaded_models_error
            else f"{len(ollama.loaded_models)} loaded · {len(ollama.available_models)} available"
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
        Binding("k", "kill_selected", "Kill Selected"),
        Binding("shift+s", "kill_switch_session", "Kill Session"),
        Binding("shift+k", "kill_switch_category", "Kill Category"),
        Binding("shift+a", "kill_switch_all", "KILL ALL"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        refresh_interval: float = REFRESH_INTERVAL,
    ) -> None:
        super().__init__()
        self.ollama_url = ollama_url
        self.refresh_interval = refresh_interval
        self._agents: list[AgentProcess] = []
        self._system_stats = SystemStats()
        self._ollama_status = OllamaStatus(online=False)
        self._selected_agent: AgentProcess | None = None
        self._selected_category: Category | None = None
        self._selected_model_name: str | None = None
        self._model_filter = ""
        self._pending_force_targets: list[AgentProcess] = []
        self._previous_loaded_names: set[str] | None = None
        self._model_poll_failed = False
        self._rebuilding_model_tables = False
        self._model_events: deque[tuple[str, str, str]] = deque(maxlen=8)

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
                    with Vertical(id="model-sidebar"):
                        yield Label("SELECTED", classes="sidebar-heading")
                        yield Static("Select a model to inspect it.", id="model-details")
                        yield Label("FIT ON THIS MACHINE", classes="sidebar-heading")
                        yield ProgressBar(
                            total=100,
                            show_percentage=False,
                            show_eta=False,
                            id="model-fit-bar",
                        )
                        yield Static("", id="model-fit-detail")
                        yield Static("", id="model-fit-warning")
                        yield Label("RECENT EVENTS", classes="sidebar-heading")
                        yield Static("Waiting for model activity.", id="model-events")
            with TabPane("4  Network", id="tab-network"):
                yield DataTable(id="network-table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self.set_interval(self.refresh_interval, self._trigger_refresh)
        self._update_responsive_class(self.size.width)
        self._trigger_refresh()

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_class(event.size.width)

    def _update_responsive_class(self, width: int) -> None:
        self.set_class(width < 108, "compact")
        self.set_class(width < 90, "narrow")

    def _setup_tables(self) -> None:
        self.query_one("#overview-table", DataTable).add_columns(
            "Category", "Count", "CPU %", "Memory", "Risk"
        )
        self.query_one("#process-table", DataTable).add_columns(
            "PID", "Category", "Tool", "CPU %", "Memory", "Uptime", "Risk"
        )
        self.query_one("#resident-models-table", DataTable).add_columns(
            "Model", "Processor", "Memory", "Context", "Expires"
        )
        self.query_one("#available-models-table", DataTable).add_columns(
            "Type", "Model", "Disk", "Params", "Quant", "Modified"
        )
        self.query_one("#network-table", DataTable).add_columns(
            "Port", "PID", "Process", "Category"
        )

    def _trigger_refresh(self) -> None:
        self.refresh_data()

    @work(exclusive=True)
    async def refresh_data(self) -> None:
        await self._do_refresh()

    async def _do_refresh(self) -> None:
        system_stats, agents, ollama_status = await asyncio.gather(
            asyncio.to_thread(collect_system_stats),
            asyncio.to_thread(collect_agent_processes),
            asyncio.to_thread(collect_ollama_status, self.ollama_url),
        )
        ports = await asyncio.to_thread(collect_listening_ports, agents)

        if (
            (ollama_status.error or ollama_status.loaded_models_error)
            and self._ollama_status.loaded_models
        ):
            ollama_status.loaded_models = list(self._ollama_status.loaded_models)
        if ollama_status.error and self._ollama_status.available_models:
            ollama_status.available_models = list(
                self._ollama_status.available_models
            )
        self._agents = agents
        self._system_stats = system_stats
        self._ollama_status = ollama_status
        self.query_one(TopBar).update_stats(system_stats, ollama_status, agents)
        self._update_model_events(ollama_status)
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
        models = self._ollama_status.available_models
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
                resident.add_row(
                    model.name,
                    model.processor,
                    f"{model.memory_gb:.1f} GB",
                    str(model.context) if model.context else "-",
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
                available.add_row(
                    _model_kind_text(model),
                    model.name,
                    f"{model.size_gb:.1f} GB",
                    model.parameter_size or "-",
                    _model_quantization(model),
                    _fmt_age(model.modified_at),
                    key=model.name,
                )
        finally:
            self._rebuilding_model_tables = False
        total_disk = sum(model.size_gb for model in ollama.available_models)
        filter_suffix = f" · {len(filtered)} shown" if self._model_filter else ""
        self.query_one("#available-heading", Label).update(
            f"AVAILABLE  {len(ollama.available_models)} models · {total_disk:.1f} GB on disk{filter_suffix}"
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
        fit_bar = self.query_one("#model-fit-bar", ProgressBar)
        fit_detail = self.query_one("#model-fit-detail", Static)
        fit_warning = self.query_one("#model-fit-warning", Static)
        if model is None:
            details.update("Select a model to inspect it.")
            fit_bar.update(progress=0)
            fit_detail.update("")
            fit_warning.update("")
            return

        size_gb = model.size_gb
        memory_gb = model.memory_gb if isinstance(model, OllamaModel) else size_gb
        details.update(
            f"[bold #E8EEF1]{escape(model.name)}[/bold #E8EEF1]\n"
            f"[#8A98A0]{'Resident in memory' if resident else 'Available locally'}[/#8A98A0]\n\n"
            f"[#46525A]PARAMS[/#46525A] [bold #E8EEF1]{escape(model.parameter_size or '-')}[/bold #E8EEF1]   "
            f"[#46525A]QUANT[/#46525A] [bold #E8EEF1]{escape(_model_quantization(model))}[/bold #E8EEF1]\n\n"
            f"[#46525A]FAMILY[/#46525A] [bold #E8EEF1]{escape(model.family or '-')}[/bold #E8EEF1]\n"
            f"[#46525A]DISK[/#46525A] [bold #E8EEF1]{size_gb:.1f} GB[/bold #E8EEF1]\n\n"
            f"[#46525A]STATE[/#46525A]\n"
            f"[bold #C6D0D6]{'loaded · ' + model.processor if resident and isinstance(model, OllamaModel) else 'not resident'}[/bold #C6D0D6]"
        )

        current = self._system_stats.mem_used_gb
        projected = current if resident else current + memory_gb
        total = self._system_stats.mem_total_gb
        projected_percent = min(100.0, projected / total * 100) if total else 0.0
        fit_bar.update(progress=projected_percent)
        fit_detail.update(
            f"[#E0A65C]■[/#E0A65C] in use {current:.1f} GB   "
            f"[#5EE6A8]■[/#5EE6A8] this model {memory_gb:.1f} GB\n"
            f"projected {projected:.1f} / {total:.1f} GB"
        )
        if projected_percent >= 90:
            fit_warning.update(
                f"[bold #F0706E]memory at {projected_percent:.0f}% — expect heavy paging on load.[/bold #F0706E]"
            )
        elif projected_percent >= 75:
            fit_warning.update(
                f"[#E0A65C]memory at {projected_percent:.0f}% — limited headroom.[/#E0A65C]"
            )
        else:
            fit_warning.update("[#5EE6A8]comfortable memory headroom.[/#5EE6A8]")

    def _update_model_events(self, ollama: OllamaStatus) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        poll_error = ollama.error or ollama.loaded_models_error
        if poll_error:
            if not self._model_poll_failed:
                self._model_events.appendleft(
                    (now, "warn", "resident model status unavailable")
                )
            self._model_poll_failed = True
            self._render_model_events()
            return
        if self._model_poll_failed:
            self._model_events.appendleft((now, "ok", "resident model status restored"))
        self._model_poll_failed = False
        loaded_names = {model.name for model in ollama.loaded_models}
        if self._previous_loaded_names is None:
            self._model_events.appendleft((now, "ok", "model monitor connected"))
        else:
            for name in sorted(loaded_names - self._previous_loaded_names):
                self._model_events.appendleft((now, "load", f"{name} became resident"))
            for name in sorted(self._previous_loaded_names - loaded_names):
                self._model_events.appendleft((now, "unload", f"{name} left memory"))
        self._previous_loaded_names = loaded_names
        self._render_model_events()

    def _render_model_events(self) -> None:
        lines = [
            f"[#3D474D]{timestamp}[/#3D474D]  [{_EVENT_STYLE[kind]}]{kind:<6}[/{_EVENT_STYLE[kind]}] {escape(message)}"
            for timestamp, kind, message in self._model_events
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
        self.query_one("#main-tabs", TabbedContent).active = tab_id

    def action_focus_model_filter(self) -> None:
        if self.query_one("#main-tabs", TabbedContent).active == "tab-models":
            self.query_one("#model-filter", Input).focus()

    def action_refresh_now(self) -> None:
        self._trigger_refresh()

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

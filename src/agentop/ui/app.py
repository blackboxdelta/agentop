"""agentop's Textual application: the main btop-style TUI."""
from __future__ import annotations

import asyncio
from collections import defaultdict

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from agentop.collectors.network import collect_listening_ports
from agentop.collectors.ollama import collect_ollama_status
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import collect_system_stats
from agentop.control import KillScope, build_kill_plan, execute_kill
from agentop.models import AgentProcess, Category, OllamaStatus, PortInfo, Risk, RISK_ORDER, SystemStats
from agentop.ui.screens import ConfirmKillScreen

REFRESH_INTERVAL = 2.0

_RISK_STYLE = {
    Risk.LOW: "green",
    Risk.MEDIUM: "yellow",
    Risk.HIGH: "bold red",
}


def _risk_text(risk: Risk) -> Text:
    return Text(risk.value, style=_RISK_STYLE[risk])


def _fmt_mem(mem_mb: float) -> str:
    return f"{mem_mb:.0f} MB" if mem_mb < 1024 else f"{mem_mb / 1024:.2f} GB"


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class StatsBar(Static):
    """Top strip: CPU/GPU/memory sparklines, swap, and Ollama status."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="stats-row"):
            with Horizontal(classes="stat-block"):
                yield Label("CPU", classes="stat-label")
                yield Sparkline([], min_color="#1f6f78", max_color="#45d4a4", id="cpu-spark")
                yield Label("--%", id="cpu-value", classes="stat-value")
            with Horizontal(classes="stat-block"):
                yield Label("GPU", classes="stat-label")
                yield Sparkline([], min_color="#5847a8", max_color="#bd77ff", id="gpu-spark")
                yield Label("--%", id="gpu-value", classes="stat-value")
            with Horizontal(classes="stat-block"):
                yield Label("MEM", classes="stat-label")
                yield Sparkline([], min_color="#715225", max_color="#f0b15a", id="mem-spark")
                yield Label("--%", id="mem-value", classes="stat-value")
            yield Label("SWAP --%", id="swap-value", classes="stat-value")
            yield Label("OLLAMA: checking...", id="ollama-badge", classes="stat-value")

    def update_stats(self, stats: SystemStats, ollama: OllamaStatus) -> None:
        self.query_one("#cpu-spark", Sparkline).data = stats.cpu_history or [0.0]
        self.query_one("#cpu-value", Label).update(f"{stats.cpu_percent:5.1f}%")
        self.query_one("#gpu-spark", Sparkline).data = stats.gpu_history or [0.0]
        self.query_one("#gpu-value", Label).update(
            f"{stats.gpu_percent:5.1f}%" if stats.gpu_percent is not None else "  N/A"
        )
        self.query_one("#mem-spark", Sparkline).data = stats.mem_history or [0.0]
        self.query_one("#mem-value", Label).update(
            f"{stats.mem_percent:5.1f}% ({stats.mem_used_gb:.1f}/{stats.mem_total_gb:.1f}GB)"
        )
        self.query_one("#swap-value", Label).update(
            f"SWAP {stats.swap_percent:4.1f}% ({stats.swap_used_gb:.1f}GB)"
        )
        badge = self.query_one("#ollama-badge", Label)
        if ollama.online:
            n = len(ollama.loaded_models)
            badge.update(f"OLLAMA v{ollama.version} · {n} loaded")
            badge.set_classes("stat-value ok")
        else:
            badge.update("OLLAMA: offline")
            badge.set_classes("stat-value offline")


class AgentopApp(App):
    """Local terminal control plane for agent processes across apps."""

    TITLE = "agentop — Local Agent Control Plane"
    CSS_PATH = "agentop.tcss"

    BINDINGS = [
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
        self._selected_agent: AgentProcess | None = None
        self._selected_category: Category | None = None
        self._pending_force_targets: list[AgentProcess] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatsBar()
        with TabbedContent(initial="tab-overview"):
            with TabPane("Overview", id="tab-overview"):
                yield DataTable(id="overview-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Processes", id="tab-processes"):
                yield DataTable(id="process-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Models", id="tab-models"):
                yield Label("Loaded models", classes="section-label")
                yield DataTable(id="loaded-models-table", cursor_type="row", zebra_stripes=True)
                yield Label("Available models", classes="section-label")
                yield DataTable(id="available-models-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Network", id="tab-network"):
                yield DataTable(id="network-table", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="toolbar"):
            yield Button("Refresh", id="btn-refresh")
            yield Button("Kill Selected", id="btn-kill-selected", variant="warning")
            yield Button("Kill Session", id="btn-kill-session", variant="warning")
            yield Button("Kill Category", id="btn-kill-category", variant="error")
            yield Button("KILL ALL", id="btn-kill-all", variant="error")
            yield Button("Force Kill Remaining (0)", id="btn-force-kill", variant="error", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self.set_interval(self.refresh_interval, self._trigger_refresh)
        self._trigger_refresh()

    def _setup_tables(self) -> None:
        self.query_one("#overview-table", DataTable).add_columns("Category", "Count", "CPU %", "Memory", "Risk")
        self.query_one("#process-table", DataTable).add_columns(
            "PID", "Category", "Tool", "CPU %", "Memory", "Uptime", "Risk"
        )
        self.query_one("#loaded-models-table", DataTable).add_columns("Model", "Size", "Processor", "Context")
        self.query_one("#available-models-table", DataTable).add_columns("Model")
        self.query_one("#network-table", DataTable).add_columns("Port", "PID", "Process", "Category")

    # -- Data refresh -----------------------------------------------------

    def _trigger_refresh(self) -> None:
        self.refresh_data()

    @work(exclusive=True)
    async def refresh_data(self) -> None:
        """Worker entry point for the periodic timer (not directly awaitable —
        `@work` methods return a Worker handle). Tests call `_do_refresh`
        directly instead."""
        await self._do_refresh()

    async def _do_refresh(self) -> None:
        system_stats, agents, ollama_status = await asyncio.gather(
            asyncio.to_thread(collect_system_stats),
            asyncio.to_thread(collect_agent_processes),
            asyncio.to_thread(collect_ollama_status, self.ollama_url),
        )
        ports = await asyncio.to_thread(collect_listening_ports, agents)

        self._agents = agents
        self.query_one(StatsBar).update_stats(system_stats, ollama_status)
        self._update_overview_table(agents)
        self._update_process_table(agents)
        self._update_models_tables(ollama_status)
        self._update_network_table(ports)
        self._refresh_force_kill_button(agents)

    def _update_overview_table(self, agents: list[AgentProcess]) -> None:
        table = self.query_one("#overview-table", DataTable)
        table.clear()
        by_cat: dict[Category, list[AgentProcess]] = defaultdict(list)
        for a in agents:
            by_cat[a.category].append(a)
        for cat, procs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            total_cpu = sum(p.cpu_percent for p in procs)
            total_mem = sum(p.mem_mb for p in procs)
            highest_risk = max(procs, key=lambda p: RISK_ORDER[p.risk]).risk
            table.add_row(
                cat.value,
                str(len(procs)),
                f"{total_cpu:.1f}",
                _fmt_mem(total_mem),
                _risk_text(highest_risk),
                key=cat.value,
            )

    def _update_process_table(self, agents: list[AgentProcess]) -> None:
        table = self.query_one("#process-table", DataTable)
        table.clear()
        for a in sorted(agents, key=lambda p: -p.cpu_percent):
            table.add_row(
                str(a.pid),
                a.category.value,
                a.subtype,
                f"{a.cpu_percent:.1f}",
                _fmt_mem(a.mem_mb),
                _fmt_uptime(a.uptime_seconds),
                _risk_text(a.risk),
                key=str(a.pid),
            )

    def _update_models_tables(self, ollama: OllamaStatus) -> None:
        loaded = self.query_one("#loaded-models-table", DataTable)
        loaded.clear()
        for m in ollama.loaded_models:
            loaded.add_row(m.name, f"{m.size_gb:.1f} GB", m.processor, str(m.context) if m.context else "-")

        available = self.query_one("#available-models-table", DataTable)
        available.clear()
        for name in ollama.available_models:
            available.add_row(name)

    def _update_network_table(self, ports: list[PortInfo]) -> None:
        table = self.query_one("#network-table", DataTable)
        table.clear()
        for p in ports:
            cat = p.category.value if p.category else "-"
            table.add_row(str(p.port), str(p.pid), p.process_name, cat, key=f"{p.port}:{p.pid}")

    # -- Selection --------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = event.data_table
        if event.row_key.value is None:
            return
        if table.id == "process-table":
            pid = int(event.row_key.value)
            self._selected_agent = next((a for a in self._agents if a.pid == pid), None)
        elif table.id == "overview-table":
            self._selected_category = Category(event.row_key.value)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        event.data_table.sort(event.column_key)

    # -- Kill actions -------------------------------------------------------

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
        pid = self._selected_agent.pid if scope is KillScope.SINGLE and self._selected_agent else None
        session_key = (
            self._selected_agent.session_key if scope is KillScope.SESSION and self._selected_agent else None
        )
        category = None
        if scope is KillScope.CATEGORY:
            category = self._selected_category or (self._selected_agent.category if self._selected_agent else None)

        if scope is KillScope.SINGLE and pid is None:
            self.notify("No process selected in the Processes tab.", severity="warning")
            return
        if scope is KillScope.SESSION and session_key is None:
            self.notify("No process selected to determine a session.", severity="warning")
            return
        if scope is KillScope.CATEGORY and category is None:
            self.notify("Select a row in Overview or Processes first.", severity="warning")
            return

        plan = build_kill_plan(scope, self._agents, pid=pid, session_key=session_key, category=category)
        if plan.count == 0:
            self.notify("Nothing to kill (already gone).", severity="information")
            return

        confirmed = await self.push_screen_wait(ConfirmKillScreen(plan, force=force))
        if not confirmed:
            return

        results = await asyncio.to_thread(execute_kill, plan, force=force)
        ok = sum(1 for r in results if r.success)
        failed = [r for r in results if not r.success]
        msg = f"Killed {ok}/{len(results)} process(es)."
        if failed:
            msg += f" Failed: {', '.join(f'{r.pid} ({r.error})' for r in failed[:5])}"
        self.notify(msg, severity="information" if not failed else "warning")

        if not force:
            # Give processes a moment to honor SIGTERM; anything still alive
            # by the next refresh becomes eligible for the Force Kill button.
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
        btn = self.query_one("#btn-force-kill", Button)
        btn.label = f"Force Kill Remaining ({len(still_alive)})"
        btn.disabled = len(still_alive) == 0

    async def _force_kill_remaining(self) -> None:
        if not self._pending_force_targets:
            self.notify("Nothing pending.", severity="information")
            return
        plan = build_kill_plan(KillScope.ALL, self._pending_force_targets)
        confirmed = await self.push_screen_wait(ConfirmKillScreen(plan, force=True))
        if not confirmed:
            return
        results = await asyncio.to_thread(execute_kill, plan, force=True)
        ok = sum(1 for r in results if r.success)
        self.notify(f"Force-killed {ok}/{len(results)} process(es).")
        self._pending_force_targets = []
        btn = self.query_one("#btn-force-kill", Button)
        btn.label = "Force Kill Remaining (0)"
        btn.disabled = True
        self._trigger_refresh()

    # -- Toolbar buttons ----------------------------------------------------

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

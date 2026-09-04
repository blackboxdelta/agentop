"""Modal screens: confirmation gate for the kill switch.

A single parameterized confirmation screen is used for every kill scope
(single process, session, category, or all). The blast radius scales the
confirmation requirement: single-process kills need one click; anything
wider requires typing the literal word "KILL" before the confirm button
becomes enabled.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from agentop.control import KillPlan, KillScope
from agentop.fit import FitPlan
from agentop.models import Risk

_RISK_STYLE = {
    Risk.LOW: "bold #5EE6A8",
    Risk.MEDIUM: "bold #E0A65C",
    Risk.HIGH: "bold #F0706E",
}

_SCOPE_LABEL = {
    KillScope.SINGLE: "this process",
    KillScope.SESSION: "this entire session",
    KillScope.CATEGORY: "this entire category",
    KillScope.ALL: "ALL detected agent processes",
}


class ConfirmKillScreen(ModalScreen[bool]):
    """Shows exactly what will be signaled and waits for explicit confirmation."""

    DEFAULT_CSS = """
    ConfirmKillScreen {
        align: center middle;
    }
    #kill-dialog {
        width: 74;
        height: auto;
        max-height: 90%;
        border: thick #F0706E;
        background: #0E1214;
        padding: 1 2;
    }
    #kill-title {
        text-style: bold;
        color: #F0706E;
        margin-bottom: 1;
    }
    #kill-target-list {
        height: auto;
        max-height: 8;
        border: solid $panel;
        margin: 1 0;
        padding: 0 1;
    }
    #kill-confirm-input {
        margin: 1 0;
    }
    #kill-buttons {
        align: right middle;
        height: auto;
        margin-top: 1;
    }
    #kill-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, plan: KillPlan, *, force: bool = False) -> None:
        super().__init__()
        self.plan = plan
        self.force = force
        self._requires_typed_confirmation = plan.count > 1

    def compose(self) -> ComposeResult:
        signal_name = "SIGKILL (force)" if self.force else "SIGTERM"
        scope_desc = _SCOPE_LABEL[self.plan.scope]
        risk_style = _RISK_STYLE[self.plan.highest_risk]

        with Container(id="kill-dialog"):
            yield Label(f"WARNING  Kill Switch — {signal_name}", id="kill-title")
            yield Static(
                f"About to signal [b]{self.plan.count}[/b] process(es) — {scope_desc}.\n"
                f"Highest risk in this set: [{risk_style}]{self.plan.highest_risk.value.upper()}[/{risk_style}]"
            )
            with VerticalScroll(id="kill-target-list"):
                for t in self.plan.targets:
                    style = _RISK_STYLE[t.risk]
                    yield Static(
                        f"PID {t.pid:>7}  [{style}]{t.risk.value:<6}[/{style}]  "
                        f"{t.category.value} / {t.subtype}  ({t.name})"
                    )
            if self._requires_typed_confirmation:
                yield Label("Type KILL to enable the confirm button:")
                yield Input(placeholder="KILL", id="kill-confirm-input")
            with Horizontal(id="kill-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button(
                    "Confirm Kill",
                    id="confirm-btn",
                    variant="error",
                    disabled=self._requires_typed_confirmation,
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "kill-confirm-input":
            confirm_btn = self.query_one("#confirm-btn", Button)
            confirm_btn.disabled = event.value.strip() != "KILL"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)


class ConfirmTextScreen(ModalScreen[bool]):
    """Typed confirmation for destructive model-wide actions."""

    DEFAULT_CSS = """
    ConfirmTextScreen { align: center middle; }
    #text-confirm-dialog {
        width: 64;
        height: auto;
        border: thick #F0706E;
        background: #0E1214;
        padding: 1 2;
    }
    #text-confirm-title {
        color: #F0706E;
        text-style: bold;
        margin-bottom: 1;
    }
    #text-confirm-input { margin: 1 0; }
    #text-confirm-buttons { height: auto; align: right middle; }
    #text-confirm-buttons Button { margin-left: 1; }
    """

    def __init__(self, title: str, message: str, expected: str) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.expected = expected

    def compose(self) -> ComposeResult:
        with Container(id="text-confirm-dialog"):
            yield Label(self.title, id="text-confirm-title")
            yield Static(self.message)
            yield Label(f"Type {self.expected} to confirm:")
            yield Input(id="text-confirm-input")
            with Horizontal(id="text-confirm-buttons"):
                yield Button("Cancel", id="text-cancel")
                yield Button(
                    "Confirm",
                    id="text-confirm",
                    variant="error",
                    disabled=True,
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "text-confirm-input":
            self.query_one("#text-confirm", Button).disabled = (
                event.value.strip().casefold() != self.expected.casefold()
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "text-confirm":
            self.dismiss(True)
        elif event.button.id == "text-cancel":
            self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)


class PreflightScreen(ModalScreen[bool]):
    """Memory plan shown before a model would evict peers or use swap."""

    DEFAULT_CSS = """
    PreflightScreen { align: center middle; }
    #preflight-dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        border: thick #E0A65C;
        background: #0E1214;
        padding: 1 2;
    }
    #preflight-title {
        color: #E0A65C;
        text-style: bold;
        margin-bottom: 1;
    }
    #preflight-buttons { height: auto; align: right middle; margin-top: 1; }
    #preflight-buttons Button { margin-left: 1; }
    """

    def __init__(self, plan: FitPlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        evictions = (
            "\n".join(f"  - {model}" for model in self.plan.evictions)
            if self.plan.evictions
            else "  none"
        )
        load_time = (
            f"{self.plan.estimated_load_seconds:.1f} s"
            if self.plan.estimated_load_seconds is not None
            else "unknown"
        )
        paging = (
            "[bold #F0706E]YES — swap paging is expected[/bold #F0706E]"
            if self.plan.will_page
            else "[#5EE6A8]no[/#5EE6A8]"
        )
        with Container(id="preflight-dialog"):
            yield Label("MODEL LOAD PRE-FLIGHT", id="preflight-title")
            yield Static(
                f"[bold #E8EEF1]{self.plan.model}[/bold #E8EEF1]\n\n"
                f"Estimated memory {self.plan.required_gb:.1f} GB\n"
                f"Safe free memory {self.plan.free_gb:.1f} GB\n"
                f"Reserved headroom {self.plan.headroom_gb:.1f} GB\n"
                f"Reclaimable       {self.plan.reclaimable_gb:.1f} GB\n"
                f"Projected use     {self.plan.projected_used_gb:.1f} / {self.plan.total_gb:.1f} GB\n"
                f"Estimated load    {load_time}\n"
                f"Will page         {paging}\n\n"
                f"Models to evict:\n{evictions}\n\n"
                "Press enter or click Continue to execute this plan; esc cancels."
            )
            with Horizontal(id="preflight-buttons"):
                yield Button("Cancel", id="preflight-cancel")
                yield Button("Continue", id="preflight-confirm", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "preflight-confirm":
            self.dismiss(True)
        elif event.button.id == "preflight-cancel":
            self.dismiss(False)

    def key_enter(self) -> None:
        self.dismiss(True)

    def key_escape(self) -> None:
        self.dismiss(False)


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    #help-dialog {
        width: 76;
        height: auto;
        max-height: 90%;
        border: thick #5EE6A8;
        background: #0E1214;
        padding: 1 2;
    }
    #help-title { color: #5EE6A8; text-style: bold; }
    #help-content { height: auto; max-height: 26; margin: 1 0; }
    #help-buttons { height: auto; align: right middle; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Label("agentop keys", id="help-title")
            with VerticalScroll(id="help-content"):
                yield Static(
                    "[#8FA9FF]1–4[/#8FA9FF]  switch tabs\n"
                    "[#5EE6A8]enter[/#5EE6A8] warm selected cold model\n"
                    "[#F0706E]k[/#F0706E]  unload selected model / kill selected process\n"
                    "[#F0706E]shift+k[/#F0706E]  unload all / kill category\n"
                    "[#8FA9FF]t[/#8FA9FF]  reload with a smaller default context\n"
                    "[#8FA9FF]p[/#8FA9FF]  toggle model pin\n"
                    "[#8FA9FF]u[/#8FA9FF]  undo the last unload (5 seconds)\n"
                    "[#8FA9FF]/[/#8FA9FF]  focus model filter\n"
                    "[#8FA9FF]r[/#8FA9FF]  refresh\n"
                    "[#8FA9FF]?[/#8FA9FF]  this help\n"
                    "[#8FA9FF]q[/#8FA9FF]  quit\n\n"
                    "Mouse clicks work for tabs, model/process rows, filter fields, "
                    "buttons, confirmations, and scrollable regions."
                )
            with Horizontal(id="help-buttons"):
                yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)

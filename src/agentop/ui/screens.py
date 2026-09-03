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
from agentop.models import Risk

_RISK_STYLE = {
    Risk.LOW: "bold green",
    Risk.MEDIUM: "bold yellow",
    Risk.HIGH: "bold red",
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
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #kill-title {
        text-style: bold;
        color: $error;
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
            yield Label(f"⚠  Kill Switch — {signal_name}", id="kill-title")
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

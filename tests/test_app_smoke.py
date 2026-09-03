"""Headless smoke tests for the Textual app, using Textual's own test harness
(`App.run_test`) which drives the app without a real terminal. This exercises
real widget mounting, data refresh, table population, mouse clicks, and the
kill confirmation flow end-to-end — not just unit-level logic.
"""
from __future__ import annotations

import asyncio
import subprocess
import time

import pytest
from textual.widgets import Button, DataTable

from agentop.models import AgentProcess, Category, Risk
from agentop.ui.app import AgentopApp


def _fake_agent(pid: int, name: str = "sleep") -> AgentProcess:
    return AgentProcess(
        pid=pid,
        ppid=1,
        name=name,
        category=Category.OTHER_AGENT,
        subtype="test",
        cmdline=name,
        cpu_percent=0.0,
        mem_mb=0.0,
        create_time=time.time(),
        risk=Risk.LOW,
        session_key=1,
    )


@pytest.mark.asyncio
async def test_app_boots_and_populates_tables():
    app = AgentopApp(refresh_interval=100)  # avoid a second auto-refresh mid-test
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()

        overview = app.query_one("#overview-table", DataTable)
        processes = app.query_one("#process-table", DataTable)
        # This dev machine always has at least the Copilot CLI session running
        # this very test, so both tables should have at least one row.
        assert overview.row_count >= 1
        assert processes.row_count >= 1


@pytest.mark.asyncio
async def test_all_tabs_are_reachable_by_click():
    from textual.widgets import Tab

    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        # TabbedContent auto-generates internal Tab widget ids (prefixed,
        # not the plain TabPane id) — click the real widget objects rather
        # than guessing the generated id scheme.
        tabs = list(app.query(Tab))
        assert len(tabs) == 4
        for tab in tabs:
            await pilot.click(tab)
            await pilot.pause()
        # No exception means every tab mounted and switched cleanly.


@pytest.mark.asyncio
async def test_kill_selected_with_no_selection_shows_warning_not_crash():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        await pilot.click("#btn-kill-selected")
        await pilot.pause()
        # No process was selected — should notify, not raise/crash the app.
        assert app.is_running


@pytest.mark.asyncio
async def test_kill_switch_all_opens_confirmation_modal_and_can_be_cancelled():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        await pilot.click("#btn-kill-all")
        await pilot.pause()
        # A modal should now be on the screen stack.
        assert len(app.screen_stack) >= 2
        await pilot.click("#cancel-btn")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        assert app.is_running  # cancelling must not kill anything or crash


@pytest.mark.asyncio
async def test_kill_all_requires_typed_confirmation_before_enabling_confirm():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        await pilot.click("#btn-kill-all")
        await pilot.pause()
        confirm_btn = app.screen.query_one("#confirm-btn", Button)
        assert confirm_btn.disabled is True  # must start disabled for bulk scope
        await pilot.click("#cancel-btn")
        await pilot.pause()


@pytest.mark.asyncio
async def test_column_header_click_sorts_without_crashing():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        table = app.query_one("#process-table", DataTable)
        column_keys = list(table.columns.keys())
        assert column_keys  # columns were set up
        table.sort(column_keys[0])
        await pilot.pause()
        assert app.is_running


@pytest.mark.asyncio
async def test_confirming_kill_selected_actually_terminates_the_process():
    """End-to-end: select a (fabricated) process, click Kill Selected, click
    Confirm Kill in the modal, and verify the real disposable subprocess
    behind it actually exits. Exercises the exact button-click path whose
    layout bug was fixed above, but for the Confirm button rather than Cancel.
    """
    proc = subprocess.Popen(["sleep", "30"])
    try:
        app = AgentopApp(refresh_interval=100)
        async with app.run_test(size=(120, 45)) as pilot:
            await app._do_refresh()
            await pilot.pause()
            # Inject the disposable subprocess as the sole "selected" agent,
            # bypassing the real OS-wide scan so this test is deterministic.
            fake = _fake_agent(proc.pid)
            app._agents = [fake]
            app._selected_agent = fake

            await pilot.click("#btn-kill-selected")
            await pilot.pause()
            assert len(app.screen_stack) >= 2, "confirmation modal should be open"

            confirm_btn = app.screen.query_one("#confirm-btn", Button)
            assert confirm_btn.disabled is False, "single-process kill needs no typed confirmation"

            await pilot.click("#confirm-btn")
            await pilot.pause(0.2)

        proc.wait(timeout=5)
        assert proc.poll() is not None, "process should have actually been terminated"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_quit_action_stops_the_app_cleanly():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        await pilot.press("q")
        await asyncio.sleep(0.05)
        assert app.return_code == 0 or not app.is_running

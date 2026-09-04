"""Headless smoke tests for the Textual app, using Textual's own test harness
(`App.run_test`) which drives the app without a real terminal. This exercises
real widget mounting, data refresh, table population, mouse clicks, and the
kill confirmation flow end-to-end — not just unit-level logic.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import psutil
import pytest
import sqlite3
from textual.widgets import Button, DataTable, Input, Label, Static

from agentop.config import AgentopConfig
from agentop.models import (
    AgentProcess,
    Category,
    OllamaAvailableModel,
    OllamaModel,
    OllamaStatus,
    Risk,
    SystemStats,
)
from agentop.events import EventStore
from agentop.ui.app import AgentopApp, _fmt_remaining


@pytest.fixture(autouse=True)
def isolated_event_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agentop.ui.app.EventStore",
        lambda path: EventStore(tmp_path / "events.db"),
    )
    monkeypatch.setattr(
        "agentop.ui.app.OllamaClient",
        lambda config: FakeOllamaClient(),
    )
    monkeypatch.setattr(
        "agentop.ui.app.collect_agent_processes",
        lambda: [
            _fake_agent(999_998, name="fixture-agent-a"),
            _fake_agent(999_999, name="fixture-agent-b"),
        ],
    )


class FakeOllamaClient:
    def __init__(self, status=None, fail_warm_for=None):
        self.calls: list[tuple] = []
        self.status = status or OllamaStatus(online=True, version="test")
        self.fail_warm_for = fail_warm_for

    async def poll(self, force_tags=False):
        return self.status

    async def warm_model(self, model, *, keep_alive="5m", context=None):
        self.calls.append(("warm", model, keep_alive, context))
        if model == self.fail_warm_for:
            raise OSError("simulated warm failure")
        return {"done": True}

    async def unload_model(self, model):
        self.calls.append(("unload", model))

    async def pin_model(self, model, pinned):
        self.calls.append(("pin", model, pinned))

    async def close(self):
        return None


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met before timeout")
        await asyncio.sleep(0.02)


def _fake_agent(
    pid: int,
    name: str = "sleep",
    create_time: float | None = None,
) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        ppid=1,
        name=name,
        category=Category.OTHER_AGENT,
        subtype="test",
        cmdline=name,
        cpu_percent=0.0,
        mem_mb=0.0,
        create_time=time.time() if create_time is None else create_time,
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
        # The autouse fixture provides a deterministic process so this passes
        # on clean Windows/macOS release runners without local AI apps.
        assert overview.row_count >= 1
        assert processes.row_count >= 1
        assert app.query_one("#gpu-value", Label).render() is not None


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
async def test_models_table_shows_memory_after_processor():
    app = AgentopApp(refresh_interval=100)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(120, 45)) as pilot:
        app._update_models_tables(
            OllamaStatus(
                online=True,
                loaded_models=[
                    OllamaModel(
                        name="test-model",
                        size_gb=10.0,
                        processor="80% GPU",
                        memory_gb=10.0,
                        gpu_memory_gb=8.0,
                        context=4096,
                    )
                ],
            )
        )
        await pilot.pause()
        table = app.query_one("#resident-models-table", DataTable)
        assert [str(column.label) for column in table.columns.values()] == [
            "State",
            "Model",
            "Tok/s",
            "TTFT",
            "Context",
            "Layers",
            "Evicts in",
        ]
        assert [str(cell) for cell in table.get_row_at(0)] == [
            "○ idle",
            "test-model",
            "-",
            "-",
            "0/4.1k",
            "80% GPU",
            "-",
        ]


@pytest.mark.asyncio
async def test_models_workspace_filters_and_updates_selected_fit_details():
    app = AgentopApp(refresh_interval=100)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)) as pilot:
        app._system_stats = SystemStats(mem_used_gb=10.0, mem_total_gb=36.0)
        status = OllamaStatus(
            online=True,
            available_models=[
                OllamaAvailableModel(
                    name="qwen2.5-coder:32b",
                    size_gb=18.5,
                    family="qwen2",
                    parameter_size="32.8B",
                    quantization="Q4_K_M",
                ),
                OllamaAvailableModel(
                    name="phi4:latest",
                    size_gb=8.4,
                    family="phi3",
                    parameter_size="14.7B",
                    quantization="Q4_K_M",
                ),
            ],
        )
        app._ollama_status = status
        app._update_models_tables(status)
        app.action_show_tab("tab-models")
        await pilot.pause()

        table = app.query_one("#available-models-table", DataTable)
        assert table.row_count == 2
        assert "qwen2.5-coder:32b" in str(
            app.query_one("#model-details", Static).render()
        )
        assert "Run clients through `agentop proxy`" in str(
            app.query_one("#throughput-detail", Static).render()
        )

        app.query_one("#model-filter", Input).value = "Q4_K_M"
        await pilot.pause()
        assert app._selected_model_name == "qwen2.5-coder:32b"

        app.query_one("#model-filter", Input).value = "phi4"
        await pilot.pause()
        assert table.row_count == 1


def test_resident_expiration_is_rendered_as_remaining_time():
    expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    assert _fmt_remaining(expires) in {"29m", "30m"}


@pytest.mark.asyncio
async def test_model_poll_failure_does_not_emit_false_unload_event():
    app = AgentopApp(refresh_interval=100)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)) as pilot:
        loaded = OllamaStatus(
            online=True,
            loaded_models=[
                OllamaModel(
                    name="test-model",
                    size_gb=10.0,
                    processor="100% GPU",
                    memory_gb=10.0,
                    gpu_memory_gb=10.0,
                    context=4096,
                )
            ],
        )
        await app._update_model_events(loaded)
        await app._update_model_events(
            OllamaStatus(
                online=True,
                loaded_models_error="temporary API failure",
            )
        )
        await pilot.pause()
        rendered = str(app.query_one("#model-events", Static).render())
        assert "status unavailable" in rendered
        assert "left memory" not in rendered
        assert app._previous_loaded_names == {"test-model"}


@pytest.mark.asyncio
async def test_ollama_offline_poll_does_not_emit_false_unload_event():
    app = AgentopApp(refresh_interval=100)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)) as pilot:
        loaded = OllamaStatus(
            online=True,
            loaded_models=[
                OllamaModel(
                    name="test-model",
                    size_gb=10.0,
                    processor="100% GPU",
                    memory_gb=10.0,
                    gpu_memory_gb=10.0,
                    context=4096,
                )
            ],
        )
        await app._update_model_events(loaded)
        await app._update_model_events(
            OllamaStatus(online=False, error="temporary version API failure")
        )
        await pilot.pause()
        rendered = str(app.query_one("#model-events", Static).render())
        assert "status unavailable" in rendered
        assert "left memory" not in rendered
        assert app._previous_loaded_names == {"test-model"}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 35), (80, 28)])
async def test_models_workspace_remains_usable_in_compact_terminals(size):
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=size) as pilot:
        await app._do_refresh()
        app.action_show_tab("tab-models")
        await pilot.pause()

        assert app.has_class("compact")
        assert app.query_one("#model-sidebar").display is False
        assert app.query_one("#available-models-table", DataTable).region.width > 0
        assert app.query_one(".vram-block").display is True
        screen = app.screen.size.region
        for metric in app.query(".metric-block"):
            assert metric.region.center in screen
        if size[0] < 90:
            assert app.has_class("narrow")
            assert app.query_one("#brand").display is False
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#models-main").display is False
            assert app.query_one("#model-sidebar").display is True
            assert app.query_one("#model-sidebar").region.width == screen.width
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#models-main").display is True


@pytest.mark.asyncio
async def test_cold_model_preflight_blocks_warm_until_confirmed():
    client = FakeOllamaClient()
    app = AgentopApp(refresh_interval=100, client=client)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)) as pilot:
        app._system_stats = SystemStats(mem_used_gb=34, mem_total_gb=36)
        status = OllamaStatus(
            online=True,
            available_models=[
                OllamaAvailableModel(name="large", size_gb=12)
            ],
        )
        app._ollama_status = status
        app._update_models_tables(status)
        app.run_worker(app._warm_selected_model())
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.click("#preflight-cancel")
        await pilot.pause()
        assert client.calls == []


@pytest.mark.asyncio
async def test_model_unload_undo_pin_and_trim_controls(tmp_path):
    client = FakeOllamaClient()
    store = EventStore(tmp_path / "actions.db")
    app = AgentopApp(refresh_interval=100, client=client, store=store)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)) as pilot:
        model = OllamaModel(
            name="resident",
            size_gb=8,
            processor="100% GPU",
            memory_gb=8,
            gpu_memory_gb=8,
            context=8192,
        )
        status = OllamaStatus(online=True, loaded_models=[model])
        app._ollama_status = status
        app._update_models_tables(status)

        await app._toggle_selected_pin()
        await app._trim_selected_context()
        await app._unload_selected_model()
        await app._undo_last_unload()
        await pilot.pause()

        assert ("pin", "resident", True) in client.calls
        assert ("warm", "resident", -1, 4096) in client.calls
        assert ("unload", "resident") in client.calls
        assert ("warm", "resident", -1, None) in client.calls


@pytest.mark.asyncio
async def test_failed_warm_rolls_back_confirmed_evictions(tmp_path, monkeypatch):
    resident = OllamaModel(
        name="resident",
        size_gb=8,
        processor="100% GPU",
        memory_gb=8,
        gpu_memory_gb=8,
        context=4096,
    )
    target = OllamaAvailableModel(name="target", size_gb=12)
    status = OllamaStatus(
        online=True,
        loaded_models=[resident],
        available_models=[target],
    )
    client = FakeOllamaClient(status=status, fail_warm_for="target")
    store = EventStore(tmp_path / "rollback.db")
    config = AgentopConfig(
        state_path=store.path,
        memory_headroom_gb=0,
        model_memory_overhead=1,
    )
    app = AgentopApp(config=config, client=client, store=store)
    app._trigger_refresh = lambda: None
    monkeypatch.setattr(
        "agentop.ui.app.collect_system_stats",
        lambda: SystemStats(mem_used_gb=30, mem_total_gb=36),
    )
    async with app.run_test(size=(140, 44)) as pilot:
        app._system_stats = SystemStats(mem_used_gb=30, mem_total_gb=36)
        app._ollama_status = status
        app._selected_model_name = "target"
        app._update_models_tables(status)
        app.run_worker(app._warm_selected_model())
        await pilot.pause()
        await pilot.click("#preflight-confirm")
        await _wait_until(
            lambda: ("warm", "resident", "5m", 4096) in client.calls
        )

    assert ("unload", "resident") in client.calls
    assert ("warm", "target", "5m", None) in client.calls
    assert ("warm", "resident", "5m", 4096) in client.calls


@pytest.mark.asyncio
async def test_pin_persistence_failure_compensates_runtime_state(tmp_path):
    class BrokenPinStore(EventStore):
        def set_pinned(self, model, pinned):
            raise sqlite3.OperationalError("disk full")

    model = OllamaModel(
        name="resident",
        size_gb=8,
        processor="100% GPU",
        memory_gb=8,
        gpu_memory_gb=8,
        context=4096,
    )
    status = OllamaStatus(online=True, loaded_models=[model])
    client = FakeOllamaClient(status=status)
    store = BrokenPinStore(tmp_path / "pins.db")
    app = AgentopApp(refresh_interval=100, client=client, store=store)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=(140, 44)):
        app._ollama_status = status
        app._update_models_tables(status)
        await app._toggle_selected_pin()

    assert ("pin", "resident", True) in client.calls
    assert ("pin", "resident", False) in client.calls
    assert app._pinned_models == set()


@pytest.mark.asyncio
async def test_cancelled_warm_restores_evicted_models(tmp_path, monkeypatch):
    class BlockingClient(FakeOllamaClient):
        def __init__(self, status):
            super().__init__(status=status)
            self.target_started = asyncio.Event()

        async def warm_model(self, model, *, keep_alive="5m", context=None):
            self.calls.append(("warm", model, keep_alive, context))
            if model == "target":
                self.target_started.set()
                await asyncio.Event().wait()
            return {"done": True}

    resident = OllamaModel(
        name="resident",
        size_gb=8,
        processor="100% GPU",
        memory_gb=8,
        gpu_memory_gb=8,
        context=4096,
    )
    target = OllamaAvailableModel(name="target", size_gb=12)
    status = OllamaStatus(
        online=True,
        loaded_models=[resident],
        available_models=[target],
    )
    client = BlockingClient(status)
    store = EventStore(tmp_path / "cancel.db")
    config = AgentopConfig(
        state_path=store.path,
        memory_headroom_gb=0,
        model_memory_overhead=1,
    )
    app = AgentopApp(config=config, client=client, store=store)
    app._trigger_refresh = lambda: None
    monkeypatch.setattr(
        "agentop.ui.app.collect_system_stats",
        lambda: SystemStats(mem_used_gb=30, mem_total_gb=36),
    )
    async with app.run_test(size=(140, 44)) as pilot:
        app._system_stats = SystemStats(mem_used_gb=30, mem_total_gb=36)
        app._ollama_status = status
        app._selected_model_name = "target"
        app._update_models_tables(status)
        worker = app.run_worker(app._warm_selected_model())
        await pilot.pause()
        await pilot.click("#preflight-confirm")
        await asyncio.wait_for(client.target_started.wait(), 2)
        worker.cancel()
        await _wait_until(
            lambda: ("warm", "resident", "5m", 4096) in client.calls
        )

    assert ("unload", "resident") in client.calls
    assert ("warm", "resident", "5m", 4096) in client.calls


@pytest.mark.asyncio
async def test_changed_eviction_plan_is_reconfirmed_before_execution(
    tmp_path, monkeypatch
):
    def resident(name):
        return OllamaModel(
            name=name,
            size_gb=8,
            processor="100% GPU",
            memory_gb=8,
            gpu_memory_gb=8,
            context=4096,
        )

    target = OllamaAvailableModel(name="target", size_gb=12)
    initial = OllamaStatus(
        online=True,
        loaded_models=[resident("model-a")],
        available_models=[target],
    )
    changed = OllamaStatus(
        online=True,
        loaded_models=[resident("model-b")],
        available_models=[target],
    )

    class ChangingClient(FakeOllamaClient):
        async def poll(self, force_tags=False):
            return changed

    client = ChangingClient(status=changed)
    store = EventStore(tmp_path / "reconfirm.db")
    config = AgentopConfig(
        state_path=store.path,
        memory_headroom_gb=0,
        model_memory_overhead=1,
    )
    app = AgentopApp(config=config, client=client, store=store)
    app._trigger_refresh = lambda: None
    monkeypatch.setattr(
        "agentop.ui.app.collect_system_stats",
        lambda: SystemStats(mem_used_gb=30, mem_total_gb=36),
    )
    async with app.run_test(size=(140, 44)) as pilot:
        app._system_stats = SystemStats(mem_used_gb=30, mem_total_gb=36)
        app._ollama_status = initial
        app._selected_model_name = "target"
        app._update_models_tables(initial)
        app.run_worker(app._warm_selected_model())
        await _wait_until(lambda: len(app.screen_stack) == 2)
        first_modal = app.screen
        await pilot.click("#preflight-confirm")
        await _wait_until(
            lambda: len(app.screen_stack) == 2 and app.screen is not first_modal
        )
        await pilot.click("#preflight-confirm")
        await _wait_until(lambda: ("unload", "model-b") in client.calls)

    assert ("unload", "model-a") not in client.calls
    assert ("unload", "model-b") in client.calls


@pytest.mark.asyncio
async def test_kill_selected_with_no_selection_shows_warning_not_crash():
    app = AgentopApp(refresh_interval=100)
    async with app.run_test(size=(120, 45)) as pilot:
        await app._do_refresh()
        await pilot.pause()
        app.action_show_tab("tab-processes")
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
        app.action_show_tab("tab-processes")
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
        app.action_show_tab("tab-processes")
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
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        app = AgentopApp(refresh_interval=100)
        async with app.run_test(size=(120, 45)) as pilot:
            await app._do_refresh()
            await pilot.pause()
            # Inject the disposable subprocess as the sole "selected" agent,
            # bypassing the real OS-wide scan so this test is deterministic.
            fake = _fake_agent(proc.pid, create_time=psutil.Process(proc.pid).create_time())
            app._agents = [fake]
            app._selected_agent = fake

            app.action_show_tab("tab-processes")
            await pilot.pause()
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

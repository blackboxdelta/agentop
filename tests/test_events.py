from __future__ import annotations

import pytest

from agentop.events import EventStore
from agentop.models import CompletionRecord


def _record(
    *,
    model: str = "model",
    client: str = "client-a",
    started: float,
    duration_ms: float,
    success: bool = True,
    error_type: str = "",
) -> CompletionRecord:
    return CompletionRecord(
        model=model,
        client=client,
        started_at=started,
        completed_at=started + duration_ms / 1000,
        success=success,
        prompt_tokens=100,
        output_tokens=50,
        prompt_eval_ns=500_000_000,
        eval_ns=2_000_000_000,
        total_ns=int(duration_ms * 1_000_000),
        ttft_ms=200,
        error_type=error_type,
        context_window=8192,
        batch_size=512,
    )


def test_event_store_persists_events_pins_and_completion_metrics(tmp_path):
    path = tmp_path / "events.db"
    store = EventStore(path)
    store.initialize()
    now = 1_000_000.0

    store.record_event("ok", "load", "model warmed", model="model", timestamp=now)
    store.set_pinned("model", True)
    store.record_completion(_record(started=now - 30, duration_ms=2200))
    store.record_completion(
        _record(
            started=now - 20,
            duration_ms=4000,
            success=False,
            error_type="oom",
        )
    )
    active_id = store.begin_request("model", "client-b", now - 5)

    reopened = EventStore(path)
    reopened.initialize()
    events = reopened.recent_events()
    metrics = reopened.model_metrics("model", now=now)

    assert events[0].message == "model warmed"
    assert reopened.pinned_models() == {"model"}
    assert metrics.run_count == 2
    assert metrics.generation_tps == 25.0
    assert metrics.prompt_tps == 200.0
    assert metrics.ttft_p50_ms == 200.0
    assert metrics.latency_p50_ms == 2200.0
    assert metrics.success_rate == 50.0
    assert metrics.oom_count == 1
    assert metrics.in_flight == 1
    assert metrics.last_context_window == 8192
    assert metrics.batch_size == 512
    assert metrics.sessions[0].client in {"client-a", "client-b"}

    # Reopening in the same live owner process must not corrupt active rows.
    with reopened._connection() as connection:
        row = connection.execute(
            "SELECT completed_at FROM completions WHERE id = ?", (active_id,)
        ).fetchone()
    assert row["completed_at"] is None


def test_event_store_cleanup_removes_expired_history(tmp_path, monkeypatch):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    store.record_event("ok", "load", "old", timestamp=100.0)
    store.record_completion(_record(started=100.0, duration_ms=100))
    monkeypatch.setattr("agentop.events.time.time", lambda: 10_000_000.0)
    store.cleanup(retention_days=1)
    assert store.recent_events() == []
    assert store.model_metrics("model", now=10_000_000.0).run_count == 1


def test_metrics_recovers_old_orphans_but_keeps_recent_requests_active(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    now = 10_000.0
    old_id = store.begin_request("model", "old-client", now - 700)
    recent_id = store.begin_request("model", "recent-client", now - 5)
    with store._connection() as connection:
        connection.execute(
            "UPDATE completions SET owner_pid = ? WHERE id = ?",
            (999_999_999, old_id),
        )

    metrics = store.model_metrics("model", now=now)
    assert metrics.in_flight == 1

    with store._connection() as connection:
        old = connection.execute(
            "SELECT error_type FROM completions WHERE id = ?", (old_id,)
        ).fetchone()
        recent = connection.execute(
            "SELECT completed_at FROM completions WHERE id = ?", (recent_id,)
        ).fetchone()
    assert old["error_type"] == "interrupted"
    assert recent["completed_at"] is None


def test_long_running_request_with_live_owner_is_not_recovered(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    now = 10_000.0
    request_id = store.begin_request("model", "client", now - 3600)
    metrics = store.model_metrics("model", now=now)
    assert metrics.in_flight == 1
    with store._connection() as connection:
        row = connection.execute(
            "SELECT completed_at FROM completions WHERE id = ?",
            (request_id,),
        ).fetchone()
    assert row["completed_at"] is None

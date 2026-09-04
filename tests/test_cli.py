from __future__ import annotations

import json

import pytest

from agentop.cli import (
    build_parser,
    collect_snapshot,
    initialize_event_store,
    proxy_would_loop,
)
from agentop.config import AgentopConfig
from agentop.events import EventStore
from agentop.models import OllamaAvailableModel, OllamaStatus


class SnapshotClient:
    async def poll(self, force_tags=False):
        assert force_tags is True
        return OllamaStatus(
            online=True,
            version="test",
            available_models=[
                OllamaAvailableModel(name="model", size_gb=1)
            ],
        )


def test_cli_parser_supports_remote_host_json_and_proxy():
    parser = build_parser()
    args = parser.parse_args(
        ["proxy", "--host", "remote.example:11434", "--port", "12000"]
    )
    assert args.command == "proxy"
    assert args.host == "remote.example:11434"
    assert args.port == 12000
    assert parser.parse_args(["--json"]).json is True


def test_proxy_loop_detection_distinguishes_remote_same_port():
    assert proxy_would_loop("http://localhost:11434", "127.0.0.1", 11434)
    assert proxy_would_loop("http://127.0.0.1:11434", "0.0.0.0", 11434)
    assert proxy_would_loop("http://[::1]:11434", "::", 11434)
    assert not proxy_would_loop(
        "http://remote-ollama.example:11434",
        "127.0.0.1",
        11434,
    )


def test_corrupt_event_database_is_quarantined_and_recreated(tmp_path):
    path = tmp_path / "events.db"
    path.write_bytes(b"not a sqlite database")
    store = initialize_event_store(path)
    store.record_event("ok", "test", "recovered")
    assert store.recent_events()[0].message == "recovered"
    assert list(tmp_path.glob("events.db.corrupt-*"))


@pytest.mark.asyncio
async def test_json_snapshot_is_serializable(tmp_path):
    config = AgentopConfig(
        host="http://ollama.test",
        state_path=tmp_path / "events.db",
    )
    store = EventStore(config.state_path)
    store.initialize()
    snapshot = await collect_snapshot(
        config,
        store,
        client=SnapshotClient(),
    )
    encoded = json.dumps(snapshot, default=str)
    assert '"agentop_version"' in encoded
    assert snapshot["host"] == "http://ollama.test"
    assert snapshot["ollama"]["available_models"][0]["name"] == "model"

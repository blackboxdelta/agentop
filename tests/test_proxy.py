from __future__ import annotations

import asyncio
import httpx
import pytest
import sqlite3

from agentop.config import AgentopConfig
from agentop.events import EventStore
from agentop.proxy import TerminalFrameParser, create_proxy_app


@pytest.mark.asyncio
async def test_proxy_forwards_completion_and_persists_metrics(tmp_path):
    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "ollama.test"
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=(
                b'{"response":"hello","done":false}\n'
                b'{"done":true,"prompt_eval_count":20,'
                b'"prompt_eval_duration":500000000,"eval_count":10,'
                b'"eval_duration":1000000000,"total_duration":1800000000}\n'
            ),
            request=request,
        )

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test", state_path=tmp_path / "events.db"),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post(
            "/api/generate",
            headers={"x-agentop-client": "pytest"},
            json={"model": "test-model", "prompt": "hello", "stream": True},
        )
        await response.aread()

    assert response.status_code == 200
    metrics = store.model_metrics("test-model")
    assert metrics.run_count == 1
    assert metrics.generation_tps == 10.0
    assert metrics.prompt_tps == 40.0
    assert metrics.tokens_in_60s == 20
    assert metrics.tokens_out_60s == 10
    assert metrics.sessions[0].client == "pytest"


def test_terminal_frame_parser_handles_sse_and_bounded_long_stream():
    parser = TerminalFrameParser(max_frame_bytes=128)
    for _ in range(20_000):
        parser.feed(b'{"response":"token","done":false}\n')
    parser.feed(
        b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert parser.finish()["usage"]["completion_tokens"] == 34
    assert len(parser.buffer) <= 128


@pytest.mark.asyncio
async def test_proxy_records_openai_sse_usage(tmp_path):
    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                b'data: {"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            ),
            request=request,
        )

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test", state_path=store.path),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "stream": True},
        )
        await response.aread()

    metrics = store.model_metrics("test-model")
    assert metrics.tokens_in_60s == 7
    assert metrics.tokens_out_60s == 3


@pytest.mark.asyncio
async def test_proxy_rejects_oversized_request_before_upstream(tmp_path):
    upstream_called = False

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_called
        upstream_called = True
        return httpx.Response(200, json={}, request=request)

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(
            host="http://ollama.test",
            state_path=store.path,
            proxy_max_body_bytes=10,
        ),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post("/api/generate", content=b"x" * 11)

    assert response.status_code == 413
    assert upstream_called is False


@pytest.mark.asyncio
async def test_interrupted_upstream_stream_is_recorded_as_failed(tmp_path):
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"response":"partial","done":false}\n'
            raise httpx.ReadError("stream broke")

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=BrokenStream(),
            request=request,
        )

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test", state_path=store.path),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    with pytest.raises(httpx.ReadError):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        ) as client:
            await client.post(
                "/api/generate",
                json={"model": "test-model", "stream": True},
            )

    with store._connection() as connection:
        row = connection.execute(
            """
            SELECT success, error_type FROM completions
            WHERE model = 'test-model' ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    assert row["success"] == 0
    assert row["error_type"] == "upstream_interrupted"


@pytest.mark.asyncio
async def test_cancellation_before_headers_finalizes_request(tmp_path):
    gate = asyncio.Event()

    async def upstream(request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json={}, request=request)

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test", state_path=store.path),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/api/generate",
                json={"model": "test-model", "stream": True},
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with store._connection() as connection:
        row = connection.execute(
            "SELECT success, error_type FROM completions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["success"] == 0
    assert row["error_type"] == "interrupted"


@pytest.mark.asyncio
async def test_telemetry_store_failure_does_not_block_inference(tmp_path):
    class BrokenStore:
        def begin_request(self, *args):
            raise sqlite3.OperationalError("disk full")

        def record_event(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk full")

        def recover_stale_requests(self):
            raise sqlite3.OperationalError("disk full")

        def cleanup(self, days):
            raise sqlite3.OperationalError("disk full")

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"done": True, "eval_count": 1},
            request=request,
        )

    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test"),
        BrokenStore(),
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post(
            "/api/generate",
            json={"model": "test-model", "stream": False},
        )
    assert response.status_code == 200
    assert response.json()["done"] is True


@pytest.mark.asyncio
async def test_proxy_preserves_request_content_encoding(tmp_path):
    seen_encoding = None

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal seen_encoding
        seen_encoding = request.headers.get("content-encoding")
        return httpx.Response(200, json={"ok": True}, request=request)

    store = EventStore(tmp_path / "events.db")
    store.initialize()
    app = create_proxy_app(
        AgentopConfig(host="http://ollama.test", state_path=store.path),
        store,
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://proxy.test",
    ) as client:
        response = await client.post(
            "/api/generate",
            headers={"content-encoding": "gzip"},
            content=b"compressed-placeholder",
        )
    assert response.status_code == 200
    assert seen_encoding == "gzip"

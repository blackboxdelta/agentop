from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentop.config import AgentopConfig
from agentop.ollama_client import OllamaClient


def _json_response(request: httpx.Request, payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=request)


@pytest.mark.asyncio
async def test_async_client_enriches_tags_and_loaded_models_with_show_metadata():
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/version":
            return _json_response(request, {"version": "0.test"})
        if request.url.path == "/api/tags":
            return _json_response(
                request,
                {
                    "models": [
                        {
                            "name": "model:Q4",
                            "digest": "abc",
                            "size": 10 * 1024**3,
                            "details": {
                                "family": "family",
                                "parameter_size": "7B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            return _json_response(
                request,
                {
                    "details": {
                        "family": "family",
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M",
                    },
                    "model_info": {
                        "family.context_length": 32768,
                        "family.block_count": 32,
                    },
                },
            )
        if request.url.path == "/api/ps":
            return _json_response(
                request,
                {
                    "models": [
                        {
                            "name": "model:Q4",
                            "size": 10 * 1024**3,
                            "size_vram": 8 * 1024**3,
                            "context_length": 16384,
                        }
                    ]
                },
            )
        raise AssertionError(request.url.path)

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test", tags_interval=30),
        transport=httpx.MockTransport(handler),
    )
    try:
        status = await client.poll()
        second = await client.poll()
    finally:
        await client.close()

    assert status.online is True
    assert status.version == "0.test"
    assert status.available_models[0].context == 32768
    assert status.available_models[0].total_layers == 32
    assert status.loaded_models[0].gpu_layers == 26
    assert status.loaded_models[0].total_layers == 32
    assert second.stale is False
    assert calls.count("/api/tags") == 1
    assert calls.count("/api/show") == 1
    assert calls.count("/api/ps") == 2


@pytest.mark.asyncio
async def test_async_client_returns_last_known_values_as_stale_on_disconnect():
    online = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal online
        if not online:
            raise httpx.ConnectError("offline", request=request)
        if request.url.path == "/api/version":
            return _json_response(request, {"version": "1"})
        if request.url.path == "/api/tags":
            return _json_response(request, {"models": [{"name": "model"}]})
        if request.url.path == "/api/show":
            return _json_response(request, {})
        if request.url.path == "/api/ps":
            return _json_response(request, {"models": []})
        raise AssertionError(request.url.path)

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        good = await client.poll()
        online = False
        stale = await client.poll()
    finally:
        await client.close()

    assert good.online is True
    assert stale.online is False
    assert stale.stale is True
    assert stale.available_models[0].name == "model"
    assert "offline" in stale.error


@pytest.mark.asyncio
async def test_model_actions_send_only_to_configured_ollama_host():
    received: list[tuple[str, dict, dict]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received.append(
            (
                str(request.url),
                json.loads(request.content),
                request.extensions.get("timeout", {}),
            )
        )
        return _json_response(request, {"done": True})

    client = OllamaClient(
        AgentopConfig(host="https://remote.example:11434"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.warm_model("model", context=4096)
        await client.unload_model("model")
    finally:
        await client.close()

    assert all(
        url.startswith("https://remote.example:11434/")
        for url, _, _ in received
    )
    assert received[0][1]["options"]["num_ctx"] == 4096
    assert received[1][1]["keep_alive"] == 0
    assert received[0][2]["read"] == 600.0


@pytest.mark.asyncio
async def test_chat_sends_non_streaming_messages_and_returns_content():
    received: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return _json_response(
            request,
            {
                "message": {
                    "role": "assistant",
                    "content": " local answer ",
                },
                "done": True,
            },
        )

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        answer = await client.chat(
            "model",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        await client.close()

    assert answer == "local answer"
    assert received == [
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "keep_alive": "5m",
        }
    ]


@pytest.mark.asyncio
async def test_chat_rejects_response_without_message_content():
    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, {"message": {"role": "assistant"}})

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="no message content"):
            await client.chat("model", [{"role": "user", "content": "hello"}])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tags_failure_keeps_resident_poll_live_and_preserves_metadata():
    fail_tags = False

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return _json_response(request, {"version": "1"})
        if request.url.path == "/api/tags":
            if fail_tags:
                return httpx.Response(503, request=request)
            return _json_response(
                request,
                {"models": [{"name": "model", "digest": "a", "size": 1}]},
            )
        if request.url.path == "/api/show":
            return _json_response(request, {})
        if request.url.path == "/api/ps":
            return _json_response(
                request,
                {"models": [{"name": "model", "size": 1, "size_vram": 1}]},
            )
        raise AssertionError(request.url.path)

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        first = await client.poll(force_tags=True)
        fail_tags = True
        second = await client.poll(force_tags=True)
    finally:
        await client.close()

    assert first.available_models[0].name == "model"
    assert second.online is True
    assert second.loaded_models[0].name == "model"
    assert second.available_models[0].name == "model"
    assert second.available_models_error != ""


@pytest.mark.asyncio
async def test_failed_show_enrichment_is_retried_on_next_tags_poll():
    show_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal show_calls
        if request.url.path == "/api/version":
            return _json_response(request, {"version": "1"})
        if request.url.path == "/api/tags":
            return _json_response(
                request,
                {"models": [{"name": "model", "digest": "same"}]},
            )
        if request.url.path == "/api/show":
            show_calls += 1
            if show_calls == 1:
                return httpx.Response(503, request=request)
            return _json_response(
                request,
                {"model_info": {"family.context_length": 8192}},
            )
        if request.url.path == "/api/ps":
            return _json_response(request, {"models": []})
        raise AssertionError(request.url.path)

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        first = await client.poll(force_tags=True)
        second = await client.poll(force_tags=True)
    finally:
        await client.close()

    assert first.available_models[0].context == 0
    assert second.available_models[0].context == 8192
    assert show_calls == 2


@pytest.mark.asyncio
async def test_overlapping_polls_are_serialized():
    active_ps = 0
    max_active_ps = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active_ps, max_active_ps
        if request.url.path == "/api/version":
            return _json_response(request, {"version": "1"})
        if request.url.path == "/api/tags":
            return _json_response(request, {"models": []})
        if request.url.path == "/api/ps":
            active_ps += 1
            max_active_ps = max(max_active_ps, active_ps)
            await asyncio.sleep(0.02)
            active_ps -= 1
            return _json_response(request, {"models": []})
        raise AssertionError(request.url.path)

    client = OllamaClient(
        AgentopConfig(host="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await asyncio.gather(client.poll(), client.poll())
    finally:
        await client.close()
    assert max_active_ps == 1

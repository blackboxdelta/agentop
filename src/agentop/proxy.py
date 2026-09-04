"""Optional Ollama-only completion proxy that records request metrics."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import sqlite3
import time
from typing import AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from agentop.config import AgentopConfig
from agentop.events import EventStore
from agentop.models import CompletionRecord

_HOP_BY_HOP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_RESPONSE_STRIP = _HOP_BY_HOP | {"content-encoding"}
_LOGGER = logging.getLogger("agentop.proxy")


async def _safe_store_call(function, *args, default=None, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except (sqlite3.Error, OSError) as exc:
        _LOGGER.warning("telemetry store unavailable: %s", exc)
        return default


async def _uncancellable_store_call(function, *args, **kwargs):
    """Finish a bounded SQLite operation before propagating cancellation."""
    task = asyncio.create_task(
        _safe_store_call(function, *args, **kwargs)
    )
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            continue
    return task.result(), cancelled


class TerminalFrameParser:
    """Incrementally retains only the terminal JSON/SSE frame."""

    def __init__(self, max_frame_bytes: int = 1_048_576) -> None:
        self.max_frame_bytes = max_frame_bytes
        self.buffer = bytearray()
        self.payload: dict = {}

    def _parse_line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]":
            return
        try:
            candidate = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if isinstance(candidate, dict):
            self.payload = candidate

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        while b"\n" in self.buffer:
            line, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            self._parse_line(line)
        if len(self.buffer) > self.max_frame_bytes:
            del self.buffer[: len(self.buffer) - self.max_frame_bytes]

    def finish(self) -> dict:
        if self.buffer:
            self._parse_line(bytes(self.buffer))
            self.buffer.clear()
        return self.payload


async def _read_bounded_body(
    request: Request,
    max_bytes: int,
) -> bytes | None:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                return None
        except ValueError:
            return None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            return None
    return bytes(body)


def _error_type(status_code: int, payload: dict) -> str:
    message = str(payload.get("error", "")).casefold()
    if "out of memory" in message or "oom" in message:
        return "oom"
    if "context" in message and ("length" in message or "window" in message):
        return "context_overflow"
    if status_code in {408, 504} or "timeout" in message:
        return "timeout"
    if status_code < 400:
        return "upstream_error" if message else ""
    return "http_error"


def _completion_record(
    *,
    model: str,
    client: str,
    started_at: float,
    completed_at: float,
    first_byte_at: float | None,
    status_code: int,
    payload: dict,
    request_options: dict,
) -> CompletionRecord:
    usage = payload.get("usage") or {}
    prompt_tokens = int(
        payload.get("prompt_eval_count")
        or usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        payload.get("eval_count")
        or usage.get("completion_tokens")
        or 0
    )
    total_ns = int(payload.get("total_duration") or (completed_at - started_at) * 1e9)
    return CompletionRecord(
        model=model,
        client=client,
        started_at=started_at,
        completed_at=completed_at,
        success=status_code < 400 and not payload.get("error"),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prompt_eval_ns=int(payload.get("prompt_eval_duration") or 0),
        eval_ns=int(payload.get("eval_duration") or 0),
        total_ns=total_ns,
        ttft_ms=(
            (first_byte_at - started_at) * 1000 if first_byte_at is not None else None
        ),
        error_type=_error_type(status_code, payload),
        context_window=int(request_options.get("num_ctx") or 0),
        batch_size=int(request_options.get("num_batch") or 0),
    )


def create_proxy_app(
    config: AgentopConfig,
    store: EventStore,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    http = httpx.AsyncClient(
        base_url=config.host,
        timeout=httpx.Timeout(
            connect=config.request_timeout,
            write=config.request_timeout,
            pool=config.request_timeout,
            read=config.model_load_timeout,
        ),
        follow_redirects=False,
        trust_env=False,
        transport=upstream_transport,
    )

    async def proxy(request: Request):
        body = await _read_bounded_body(request, config.proxy_max_body_bytes)
        if body is None:
            return JSONResponse(
                {"error": "request body exceeds configured proxy limit"},
                status_code=413,
            )
        try:
            request_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_json = {}
        model = str(request_json.get("model", ""))
        tracked = request.url.path in {
            "/api/generate",
            "/api/chat",
            "/v1/chat/completions",
        } and bool(model)
        options = request_json.get("options") or {}
        client_name = (
            request.headers.get("x-agentop-client")
            or request.headers.get("user-agent")
            or (request.client.host if request.client else "unknown")
        )
        started_at = time.time()
        begin_cancelled = False
        if tracked:
            request_id, begin_cancelled = await _uncancellable_store_call(
                store.begin_request,
                model,
                client_name,
                started_at,
            )
        else:
            request_id = None
        if begin_cancelled:
            if request_id is not None:
                interrupted = CompletionRecord(
                    model=model,
                    client=client_name,
                    started_at=started_at,
                    completed_at=time.time(),
                    success=False,
                    error_type="interrupted",
                )
                await _uncancellable_store_call(
                    store.finish_request,
                    request_id,
                    interrupted,
                )
            raise asyncio.CancelledError
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.casefold() not in _HOP_BY_HOP
        }
        url = request.url.path
        if request.url.query:
            url += f"?{request.url.query}"
        upstream_request = http.build_request(
            request.method,
            url,
            content=body,
            headers=headers,
        )
        try:
            upstream = await http.send(upstream_request, stream=True)
        except asyncio.CancelledError:
            completed_at = time.time()
            if request_id is not None:
                record = CompletionRecord(
                    model=model,
                    client=client_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    success=False,
                    total_ns=int((completed_at - started_at) * 1e9),
                    error_type="interrupted",
                )
                await _uncancellable_store_call(
                    store.finish_request,
                    request_id,
                    record,
                )
            raise
        except httpx.HTTPError as exc:
            completed_at = time.time()
            record = CompletionRecord(
                model=model,
                client=client_name,
                started_at=started_at,
                completed_at=completed_at,
                success=False,
                total_ns=int((completed_at - started_at) * 1e9),
                error_type="upstream_unavailable",
            )
            if request_id is not None:
                await _safe_store_call(store.finish_request, request_id, record)
            await _safe_store_call(
                store.record_event,
                "error",
                "proxy",
                f"upstream request failed: {exc}",
                model=model,
            )
            return JSONResponse(
                {"error": f"Ollama upstream unavailable: {exc}"},
                status_code=502,
            )

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.casefold() not in _RESPONSE_STRIP
        }

        async def stream() -> AsyncIterator[bytes]:
            parser = TerminalFrameParser()
            first_byte_at: float | None = None
            reached_eof = False
            stream_error = ""
            try:
                async for chunk in upstream.aiter_bytes():
                    if chunk and first_byte_at is None:
                        first_byte_at = time.time()
                    parser.feed(chunk)
                    yield chunk
                reached_eof = True
            except asyncio.CancelledError:
                stream_error = "interrupted"
                raise
            except (httpx.HTTPError, OSError):
                stream_error = "upstream_interrupted"
                raise
            finally:
                await asyncio.shield(upstream.aclose())
                completed_at = time.time()
                payload = parser.finish()
                record = _completion_record(
                    model=model,
                    client=client_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    first_byte_at=first_byte_at,
                    status_code=upstream.status_code,
                    payload=payload,
                    request_options=options,
                )
                if not reached_eof:
                    record.success = False
                    record.error_type = stream_error or "interrupted"
                if request_id is not None:
                    await _uncancellable_store_call(
                        store.finish_request,
                        request_id,
                        record,
                    )

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=None,
        )

    @asynccontextmanager
    async def lifespan(app: Starlette):
        await _safe_store_call(store.recover_stale_requests)
        await _safe_store_call(store.cleanup, 30)
        yield
        await http.aclose()

    return Starlette(
        routes=[
            Route(
                "/{path:path}",
                proxy,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            )
        ],
        lifespan=lifespan,
    )

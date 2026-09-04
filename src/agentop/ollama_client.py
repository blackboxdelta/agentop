"""Async Ollama client with metadata caching and reconnect backoff."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import time
from typing import Any

import httpx

from agentop.config import AgentopConfig
from agentop.models import OllamaAvailableModel, OllamaModel, OllamaStatus


def _context_and_layers(model_info: dict[str, Any]) -> tuple[int, int]:
    contexts = [
        int(value)
        for key, value in model_info.items()
        if str(key).endswith(".context_length") and isinstance(value, (int, float))
    ]
    layers = [
        int(value)
        for key, value in model_info.items()
        if str(key).endswith(".block_count") and isinstance(value, (int, float))
    ]
    return (max(contexts, default=0), max(layers, default=0))


def parse_available_model(
    item: dict[str, Any],
    show: dict[str, Any] | None = None,
) -> OllamaAvailableModel:
    show = show or {}
    details = show.get("details") or item.get("details") or {}
    context, layers = _context_and_layers(show.get("model_info") or {})
    family = str(details.get("family", ""))
    params = str(details.get("parameter_size", ""))
    quant = str(details.get("quantization_level", ""))
    description = " ".join(part for part in (params, family, "model") if part)
    return OllamaAvailableModel(
        name=str(item.get("name") or item.get("model") or "unknown"),
        size_gb=float(item.get("size", 0) or 0) / (1024**3),
        modified_at=str(item.get("modified_at", "")),
        family=family,
        parameter_size=params,
        quantization=quant,
        context=context,
        total_layers=layers,
        digest=str(item.get("digest", "")),
        description=description,
    )


def parse_loaded_model(
    item: dict[str, Any],
    metadata: OllamaAvailableModel | None = None,
) -> OllamaModel:
    size = int(item.get("size", 0) or 0)
    size_vram = int(item.get("size_vram", 0) or 0)
    details = item.get("details") or {}
    ratio = size_vram / size if size else 0.0
    if size > 0 and size_vram >= size:
        processor = "100% GPU"
    elif size_vram <= 0:
        processor = "100% CPU"
    else:
        processor = f"{round(ratio * 100)}% GPU"
    total_layers = metadata.total_layers if metadata else 0
    gpu_layers = round(total_layers * ratio) if total_layers else 0
    return OllamaModel(
        name=str(item.get("name") or item.get("model") or "unknown"),
        size_gb=size / (1024**3),
        processor=processor,
        memory_gb=size / (1024**3),
        gpu_memory_gb=size_vram / (1024**3),
        context=int(item.get("context_length", 0) or 0),
        family=(metadata.family if metadata else str(details.get("family", ""))),
        parameter_size=(
            metadata.parameter_size
            if metadata
            else str(details.get("parameter_size", ""))
        ),
        quantization=(
            metadata.quantization
            if metadata
            else str(details.get("quantization_level", ""))
        ),
        expires_at=str(item.get("expires_at", "")),
        total_layers=total_layers,
        gpu_layers=gpu_layers,
    )


class OllamaClient:
    def __init__(
        self,
        config: AgentopConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.host,
            timeout=httpx.Timeout(config.request_timeout),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )
        self._version = ""
        self._available: list[OllamaAvailableModel] = []
        self._show_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._last_tags_poll = 0.0
        self._last_status: OllamaStatus | None = None
        self._last_success = 0.0
        self._failures = 0
        self._next_attempt = 0.0
        self._show_semaphore = asyncio.Semaphore(8)
        self._poll_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._http.aclose()

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        long_running: bool = False,
    ) -> dict[str, Any]:
        response = await self._http.request(
            method,
            path,
            json=json,
            timeout=(
                self.config.model_load_timeout
                if long_running
                else self.config.request_timeout
            ),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Ollama returned a non-object response for {path}")
        return payload

    def _stale_status(self, error: Exception | str) -> OllamaStatus:
        message = str(error)
        stale_seconds = max(0.0, time.monotonic() - self._last_success)
        if self._last_status is None:
            return OllamaStatus(
                online=False,
                error=message,
                stale=False,
                stale_seconds=0.0,
                sampled_at=time.time(),
            )
        return replace(
            self._last_status,
            online=False,
            error=message,
            stale=True,
            stale_seconds=stale_seconds,
            sampled_at=time.time(),
        )

    async def _poll_tags(self) -> list[OllamaAvailableModel]:
        payload = await self._json("GET", "/api/tags")
        items = payload.get("models") or []
        if not isinstance(items, list):
            raise ValueError("Ollama /api/tags models field is not a list")

        async def enrich(item: dict[str, Any]) -> OllamaAvailableModel:
            name = str(item.get("name") or item.get("model") or "")
            digest = str(item.get("digest", ""))
            cached = self._show_cache.get(name)
            if cached and cached[0] == digest:
                show = cached[1]
            else:
                try:
                    async with self._show_semaphore:
                        show = await self._json(
                            "POST",
                            "/api/show",
                            json={"model": name},
                        )
                except (httpx.HTTPError, ValueError):
                    show = {}
                if show:
                    self._show_cache[name] = (digest, show)
            return parse_available_model(item, show)

        models = await asyncio.gather(
            *(enrich(item) for item in items if isinstance(item, dict))
        )
        self._available = list(models)
        self._last_tags_poll = time.monotonic()
        return self._available

    async def poll(self, *, force_tags: bool = False) -> OllamaStatus:
        async with self._poll_lock:
            return await self._poll(force_tags=force_tags)

    async def _poll(self, *, force_tags: bool = False) -> OllamaStatus:
        now = time.monotonic()
        if now < self._next_attempt:
            return self._stale_status("reconnect backoff")
        tags_error = ""
        try:
            if not self._version:
                version_payload = await self._json("GET", "/api/version")
                self._version = str(version_payload.get("version", ""))
            if (
                force_tags
                or not self._available
                or now - self._last_tags_poll >= self.config.tags_interval
            ):
                try:
                    await self._poll_tags()
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    tags_error = str(exc)
            ps_payload = await self._json("GET", "/api/ps")
            ps_items = ps_payload.get("models") or []
            if not isinstance(ps_items, list):
                raise ValueError("Ollama /api/ps models field is not a list")
        except (httpx.HTTPError, OSError, ValueError) as exc:
            self._failures += 1
            self._next_attempt = time.monotonic() + min(
                30.0, 0.5 * (2 ** (self._failures - 1))
            )
            return self._stale_status(exc)

        metadata = {model.name: model for model in self._available}
        loaded = [
            parse_loaded_model(item, metadata.get(str(item.get("name") or item.get("model"))))
            for item in ps_items
            if isinstance(item, dict)
        ]
        status = OllamaStatus(
            online=True,
            version=self._version,
            loaded_models=loaded,
            available_models=list(self._available),
            available_models_error=tags_error,
            sampled_at=time.time(),
        )
        self._last_status = status
        self._last_success = time.monotonic()
        self._failures = 0
        self._next_attempt = 0.0
        return status

    async def warm_model(
        self,
        model: str,
        *,
        keep_alive: str | int = "5m",
        context: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        if context:
            payload["options"] = {"num_ctx": context}
        return await self._json(
            "POST",
            "/api/generate",
            json=payload,
            long_running=True,
        )

    async def unload_model(self, model: str) -> None:
        await self._json(
            "POST",
            "/api/generate",
            json={"model": model, "keep_alive": 0},
        )

    async def pin_model(self, model: str, pinned: bool) -> None:
        await self.warm_model(model, keep_alive=-1 if pinned else "5m")

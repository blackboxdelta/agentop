"""Synchronous compatibility snapshot for scripts and focused tests.

The Textual application uses :class:`agentop.ollama_client.OllamaClient`
directly so polling never blocks its event loop.
"""
from __future__ import annotations

import time

import httpx

from agentop.models import OllamaStatus
from agentop.ollama_client import parse_available_model, parse_loaded_model

_TIMEOUT = 1.5


def collect_ollama_status(base_url: str = "http://localhost:11434") -> OllamaStatus:
    host = base_url.rstrip("/")
    try:
        version_response = httpx.get(
            f"{host}/api/version",
            timeout=_TIMEOUT,
            trust_env=False,
        )
        version_response.raise_for_status()
        version = str(version_response.json().get("version", ""))
        tags_response = httpx.get(
            f"{host}/api/tags",
            timeout=_TIMEOUT,
            trust_env=False,
        )
        tags_response.raise_for_status()
        available = [
            parse_available_model(item)
            for item in tags_response.json().get("models", [])
            if isinstance(item, dict)
        ]
        ps_response = httpx.get(
            f"{host}/api/ps",
            timeout=_TIMEOUT,
            trust_env=False,
        )
        ps_response.raise_for_status()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return OllamaStatus(
            online=False,
            error=str(exc),
            sampled_at=time.time(),
        )

    metadata = {model.name: model for model in available}
    loaded = [
        parse_loaded_model(
            item,
            metadata.get(str(item.get("name") or item.get("model"))),
        )
        for item in ps_response.json().get("models", [])
        if isinstance(item, dict)
    ]
    return OllamaStatus(
        online=True,
        version=version,
        loaded_models=loaded,
        available_models=available,
        sampled_at=time.time(),
    )

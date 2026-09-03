"""Polls the local Ollama REST API for loaded/available models.

Ollama may not be running at all (fresh machine, manually stopped, etc.) —
every network call here is guarded so the UI can simply render an
"offline" badge instead of crashing.
"""
from __future__ import annotations

import requests

from agentop.models import OllamaModel, OllamaStatus

_TIMEOUT = 1.5


def collect_ollama_status(base_url: str = "http://localhost:11434") -> OllamaStatus:
    try:
        version_resp = requests.get(f"{base_url}/api/version", timeout=_TIMEOUT)
        version_resp.raise_for_status()
        version = version_resp.json().get("version", "")
    except requests.RequestException as exc:
        return OllamaStatus(online=False, error=str(exc))

    loaded_models: list[OllamaModel] = []
    try:
        ps_resp = requests.get(f"{base_url}/api/ps", timeout=_TIMEOUT)
        ps_resp.raise_for_status()
        for m in ps_resp.json().get("models", []):
            size = m.get("size", 0) or 0
            size_vram = m.get("size_vram", 0) or 0
            if size > 0 and size_vram >= size:
                processor = "100% GPU"
            elif size_vram <= 0:
                processor = "100% CPU"
            else:
                pct = round(size_vram / size * 100)
                processor = f"{pct}% GPU"
            loaded_models.append(
                OllamaModel(
                    name=m.get("name", "unknown"),
                    size_gb=size / (1024**3),
                    processor=processor,
                    context=m.get("context_length", 0) or 0,
                )
            )
    except requests.RequestException:
        pass  # server up but /api/ps hiccuped — still report what we have

    available_models: list[str] = []
    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=_TIMEOUT)
        tags_resp.raise_for_status()
        available_models = [m.get("name", "unknown") for m in tags_resp.json().get("models", [])]
    except requests.RequestException:
        pass

    return OllamaStatus(
        online=True,
        version=version,
        loaded_models=loaded_models,
        available_models=available_models,
    )

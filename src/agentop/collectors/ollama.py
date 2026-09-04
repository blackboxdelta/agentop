"""Polls the local Ollama REST API for loaded/available models.

Ollama may not be running at all (fresh machine, manually stopped, etc.) —
every network call here is guarded so the UI can simply render an
"offline" badge instead of crashing.
"""
from __future__ import annotations

import requests

from agentop.models import OllamaAvailableModel, OllamaModel, OllamaStatus

_TIMEOUT = 1.5


def collect_ollama_status(base_url: str = "http://localhost:11434") -> OllamaStatus:
    try:
        version_resp = requests.get(f"{base_url}/api/version", timeout=_TIMEOUT)
        version_resp.raise_for_status()
        version = version_resp.json().get("version", "")
    except requests.RequestException as exc:
        return OllamaStatus(online=False, error=str(exc))

    loaded_models: list[OllamaModel] = []
    loaded_models_error = ""
    try:
        ps_resp = requests.get(f"{base_url}/api/ps", timeout=_TIMEOUT)
        ps_resp.raise_for_status()
        for m in ps_resp.json().get("models", []):
            size = m.get("size", 0) or 0
            size_vram = m.get("size_vram", 0) or 0
            details = m.get("details") or {}
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
                    memory_gb=size / (1024**3),
                    gpu_memory_gb=size_vram / (1024**3),
                    context=m.get("context_length", 0) or 0,
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    expires_at=m.get("expires_at", ""),
                )
            )
    except requests.RequestException as exc:
        loaded_models_error = str(exc)

    available_models: list[OllamaAvailableModel] = []
    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=_TIMEOUT)
        tags_resp.raise_for_status()
        for m in tags_resp.json().get("models", []):
            details = m.get("details") or {}
            available_models.append(
                OllamaAvailableModel(
                    name=m.get("name", "unknown"),
                    size_gb=(m.get("size", 0) or 0) / (1024**3),
                    modified_at=m.get("modified_at", ""),
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                )
            )
    except requests.RequestException:
        pass

    return OllamaStatus(
        online=True,
        version=version,
        loaded_models=loaded_models,
        available_models=available_models,
        loaded_models_error=loaded_models_error,
    )

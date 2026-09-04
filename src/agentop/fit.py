"""Memory fit planning for model warm-up actions."""
from __future__ import annotations

from dataclasses import dataclass, field

from agentop.models import OllamaModel


@dataclass
class FitPlan:
    model: str
    required_gb: float
    free_gb: float
    reclaimable_gb: float
    projected_used_gb: float
    total_gb: float
    headroom_gb: float
    will_page: bool
    evictions: list[str] = field(default_factory=list)
    estimated_load_seconds: float | None = None

    @property
    def requires_confirmation(self) -> bool:
        return self.required_gb > self.free_gb or bool(self.evictions)


def calculate_fit_plan(
    *,
    model: str,
    required_gb: float,
    used_gb: float,
    total_gb: float,
    loaded_models: list[OllamaModel],
    pinned_models: set[str],
    headroom_gb: float = 0.0,
    estimated_load_seconds: float | None = None,
) -> FitPlan:
    free_gb = max(0.0, total_gb - used_gb - headroom_gb)
    deficit = max(0.0, required_gb - free_gb)
    reclaimable = 0.0
    evictions: list[str] = []
    if deficit > 0:
        candidates = [
            model_info
            for model_info in loaded_models
            if model_info.name != model and model_info.name not in pinned_models
        ]
        for candidate in candidates:
            evictions.append(candidate.name)
            reclaimable += candidate.memory_gb
            if reclaimable >= deficit:
                break

    projected = max(0.0, used_gb - reclaimable) + required_gb
    return FitPlan(
        model=model,
        required_gb=required_gb,
        free_gb=free_gb,
        reclaimable_gb=reclaimable,
        projected_used_gb=projected,
        total_gb=total_gb,
        headroom_gb=headroom_gb,
        will_page=projected > total_gb,
        evictions=evictions,
        estimated_load_seconds=estimated_load_seconds,
    )

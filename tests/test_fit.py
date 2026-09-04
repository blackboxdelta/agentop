from __future__ import annotations

from agentop.fit import calculate_fit_plan
from agentop.models import OllamaModel


def _loaded(name: str, memory_gb: float) -> OllamaModel:
    return OllamaModel(
        name=name,
        size_gb=memory_gb,
        processor="100% GPU",
        memory_gb=memory_gb,
        gpu_memory_gb=memory_gb,
        context=4096,
    )


def test_fit_plan_requires_no_confirmation_when_model_fits():
    plan = calculate_fit_plan(
        model="small",
        required_gb=4,
        used_gb=10,
        total_gb=36,
        loaded_models=[],
        pinned_models=set(),
    )
    assert plan.requires_confirmation is False
    assert plan.will_page is False
    assert plan.projected_used_gb == 14


def test_fit_plan_selects_unpinned_evictions_before_swap():
    plan = calculate_fit_plan(
        model="large",
        required_gb=12,
        used_gb=30,
        total_gb=36,
        loaded_models=[_loaded("pinned", 5), _loaded("evictable", 8)],
        pinned_models={"pinned"},
    )
    assert plan.evictions == ["evictable"]
    assert plan.reclaimable_gb == 8
    assert plan.will_page is False
    assert plan.requires_confirmation is True


def test_fit_plan_reports_paging_when_pins_prevent_enough_reclaim():
    plan = calculate_fit_plan(
        model="large",
        required_gb=12,
        used_gb=34,
        total_gb=36,
        loaded_models=[_loaded("pinned", 10)],
        pinned_models={"pinned"},
    )
    assert plan.evictions == []
    assert plan.will_page is True
    assert plan.projected_used_gb == 46

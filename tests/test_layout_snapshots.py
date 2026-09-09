from __future__ import annotations

import pytest

from agentop.events import EventStore
from agentop.models import OllamaStatus
from agentop.ui.app import AgentopApp


class EmptyClient:
    async def poll(self, force_tags=False):
        return OllamaStatus(online=True, version="test")


SNAPSHOTS = {
    (80, 24): {
        "tab-overview": {
            "#top-bar": (0, 0, 80, 3),
            "#overview-table": (0, 5, 80, 18),
            "#keybar-actions": (0, 23, 80, 1),
        },
        "tab-processes": {
            "#top-bar": (0, 0, 80, 3),
            "#process-table": (0, 5, 80, 12),
            "#process-toolbar": (0, 17, 80, 6),
        },
        "tab-models": {
            "#top-bar": (0, 0, 80, 3),
            "#models-main": (0, 5, 80, 18),
            "#available-models-table": (0, 17, 78, 6),
        },
        "tab-playground": {
            "#top-bar": (0, 0, 80, 3),
            "#playground-layout": (0, 5, 80, 18),
            "#playground-transcript": (0, 8, 80, 11),
            "#playground-compose": (0, 20, 80, 3),
        },
        "tab-network": {"#top-bar": (0, 0, 80, 3), "#network-table": (0, 5, 80, 18)},
    },
    (120, 30): {
        "tab-overview": {
            "#top-bar": (0, 0, 120, 3),
            "#overview-table": (0, 5, 120, 24),
            "#keybar-actions": (0, 29, 82, 1),
        },
        "tab-processes": {
            "#top-bar": (0, 0, 120, 3),
            "#process-table": (0, 5, 120, 21),
            "#process-toolbar": (0, 26, 120, 3),
        },
        "tab-models": {
            "#top-bar": (0, 0, 120, 3),
            "#models-main": (0, 5, 120, 24),
            "#available-models-table": (0, 17, 118, 12),
        },
        "tab-playground": {
            "#top-bar": (0, 0, 120, 3),
            "#playground-layout": (0, 5, 120, 24),
            "#playground-transcript": (0, 8, 120, 17),
            "#playground-compose": (0, 26, 120, 3),
        },
        "tab-network": {"#top-bar": (0, 0, 120, 3), "#network-table": (0, 5, 120, 24)},
    },
    (200, 60): {
        "tab-overview": {
            "#top-bar": (0, 0, 200, 3),
            "#overview-table": (0, 5, 200, 54),
            "#keybar-actions": (0, 59, 162, 1),
        },
        "tab-processes": {
            "#top-bar": (0, 0, 200, 3),
            "#process-table": (0, 5, 200, 51),
            "#process-toolbar": (0, 56, 200, 3),
        },
        "tab-models": {
            "#top-bar": (0, 0, 200, 3),
            "#models-main": (0, 5, 156, 54),
            "#model-sidebar": (156, 5, 44, 54),
            "#available-models-table": (0, 26, 152, 33),
        },
        "tab-playground": {
            "#top-bar": (0, 0, 200, 3),
            "#playground-layout": (0, 5, 200, 54),
            "#playground-transcript": (0, 9, 200, 46),
            "#playground-compose": (0, 56, 200, 3),
        },
        "tab-network": {"#top-bar": (0, 0, 200, 3), "#network-table": (0, 5, 200, 54)},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 30), (200, 60)])
async def test_all_pane_layout_snapshots(size, tmp_path):
    store = EventStore(tmp_path / f"events-{size[0]}.db")
    app = AgentopApp(refresh_interval=100, client=EmptyClient(), store=store)
    app._trigger_refresh = lambda: None
    async with app.run_test(size=size) as pilot:
        for tab, expected in SNAPSHOTS[size].items():
            app.action_show_tab(tab)
            await pilot.pause()
            actual = {
                selector: tuple(app.query_one(selector).region)
                for selector in expected
            }
            assert actual == expected

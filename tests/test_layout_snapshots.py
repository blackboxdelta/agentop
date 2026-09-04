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
        "tab-overview": {"#top-bar": (0, 0, 80, 3), "#overview-table": (0, 6, 80, 17)},
        "tab-processes": {
            "#top-bar": (0, 0, 80, 3),
            "#process-table": (0, 6, 80, 11),
            "#process-toolbar": (0, 17, 80, 6),
        },
        "tab-models": {
            "#top-bar": (0, 0, 80, 3),
            "#models-main": (0, 6, 80, 17),
            "#available-models-table": (0, 19, 78, 4),
        },
        "tab-network": {"#top-bar": (0, 0, 80, 3), "#network-table": (0, 6, 80, 17)},
    },
    (120, 30): {
        "tab-overview": {"#top-bar": (0, 0, 120, 3), "#overview-table": (0, 6, 120, 23)},
        "tab-processes": {
            "#top-bar": (0, 0, 120, 3),
            "#process-table": (0, 6, 120, 20),
            "#process-toolbar": (0, 26, 120, 3),
        },
        "tab-models": {
            "#top-bar": (0, 0, 120, 3),
            "#models-main": (0, 6, 120, 23),
            "#available-models-table": (0, 19, 118, 10),
        },
        "tab-network": {"#top-bar": (0, 0, 120, 3), "#network-table": (0, 6, 120, 23)},
    },
    (200, 60): {
        "tab-overview": {"#top-bar": (0, 0, 200, 3), "#overview-table": (0, 6, 200, 53)},
        "tab-processes": {
            "#top-bar": (0, 0, 200, 3),
            "#process-table": (0, 6, 200, 50),
            "#process-toolbar": (0, 56, 200, 3),
        },
        "tab-models": {
            "#top-bar": (0, 0, 200, 3),
            "#models-main": (0, 6, 158, 53),
            "#model-sidebar": (158, 6, 42, 53),
            "#available-models-table": (0, 27, 154, 32),
        },
        "tab-network": {"#top-bar": (0, 0, 200, 3), "#network-table": (0, 6, 200, 53)},
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

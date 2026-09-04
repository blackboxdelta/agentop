"""Command-line entry point for agentop."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import urlparse

import uvicorn

from agentop import __version__
from agentop.collectors.processes import collect_agent_processes
from agentop.collectors.system import collect_system_stats
from agentop.config import AgentopConfig, load_config
from agentop.events import EventStore, NullEventStore
from agentop.ollama_client import OllamaClient
from agentop.proxy import create_proxy_app
from agentop.ui.app import AgentopApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentop",
        description="Monitor and control Ollama models and local AI agent processes.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("tui", "proxy"),
        default="tui",
        help="run the TUI (default) or the metrics-recording Ollama proxy",
    )
    parser.add_argument(
        "--host",
        "--ollama-url",
        dest="host",
        help="Ollama host; overrides OLLAMA_HOST and config.toml",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="TUI refresh interval in seconds",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="configuration file (default: ~/.config/agentop/config.toml)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="event database path (default: ~/.local/state/agentop/events.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON snapshot and exit",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable color (also honors NO_COLOR)",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1",
        help="proxy listen address (proxy command only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11435,
        help="proxy listen port (proxy command only; default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"agentop {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> AgentopConfig:
    return load_config(
        config_path=args.config,
        host=args.host,
        refresh_interval=args.interval,
        state_path=args.state,
        no_color=args.no_color,
    )


def initialize_event_store(path: Path) -> EventStore:
    store = EventStore(path)
    try:
        store.initialize()
        return store
    except sqlite3.DatabaseError as exc:
        quarantine = path.with_name(
            f"{path.name}.corrupt-{int(time.time())}"
        )
        try:
            path.replace(quarantine)
            recovered = EventStore(path)
            recovered.initialize()
            print(
                f"agentop: quarantined corrupt event DB at {quarantine}",
                file=sys.stderr,
            )
            return recovered
        except (OSError, sqlite3.Error) as recovery_error:
            print(
                f"agentop: event history disabled: {recovery_error}",
                file=sys.stderr,
            )
            return NullEventStore(path, str(exc))
    except (OSError, sqlite3.Error) as exc:
        print(f"agentop: event history disabled: {exc}", file=sys.stderr)
        return NullEventStore(path, str(exc))


def proxy_would_loop(upstream: str, listen: str, port: int) -> bool:
    parsed = urlparse(upstream)
    upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if upstream_port != port:
        return False
    upstream_host = (parsed.hostname or "").casefold()
    listen_host = listen.strip("[]").casefold()
    loopback = {"localhost", "127.0.0.1", "::1"}
    wildcard = {"0.0.0.0", "::"}
    if listen_host in wildcard:
        return upstream_host in loopback or upstream_host in wildcard
    if listen_host in loopback:
        return upstream_host in loopback
    return upstream_host == listen_host


async def collect_snapshot(
    config: AgentopConfig,
    store: EventStore,
    *,
    client: OllamaClient | None = None,
) -> dict:
    resolved_client = client or OllamaClient(config)
    try:
        system, agents, ollama = await asyncio.gather(
            asyncio.to_thread(collect_system_stats),
            asyncio.to_thread(collect_agent_processes),
            resolved_client.poll(force_tags=True),
        )
        model_names = [model.name for model in ollama.available_models]
        metrics = await asyncio.to_thread(store.all_model_metrics, model_names)
        events = await asyncio.to_thread(store.recent_events, 20)
        return {
            "agentop_version": __version__,
            "host": config.host,
            "system": asdict(system),
            "ollama": asdict(ollama),
            "agents": [asdict(agent) for agent in agents],
            "model_metrics": {
                name: asdict(value) for name, value in metrics.items()
            },
            "recent_events": [asdict(event) for event in events],
        }
    finally:
        if client is None:
            await resolved_client.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    store = initialize_event_store(config.state_path)

    if args.command == "proxy":
        if proxy_would_loop(config.host, args.listen, args.port):
            parser.error("proxy listen address would loop back to its own Ollama host")
        app = create_proxy_app(config, store)
        uvicorn.run(app, host=args.listen, port=args.port, log_level="warning")
        return 0

    if args.json:
        snapshot = asyncio.run(collect_snapshot(config, store))
        json.dump(snapshot, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if config.no_color:
        os.environ.setdefault("NO_COLOR", "1")
    app = AgentopApp(config=config, store=store)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

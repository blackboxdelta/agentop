"""Command-line entry point for `agentop`."""
from __future__ import annotations

import argparse
import sys

from agentop import __version__
from agentop.ui.app import REFRESH_INTERVAL, AgentopApp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentop",
        description="Local terminal control plane for monitoring and killing AI agent processes across apps.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL of the local Ollama server (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=REFRESH_INTERVAL,
        help="Refresh interval in seconds (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"agentop {__version__}")
    args = parser.parse_args(argv)

    app = AgentopApp(ollama_url=args.ollama_url, refresh_interval=args.interval)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

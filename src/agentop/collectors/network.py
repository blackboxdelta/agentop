"""Listening-port discovery, cross-referenced against already-categorized agents.

Deliberately reuses the process categorization from `processes.py` as the
single source of truth for "is this an agent" — a port is only surfaced here
if its owning pid was already classified as an agent process, which
automatically excludes unrelated system services (ControlCenter, rapportd,
OneDrive, ...) without a separate denylist to maintain.
"""
from __future__ import annotations

import psutil

from agentop.models import AgentProcess, PortInfo


def collect_listening_ports(agents: list[AgentProcess]) -> list[PortInfo]:
    by_pid = {a.pid: a for a in agents}
    results: list[PortInfo] = []

    # Preferred path: one system-wide scan. macOS can raise AccessDenied here
    # for a sandboxed/unprivileged caller, in which case we fall back to
    # asking each agent process individually for its own sockets (allowed
    # without elevated privileges for processes you own).
    try:
        conns = psutil.net_connections(kind="inet")
        for c in conns:
            if c.status != psutil.CONN_LISTEN or not c.laddr or c.pid is None:
                continue
            agent = by_pid.get(c.pid)
            if agent is None:
                continue
            results.append(
                PortInfo(port=c.laddr.port, pid=c.pid, process_name=agent.name, category=agent.category)
            )
    except (psutil.AccessDenied, PermissionError):
        for agent in agents:
            try:
                proc = psutil.Process(agent.pid)
                for c in proc.net_connections(kind="inet"):
                    if c.status != psutil.CONN_LISTEN or not c.laddr:
                        continue
                    results.append(
                        PortInfo(
                            port=c.laddr.port,
                            pid=agent.pid,
                            process_name=agent.name,
                            category=agent.category,
                        )
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

    seen: set[tuple[int, int]] = set()
    deduped: list[PortInfo] = []
    for p in results:
        key = (p.port, p.pid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return sorted(deduped, key=lambda p: p.port)

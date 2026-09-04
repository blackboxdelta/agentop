"""Kill-switch logic: plan a termination, then (separately) execute it.

Split deliberately into two halves:

- `build_kill_plan` is pure — given a scope and the current process list, it
  resolves exactly which processes would be signaled. Fully unit-testable
  with fabricated AgentProcess lists, no OS interaction at all.
- `execute_kill` is the only function in this codebase that calls
  `os.kill`. It always targets an explicit, already-resolved PID list (never
  a name/pattern), signals one PID at a time, and turns every failure mode
  (already exited, permission denied) into a structured result instead of
  raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import psutil

from agentop.models import AgentProcess, Category, Risk, RISK_ORDER


class KillScope(str, Enum):
    SINGLE = "single"      # exactly one process, by pid
    SESSION = "session"    # every process sharing the same session_key (ppid)
    CATEGORY = "category"  # every process in one Category
    ALL = "all"            # every currently-detected agent process


@dataclass
class KillPlan:
    scope: KillScope
    targets: list[AgentProcess] = field(default_factory=list)
    highest_risk: Risk = Risk.LOW

    @property
    def count(self) -> int:
        return len(self.targets)


@dataclass
class KillResult:
    pid: int
    name: str
    success: bool
    signal_used: str
    error: str = ""


def build_kill_plan(
    scope: KillScope,
    agents: list[AgentProcess],
    *,
    pid: int | None = None,
    session_key: int | None = None,
    category: Category | None = None,
) -> KillPlan:
    """Resolve a scope into a concrete, explicit list of target processes."""
    if scope is KillScope.SINGLE:
        targets = [a for a in agents if a.pid == pid]
    elif scope is KillScope.SESSION:
        targets = [a for a in agents if a.session_key == session_key]
    elif scope is KillScope.CATEGORY:
        targets = [a for a in agents if a.category == category]
    elif scope is KillScope.ALL:
        targets = list(agents)
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unknown kill scope: {scope!r}")

    highest = Risk.LOW
    for a in targets:
        if RISK_ORDER[a.risk] > RISK_ORDER[highest]:
            highest = a.risk

    return KillPlan(scope=scope, targets=targets, highest_risk=highest)


def execute_kill(plan: KillPlan, *, force: bool = False) -> list[KillResult]:
    """Signal every process in the plan. SIGTERM by default, SIGKILL if force=True.

    Never raises: every OS-level failure for a given pid is captured as a
    per-process KillResult so a bulk kill can partially succeed and report
    exactly which PIDs failed and why.
    """
    signal_used = "SIGKILL" if force else "SIGTERM"
    results: list[KillResult] = []
    for target in plan.targets:
        try:
            process = psutil.Process(target.pid)
            live_create_time = process.create_time()
        except psutil.NoSuchProcess:
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=False,
                    signal_used=signal_used,
                    error="already exited",
                )
            )
            continue
        except (psutil.AccessDenied, psutil.ZombieProcess):
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=False,
                    signal_used=signal_used,
                    error="could not verify process identity",
                )
            )
            continue
        if abs(live_create_time - target.create_time) > 0.01:
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=False,
                    signal_used=signal_used,
                    error="PID reused; identity mismatch",
                )
            )
            continue
        try:
            if force:
                process.kill()
            else:
                process.terminate()
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=True,
                    signal_used=signal_used,
                )
            )
        except psutil.NoSuchProcess:
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=False,
                    signal_used=signal_used,
                    error="already exited",
                )
            )
        except psutil.AccessDenied:
            results.append(
                KillResult(
                    pid=target.pid,
                    name=target.name,
                    success=False,
                    signal_used=signal_used,
                    error="permission denied",
                )
            )
        except OSError as exc:
            results.append(
                KillResult(pid=target.pid, name=target.name, success=False,                 signal_used=signal_used, error=str(exc))
            )
    return results

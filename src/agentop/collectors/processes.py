"""Process discovery and categorization for agent/tool runtimes.

The classification logic (`classify`) is a pure function with no OS
dependency, so it can be exhaustively unit tested with fabricated
(exe, cmdline, name) triples. `collect_agent_processes` is the impure
half that walks live OS processes via psutil and calls `classify` on each.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from agentop.models import AgentProcess, Category, Risk

# Cache of psutil.Process objects across polls, keyed by pid. psutil requires
# a "priming" call to cpu_percent() before it returns a meaningful (non-zero)
# delta on the next call — recreating Process objects every poll would make
# every reading show 0.0%, so we deliberately keep this cache alive between
# collection cycles for the lifetime of the running agentop instance.
_process_cache: dict[int, psutil.Process] = {}


@dataclass(frozen=True)
class ClassifyResult:
    category: Category
    subtype: str
    risk: Risk


def _extract_mcp_subtype(cmdline: list[str]) -> str:
    """Derive a human-readable tool name from `agency mcp [native] <tool> ...`."""
    try:
        idx = cmdline.index("mcp")
    except ValueError:
        return "unknown"
    rest = cmdline[idx + 1 :]
    native = False
    for tok in rest:
        if tok.startswith("-"):
            break
        if tok == "native":
            native = True
            continue
        return f"{tok} (native)" if native else tok
    return "unknown"


def _executable_stem(value: str) -> str:
    base = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return base[:-4] if base.endswith(".exe") else base


def classify(exe: str | None, cmdline: list[str] | None, name: str | None) -> ClassifyResult | None:
    """Return a (category, subtype, risk) classification, or None to exclude.

    Deliberately allowlist-based (explicit known signatures) rather than a
    generic "agent" substring match — that would also catch unrelated macOS
    system agents (trustd, ReportCrash, assessmentagent, rapportd, ...).
    """
    exe = exe or ""
    name = name or ""
    cmdline = cmdline or []
    joined = " ".join(cmdline).lower()
    command_tokens = [token.casefold() for token in cmdline]
    base = _executable_stem(exe or name)
    normalized_exe = exe.replace("\\", "/").casefold()

    # --- Model servers (Ollama) ---
    if base == "ollama" and "serve" in command_tokens:
        return ClassifyResult(Category.MODEL_SERVER, "ollama-server", Risk.MEDIUM)
    if "ollama.app/contents/macos/ollama" in normalized_exe:
        return ClassifyResult(Category.MODEL_SERVER, "ollama-app", Risk.MEDIUM)
    if base in {
        "llama-server",
        "llama_server",
        "ollama_llama_server",
    }:
        return ClassifyResult(Category.MODEL_SERVER, "llama-server (inference)", Risk.HIGH)
    if base == "ollama":
        return ClassifyResult(Category.MODEL_SERVER, "ollama-app", Risk.MEDIUM)

    # --- Claude Desktop ---
    if "claude.app/contents/macos/claude" in exe.lower():
        return ClassifyResult(Category.CLAUDE_DESKTOP, "app", Risk.HIGH)
    if "claude.app" in exe.lower():
        return ClassifyResult(Category.CLAUDE_DESKTOP, "helper", Risk.LOW)

    # --- ChatGPT / Codex ---
    if "chatgpt.app/contents/macos/chatgpt" in exe.lower():
        return ClassifyResult(Category.CHATGPT_CODEX, "app", Risk.HIGH)
    if "codex framework" in exe.lower():
        if "service" in exe.lower():
            return ClassifyResult(Category.CHATGPT_CODEX, "codex-service", Risk.MEDIUM)
        if "renderer" in exe.lower():
            return ClassifyResult(Category.CHATGPT_CODEX, "codex-renderer", Risk.LOW)
        return ClassifyResult(Category.CHATGPT_CODEX, "codex-helper", Risk.LOW)
    if "chatgpt.app" in exe.lower():
        return ClassifyResult(Category.CHATGPT_CODEX, "helper", Risk.LOW)

    # --- Cursor ---
    if "cursor.app/contents/macos/cursor" in exe.lower():
        return ClassifyResult(Category.CURSOR, "app", Risk.HIGH)
    if "cursor.app" in exe.lower():
        return ClassifyResult(Category.CURSOR, "helper", Risk.LOW)

    # --- Copilot CLI session (this tool's own family) ---
    if "github-copilot-sdk/cli/" in normalized_exe and base == "copilot":
        return ClassifyResult(Category.COPILOT_SESSION, "cli", Risk.HIGH)

    # --- Copilot VS Code extension host (headless) ---
    if "copilot" in joined and "--headless" in cmdline:
        return ClassifyResult(Category.COPILOT_EXTENSION, "extension-host", Risk.HIGH)

    # --- Copilot CLI plugin: computer-use MCP ---
    if "computer-use-mcp" in joined:
        return ClassifyResult(Category.MCP_TOOL, "computer-use", Risk.LOW)

    # --- agency runtime MCP tool servers ---
    if base == "agency" and "mcp" in cmdline:
        subtype = _extract_mcp_subtype(cmdline)
        return ClassifyResult(Category.MCP_TOOL, subtype, Risk.LOW)

    # --- WorkIQ MCP (direct, via node, or via `npm exec`) ---
    if "workiq" in joined and "mcp" in cmdline:
        return ClassifyResult(Category.WORKIQ, "workiq", Risk.LOW)

    # --- Generic/unrecognized MCP process: still surfaced, not silently dropped ---
    if "mcp" in cmdline:
        return ClassifyResult(Category.OTHER_AGENT, "unknown-mcp", Risk.LOW)

    return None


def collect_agent_processes() -> list[AgentProcess]:
    """Enumerate live processes and return the ones classified as agent-related.

    Every per-process operation is individually guarded: processes can exit
    between enumeration and detail-fetch (a very common race with short-lived
    MCP server subprocesses), and macOS can raise AccessDenied for processes
    owned by other users — neither should ever crash the collector.
    """
    live_pids: set[int] = set()
    results: list[AgentProcess] = []

    for proc in psutil.process_iter(["pid", "ppid", "name"]):
        pid = proc.info["pid"]
        live_pids.add(pid)
        try:
            cached = _process_cache.get(pid)
            if cached is None or cached.create_time() != proc.create_time():
                cached = psutil.Process(pid)
                _process_cache[pid] = cached
                cached.cpu_percent(None)  # prime; first read is always 0.0

            exe = ""
            try:
                exe = cached.exe()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                pass

            try:
                cmdline = cached.cmdline()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                cmdline = []

            name = proc.info.get("name") or ""
            match = classify(exe, cmdline, name)
            if match is None:
                continue

            cpu = cached.cpu_percent(None)
            try:
                mem_mb = cached.memory_info().rss / (1024 * 1024)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                mem_mb = 0.0
            try:
                create_time = cached.create_time()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                create_time = time.time()
            ppid = proc.info.get("ppid") or 0

            results.append(
                AgentProcess(
                    pid=pid,
                    ppid=ppid,
                    name=name,
                    category=match.category,
                    subtype=match.subtype,
                    cmdline=" ".join(cmdline) if cmdline else name,
                    cpu_percent=cpu,
                    mem_mb=mem_mb,
                    create_time=create_time,
                    risk=match.risk,
                    session_key=ppid,
                )
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue

    # Evict cache entries for processes that no longer exist to bound memory.
    for stale_pid in list(_process_cache.keys()):
        if stale_pid not in live_pids:
            _process_cache.pop(stale_pid, None)

    return results

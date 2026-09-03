# agentop

A local, terminal-based control plane for monitoring — and killing — AI
agent processes running across apps on this machine: Copilot CLI sessions,
MCP tool servers, WorkIQ, Ollama/model servers, Claude Desktop, ChatGPT/Codex,
and Cursor.

Built with [Textual](https://textual.textualize.io/) for a btop-style TUI
with full mouse support (click, scroll, drag).

## Install

```bash
cd agentop
uv tool install --editable .
```

This puts an `agentop` command on your `PATH` (via `~/.local/bin`, backed by
an isolated uv-managed virtualenv — editable, so code changes take effect
immediately without reinstalling).

## Run

```bash
agentop
```

## What it shows

- **Overview** — agent processes grouped by category (Copilot CLI, MCP tool
  servers, WorkIQ, model servers, Claude/ChatGPT/Cursor desktop apps), each
  with a live process count, total CPU%, and total memory.
- **System strip** — live CPU, Apple GPU, memory, swap, and Ollama status.
  GPU telemetry uses macOS's AGXAccelerator counters and shows `N/A` on
  unsupported platforms.
- **Processes** — every detected agent process individually: PID, category,
  tool name, CPU%, memory, uptime, and a risk rating for what happens if you
  kill it. Click a column header to sort, click a row to select it.
- **Models** — Ollama status: loaded models (with GPU/CPU split and context
  size) and all locally available models.
- **Network** — listening ports owned by agent processes.

## Kill switch

Select a process and use **Kill Selected**, or use **Kill Switch** for a
scoped bulk action: an entire session, an entire category (e.g. every MCP
tool server), or every detected agent process at once. Every kill:

1. Shows the exact PID list that will be signaled before you confirm.
2. Requires an explicit confirmation click — bulk scopes above "session"
   require typing `KILL` into the confirmation dialog.
3. Sends `SIGTERM` first. If a process ignores it, a separate, separately
   confirmed **Force Kill (SIGKILL)** action is offered.

Nothing is ever targeted by name/pattern (no `pkill`/`killall` equivalents)
— only exact PIDs already enumerated by the app's own process scan.

## Keybindings

| Key | Action |
|-----|--------|
| `r` | Refresh now |
| `k` | Kill selected process |
| `K` | Open kill switch (bulk) |
| `tab` / `shift+tab` | Switch panel |
| `q` | Quit |

All actions are also available as clickable buttons in the toolbar/footer.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

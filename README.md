# agentop

A local, terminal-based control plane for monitoring — and killing — AI
agent processes running across apps on this machine: Copilot CLI sessions,
MCP tool servers, WorkIQ, Ollama/model servers, Claude Desktop, ChatGPT/Codex,
and Cursor.

Built with [Textual](https://textual.textualize.io/) as a compact, dark
operations console with full mouse support.

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

- **Resource header** — live CPU, Apple GPU, memory, swap, and Ollama status
  remain visible while switching workspaces.
  GPU telemetry uses macOS's AGXAccelerator counters and shows `N/A` on
  unsupported platforms.
- **Overview** — agent processes grouped by category (Copilot CLI, MCP tool
  servers, WorkIQ, model servers, Claude/ChatGPT/Cursor desktop apps), each
  with a live process count, total CPU%, and total memory.
- **Processes** — every detected agent process individually: PID, category,
  tool name, CPU%, memory, uptime, and a risk rating for what happens if you
  kill it. Kill controls live in this workspace so they remain contextual.
- **Models** — separate resident and available model tables, model filtering,
  processor and memory details, family/parameter/quantization metadata,
  projected fit against current system memory, and recent load/unload events.
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
| `1`–`4` | Open Overview, Processes, Models, or Network |
| `/` | Focus the model filter while in Models |
| `r` | Refresh now |
| `k` | Kill selected process |
| `S` | Kill selected session |
| `K` | Kill selected category |
| `A` | Kill all detected agent processes |
| `q` | Quit |

Tabs, tables, filters, and process kill controls are mouse-accessible.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

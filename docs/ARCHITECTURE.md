# Architecture and data integrity

## Runtime boundaries

`agentop` has three independent modes:

1. **TUI** — polls Ollama asynchronously, collects host metrics in worker
   threads, and reads summarized local history.
2. **JSON snapshot** — performs one bounded collection and exits.
3. **Proxy** — forwards requests only to the configured Ollama host and
   persists completion metadata.

The proxy is optional. Without it, completion-derived UI sections are hidden
or explicitly labeled unavailable; agentop does not manufacture reliability
or throughput values.

## Polling and stale state

Resident state is sampled every configured refresh interval. Installed model
metadata is cached and refreshed every 30 seconds by default; `/api/show`
fan-out is bounded. Connection failures use exponential backoff. The TUI keeps
last-known values, labels them stale, and never blocks its event loop on HTTP,
SQLite, process enumeration, or GPU commands.

## Event store

The SQLite database uses WAL mode. It stores:

- completion timing and token counters,
- client identifiers supplied to the proxy,
- model action and health events,
- model pin preferences.

Stale in-flight requests are recovered periodically. Data older than 30 days
is deleted by scheduled maintenance. The database never contains prompt or
response bodies.

## Proxy safety

- Request bodies are bounded before forwarding.
- The upstream endpoint is fixed by configuration; client input cannot choose
  another host.
- Hop-by-hop headers are stripped.
- NDJSON and OpenAI SSE are parsed incrementally.
- Only the terminal frame is retained in memory; long completions do not lose
  final usage data.
- Cancelled or broken streams are recorded as interrupted, not successful.

## Model actions

All model mutations are serialized. Warm-up pre-flight calculations reserve
configurable host headroom and apply a configurable runtime-overhead estimate.
The plan is re-polled after confirmation. If a target load fails after
confirmed evictions, agentop attempts to restore those runners.

Context reload changes a runner's default `num_ctx`; it does not rewrite active
conversation state. The UI says this explicitly.

## Platform metrics

- **Apple Silicon:** IORegistry AGX utilization and GPU allocation. Apple
  unified memory is labeled `UNIFIED`, never represented as dedicated VRAM.
- **NVIDIA:** NVML utilization, VRAM, temperature, and power.
- **AMD:** `rocm-smi` utilization, VRAM, temperature, and power when available.

Unavailable metrics remain `N/A`.

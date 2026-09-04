"""Persistent local event and completion history."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager
from collections.abc import Iterator

import psutil

from agentop.models import CompletionRecord, EventRecord, ModelMetrics, SessionMetric


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


class EventStore:
    """SQLite-backed rolling history. A connection is opened per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_timestamp_idx
                    ON events(timestamp DESC);
                CREATE TABLE IF NOT EXISTS completions (
                    id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    client TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    success INTEGER,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    prompt_eval_ns INTEGER NOT NULL DEFAULT 0,
                    eval_ns INTEGER NOT NULL DEFAULT 0,
                    total_ns INTEGER NOT NULL DEFAULT 0,
                    ttft_ms REAL,
                    error_type TEXT NOT NULL DEFAULT '',
                    context_window INTEGER NOT NULL DEFAULT 0,
                    batch_size INTEGER NOT NULL DEFAULT 0,
                    owner_pid INTEGER,
                    owner_started_at REAL
                );
                CREATE INDEX IF NOT EXISTS completions_model_time_idx
                    ON completions(model, started_at DESC);
                CREATE TABLE IF NOT EXISTS model_totals (
                    model TEXT PRIMARY KEY,
                    run_count INTEGER NOT NULL,
                    last_run_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pins (
                    model TEXT PRIMARY KEY,
                    pinned INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                );
                INSERT OR IGNORE INTO model_totals(model, run_count, last_run_at)
                SELECT model, COUNT(*), MAX(started_at)
                FROM completions
                WHERE completed_at IS NOT NULL
                GROUP BY model;
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(completions)"
                ).fetchall()
            }
            if "owner_pid" not in columns:
                connection.execute(
                    "ALTER TABLE completions ADD COLUMN owner_pid INTEGER"
                )
            if "owner_started_at" not in columns:
                connection.execute(
                    "ALTER TABLE completions ADD COLUMN owner_started_at REAL"
                )
        self.recover_stale_requests()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_event(
        self,
        level: str,
        kind: str,
        message: str,
        *,
        model: str = "",
        timestamp: float | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO events(timestamp, level, kind, model, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp or time.time(), level, kind, model, message),
            )

    def recent_events(self, limit: int = 20) -> list[EventRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT timestamp, level, kind, model, message
                FROM events ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            EventRecord(
                timestamp=row["timestamp"],
                level=row["level"],
                kind=row["kind"],
                model=row["model"],
                message=row["message"],
            )
            for row in rows
        ]

    def begin_request(self, model: str, client: str, started_at: float) -> int:
        owner_pid = os.getpid()
        try:
            owner_started_at = psutil.Process(owner_pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            owner_started_at = None
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO completions(
                    model, client, started_at, owner_pid, owner_started_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model,
                    client,
                    started_at,
                    owner_pid,
                    owner_started_at,
                ),
            )
            return int(cursor.lastrowid)

    def finish_request(self, request_id: int, record: CompletionRecord) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE completions
                SET completed_at = ?, success = ?, prompt_tokens = ?,
                    output_tokens = ?, prompt_eval_ns = ?, eval_ns = ?,
                    total_ns = ?, ttft_ms = ?, error_type = ?,
                    context_window = ?, batch_size = ?
                WHERE id = ? AND completed_at IS NULL
                """,
                (
                    record.completed_at,
                    int(record.success),
                    record.prompt_tokens,
                    record.output_tokens,
                    record.prompt_eval_ns,
                    record.eval_ns,
                    record.total_ns,
                    record.ttft_ms,
                    record.error_type,
                    record.context_window,
                    record.batch_size,
                    request_id,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    INSERT INTO model_totals(model, run_count, last_run_at)
                    VALUES (?, 1, ?)
                    ON CONFLICT(model) DO UPDATE
                    SET run_count = model_totals.run_count + 1,
                        last_run_at = excluded.last_run_at
                    """,
                    (record.model, record.started_at),
                )

    def record_completion(self, record: CompletionRecord) -> None:
        request_id = self.begin_request(record.model, record.client, record.started_at)
        self.finish_request(request_id, record)

    def set_pinned(self, model: str, pinned: bool) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO pins(model, pinned, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(model) DO UPDATE
                SET pinned = excluded.pinned, updated_at = excluded.updated_at
                """,
                (model, int(pinned), time.time()),
            )

    def pinned_models(self) -> set[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT model FROM pins WHERE pinned = 1"
            ).fetchall()
        return {str(row["model"]) for row in rows}

    def model_metrics(
        self,
        model: str,
        *,
        now: float | None = None,
        reliability_window: float = 86400.0,
        _recover: bool = True,
    ) -> ModelMetrics:
        now = now or time.time()
        if _recover:
            self.recover_stale_requests(now=now)
        since = now - reliability_window
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM completions
                WHERE model = ? AND started_at >= ? AND completed_at IS NOT NULL
                ORDER BY started_at DESC
                """,
                (model, since),
            ).fetchall()
            total_row = connection.execute(
                "SELECT run_count, last_run_at FROM model_totals WHERE model = ?",
                (model,),
            ).fetchone()
            lifetime_count = int(total_row["run_count"]) if total_row else 0
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM completions WHERE model = ? AND completed_at IS NULL",
                    (model,),
                ).fetchone()[0]
            )

        successes = [row for row in rows if row["success"] == 1]
        latest = successes[0] if successes else None
        latencies = [
            row["total_ns"] / 1_000_000
            for row in successes
            if row["total_ns"] > 0
        ]
        ttfts = [
            float(row["ttft_ms"])
            for row in successes
            if row["ttft_ms"] is not None
        ]
        recent = [row for row in rows if row["started_at"] >= now - 60]
        sessions_by_client: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            sessions_by_client.setdefault(str(row["client"]), []).append(row)
        sessions = []
        for client, client_rows in sorted(
            sessions_by_client.items(),
            key=lambda item: max(row["started_at"] for row in item[1]),
            reverse=True,
        )[:5]:
            first = min(row["started_at"] for row in client_rows)
            last = max(row["completed_at"] or row["started_at"] for row in client_rows)
            sessions.append(
                SessionMetric(
                    client=client,
                    request_count=len(client_rows),
                    held_seconds=max(0.0, last - first),
                    idle_seconds=max(0.0, now - last),
                )
            )

        def error_count(name: str) -> int:
            return sum(1 for row in rows if row["error_type"] == name)

        return ModelMetrics(
            model=model,
            run_count=lifetime_count,
            last_run_at=(
                float(total_row["last_run_at"]) if total_row else None
            ),
            generation_tps=(
                latest["output_tokens"] / (latest["eval_ns"] / 1_000_000_000)
                if latest and latest["eval_ns"] > 0
                else None
            ),
            prompt_tps=(
                latest["prompt_tokens"]
                / (latest["prompt_eval_ns"] / 1_000_000_000)
                if latest and latest["prompt_eval_ns"] > 0
                else None
            ),
            ttft_p50_ms=_percentile(ttfts, 0.50),
            tokens_in_60s=sum(row["prompt_tokens"] for row in recent),
            tokens_out_60s=sum(row["output_tokens"] for row in recent),
            in_flight=active,
            latency_p50_ms=_percentile(latencies, 0.50),
            latency_p95_ms=_percentile(latencies, 0.95),
            success_rate=(len(successes) / len(rows) * 100) if rows else None,
            oom_count=error_count("oom"),
            timeout_count=error_count("timeout"),
            context_overflow_count=error_count("context_overflow"),
            last_context_used=(
                latest["prompt_tokens"] + latest["output_tokens"] if latest else 0
            ),
            last_context_window=latest["context_window"] if latest else 0,
            batch_size=latest["batch_size"] if latest else 0,
            sessions=sessions,
        )

    def all_model_metrics(self, models: list[str]) -> dict[str, ModelMetrics]:
        now = time.time()
        self.recover_stale_requests(now=now)
        return {
            model: self.model_metrics(model, now=now, _recover=False)
            for model in models
        }

    def recover_stale_requests(
        self,
        *,
        now: float | None = None,
        max_age_seconds: float = 600,
    ) -> int:
        now = now or time.time()
        with self._connection() as connection:
            stale_rows = connection.execute(
                """
                SELECT id, model, started_at, owner_pid, owner_started_at
                FROM completions
                WHERE completed_at IS NULL
                """,
            ).fetchall()
            recovered = 0
            for row in stale_rows:
                owner_pid = row["owner_pid"]
                owner_started_at = row["owner_started_at"]
                owner_alive = False
                if owner_pid is not None and owner_started_at is not None:
                    try:
                        live_started_at = psutil.Process(
                            int(owner_pid)
                        ).create_time()
                        owner_alive = (
                            abs(live_started_at - float(owner_started_at))
                            <= 0.01
                        )
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        owner_alive = False
                    except psutil.AccessDenied:
                        owner_alive = True
                elif float(row["started_at"]) >= now - max_age_seconds:
                    owner_alive = True
                if owner_alive:
                    continue

                cursor = connection.execute(
                    """
                    UPDATE completions
                    SET completed_at = ?, success = 0,
                        error_type = 'interrupted'
                    WHERE id = ? AND completed_at IS NULL
                    """,
                    (now, row["id"]),
                )
                if not cursor.rowcount:
                    continue
                recovered += 1
                connection.execute(
                    """
                    INSERT INTO model_totals(model, run_count, last_run_at)
                    VALUES (?, 1, ?)
                    ON CONFLICT(model) DO UPDATE
                    SET run_count = model_totals.run_count + 1,
                        last_run_at = MAX(model_totals.last_run_at, excluded.last_run_at)
                    """,
                    (row["model"], row["started_at"]),
                )
            return recovered

    def cleanup(self, retention_days: int = 30) -> None:
        cutoff = time.time() - retention_days * 86400
        with self._connection() as connection:
            connection.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            connection.execute(
                "DELETE FROM completions WHERE started_at < ?",
                (cutoff,),
            )


class NullEventStore(EventStore):
    """Fail-open store used when the persistent database is unavailable."""

    def __init__(self, path: str | Path, reason: str = "") -> None:
        super().__init__(path)
        self.reason = reason

    def initialize(self) -> None:
        return None

    def record_event(self, *args, **kwargs) -> None:
        return None

    def recent_events(self, limit: int = 20) -> list[EventRecord]:
        return []

    def begin_request(self, model: str, client: str, started_at: float):
        return None

    def finish_request(self, request_id, record: CompletionRecord) -> None:
        return None

    def record_completion(self, record: CompletionRecord) -> None:
        return None

    def set_pinned(self, model: str, pinned: bool) -> None:
        return None

    def pinned_models(self) -> set[str]:
        return set()

    def model_metrics(self, model: str, **kwargs) -> ModelMetrics:
        return ModelMetrics(model=model)

    def all_model_metrics(self, models: list[str]) -> dict[str, ModelMetrics]:
        return {model: ModelMetrics(model=model) for model in models}

    def recover_stale_requests(self, **kwargs) -> int:
        return 0

    def cleanup(self, retention_days: int = 30) -> None:
        return None

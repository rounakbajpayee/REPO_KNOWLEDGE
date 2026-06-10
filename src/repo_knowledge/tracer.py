"""
tracer.py — Structured JSONL tracer for REPO_KNOWLEDGE.

Replaces the flat logger with a schema that carries:
  ts, trace_id, event, severity, subsystem, duration_ms (optional), payload

Every MCP tool invocation generates one trace_id (8-char hex). All log lines
emitted during that call share the same trace_id, making it trivial to grep
a complete trace for any tool invocation.

Public API
----------
new_trace_id() -> str
    Generate a fresh 8-char random hex trace ID. Call once per tool invocation.

trace(event, *, subsystem, trace_id=None, severity="INFO", duration_ms=None, **payload) -> None
    Write one structured JSONL line. Non-blocking. Never raises.

Subsystem labels (by convention):
    "mcp"       — MCP server entry/exit
    "knowledge" — KnowledgeService methods
    "embedder"  — Ollama embed calls
    "store"     — Qdrant read/write

Severity values: DEBUG | INFO | WARNING | ERROR

Log file: <repo_root>/logs/repo_knowledge.jsonl  (same file as legacy logger)
"""

from __future__ import annotations

import contextvars
import json
import queue
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Log file location ─────────────────────────────────────────────────────────
# Resolve relative to this file: src/repo_knowledge/tracer.py → ../../logs/
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "repo_knowledge.jsonl"

# ── Background writer ─────────────────────────────────────────────────────────
# The JSONL write queue. The writer thread drains this quickly (no I/O blocking).
# A separate DB-flush thread handles Postgres batching so slow/absent Postgres
# never stalls the main writer loop.
_queue: queue.Queue[str] = queue.Queue()

# Separate queue feeding the Postgres batch-flush thread.
_db_queue: queue.Queue[dict] = queue.Queue()

# Batch flush parameters
_BATCH_MAX = 50  # max records per DB flush
_BATCH_TIMEOUT = 0.5  # seconds to wait before flushing a partial batch


def _writer_loop() -> None:
    """Drain _queue and write each record to JSONL. Never touches Postgres.

    Parsed records are forwarded to _db_queue for the DB flush thread.
    This loop must never block on I/O that could be slow or absent (e.g. Postgres).
    """
    while True:
        try:
            line = _queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # Write to JSONL log file immediately.
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

        # Forward parsed record to the DB flush thread (non-blocking).
        try:
            record = json.loads(line)
            _db_queue.put_nowait(
                {
                    "ts_str": record["ts"],
                    "trace_id": record.get("trace_id"),
                    "event": record["event"],
                    "severity": record.get("severity", "INFO"),
                    "subsystem": record.get("subsystem", "unknown"),
                    "duration_ms": record.get("duration_ms"),
                    "payload": record.get("payload"),
                }
            )
        except Exception:
            pass


def _db_flush_loop() -> None:
    """Batch-flush parsed records to Postgres. May block on network I/O; that is fine
    because this loop never touches _queue, so the JSONL writer is never stalled.
    """
    from repo_knowledge.postgres_store import PostgresStore

    pg: PostgresStore | None = None
    try:
        pg = PostgresStore()
    except Exception:
        pass

    pending: list[dict] = []
    last_flush = time.monotonic()

    while True:
        try:
            record = _db_queue.get(timeout=_BATCH_TIMEOUT)
            pending.append(record)
        except queue.Empty:
            pass

        now = time.monotonic()
        should_flush = len(pending) >= _BATCH_MAX or (
            pending and (now - last_flush) >= _BATCH_TIMEOUT
        )

        if should_flush:
            batch = pending[:]
            pending.clear()
            last_flush = now
            try:
                if pg is None:
                    pg = PostgresStore()
                pg.log_audit_traces_batch(batch)
            except Exception:
                pg = None  # Force re-init on next flush


_writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="tracer-writer")
_writer_thread.start()

_db_flush_thread = threading.Thread(target=_db_flush_loop, daemon=True, name="tracer-db-flush")
_db_flush_thread.start()


# ── Public API ────────────────────────────────────────────────────────────────


def new_trace_id() -> str:
    """Return a fresh 8-character lowercase hex string. Cryptographically random."""
    return secrets.token_hex(4)  # 4 bytes → 8 hex chars


_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def set_trace_id(trace_id: str | None) -> contextvars.Token:
    return _trace_id_var.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id_var.get()


def reset_trace_id(token: contextvars.Token) -> None:
    _trace_id_var.reset(token)


def trace(
    event: str,
    *,
    subsystem: str,
    trace_id: str | None = None,
    severity: str = "INFO",
    duration_ms: int | None = None,
    **payload: Any,
) -> None:
    """Enqueue one structured JSONL record. Non-blocking. Never raises.

    duration_ms is omitted from the output entirely when None (not written as null).
    payload fields are nested under a "payload" key.
    """
    if trace_id is None:
        trace_id = get_trace_id()
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event,
        "severity": severity,
        "subsystem": subsystem,
    }
    if duration_ms is not None:
        record["duration_ms"] = duration_ms
    if payload:
        record["payload"] = payload

    try:
        line = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        # Unserializable payload — degrade gracefully
        try:
            safe_payload = {k: repr(v) for k, v in payload.items()}
            record["payload"] = safe_payload
            record["_serialization_error"] = True
            line = json.dumps(record, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return  # last resort — silently discard

    _queue.put(line)

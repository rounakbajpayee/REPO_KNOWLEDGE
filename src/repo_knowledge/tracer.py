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

import json
import queue
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Log file location ─────────────────────────────────────────────────────────
# Resolve relative to this file: src/repo_knowledge/tracer.py → ../../logs/
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "repo_knowledge.jsonl"

# ── Background writer ─────────────────────────────────────────────────────────
_queue: queue.SimpleQueue[str] = queue.SimpleQueue()


def _writer_loop() -> None:
    """Drain the queue and append each line to the log file and PostgreSQL. Never dies."""
    from repo_knowledge.postgres_store import PostgresStore
    pg = None
    try:
        pg = PostgresStore()
    except Exception:
        pass

    while True:
        line = _queue.get()  # blocks until an item arrives
        # 1. Write to JSONL log file
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # silently discard on disk error

        # 2. Write to PostgreSQL database
        try:
            record = json.loads(line)
            if pg is None:
                pg = PostgresStore()
            pg.log_audit_trace(
                ts_str=record["ts"],
                trace_id=record.get("trace_id"),
                event=record["event"],
                severity=record.get("severity", "INFO"),
                subsystem=record.get("subsystem", "unknown"),
                duration_ms=record.get("duration_ms"),
                payload=record.get("payload")
            )
        except Exception:
            pg = None  # Force re-initialization next time if it fails
            pass



_writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="tracer-writer")
_writer_thread.start()


# ── Public API ────────────────────────────────────────────────────────────────

def new_trace_id() -> str:
    """Return a fresh 8-character lowercase hex string. Cryptographically random."""
    return secrets.token_hex(4)  # 4 bytes → 8 hex chars


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

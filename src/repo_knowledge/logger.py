"""
logger.py — Lightweight structured logger for REPO_KNOWLEDGE.

Adapted from APRIL/src/debug_log.py.

Writes newline-delimited JSON to logs/repo_knowledge.jsonl.
Each record: {ts, event, **payload}

Implementation note: log() is non-blocking — it enqueues a pre-serialised
JSON string onto a SimpleQueue. A single daemon thread drains the queue and
appends to the log file. This avoids 100+ open/close cycles during reindex.

Usage:
    from repo_knowledge.logger import log
    log("index_start", project="LENS", file_count=42)
    log("embed_batch", project="LENS", batch=1, size=32, duration_ms=840)
    log("index_complete", project="LENS", chunks=187, duration_ms=12400)
    log("search", query="auth flow", project="LENS", top_k=5, results=5)
    log("error", event_source="embedder", message="Ollama timed out", project="LENS")

Log file location: <repo_root>/logs/repo_knowledge.jsonl
Read recent logs: tail_log(n=50)
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve log dir relative to this file: src/repo_knowledge/ -> ../../logs/
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "repo_knowledge.jsonl"

# Internal sentinel used to signal the writer thread to stop (tests only).
_STOP = object()

_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()


def _writer_loop() -> None:
    """Background daemon: drain queue and append records to the log file."""
    while True:
        item = _queue.get()  # blocks until something arrives
        if item is _STOP:
            break
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(item + "\n")
        except OSError:
            pass  # never let the writer thread die


_writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="log-writer")
_writer_thread.start()


def log(event: str, **payload: Any) -> None:
    """Enqueue a structured event for background write. Never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    try:
        line = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        # Unserializable payload — log a degraded version rather than raising.
        try:
            safe_payload = {k: repr(v) for k, v in payload.items()}
            fallback = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "_serialization_error": True,
                **safe_payload,
            }
            line = json.dumps(fallback, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return  # absolute last resort — silently discard
    _queue.put(line)


def tail_log(n: int = 50) -> list[dict[str, Any]]:
    """Return the last n log records as parsed dicts.

    Reads directly from disk; does NOT drain the queue first.
    Call after a short sleep / _drain() in tests to ensure recency.
    """
    if not _LOG_PATH.exists():
        return []
    try:
        lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records

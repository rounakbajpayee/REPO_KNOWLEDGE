"""
logger.py — Lightweight structured logger for REPO_KNOWLEDGE.

Adapted from APRIL/src/debug_log.py.

Writes newline-delimited JSON to logs/repo_knowledge.jsonl.
Each record: {ts, event, **payload}

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
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

# Resolve log dir relative to this file: src/repo_knowledge/ -> ../../logs/
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_PATH = _LOG_DIR / "repo_knowledge.jsonl"
_lock = Lock()


def log(event: str, **payload: Any) -> None:
    """Write a structured event to the log file. Never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            with _LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def tail_log(n: int = 50) -> list[dict[str, Any]]:
    """Return the last n log records as parsed dicts."""
    if not _LOG_PATH.exists():
        return []
    try:
        with _lock:
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

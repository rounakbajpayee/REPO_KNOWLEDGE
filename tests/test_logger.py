"""
tests/test_logger.py — Unit tests for the background-thread logger.

The logger is async in the sense that log() enqueues and a daemon thread
writes. Tests use the internal _flush() helper (or join the writer thread)
to drain the queue before asserting on file contents.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_logger(tmp_path: Path):
    """
    Re-import logger with LOG_DIR redirected to tmp_path so tests don't
    touch the real logs/ directory and don't interfere with each other.
    """
    import repo_knowledge.logger as logger_mod
    # Patch the private path constants and restart the background thread.
    logger_mod._LOG_DIR = tmp_path
    logger_mod._LOG_PATH = tmp_path / "repo_knowledge.jsonl"
    return logger_mod


def _drain(logger_mod, timeout: float = 2.0) -> None:
    """Block until the background writer queue is empty (or timeout)."""
    deadline = time.monotonic() + timeout
    while not logger_mod._queue.empty():
        if time.monotonic() > deadline:
            raise TimeoutError("Logger queue did not drain within timeout")
        time.sleep(0.01)
    # Give the thread one extra tick to finish the last write.
    time.sleep(0.05)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_log_writes_event(tmp_path: Path) -> None:
    """log() eventually writes a JSONL record with the correct event name."""
    logger_mod = _reload_logger(tmp_path)

    logger_mod.log("test_event", key="value")
    _drain(logger_mod)

    log_file = tmp_path / "repo_knowledge.jsonl"
    assert log_file.exists(), "Log file was not created"
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["event"] == "test_event"
    assert record["key"] == "value"
    assert "ts" in record


def test_log_never_raises(tmp_path: Path) -> None:
    """log() must not raise even when given an unserializable payload."""
    logger_mod = _reload_logger(tmp_path)

    # Pass an object that is not JSON-serializable.
    class _NotSerializable:
        pass

    try:
        logger_mod.log("bad_payload", obj=_NotSerializable())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"log() raised an exception: {exc}")


def test_tail_log_returns_dicts(tmp_path: Path) -> None:
    """tail_log(n) returns the last n records as parsed dicts."""
    logger_mod = _reload_logger(tmp_path)

    for i in range(3):
        logger_mod.log(f"event_{i}", index=i)
    _drain(logger_mod)

    records = logger_mod.tail_log(2)
    assert len(records) == 2
    assert all(isinstance(r, dict) for r in records)
    # Last two events must be event_1 and event_2 (in order)
    assert records[0]["event"] == "event_1"
    assert records[1]["event"] == "event_2"


def test_tail_log_empty_if_no_file(tmp_path: Path) -> None:
    """tail_log() returns [] when the log file does not exist."""
    logger_mod = _reload_logger(tmp_path)

    # Ensure the file really does not exist
    log_file = tmp_path / "repo_knowledge.jsonl"
    if log_file.exists():
        log_file.unlink()

    records = logger_mod.tail_log()
    assert records == []

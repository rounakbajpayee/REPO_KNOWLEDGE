"""
tests/test_tracer.py — Unit tests for tracer.py (Issue #2).

The tracer is non-blocking: trace() enqueues and a daemon thread writes.
Tests use _drain() to flush the queue before asserting on output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_tracer(tmp_path: Path):
    """
    Re-import tracer with LOG_PATH redirected to tmp_path so tests don't
    touch the real logs/ directory and don't interfere with each other.
    """
    import repo_knowledge.tracer as tracer_mod

    while not tracer_mod._queue.empty():
        try:
            tracer_mod._queue.get_nowait()
        except:
            break
    tracer_mod._LOG_DIR = tmp_path
    tracer_mod._LOG_PATH = tmp_path / "repo_knowledge.jsonl"
    return tracer_mod


def _drain(tracer_mod, timeout: float = 5.0) -> None:
    """Block until the background writer queue is empty (or timeout)."""
    deadline = time.monotonic() + timeout
    while not tracer_mod._queue.empty():
        if time.monotonic() > deadline:
            raise TimeoutError("Tracer queue did not drain within timeout")
        time.sleep(0.01)
    # One extra tick for the writer thread to finish the final write.
    time.sleep(0.05)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trace_writes_jsonl_line(tmp_path: Path) -> None:
    """trace() eventually writes a JSONL record with all required schema fields."""
    tracer_mod = _reload_tracer(tmp_path)

    tracer_mod.trace(
        "search", subsystem="knowledge", trace_id="aabbccdd", duration_ms=42, query="auth flow"
    )
    _drain(tracer_mod)

    log_file = tmp_path / "repo_knowledge.jsonl"
    assert log_file.exists(), "Log file was not created"
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 1
    record = json.loads(lines[-1])

    assert record["event"] == "search"
    assert record["severity"] == "INFO"
    assert record["subsystem"] == "knowledge"
    assert record["trace_id"] == "aabbccdd"
    assert record["duration_ms"] == 42
    assert "ts" in record
    assert record["payload"]["query"] == "auth flow"


def test_trace_id_in_output(tmp_path: Path) -> None:
    """trace(..., trace_id='abc123') → parsed line has trace_id: 'abc123'."""
    tracer_mod = _reload_tracer(tmp_path)

    tracer_mod.trace("tool_start", subsystem="mcp", trace_id="abc123", tool="search_codebase")
    _drain(tracer_mod)

    lines = (tmp_path / "repo_knowledge.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["trace_id"] == "abc123"


def test_trace_never_raises(tmp_path: Path) -> None:
    """trace() must not raise even when given an unserializable payload."""
    tracer_mod = _reload_tracer(tmp_path)

    class _NotSerializable:
        pass

    try:
        tracer_mod.trace("bad_payload", subsystem="knowledge", obj=_NotSerializable())
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"trace() raised an exception: {exc}")


def test_new_trace_id_is_unique(tmp_path: Path) -> None:
    """new_trace_id() called 100 times produces 100 distinct values."""
    tracer_mod = _reload_tracer(tmp_path)

    ids = [tracer_mod.new_trace_id() for _ in range(100)]
    assert len(set(ids)) == 100, "trace IDs are not unique"


def test_new_trace_id_is_hex_string(tmp_path: Path) -> None:
    """new_trace_id() returns an 8-character lowercase hex string."""
    tracer_mod = _reload_tracer(tmp_path)

    tid = tracer_mod.new_trace_id()
    assert len(tid) == 8
    assert all(c in "0123456789abcdef" for c in tid), f"Not hex: {tid!r}"


def test_severity_default_is_info(tmp_path: Path) -> None:
    """Omitting severity → parsed line has severity: 'INFO'."""
    tracer_mod = _reload_tracer(tmp_path)

    tracer_mod.trace("some_event", subsystem="store")
    _drain(tracer_mod)

    lines = (tmp_path / "repo_knowledge.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["severity"] == "INFO"


def test_duration_ms_optional(tmp_path: Path) -> None:
    """Omitting duration_ms → field is absent from the output (not null)."""
    tracer_mod = _reload_tracer(tmp_path)

    tracer_mod.trace("no_duration", subsystem="mcp")
    _drain(tracer_mod)

    lines = (tmp_path / "repo_knowledge.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert "duration_ms" not in record, "duration_ms should be absent when not provided"


def test_trace_subsystem_required(tmp_path: Path) -> None:
    """Omitting subsystem raises TypeError (keyword-only required argument)."""
    tracer_mod = _reload_tracer(tmp_path)

    with pytest.raises(TypeError):
        tracer_mod.trace("some_event")  # missing required kwarg: subsystem

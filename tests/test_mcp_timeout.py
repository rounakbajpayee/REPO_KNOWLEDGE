"""
tests/test_mcp_timeout.py — Unit tests for the extracted _dispatch() function
in mcp_server.py.

_dispatch is a pure synchronous function; no event loop needed here.
All KnowledgeService calls are mocked.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

from repo_knowledge.mcp_server import _dispatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_svc() -> MagicMock:
    svc = MagicMock()
    svc.list_projects.return_value = [{"name": "ALPHA"}]
    svc.get_project_context.return_value = {"project": "ALPHA"}
    svc.search.return_value = [{"symbol": "run"}]
    svc.get_file.return_value = {"content": "def run(): pass"}
    svc.reindex_project.return_value = {"chunks_indexed": 10}
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dispatch_unknown_tool_returns_error(mock_svc: MagicMock) -> None:
    """Unknown tool name → returns dict with 'error' key."""
    result = _dispatch(mock_svc, "nonexistent_tool", {})
    assert "error" in result
    assert "nonexistent_tool" in result["error"]


def test_dispatch_routes_list_projects(mock_svc: MagicMock) -> None:
    """_dispatch('list_projects') calls svc.list_projects() and returns its value."""
    result = _dispatch(mock_svc, "list_projects", {})
    mock_svc.list_projects.assert_called_once_with()
    assert result == [{"name": "ALPHA"}]


def test_dispatch_routes_get_project_context(mock_svc: MagicMock) -> None:
    """_dispatch('get_project_context') calls svc.get_project_context(project)."""
    result = _dispatch(mock_svc, "get_project_context", {"project": "ALPHA"})
    mock_svc.get_project_context.assert_called_once_with("ALPHA")
    assert result == {"project": "ALPHA"}


def test_dispatch_get_project_context_missing_project(mock_svc: MagicMock) -> None:
    """Missing required arg 'project' → returns error dict without calling svc."""
    result = _dispatch(mock_svc, "get_project_context", {})
    assert "error" in result
    mock_svc.get_project_context.assert_not_called()


def test_dispatch_routes_search(mock_svc: MagicMock) -> None:
    """_dispatch('search_codebase') calls svc.search() with correct args."""
    result = _dispatch(mock_svc, "search_codebase", {"query": "auth flow", "project": "ALPHA", "top_k": 3})
    mock_svc.search.assert_called_once_with(query="auth flow", project="ALPHA", top_k=3)
    assert result == [{"symbol": "run"}]


def test_dispatch_search_missing_query(mock_svc: MagicMock) -> None:
    """Missing 'query' → returns error dict without calling svc.search."""
    result = _dispatch(mock_svc, "search_codebase", {})
    assert "error" in result
    mock_svc.search.assert_not_called()


def test_dispatch_routes_get_file(mock_svc: MagicMock) -> None:
    """_dispatch('get_file') calls svc.get_file(project, path)."""
    result = _dispatch(mock_svc, "get_file", {"project": "ALPHA", "path": "src/main.py"})
    mock_svc.get_file.assert_called_once_with("ALPHA", "src/main.py", start_line=None, end_line=None)
    assert result == {"content": "def run(): pass"}


def test_dispatch_routes_get_file_with_lines(mock_svc: MagicMock) -> None:
    """_dispatch('get_file') with line bounds passes them to svc.get_file."""
    result = _dispatch(mock_svc, "get_file", {"project": "ALPHA", "path": "src/main.py", "start_line": 10, "end_line": 20})
    mock_svc.get_file.assert_called_once_with("ALPHA", "src/main.py", start_line=10, end_line=20)
    assert result == {"content": "def run(): pass"}


def test_dispatch_get_file_missing_args(mock_svc: MagicMock) -> None:
    """Missing project or path → returns error dict without calling svc.get_file."""
    result = _dispatch(mock_svc, "get_file", {"project": "ALPHA"})
    assert "error" in result
    mock_svc.get_file.assert_not_called()


def test_dispatch_routes_reindex(mock_svc: MagicMock) -> None:
    """_dispatch('reindex_project') calls svc.reindex_project(project)."""
    result = _dispatch(mock_svc, "reindex_project", {"project": "ALPHA"})
    mock_svc.reindex_project.assert_called_once_with("ALPHA")
    assert result == {"chunks_indexed": 10}


def test_dispatch_reindex_missing_project(mock_svc: MagicMock) -> None:
    """Missing 'project' → returns error dict without calling svc.reindex_project."""
    result = _dispatch(mock_svc, "reindex_project", {})
    assert "error" in result
    mock_svc.reindex_project.assert_not_called()


def test_dispatch_propagates_runtime_error(mock_svc: MagicMock) -> None:
    """If svc raises RuntimeError, _dispatch re-raises it (caller handles it)."""
    mock_svc.list_projects.side_effect = RuntimeError("Qdrant is down")

    with pytest.raises(RuntimeError, match="Qdrant is down"):
        _dispatch(mock_svc, "list_projects", {})


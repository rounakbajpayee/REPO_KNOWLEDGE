import pytest
from pathlib import Path
from unittest.mock import MagicMock
from repo_knowledge.knowledge import KnowledgeService


@pytest.fixture
def fake_projects_root(tmp_path: Path) -> Path:
    for name in ["ALPHA", "BETA"]:
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "README.md").write_text(f"# {name}\n\n## Overview\nThis is {name}.")
        src = repo / "src"
        src.mkdir()
        (src / "main.py").write_text("def run():\n    pass\n")
    return tmp_path


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.list_projects.return_value = ["ALPHA"]
    store.search.return_value = [
        {"project": "ALPHA", "path": "src/main.py", "symbol": "run",
         "content": "def run():\n    pass", "score": 0.95,
         "chunk_type": "function", "start_line": 1, "end_line": 2}
    ]
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 1024
    embedder.embed_batch.return_value = [[0.1] * 1024]
    return embedder


@pytest.fixture
def svc(fake_projects_root, mock_store, mock_embedder) -> KnowledgeService:
    return KnowledgeService(
        store=mock_store,
        embedder=mock_embedder,
        projects_root=str(fake_projects_root),
    )


def test_list_projects_returns_all_scanned(svc):
    projects = svc.list_projects()
    names = {p["name"] for p in projects}
    assert names == {"ALPHA", "BETA"}


def test_list_projects_indexed_flag(svc):
    projects = svc.list_projects()
    by_name = {p["name"]: p for p in projects}
    assert by_name["ALPHA"]["indexed"] is True
    assert by_name["BETA"]["indexed"] is False


def test_get_project_context_returns_readme(svc):
    ctx = svc.get_project_context("ALPHA")
    assert "ALPHA" in ctx["readme_excerpt"]


def test_get_project_context_returns_tree(svc):
    ctx = svc.get_project_context("ALPHA")
    assert isinstance(ctx["directory_tree"], list)
    assert len(ctx["directory_tree"]) > 0


def test_get_project_context_returns_file_count(svc):
    ctx = svc.get_project_context("ALPHA")
    assert ctx["file_count"] > 0


def test_get_project_context_unknown_project(svc):
    ctx = svc.get_project_context("NONEXISTENT")
    assert "error" in ctx


def test_get_project_context_indexed_flag(svc):
    ctx = svc.get_project_context("ALPHA")
    assert ctx["indexed"] is True
    ctx2 = svc.get_project_context("BETA")
    assert ctx2["indexed"] is False


def test_search_calls_embedder(svc, mock_embedder):
    svc.search("authentication flow")
    mock_embedder.embed.assert_called_once_with("authentication flow")


def test_search_calls_store_with_vector(svc, mock_store, mock_embedder):
    svc.search("authentication flow", top_k=3)
    mock_store.search.assert_called_once_with(
        mock_embedder.embed.return_value, top_k=3, project=None,
    )


def test_search_with_project_filter(svc, mock_store):
    svc.search("auth", project="ALPHA")
    call_kwargs = mock_store.search.call_args[1]
    assert call_kwargs["project"] == "ALPHA"


def test_search_returns_results(svc):
    results = svc.search("run function")
    assert len(results) == 1
    assert results[0]["symbol"] == "run"


def test_get_file_returns_content(svc):
    result = svc.get_file("ALPHA", "src/main.py")
    assert "def run" in result["content"]
    assert result["project"] == "ALPHA"


def test_get_file_unknown_project(svc):
    result = svc.get_file("NONEXISTENT", "src/main.py")
    assert "error" in result


def test_get_file_missing_file(svc):
    result = svc.get_file("ALPHA", "src/does_not_exist.py")
    assert "error" in result


def test_reindex_calls_delete_first(svc, mock_store):
    svc.reindex_project("ALPHA")
    mock_store.delete_project.assert_called_once_with("ALPHA")


def test_reindex_calls_upsert(svc, mock_store):
    svc.reindex_project("ALPHA")
    assert mock_store.upsert_chunks.called


def test_reindex_unknown_project(svc):
    result = svc.reindex_project("NONEXISTENT")
    assert "error" in result


def test_reindex_returns_chunk_count(svc):
    svc._embedder.embed_batch.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    result = svc.reindex_project("ALPHA")
    assert "chunks_indexed" in result
    assert result["chunks_indexed"] > 0


def test_reindex_embedder_failure_returns_error(svc, mock_embedder):
    mock_embedder.embed_batch.side_effect = RuntimeError("Ollama is down")
    result = svc.reindex_project("ALPHA")
    assert "error" in result
    assert "Ollama" in result["error"]

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repo_knowledge.knowledge import KnowledgeService


@pytest.fixture(autouse=True)
def mock_reranker_model():
    """Mock reranker model loading to degrade gracefully in unit tests."""
    with patch("repo_knowledge.reranker._load_model", return_value=None):
        yield


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
        {
            "project": "ALPHA",
            "path": "src/main.py",
            "symbol": "run",
            "content": "def run():\n    pass",
            "score": 0.95,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 2,
        }
    ]
    # Simulate no previously indexed files (all files treated as new)
    store.get_indexed_file_hashes.return_value = {}
    store.get_indexed_file_mtimes.return_value = {}
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
        mock_embedder.embed.return_value, top_k=3, project=None, query_text="authentication flow"
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


def test_get_file_returns_sliced_content(svc):
    result = svc.get_file("ALPHA", "src/main.py", start_line=1, end_line=1)
    assert result["content"] == "def run():\n"
    assert result["line_count"] == 2
    assert result["start_line"] == 1
    assert result["end_line"] == 1

    result2 = svc.get_file("ALPHA", "src/main.py", start_line=2, end_line=2)
    assert result2["content"] == "    pass\n"


def test_get_file_unknown_project(svc):
    result = svc.get_file("NONEXISTENT", "src/main.py")
    assert "error" in result


def test_get_file_missing_file(svc):
    result = svc.get_file("ALPHA", "src/does_not_exist.py")
    assert "error" in result


def test_reindex_incremental_calls_get_hashes_not_delete(svc, mock_store):
    """Default (incremental) reindex must NOT call delete_project."""
    svc.reindex_project("ALPHA")
    mock_store.delete_project.assert_not_called()
    mock_store.get_indexed_file_hashes.assert_called_once_with("ALPHA")


def test_reindex_force_calls_delete_project(svc, mock_store):
    """force=True must wipe all indexed data before re-embedding."""
    svc.reindex_project("ALPHA", force=True)
    mock_store.delete_project.assert_called_once_with("ALPHA")
    mock_store.get_indexed_file_hashes.assert_not_called()


def test_reindex_calls_upsert(svc, mock_store):
    """Both incremental and force paths must upsert chunks."""
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


def test_reindex_embedder_failure_returns_error(svc, mock_embedder, mock_store):
    """Embedder failures must be caught and returned as error dicts."""
    mock_embedder.embed_batch.side_effect = RuntimeError("RapidMLX is down")
    result = svc.reindex_project("ALPHA")
    assert "error" in result
    assert "RapidMLX" in result["error"]


def test_reindex_incremental_no_changes_returns_zero(svc, mock_store, fake_projects_root):
    """If all indexed hashes and mtimes match current files, chunks_indexed must be 0."""
    # Pre-populate hashes and mtimes matching the actual file content in fake_projects_root
    alpha_path = fake_projects_root / "ALPHA"
    import hashlib

    hashes = {}
    mtimes = {}
    for fp in alpha_path.rglob("*"):
        if fp.is_file():
            rel = str(fp.relative_to(alpha_path))
            source = fp.read_text(encoding="utf-8", errors="ignore")
            hashes[rel] = hashlib.sha256(source.encode()).hexdigest()
            mtimes[rel] = fp.stat().st_mtime
    mock_store.get_indexed_file_hashes.return_value = hashes
    mock_store.get_indexed_file_mtimes.return_value = mtimes
    result = svc.reindex_project("ALPHA")
    assert result["chunks_indexed"] == 0
    assert "No changes" in result.get("message", "")
    mock_store.upsert_chunks.assert_not_called()


def test_reindex_search_quality_good(svc):
    """Results with best score >= 0.65 must report search_quality='good'."""
    results = svc.search("run function")
    assert all(r["search_quality"] == "good" for r in results)


def test_reindex_search_quality_low(svc, mock_store):
    """Results with best score in [0.40, 0.65) must report search_quality='low'."""
    mock_store.search.return_value = [
        {
            "project": "ALPHA",
            "path": "src/main.py",
            "symbol": "run",
            "content": "def run(): pass",
            "score": 0.55,
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 2,
        }
    ]
    results = svc.search("run function")
    assert all(r["search_quality"] == "low" for r in results)


def test_reindex_search_quality_none_on_empty(svc, mock_store):
    """Empty result set must report search_quality='none' (but return empty list)."""
    mock_store.search.return_value = []
    results = svc.search("something obscure")
    assert results == []


def test_list_projects_ttl_cache_returns_cached(svc, mock_store):
    """Second list_projects call within TTL must not re-scan the store."""
    svc.list_projects()
    svc.list_projects()
    # scan_projects hits the filesystem, not the store — but list_projects hits
    # store.list_projects once per cache miss.  Two calls should yield one store hit.
    assert mock_store.list_projects.call_count == 1


def test_list_projects_cache_invalidated_after_reindex(svc, mock_store):
    """reindex_project must invalidate the list_projects TTL cache."""
    svc.list_projects()
    assert mock_store.list_projects.call_count == 1
    svc.reindex_project("ALPHA")
    svc.list_projects()
    assert mock_store.list_projects.call_count == 2


def test_get_project_context_file_count_excludes_ignore_dirs(svc, fake_projects_root):
    """file_count must not include files inside IGNORE_DIRS like __pycache__."""
    alpha = fake_projects_root / "ALPHA"
    cache_dir = alpha / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "main.cpython-311.pyc").write_bytes(b"bytecode")
    ctx = svc.get_project_context("ALPHA")
    # file_count must not include the .pyc inside __pycache__
    normal_count = sum(1 for p in alpha.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert ctx["file_count"] == normal_count


def test_list_files_returns_all_supported(svc, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()

    (proj_dir / "test.py").write_text("print(1)")
    (proj_dir / "doc.md").write_text("# Doc")
    (proj_dir / "img.png").write_text("binary")

    res = svc.list_files("LENS")
    assert "error" not in res

    files = res["files"]
    paths = [f["path"] for f in files]
    assert "test.py" in paths
    assert "doc.md" in paths
    assert "img.png" not in paths
    assert res["total"] == 2


def test_list_files_extension_filter(svc, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    (proj_dir / "a.py").write_text("a")
    (proj_dir / "b.md").write_text("b")

    res = svc.list_files("LENS", extension=".py")
    files = res["files"]
    assert len(files) == 1
    assert files[0]["path"] == "a.py"


def test_list_files_path_prefix_filter(svc, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    src_dir = proj_dir / "src"
    src_dir.mkdir()
    (src_dir / "a.py").write_text("a")
    (proj_dir / "b.py").write_text("b")

    res = svc.list_files("LENS", path_prefix="src/")
    files = res["files"]
    assert len(files) == 1
    assert files[0]["path"] == "src/a.py"


def test_list_files_unknown_project(svc, tmp_path):
    res = svc.list_files("NOT_EXIST")
    assert "error" in res


def test_list_files_excludes_ignore_dirs(svc, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    node_modules = proj_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "a.py").write_text("a")

    res = svc.list_files("LENS")
    assert res["total"] == 0


def test_list_files_returns_line_count(svc, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    (proj_dir / "test.py").write_text("1\n2\n3\n")

    res = svc.list_files("LENS")
    assert res["files"][0]["line_count"] == 3


def test_search_symbols_no_content_in_results(svc, mock_store, mock_embedder):
    mock_store.search.return_value = [
        {"path": "a.py", "symbol": "foo", "content": "body", "score": 0.9}
    ]
    mock_embedder.embed.return_value = [0.1, 0.2]

    res = svc.search_symbols("foo")
    assert "content" not in res[0]
    assert res[0]["symbol"] == "foo"


def test_search_symbols_has_required_fields(svc, mock_store, mock_embedder):
    mock_store.search.return_value = [
        {
            "path": "a.py",
            "symbol": "foo",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 10,
            "score": 0.9,
            "project": "P",
        }
    ]
    mock_embedder.embed.return_value = [0.1, 0.2]

    res = svc.search_symbols("foo")
    r = res[0]
    for k in ["path", "symbol", "chunk_type", "start_line", "end_line", "score"]:
        assert k in r


def test_search_symbols_calls_embedder(svc, mock_store, mock_embedder):
    mock_store.search.return_value = []
    svc.search_symbols("query")
    mock_embedder.embed.assert_called_once_with("query")


def test_search_symbols_project_filter_passed(svc, mock_store, mock_embedder):
    mock_store.search.return_value = []
    svc.search_symbols("query", project="ALPHA")
    mock_store.search.assert_called_once_with(
        mock_embedder.embed.return_value, top_k=10, project="ALPHA", query_text="query"
    )


def test_get_chunks_for_file_returns_symbol_map(svc, mock_store, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    mock_store.get_chunks_for_path.return_value = [
        {"symbol": "foo", "chunk_type": "func", "start_line": 10, "end_line": 20, "content": "x"}
    ]

    res = svc.get_chunks_for_file("LENS", "src/a.py")
    assert res["total"] == 1
    assert len(res["chunks"]) == 1


def test_get_chunks_for_file_no_content_in_chunks(svc, mock_store, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    mock_store.get_chunks_for_path.return_value = [
        {"symbol": "foo", "chunk_type": "func", "start_line": 10, "end_line": 20, "content": "x"}
    ]

    res = svc.get_chunks_for_file("LENS", "src/a.py")
    assert "content" not in res["chunks"][0]


def test_get_chunks_for_file_sorted_by_start_line(svc, mock_store, tmp_path):
    proj_dir = tmp_path / "LENS"
    proj_dir.mkdir()
    mock_store.get_chunks_for_path.return_value = [
        {"symbol": "b", "start_line": 20},
        {"symbol": "a", "start_line": 10},
    ]

    res = svc.get_chunks_for_file("LENS", "src/a.py")
    assert res["chunks"][0]["symbol"] == "a"
    assert res["chunks"][1]["symbol"] == "b"


def test_get_chunks_for_file_unknown_project(svc, mock_store):
    mock_store.get_chunks_for_path.return_value = []
    res = svc.get_chunks_for_file("NOT_EXIST", "src/a.py")
    assert "error" in res


def test_benchmark_embeddings_empty(svc):
    res = svc.benchmark_embeddings([])
    assert res["recall_at_5"] == 0.0
    assert res["total_queries"] == 0
    assert res["hits"] == 0


def test_benchmark_embeddings_hits(svc, mock_store, mock_embedder):
    mock_store.search.return_value = [
        {"path": "src/main.py", "score": 0.9}
    ]
    qa_pairs = [{"query": "run", "expected_path": "src/main.py"}]
    res = svc.benchmark_embeddings(qa_pairs)
    assert res["recall_at_5"] == 100.0
    assert res["total_queries"] == 1
    assert res["hits"] == 1
    assert res["breakdown"][0]["hit"] is True
    assert res["breakdown"][0]["rank"] == 1


def test_benchmark_embeddings_misses(svc, mock_store, mock_embedder):
    mock_store.search.return_value = [
        {"path": "src/other.py", "score": 0.9}
    ]
    qa_pairs = [{"query": "run", "expected_path": "src/main.py"}]
    res = svc.benchmark_embeddings(qa_pairs)
    assert res["recall_at_5"] == 0.0
    assert res["total_queries"] == 1
    assert res["hits"] == 0
    assert res["breakdown"][0]["hit"] is False
    assert res["breakdown"][0]["rank"] is None

import pytest
from pathlib import Path
from repo_knowledge.chunker import chunk_file, chunk_project, Chunk

PY_SOURCE = '''
import os
import sys

def foo():
    return 1

def bar(x: int) -> int:
    return x + 1

class MyClass:
    def method(self):
        pass
'''

MD_SOURCE = """# Title

## Section One

Some content here.

## Section Two

More content.
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def test_python_chunk_count(tmp_path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) == 3


def test_python_chunk_types(tmp_path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    types = {c.chunk_type for c in chunks}
    assert "function" in types
    assert "class" in types


def test_python_chunk_symbols(tmp_path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    symbols = {c.symbol for c in chunks}
    assert symbols == {"foo", "bar", "MyClass"}


def test_python_imports_prepended(tmp_path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    for chunk in chunks:
        assert "import os" in chunk.content
        assert "import sys" in chunk.content


def test_python_line_numbers_are_set(tmp_path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    for chunk in chunks:
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line


def test_python_invalid_syntax_falls_back_to_fixed(tmp_path):
    f = _write(tmp_path, "bad.py", "def foo(:\n    pass")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) >= 1
    assert all(c.chunk_type == "block" for c in chunks)


def test_python_no_top_level_defs_falls_back(tmp_path):
    f = _write(tmp_path, "script.py", "X = 1\nY = 2\n")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) >= 1


def test_python_async_function_chunked(tmp_path):
    src = "async def handler():\n    pass\n"
    f = _write(tmp_path, "async_mod.py", src)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert any(c.symbol == "handler" for c in chunks)


def test_markdown_chunk_count(tmp_path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    # MD_SOURCE has # Title, ## Section One, ## Section Two — all 3 are sections now
    assert len(chunks) == 3


def test_markdown_chunk_symbols(tmp_path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    symbols = [c.symbol for c in chunks]
    assert "Section One" in symbols
    assert "Section Two" in symbols


def test_markdown_chunk_type(tmp_path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert all(c.chunk_type == "section" for c in chunks)


def test_fixed_chunker_single_block_for_short_file(tmp_path):
    content = "\n".join([f"line {i}" for i in range(10)])
    f = _write(tmp_path, "config.yaml", content)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "block"


def test_fixed_chunker_multiple_blocks_for_long_file(tmp_path):
    content = "\n".join([f"line {i}" for i in range(130)])
    f = _write(tmp_path, "big.yaml", content)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) > 1


def test_empty_file_returns_no_chunks(tmp_path):
    f = _write(tmp_path, "empty.py", "")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_unsupported_extension_returns_no_chunks(tmp_path):
    f = _write(tmp_path, "image.png", "binary")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_chunk_project_skips_ignore_dirs(tmp_path):
    _write(tmp_path, "main.py", "def main():\n    pass\n")
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    _write(ignored, "lib.py", "def lib():\n    pass\n")
    chunks = chunk_project(tmp_path, "PROJ")
    paths = [c.path for c in chunks]
    assert all("node_modules" not in p for p in paths)


def test_chunk_project_metadata(tmp_path):
    _write(tmp_path, "service.py", "def serve():\n    pass\n")
    chunks = chunk_project(tmp_path, "MYPROJECT")
    assert all(c.project == "MYPROJECT" for c in chunks)
    assert all(c.language == "python" for c in chunks)


# ── Issue #3: Markdown h1 fixes ────────────────────────────────────────────────────

MD_H1_SOURCE = """# Title

Some intro content.

## Sub Section

More content here.
"""

MD_H1_ONLY = """# Title

Just some content under a top-level header.
No sub-sections at all.
"""


def test_markdown_h1_header_chunked(tmp_path):
    """# headers must be split as sections, not fall through to fixed chunking."""
    f = _write(tmp_path, "readme.md", MD_H1_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    symbols = [c.symbol for c in chunks]
    assert "Title" in symbols


def test_markdown_h1_only_no_h2_still_chunks(tmp_path):
    """A file with only a # header must produce at least one section chunk."""
    f = _write(tmp_path, "doc.md", MD_H1_ONLY)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) >= 1
    # Must not fall through to fixed chunking entirely
    assert any(c.chunk_type == "section" for c in chunks)


def test_markdown_long_section_splits_into_blocks(tmp_path):
    """A section body > 80 lines must be further split into block chunks."""
    long_body = "# BigSection\n" + "\n".join(f"line {i}" for i in range(100))
    f = _write(tmp_path, "long.md", long_body)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) > 1


# ── Issue #3: Import extraction scope ────────────────────────────────────────────

PY_METHOD_IMPORT = '''
import os

def foo():
    import sys  # method-level import — must NOT appear in sibling chunk headers
    return sys.argv

def bar():
    return os.getcwd()
'''


def test_python_method_imports_not_in_header(tmp_path):
    """Imports inside method bodies must not pollute the shared import header."""
    f = _write(tmp_path, "mod.py", PY_METHOD_IMPORT)
    chunks = chunk_file(f, tmp_path, "PROJ")
    bar_chunk = next(c for c in chunks if c.symbol == "bar")
    assert "import sys" not in bar_chunk.content


# ── Issue #3: Ignore extensions + egg-info dirs ───────────────────────────────────────────

def test_lock_file_returns_no_chunks(tmp_path):
    """package-lock.json (extension .lock after renaming) — .lock files ignored."""
    f = _write(tmp_path, "poetry.lock", '[metadata]\ncontent-hash = "abc"\n')
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_jsonl_file_returns_no_chunks(tmp_path):
    """Trace log files (.jsonl) must not be indexed."""
    f = _write(tmp_path, "trace.jsonl", '{"event": "tool_start"}\n')
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_egg_info_dir_skipped_in_project(tmp_path):
    """chunk_project() must skip .egg-info directories."""
    _write(tmp_path, "main.py", "def main():\n    pass\n")
    egg_dir = tmp_path / "mypackage.egg-info"
    egg_dir.mkdir()
    _write(egg_dir, "top_level.txt", "mypackage\n")
    chunks = chunk_project(tmp_path, "PROJ")
    paths = [c.path for c in chunks]
    assert all(".egg-info" not in p for p in paths)


# ── Issue #4: content_hash and file_mtime stamping ────────────────────────────

def test_chunk_file_stamps_content_hash(tmp_path):
    """Every chunk from a file must carry a non-empty sha256 content_hash."""
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks
    for chunk in chunks:
        assert len(chunk.content_hash) == 64  # sha256 hex digest length
        assert all(c in "0123456789abcdef" for c in chunk.content_hash)


def test_chunk_file_all_chunks_share_same_hash(tmp_path):
    """All chunks from the same file must share the same content_hash."""
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks
    hashes = {c.content_hash for c in chunks}
    assert len(hashes) == 1


def test_chunk_file_stamps_file_mtime(tmp_path):
    """Every chunk must have a non-zero file_mtime matching the file's st_mtime."""
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    expected_mtime = f.stat().st_mtime
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks
    for chunk in chunks:
        assert chunk.file_mtime == pytest.approx(expected_mtime, rel=1e-3)


def test_chunk_file_accepts_precomputed_hash(tmp_path):
    """If content_hash is supplied by the caller, chunk_file must use it as-is."""
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    sentinel = "aabbccdd" * 8  # 64-char fake sha256
    chunks = chunk_file(f, tmp_path, "PROJ", content_hash=sentinel)
    assert chunks
    for chunk in chunks:
        assert chunk.content_hash == sentinel


def test_chunk_file_accepts_precomputed_mtime(tmp_path):
    """If file_mtime is supplied by the caller, chunk_file must use it as-is."""
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    sentinel_mtime = 1_700_000_000.0
    chunks = chunk_file(f, tmp_path, "PROJ", file_mtime=sentinel_mtime)
    assert chunks
    for chunk in chunks:
        assert chunk.file_mtime == sentinel_mtime


def test_different_files_have_different_hashes(tmp_path):
    """Two files with different content must produce chunks with different hashes."""
    f1 = _write(tmp_path, "a.py", "def alpha():\n    return 1\n")
    f2 = _write(tmp_path, "b.py", "def beta():\n    return 2\n")
    chunks_a = chunk_file(f1, tmp_path, "PROJ")
    chunks_b = chunk_file(f2, tmp_path, "PROJ")
    assert chunks_a and chunks_b
    assert chunks_a[0].content_hash != chunks_b[0].content_hash

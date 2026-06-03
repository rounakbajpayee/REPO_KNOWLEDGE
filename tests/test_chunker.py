"""
test_chunker.py — Unit tests for chunker.py

Tests:
  - Python AST chunker splits by function and class
  - Python AST chunker prepends imports to each chunk
  - Python AST chunker falls back to fixed for files with no top-level defs
  - Python AST chunker falls back to fixed for invalid syntax
  - Markdown chunker splits by ## headers
  - Fixed chunker produces correct line ranges
  - chunk_file skips unsupported extensions
  - chunk_file handles empty files
  - chunk_project walks directory and skips IGNORE_DIRS
"""

import pytest
from pathlib import Path

from repo_knowledge.chunker import chunk_file, chunk_project, Chunk


# ── Python AST ──────────────────────────────────────────────────────────────────

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


def _write(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def test_python_chunk_count(tmp_path: Path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    # Expects: foo, bar, MyClass
    assert len(chunks) == 3


def test_python_chunk_types(tmp_path: Path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    types = {c.chunk_type for c in chunks}
    assert "function" in types
    assert "class" in types


def test_python_chunk_symbols(tmp_path: Path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    symbols = {c.symbol for c in chunks}
    assert symbols == {"foo", "bar", "MyClass"}


def test_python_imports_prepended(tmp_path: Path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    for chunk in chunks:
        assert "import os" in chunk.content
        assert "import sys" in chunk.content


def test_python_line_numbers_are_set(tmp_path: Path):
    f = _write(tmp_path, "mod.py", PY_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    for chunk in chunks:
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line


def test_python_invalid_syntax_falls_back_to_fixed(tmp_path: Path):
    f = _write(tmp_path, "bad.py", "def foo(:\n    pass")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) >= 1
    assert all(c.chunk_type == "block" for c in chunks)


def test_python_no_top_level_defs_falls_back(tmp_path: Path):
    # Script with only assignments, no functions or classes
    f = _write(tmp_path, "script.py", "X = 1\nY = 2\n")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) >= 1


def test_python_async_function_chunked(tmp_path: Path):
    src = "async def handler():\n    pass\n"
    f = _write(tmp_path, "async_mod.py", src)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert any(c.symbol == "handler" for c in chunks)


# ── Markdown ────────────────────────────────────────────────────────────────────
MD_SOURCE = """# Title

## Section One

Some content here.

## Section Two

More content.
"""


def test_markdown_chunk_count(tmp_path: Path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) == 2


def test_markdown_chunk_symbols(tmp_path: Path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    symbols = [c.symbol for c in chunks]
    assert "Section One" in symbols
    assert "Section Two" in symbols


def test_markdown_chunk_type(tmp_path: Path):
    f = _write(tmp_path, "doc.md", MD_SOURCE)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert all(c.chunk_type == "section" for c in chunks)


# ── Fixed fallback ─────────────────────────────────────────────────────────────────

def test_fixed_chunker_single_block_for_short_file(tmp_path: Path):
    content = "\n".join([f"line {i}" for i in range(10)])
    f = _write(tmp_path, "config.yaml", content)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "block"


def test_fixed_chunker_multiple_blocks_for_long_file(tmp_path: Path):
    # 130 lines should produce multiple 60-line chunks with 10-line overlap
    content = "\n".join([f"line {i}" for i in range(130)])
    f = _write(tmp_path, "big.yaml", content)
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert len(chunks) > 1


# ── Edge cases ───────────────────────────────────────────────────────────────────

def test_empty_file_returns_no_chunks(tmp_path: Path):
    f = _write(tmp_path, "empty.py", "")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_unsupported_extension_returns_no_chunks(tmp_path: Path):
    f = _write(tmp_path, "image.png", "binary")
    chunks = chunk_file(f, tmp_path, "PROJ")
    assert chunks == []


def test_chunk_project_skips_ignore_dirs(tmp_path: Path):
    # Valid file in project root
    _write(tmp_path, "main.py", "def main():\n    pass\n")
    # File inside an ignored directory
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    _write(ignored, "lib.py", "def lib():\n    pass\n")

    chunks = chunk_project(tmp_path, "PROJ")
    paths = [c.path for c in chunks]
    assert all("node_modules" not in p for p in paths)


def test_chunk_project_metadata(tmp_path: Path):
    _write(tmp_path, "service.py", "def serve():\n    pass\n")
    chunks = chunk_project(tmp_path, "MYPROJECT")
    assert all(c.project == "MYPROJECT" for c in chunks)
    assert all(c.language == "python" for c in chunks)

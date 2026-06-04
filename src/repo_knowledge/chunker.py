"""
chunker.py — Converts source files into indexable chunks.

Strategy by file type:
  .py         → AST-based: one chunk per top-level function/class.
                Imports prepended to each chunk for standalone context.
  .ts/.tsx    → Regex boundary split on function/class/arrow fn declarations.
  .js/.jsx    → Same regex strategy as TS.
  .md         → Split on ## headers.
  Everything  → Fixed 60-line chunks with 10-line overlap.

Each chunk carries enough metadata for an agent to locate and understand it
without reading the full file.
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from repo_knowledge.config import IGNORE_DIRS, IGNORE_EXTENSIONS, SUPPORTED_EXTENSIONS


@dataclass
class Chunk:
    project: str
    path: str               # Relative to project root
    language: str
    chunk_type: str         # "function" | "class" | "section" | "block"
    symbol: str             # Function/class name, header text, or ""
    content: str
    start_line: int
    end_line: int
    content_hash: str = field(default="")   # sha256 of file source, stamped per-file
    file_mtime: float = field(default=0.0)  # st_mtime of the source file


# ── Python AST chunker ────────────────────────────────────────────────────────

def _extract_imports(source_lines: list[str], tree: ast.Module) -> str:
    """Collect all top-level import lines as a header block."""
    import_lines: list[str] = []
    for node in tree.body:  # top-level only — excludes method-level imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # node.lineno is 1-indexed
            import_lines.append(source_lines[node.lineno - 1])
    return "\n".join(dict.fromkeys(import_lines))  # deduplicate, preserve order


def _chunk_python(source: str, rel_path: str, project: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to fixed chunking if file can't be parsed
        return _chunk_fixed(source, rel_path, project, language="python")

    source_lines = source.splitlines()
    import_header = _extract_imports(source_lines, tree)

    top_level_nodes = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    if not top_level_nodes:
        # File has no top-level functions/classes (e.g. script, config)
        return _chunk_fixed(source, rel_path, project, language="python")

    for node in top_level_nodes:
        start = node.lineno - 1           # 0-indexed
        end = node.end_lineno             # exclusive slice end
        body_lines = source_lines[start:end]
        content = "\n".join(body_lines)

        if import_header:
            content = f"{import_header}\n\n{content}"

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

        chunks.append(Chunk(
            project=project,
            path=rel_path,
            language="python",
            chunk_type=chunk_type,
            symbol=node.name,
            content=content,
            start_line=node.lineno,
            end_line=node.end_lineno,
        ))

    return chunks


# ── JS/TS regex chunker ───────────────────────────────────────────────────────

# Matches: function foo, async function foo, class Foo,
#          const foo = () =>, const foo = async () =>, export default function
_JS_BOUNDARY_RE = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:\(|\w+\s*=>))",
    re.MULTILINE,
)


def _chunk_js_ts(source: str, rel_path: str, project: str, language: str) -> list[Chunk]:
    lines = source.splitlines()
    matches = list(_JS_BOUNDARY_RE.finditer(source))

    if not matches:
        return _chunk_fixed(source, rel_path, project, language=language)

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(source)

        block = source[start_char:end_char].strip()
        start_line = source[:start_char].count("\n") + 1
        end_line = start_line + block.count("\n")

        # Best-effort symbol extraction from the match text
        symbol_match = re.search(r"(?:function|class|const)\s+(\w+)", match.group())
        symbol = symbol_match.group(1) if symbol_match else ""

        chunk_type = "class" if "class" in match.group() else "function"

        chunks.append(Chunk(
            project=project,
            path=rel_path,
            language=language,
            chunk_type=chunk_type,
            symbol=symbol,
            content=block,
            start_line=start_line,
            end_line=end_line,
        ))

    return chunks


# ── Markdown header chunker ───────────────────────────────────────────────────

_MD_HEADER_RE = re.compile(r"^#{1,6} .+", re.MULTILINE)


_MD_OVERSIZED_LINES = 80


def _chunk_markdown(source: str, rel_path: str, project: str) -> list[Chunk]:
    splits = list(_MD_HEADER_RE.finditer(source))

    if not splits:
        return _chunk_fixed(source, rel_path, project, language="markdown")

    chunks: list[Chunk] = []
    for i, match in enumerate(splits):
        start_char = match.start()
        end_char = splits[i + 1].start() if i + 1 < len(splits) else len(source)
        block = source[start_char:end_char].strip()
        start_line = source[:start_char].count("\n") + 1
        end_line = start_line + block.count("\n")

        if block.count("\n") + 1 > _MD_OVERSIZED_LINES:
            # Section is too large — split into fixed blocks, preserving line offsets
            sub_chunks = _chunk_fixed(block, rel_path, project, language="markdown")
            for sc in sub_chunks:
                sc.start_line += start_line - 1
                sc.end_line += start_line - 1
            chunks.extend(sub_chunks)
        else:
            chunks.append(Chunk(
                project=project,
                path=rel_path,
                language="markdown",
                chunk_type="section",
                symbol=match.group().lstrip("#").strip(),
                content=block,
                start_line=start_line,
                end_line=end_line,
            ))

    return chunks


# ── Fixed-size fallback chunker ───────────────────────────────────────────────

_CHUNK_LINES = 60
_OVERLAP_LINES = 10


def _chunk_fixed(source: str, rel_path: str, project: str, language: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    step = _CHUNK_LINES - _OVERLAP_LINES
    i = 0
    while i < len(lines):
        block_lines = lines[i: i + _CHUNK_LINES]
        chunks.append(Chunk(
            project=project,
            path=rel_path,
            language=language,
            chunk_type="block",
            symbol="",
            content="\n".join(block_lines),
            start_line=i + 1,
            end_line=i + len(block_lines),
        ))
        i += step
    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


def chunk_file(
    file_path: Path,
    project_root: Path,
    project_name: str,
    *,
    content_hash: str = "",
    file_mtime: float = 0.0,
) -> list[Chunk]:
    """
    Read a single file and return its chunks.
    Returns [] if the file is unsupported, unreadable, or empty.

    content_hash and file_mtime are computed here if not supplied by the caller.
    They are stamped onto every Chunk produced from this file.
    """
    suffix = file_path.suffix.lower()
    if suffix in IGNORE_EXTENSIONS:
        return []
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not source.strip():
        return []

    # Compute hash + mtime if the caller didn't supply them
    if not content_hash:
        content_hash = hashlib.sha256(source.encode()).hexdigest()
    if file_mtime == 0.0:
        try:
            file_mtime = file_path.stat().st_mtime
        except OSError:
            file_mtime = 0.0

    rel_path = str(file_path.relative_to(project_root))
    language = _LANG_MAP.get(suffix, "text")

    if suffix == ".py":
        chunks = _chunk_python(source, rel_path, project_name)
    elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
        chunks = _chunk_js_ts(source, rel_path, project_name, language)
    elif suffix == ".md":
        chunks = _chunk_markdown(source, rel_path, project_name)
    else:
        chunks = _chunk_fixed(source, rel_path, project_name, language)

    # Stamp every chunk with the file-level hash and mtime
    for chunk in chunks:
        chunk.content_hash = content_hash
        chunk.file_mtime = file_mtime

    return chunks


def chunk_project(project_root: Path, project_name: str) -> list[Chunk]:
    """
    Walk an entire project directory and return all chunks.
    Skips IGNORE_DIRS automatically.
    """
    all_chunks: list[Chunk] = []

    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue
        # Skip ignored directories anywhere in the path
        if any(
            part in IGNORE_DIRS or part.endswith(".egg-info")
            for part in file_path.parts
        ):
            continue
        all_chunks.extend(chunk_file(file_path, project_root, project_name))

    return all_chunks

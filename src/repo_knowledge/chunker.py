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

from repo_knowledge.config import IGNORE_EXTENSIONS, SUPPORTED_EXTENSIONS


@dataclass
class Chunk:
    project: str
    path: str  # Relative to project root
    language: str
    chunk_type: str  # "function" | "class" | "section" | "block"
    symbol: str  # Function/class name, header text, or ""
    content: str
    start_line: int
    end_line: int
    content_hash: str = field(default="")  # sha256 of file source, stamped per-file
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
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    if not top_level_nodes:
        # File has no top-level functions/classes (e.g. script, config)
        return _chunk_fixed(source, rel_path, project, language="python")

    for node in top_level_nodes:
        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno  # exclusive slice end
        body_lines = source_lines[start:end]
        content = "\n".join(body_lines)

        if import_header:
            content = f"{import_header}\n\n{content}"

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language="python",
                chunk_type=chunk_type,
                symbol=node.name,
                content=content,
                start_line=node.lineno,
                end_line=node.end_lineno,
            )
        )

    return chunks


# ── JS/TS AST chunker ───────────────────────────────────────────────────────
from tree_sitter import Language, Parser


def _get_ts_parser(language: str) -> Parser | None:
    try:
        if language == "typescript":
            import tree_sitter_typescript as tsts

            if hasattr(tsts, "language_typescript"):
                lang = Language(tsts.language_typescript())
            else:
                lang = Language(tsts.language())
        else:
            import tree_sitter_javascript as tsjs

            if hasattr(tsjs, "language_javascript"):
                lang = Language(tsjs.language_javascript())
            else:
                lang = Language(tsjs.language())

        parser = Parser(lang)
        return parser
    except (ImportError, AttributeError):
        return None


def _get_symbol_from_node(node) -> str:
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return child.text.decode("utf8")
    if node.type == "lexical_declaration" or node.type == "variable_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                return _get_symbol_from_node(child)
    return ""


def _chunk_js_ts(source: str, rel_path: str, project: str, language: str) -> list[Chunk]:
    parser = _get_ts_parser(language)
    if not parser:
        return _chunk_fixed(source, rel_path, project, language=language)

    try:
        tree = parser.parse(source.encode("utf8"))
    except Exception:
        return _chunk_fixed(source, rel_path, project, language=language)

    chunks: list[Chunk] = []

    target_types = {
        "class_declaration",
        "function_declaration",
        "method_definition",
        "interface_declaration",
    }
    arrow_fn_declarations = {"lexical_declaration", "variable_declaration"}

    def extract_nodes(node):
        result = []
        if node.type in target_types:
            result.append(node)
        elif node.type in arrow_fn_declarations:
            for child in node.children:
                if child.type == "variable_declarator":
                    for gc in child.children:
                        if gc.type == "arrow_function":
                            result.append(node)
                            break
        for child in node.children:
            result.extend(extract_nodes(child))
        return result

    target_nodes = extract_nodes(tree.root_node)
    if not target_nodes:
        return _chunk_fixed(source, rel_path, project, language=language)

    source_lines = source.splitlines()

    for node in target_nodes:
        start_line_idx = node.start_point[0]
        end_line_idx = node.end_point[0]
        if end_line_idx >= len(source_lines):
            end_line_idx = len(source_lines) - 1

        body_lines = source_lines[start_line_idx : end_line_idx + 1]
        content = "\n".join(body_lines)
        symbol = _get_symbol_from_node(node)

        chunk_type = "function"
        if node.type in ("class_declaration", "interface_declaration"):
            chunk_type = "class"
        elif node.type == "method_definition":
            chunk_type = "function"

        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language=language,
                chunk_type=chunk_type,
                symbol=symbol,
                content=content,
                start_line=start_line_idx + 1,
                end_line=end_line_idx + 1,
            )
        )

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
            chunks.append(
                Chunk(
                    project=project,
                    path=rel_path,
                    language="markdown",
                    chunk_type="section",
                    symbol=match.group().lstrip("#").strip(),
                    content=block,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

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
        block_lines = lines[i : i + _CHUNK_LINES]
        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language=language,
                chunk_type="block",
                symbol="",
                content="\n".join(block_lines),
                start_line=i + 1,
                end_line=i + len(block_lines),
            )
        )
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


# ── Infrastructure file chunkers ──────────────────────────────────────────────

_DOCKER_COMPOSE_RE = re.compile(r"^\s{2}([a-zA-Z0-9_-]+):", re.MULTILINE)
_CONF_SECTION_RE = re.compile(r"^\[(.*?)\]", re.MULTILINE)


def _chunk_docker_compose(source: str, rel_path: str, project: str) -> list[Chunk]:
    matches = list(_DOCKER_COMPOSE_RE.finditer(source))
    if not matches:
        return _chunk_fixed(source, rel_path, project, language="yaml")

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        block = source[start_char:end_char].rstrip()

        start_line = source[:start_char].count("\n") + 1
        end_line = start_line + block.count("\n")

        symbol = f"service: {match.group(1)}"

        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language="yaml",
                chunk_type="section",
                symbol=symbol,
                content=block,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return chunks


def _chunk_plist(source: str, file_path: Path, rel_path: str, project: str) -> list[Chunk]:
    import json
    import plistlib

    try:
        with open(file_path, "rb") as f:
            pl = plistlib.load(f)
    except Exception:
        return _chunk_fixed(source, rel_path, project, language="xml")

    if not isinstance(pl, dict):
        return _chunk_fixed(source, rel_path, project, language="xml")

    chunks: list[Chunk] = []
    for key, value in pl.items():
        try:
            content = json.dumps({key: value}, indent=2)
        except TypeError:
            content = str({key: value})

        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language="xml",
                chunk_type="section",
                symbol=f"key: {key}",
                content=content,
                start_line=1,
                end_line=content.count("\n") + 1,
            )
        )

    if not chunks:
        return _chunk_fixed(source, rel_path, project, language="xml")
    return chunks


def _chunk_conf(source: str, rel_path: str, project: str) -> list[Chunk]:
    matches = list(_CONF_SECTION_RE.finditer(source))
    if not matches:
        return _chunk_fixed(source, rel_path, project, language="ini")

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        block = source[start_char:end_char].rstrip()

        start_line = source[:start_char].count("\n") + 1
        end_line = start_line + block.count("\n")

        symbol = f"section: {match.group(1)}"

        chunks.append(
            Chunk(
                project=project,
                path=rel_path,
                language="ini",
                chunk_type="section",
                symbol=symbol,
                content=block,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return chunks


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

    filename = file_path.name.lower()
    is_docker_compose = filename in ("docker-compose.yml", "docker-compose.yaml")

    if (
        suffix not in SUPPORTED_EXTENSIONS
        and not is_docker_compose
        and suffix not in (".plist", ".conf", ".ini")
    ):
        return []

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if not source.strip():
        return []

    # Skip files larger than 500 KB.
    try:
        size = file_path.stat().st_size
    except OSError:
        size = 0
    if size > 500000:
        return []

    # Skip files with extremely long lines (classic signature of minified code/data blobs).
    # Programming code files have strict formatting, whereas docs/data files might have longer lines.  # noqa: E501
    max_line_len = 1000 if suffix in {".py", ".js", ".jsx", ".ts", ".tsx"} else 10000
    if any(len(line) > max_line_len for line in source.splitlines()):
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
    if is_docker_compose:
        language = "yaml"

    if suffix == ".py":
        chunks = _chunk_python(source, rel_path, project_name)
    elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
        chunks = _chunk_js_ts(source, rel_path, project_name, language)
    elif suffix == ".md":
        chunks = _chunk_markdown(source, rel_path, project_name)
    elif is_docker_compose:
        chunks = _chunk_docker_compose(source, rel_path, project_name)
    elif suffix == ".plist":
        chunks = _chunk_plist(source, file_path, rel_path, project_name)
    elif suffix in {".conf", ".ini"}:
        chunks = _chunk_conf(source, rel_path, project_name)
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
    Uses Git-aware file listing if available.
    """
    from repo_knowledge.scanner import list_project_files

    all_chunks: list[Chunk] = []

    for file_path in list_project_files(project_root):
        all_chunks.extend(chunk_file(file_path, project_root, project_name))

    return all_chunks

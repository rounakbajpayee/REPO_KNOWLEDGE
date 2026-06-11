# ADR-004: Tree-sitter AST Parsing

**Status**: Active
**Date**: 2026-06-12

## Context
Accurately splitting code files into semantic chunks (e.g., classes, functions) is critical for effective retrieval. Currently, JavaScript and TypeScript parsing relies on regular expressions or simple line-based splitting, which is brittle and fails on complex syntax, nested functions, or minified code (`src/repo_knowledge/chunker.py`).

## Decision
Adopt `tree-sitter` for robust Abstract Syntax Tree (AST) parsing for JavaScript and TypeScript (and potentially other languages in the future).

## Consequences
1. Adds native binary dependencies (`tree-sitter`, `tree-sitter-javascript`, `tree-sitter-typescript`) to the environment.
2. The chunker must degrade gracefully to regex or fixed-size line blocks if the `tree-sitter` parsers are unavailable or fail to parse a specific file (as seen in conditional imports in `chunker.py`).
3. Introduces new dependencies to `requirements.txt` / `pyproject.toml`.
4. Greatly improves chunk quality, symbol extraction, and context preservation for supported languages.

## Alternatives Considered
- **Regex-based parsing (current):** Too brittle for full language grammars, especially for JavaScript/TypeScript's varied declaration syntax.
- **LibCST or `ast` module:** Excellent for Python but do not support cross-language parsing for JS/TS.
- **Language-specific parsers:** Requires maintaining multiple distinct parsing frameworks rather than a unified interface like tree-sitter.

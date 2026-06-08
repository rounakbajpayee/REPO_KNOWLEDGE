# Issue: Tree-sitter AST Chunking for JS/TS

## Context
Currently, `src/repo_knowledge/chunker.py` uses naive/regex-based chunking for JavaScript and TypeScript files (`.js`, `.ts`, `.jsx`, `.tsx`). Regex is brittle for complex nested scopes and often leads to broken semantic boundaries or missing functions.

## Goal
Replace the regex logic with a robust AST parser using `tree-sitter`.

## Implementation Details for Jules
1. **Dependencies**: Add `tree-sitter`, `tree-sitter-javascript`, and `tree-sitter-typescript` to `requirements.txt`.
2. **Chunker Update**: In `src/repo_knowledge/chunker.py`, locate the chunking logic for JS/TS extensions.
3. **Parser Initialization**: Initialize the Tree-sitter parser with the appropriate language binary based on the file extension.
4. **AST Traversal**: Walk the AST to find semantic definitions, for example:
   - `class_declaration`
   - `function_declaration`
   - `method_definition`
   - `interface_declaration` (for TS)
5. **Chunk Creation**: For each discovered node, instantiate a `Chunk` object (imported from `repo_knowledge.chunker` or wherever defined in the repo).
   - Extract the node's raw text content.
   - Capture `line_start` and `line_end` (Note: Tree-sitter provides 0-indexed rows; verify if `Chunk` schema expects 1-indexed rows).
   - Assign the `symbol` name (e.g., class name or function name) from the AST identifier node.
6. **Fallback**: Implement graceful degradation. If tree-sitter fails to parse a file due to syntax errors, fallback to standard line-based chunking or the legacy regex logic.

## Acceptance Criteria
- JS and TS files are chunked by exact syntactic boundaries.
- The `Chunk` objects contain the correct line ranges matching the original source code.
- Running the chunker test suite (e.g., `pytest tests/test_chunker.py`) passes. You should also add a specific unit test covering tree-sitter chunking of a complex nested TS file.

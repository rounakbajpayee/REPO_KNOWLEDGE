# Issue: Infrastructure File Chunkers

## Context
Repo Knowledge is expanding to index Homelab infrastructure (GitOps repos). Infrastructure files like Docker Compose, Plist, and configuration files have distinct semantic structures that our current naive text-chunker doesn't handle well. They are often arbitrarily split mid-configuration.

## Goal
Implement specialized chunking logic in `src/repo_knowledge/chunker.py` for `.plist`, `docker-compose.yml`, and `.conf` files.

## Implementation Details for Jules
1. **Docker Compose (`docker-compose.yml`, `docker-compose.yaml`)**:
   - The goal is to chunk the file by `services` blocks.
   - You can use `pyyaml` to parse the structure, but beware that dumping YAML often loses original comments and formatting.
   - *Alternative/Better approach*: Use a regex or string split pattern on `^\s{2}[a-zA-Z0-9_-]+:` (the start of a service block) to preserve comments and raw text formatting, slicing the raw string into distinct chunks.
2. **macOS Plist (`.plist`)**:
   - Use Python's built-in `plistlib` to load the file.
   - Since plists can be binary or XML, load them and dump them to a clean JSON/XML text representation for the chunk content, chunked by primary dictionary keys (e.g. `Label`, `ProgramArguments`).
3. **Configuration files (`.conf`, `.ini`)**:
   - Chunk by section headers (e.g., `[server]`, `[database]`).
   - Use a regex like `^\[(.*?)\]` to identify chunk boundaries and slice the file contents accordingly. Keep the section header in the text.

## Acceptance Criteria
- `docker-compose.yml` files are split logically by service rather than arbitrarily by max line length.
- `.plist` files and `.conf` files are successfully indexed with meaningful semantic boundaries.
- The resulting `Chunk` objects have meaningful names in the `symbol` field (e.g., "service: web", "section: database").
- Integration into the `reindex_project` pipeline works seamlessly.

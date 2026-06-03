# Tests for REPO_KNOWLEDGE

Run with:
```
$env:PYTHONPATH = "src"
pytest tests/ -v
```

Test structure:
- `test_scanner.py` — project discovery, stack detection
- `test_chunker.py` — AST chunking (Python), regex (JS/TS), header (MD), fixed fallback
- `test_knowledge_unit.py` — KnowledgeService logic with mocked store + embedder

Integration tests (require live Qdrant + Ollama) are not in this suite.
Run those manually after confirming infrastructure is up.

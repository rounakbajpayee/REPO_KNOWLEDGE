"""
config.py — Central configuration for REPO_KNOWLEDGE.

All values are read from environment variables with sensible defaults.
Copy .env.example to .env and adjust before running.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://100.70.3.86:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "code_chunks_nomic")

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://100.70.3.86:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Dimension must match the embedding model:
#   nomic-embed-text  → 768
#   qwen3-embed       → 1024
# Changing this requires recreating the collection and reindexing all projects.
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Search ───────────────────────────────────────────────────────────────────
SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "5"))

# ── Scanner ──────────────────────────────────────────────────────────────────
PROJECTS_ROOT: str = os.getenv("PROJECTS_ROOT", os.path.expanduser("~/Projects"))

# Directories to skip during scanning and chunking
IGNORE_DIRS: set[str] = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".next",
    ".cache",
    ".ruff_cache",
    "__pycache__",
    ".mypy_cache",
}

# File extensions to chunk and index
SUPPORTED_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
}

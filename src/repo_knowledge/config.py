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

# Timeout in seconds for Ollama embed calls.
# Large models (qwen3-embedding:4b = 2.5GB) take 60-90s on first load.
# Increase if you see ReadTimeout errors on first run after Ollama restart.
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

# Dimension must match the embedding model:
#   nomic-embed-text       → 768
#   qwen3-embedding:4b     → 1024
# Changing this requires recreating the collection and reindexing all projects.
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Search ───────────────────────────────────────────────────────────────────
SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "5"))

# ── MCP Server ───────────────────────────────────────────────────────────────
# Maximum seconds a single tool call may run before the server returns a clean
# timeout error. 90s is generous: Ollama cold-start on qwen3-embedding:4b takes
# up to 90s over Tailscale. Increase if you see spurious timeouts on first run.
TOOL_TIMEOUT_S: float = float(os.getenv("TOOL_TIMEOUT_S", "90.0"))

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
    "egg-info",
    ".egg-info",
}

# File extensions that must never be indexed (noise / lock files / logs)
IGNORE_EXTENSIONS: set[str] = {
    ".lock",
    ".sum",
    ".log",
    ".jsonl",
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

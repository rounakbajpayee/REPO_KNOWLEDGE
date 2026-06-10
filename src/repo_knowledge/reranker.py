"""
reranker.py — Cross-encoder reranking stage for the two-stage retrieval pipeline.

Wraps sentence-transformers CrossEncoder with:
  - Lazy model loading (downloaded on first call, then cached in-process)
  - Thread-safe singleton initialisation
  - Graceful degradation if sentence-transformers is not installed
  - Configurable model via RERANK_MODEL env var

Usage:
    from repo_knowledge.reranker import rerank
    results = rerank(query="auth middleware", candidates=chunks, top_k=5)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from repo_knowledge.config import RERANK_MODEL

log = logging.getLogger(__name__)

try:
    import sentence_transformers  # type: ignore[import]  # noqa: F401
except ImportError:
    log.warning(
        "Reranker unavailable: sentence-transformers not installed. Install with: pip install repo-knowledge[reranker]"  # noqa: E501
    )


# ── Singleton loader ──────────────────────────────────────────────────────────

_model: Any = None  # CrossEncoder instance once loaded
_init_lock = threading.Lock()
_model_failed = False  # Set True if import/load fails; skips retries


def _load_model() -> Any | None:
    """Load the CrossEncoder model once; return None if unavailable."""
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None

    with _init_lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            try:
                # Try loading from local cache first to avoid network checks/latency
                _model = CrossEncoder(RERANK_MODEL, local_files_only=True)
            except Exception:
                # Fallback to downloading/checking online if not cached locally
                _model = CrossEncoder(RERANK_MODEL, local_files_only=False)
        except ImportError:
            _model_failed = True
            return None
        except Exception:
            _model_failed = True
            return None
    return _model


# ── Public API ────────────────────────────────────────────────────────────────


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Rerank *candidates* using a cross-encoder and return the top_k results.

    Each candidate dict must have a ``"content"`` key (the chunk text).
    The function adds/overwrites ``"rerank_score"`` on each returned item and
    sets ``"score"`` to the rerank score so downstream consumers see a single
    coherent score field.

    Degrades gracefully: if the model is unavailable the original order is
    preserved and the first top_k items are returned unchanged.

    Args:
        query:      The original user query string.
        candidates: List of chunk dicts (each must have ``"content"``).
        top_k:      Maximum number of results to return.

    Returns:
        Reranked (or original) list, length ≤ top_k.
    """
    if not candidates:
        return candidates

    model = _load_model()
    if model is None:
        # Graceful degradation — return original order truncated to top_k
        return candidates[:top_k]

    pairs = [(query, c.get("content", "")) for c in candidates]
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    # Attach rerank scores and sort descending
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for rerank_score, chunk in scored[:top_k]:
        item = dict(chunk)
        item["rerank_score"] = round(float(rerank_score), 4)
        item["score"] = item["rerank_score"]  # normalise field name
        results.append(item)

    return results


def is_available() -> bool:
    """Return True if the reranker model loaded successfully."""
    return _load_model() is not None

"""
embedder.py — Embedding abstraction + Ollama implementation.

Protocol-based design: swap OllamaEmbedder for any other implementation
(OpenAI, HuggingFace local, etc.) without touching the rest of the system.

Changing embedding models also requires:
  1. Update EMBEDDING_MODEL and EMBEDDING_DIM in .env
  2. Update QDRANT_COLLECTION to a new model-slug name
  3. Reindex all projects via: python index.py --all
"""

from typing import Protocol

import httpx

from repo_knowledge.config import EMBEDDING_DIM, EMBEDDING_MODEL, OLLAMA_URL


class Embedder(Protocol):
    """Any embedder must implement this interface."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def model(self) -> str: ...
    @property
    def dimensions(self) -> int: ...


class OllamaEmbedder:
    """
    Embedder backed by a locally running Ollama instance.

    Raises RuntimeError on connection failure or unexpected response —
    callers should handle this and surface clean errors to agents.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        ollama_url: str = OLLAMA_URL,
        dimensions: int = EMBEDDING_DIM,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._url = ollama_url.rstrip("/")
        self._dimensions = dimensions
        self._client = httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Raises RuntimeError on failure."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in a single Ollama request.
        Ollama /api/embed supports batched input natively.
        """
        try:
            response = self._client.post(
                f"{self._url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._url}. "
                "Check that Ollama is running and accessible via Tailscale."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama returned no embeddings. Response: {data}")

        return embeddings

    def health_check(self) -> bool:
        """Returns True if Ollama is reachable, False otherwise."""
        try:
            self._client.get(f"{self._url}/api/tags", timeout=5.0).raise_for_status()
            return True
        except Exception:
            return False

    def __del__(self) -> None:
        self._client.close()


def default_embedder() -> OllamaEmbedder:
    """Convenience factory — returns the configured embedder."""
    return OllamaEmbedder()

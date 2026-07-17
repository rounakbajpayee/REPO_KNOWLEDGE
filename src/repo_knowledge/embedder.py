"""
embedder.py — Embedding abstraction + Ollama implementation.

Protocol-based design: swap OllamaEmbedder for any other implementation
(OpenAI, HuggingFace local, etc.) without touching the rest of the system.

Changing embedding models also requires:
  1. Update EMBEDDING_MODEL and EMBEDDING_DIM in .env
  2. Update QDRANT_COLLECTION to a new model-slug name
  3. Reindex all projects via: python index.py

Timeout note:
  Large models (qwen3-embedding:4b at 2.5GB) take 60-90s to load on first call.
  OLLAMA_TIMEOUT defaults to 120s. Override in .env if needed.
"""

from typing import Protocol

import httpx

from repo_knowledge.config import EMBEDDING_DIM, EMBEDDING_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL
from repo_knowledge.tracer import get_trace_id


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
    Embedder backed by a locally running RapidMLX / OpenAI-compatible instance.
    (Kept name OllamaEmbedder for backward compatibility with existing usages).

    Raises RuntimeError on connection failure or unexpected response —
    callers should handle this and surface clean errors to agents.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        ollama_url: str = OLLAMA_URL,
        dimensions: int = EMBEDDING_DIM,
        timeout: float = OLLAMA_TIMEOUT,
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
        Embed a list of texts in a single RapidMLX/OpenAI request.
        """
        try:
            trace_id = get_trace_id()
            headers = {"X-Trace-ID": trace_id} if trace_id else None
            response = self._client.post(
                f"{self._url}/v1/embeddings",
                json={"model": self._model, "input": texts},
                headers=headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"RapidMLX timed out embedding with model '{self._model}'. "
                f"Large models take 60-90s on first load. "
                f"Current timeout: {self._client.timeout.read}s. "
                "Increase OLLAMA_TIMEOUT in .env if this persists."
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot reach RapidMLX at {self._url}. "
                "Check that it is running and accessible."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"RapidMLX returned HTTP {e.response.status_code}: {e.response.text}"
            ) from e

        data = response.json()
        data_list = data.get("data")
        
        if not data_list:
            raise RuntimeError(f"RapidMLX returned no embeddings. Response: {data}")

        embeddings = [item.get("embedding") for item in data_list]
        return embeddings

    def health_check(self) -> bool:
        """Returns True if RapidMLX is reachable, False otherwise."""
        try:
            self._client.get(f"{self._url}/v1/models", timeout=5.0).raise_for_status()
            return True
        except Exception:
            return False

    def __del__(self) -> None:
        self._client.close()


def default_embedder() -> OllamaEmbedder:
    """Convenience factory — returns the configured embedder."""
    return OllamaEmbedder()

"""Embedding client + cosine, used by the semantic/hybrid retrievers.

Two embedders sit behind the same `Embedder` protocol:
- `EmbeddingClient` -- the real OpenAI/OpenRouter embeddings endpoint (a key).
- `HashingEmbedder` -- a deterministic OFFLINE fallback (hashed bag-of-words)
  so semantic/hybrid retrieval, and the recall comparison, run with NO key.

The OpenAI call is isolated here so the retrieval logic (ranking, fusion) stays
pure and unit-testable without a network. Needs a key only when actually used.
"""

from __future__ import annotations

import hashlib
import math
import os

from .text import tokens as _tokens


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingClient:
    """Thin wrapper over an embeddings endpoint (needs a key when called)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        if api_key is None:
            api_key = os.environ.get(
                "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class HashingEmbedder:
    """Deterministic OFFLINE embedder: hashed bag-of-words vectors, no key.

    Each content token is hashed into one of `dim` buckets and counted. Two
    texts that share tokens get overlapping vectors, so cosine gives a real (if
    crude) lexical-semantic signal -- enough for semantic/hybrid retrieval and
    the recall comparison to run with no network. Not a substitute for a real
    embedding model; the `EmbeddingClient` path is used whenever a key is set.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        return vec

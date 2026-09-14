"""Measure retrieval recall of lexical vs semantic vs hybrid on the gold set.

Recall here = fraction of labelled questions whose gold topic appears anywhere
in the top-k retrieved articles. Deterministic and offline via the hashed
bag-of-words embedder; pass a real `EmbeddingClient` for the real numbers.
"""

from __future__ import annotations

from .questions import QUESTIONS
from .retrieve import (
    Embedder,
    HybridRetriever,
    KeywordRetriever,
    Retriever,
    SemanticRetriever,
)


def recall_at_k(retriever: Retriever, k: int) -> float:
    """Fraction of questions whose gold topic is in the top-k retrieved set."""
    if not QUESTIONS:
        return 1.0
    hits = 0
    for question in QUESTIONS:
        retrieved = retriever.retrieve(question, k)
        if any(a.topic == question.gold_topic for a in retrieved):
            hits += 1
    return hits / len(QUESTIONS)


def measure_recall(embedder: Embedder, k: int = 4) -> list[tuple[str, float]]:
    """Recall of lexical / semantic / hybrid, in that order, with one embedder."""
    lexical = KeywordRetriever()
    semantic = SemanticRetriever(embedder)
    hybrid = HybridRetriever(KeywordRetriever(), SemanticRetriever(embedder))
    return [
        ("keyword", recall_at_k(lexical, k)),
        ("semantic", recall_at_k(semantic, k)),
        ("hybrid", recall_at_k(hybrid, k)),
    ]

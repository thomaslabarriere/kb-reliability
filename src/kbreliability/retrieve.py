"""Retrieval layer: score KB articles for a question.

Three real strategies sit behind the `Retriever` protocol:
- `KeywordRetriever` -- a lexical term-frequency baseline, PERMISSION-AWARE but
  FRESHNESS-NAIVE (an older, wordier article out-scores the concise current one:
  a common real RAG failure). A freshness-aware variant fixes it.
- `SemanticRetriever` -- embeddings + cosine (needs a key).
- `HybridRetriever` -- reciprocal-rank fusion of the lexical and semantic
  rankings (the standard hybrid technique; no score-normalization needed).

Swap a production retriever (vector DB, cross-encoder rerank) behind the same
protocol and the diagnostics are unchanged. Fixtures (blind / leaky) drive the
mutation-proof tests.
"""

from __future__ import annotations

from typing import Protocol

from .embeddings import cosine
from .kb import ARTICLES, is_permitted
from .models import Article, Question
from .text import tokens as _tokens


class Retriever(Protocol):
    name: str

    def retrieve(self, question: Question, k: int) -> list[Article]: ...


class KeywordRetriever:
    """Lexical top-k. permission_aware filters forbidden articles;
    freshness_aware drops a stale version when its current sibling is present."""

    def __init__(self, freshness_aware: bool = False, permission_aware: bool = True) -> None:
        self.freshness_aware = freshness_aware
        self.permission_aware = permission_aware
        self.name = "keyword" + ("+fresh" if freshness_aware else "")

    def retrieve(self, question: Question, k: int) -> list[Article]:
        # Term-frequency scoring: a query term is worth its number of
        # occurrences in the article. This is why a wordy, outdated article can
        # out-score the concise current one -- the freshness trap this measures.
        q_terms = set(_tokens(question.text))
        scored: list[tuple[int, str, Article]] = []
        for article in ARTICLES:
            if self.permission_aware and not is_permitted(article, question.user_scopes):
                continue
            doc_terms = _tokens(f"{article.title} {article.body} {article.topic}")
            score = sum(doc_terms.count(term) for term in q_terms)
            if score > 0:
                scored.append((score, article.article_id, article))
        scored.sort(key=lambda s: (-s[0], s[1]))
        candidates = [a for _, _, a in scored]
        if self.freshness_aware:
            candidates = _drop_stale_when_current_present(candidates)
        return candidates[:k]


def _drop_stale_when_current_present(articles: list[Article]) -> list[Article]:
    current_topics = {a.topic for a in articles if a.is_current}
    return [a for a in articles if a.is_current or a.topic not in current_topics]


class BlindRetriever:
    """Mutation fixture: always returns the same off-topic article -> the gold
    topic is missed for every other question (retrieval fault)."""

    name = "blind"

    def retrieve(self, question: Question, k: int) -> list[Article]:
        from .kb import get_article

        return [get_article("iban-v1")][:k]


class LeakyRetriever:
    """Mutation fixture: ignores permissions and always surfaces the internal
    fraud playbook -> a permission leak whenever the asker lacks the scope."""

    name = "leaky"

    def retrieve(self, question: Question, k: int) -> list[Article]:
        from .kb import current_article, get_article

        leaked = get_article("fraud-playbook")
        try:
            topical = current_article(question.gold_topic)
            return [topical, leaked][:k]
        except KeyError:
            return [leaked][:k]


# --- Semantic + hybrid retrieval --------------------------------------------


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse several ranked id-lists into one. RRF score = sum 1/(k + rank).

    Rank-based, so it needs no score normalization between lexical and semantic
    -- the standard way to combine heterogeneous retrievers. Ties break by id.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


class SemanticRetriever:
    """Embeddings + cosine. Permission-aware. Needs a key (via the embedder)."""

    def __init__(self, embedder: Embedder, permission_aware: bool = True) -> None:
        self.name = "semantic"
        self._embedder = embedder
        self.permission_aware = permission_aware
        self._vectors: dict[str, list[float]] | None = None

    def _article_vectors(self) -> dict[str, list[float]]:
        if self._vectors is None:
            texts = [f"{a.title}. {a.body}" for a in ARTICLES]
            vecs = self._embedder.embed(texts)
            self._vectors = {a.article_id: v for a, v in zip(ARTICLES, vecs, strict=False)}
        return self._vectors

    def retrieve(self, question: Question, k: int) -> list[Article]:
        vectors = self._article_vectors()
        q_vec = self._embedder.embed([question.text])[0]
        scored: list[tuple[float, str, Article]] = []
        for article in ARTICLES:
            if self.permission_aware and not is_permitted(article, question.user_scopes):
                continue
            scored.append((cosine(q_vec, vectors[article.article_id]), article.article_id, article))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [a for _, _, a in scored][:k]


class HybridRetriever:
    """Reciprocal-rank fusion of a lexical and a semantic retriever."""

    def __init__(self, lexical: Retriever, semantic: Retriever, k_rrf: int = 60) -> None:
        self.name = "hybrid"
        self._lexical = lexical
        self._semantic = semantic
        self._k_rrf = k_rrf

    def retrieve(self, question: Question, k: int) -> list[Article]:
        depth = len(ARTICLES)
        lex = [a.article_id for a in self._lexical.retrieve(question, depth)]
        sem = [a.article_id for a in self._semantic.retrieve(question, depth)]
        fused = reciprocal_rank_fusion([lex, sem], self._k_rrf)
        from .kb import get_article

        return [get_article(i) for i in fused][:k]

"""Retrieval layer: score KB articles for a question.

The baseline is a lexical retriever (token overlap) that is PERMISSION-AWARE
(never returns an article the asker may not see) but FRESHNESS-NAIVE (it does
not prefer the current version). That naivety is a real, common RAG failure:
an older, wordier article out-scores the concise current one. A freshness-aware
variant fixes it. Fixtures (blind / leaky) drive the mutation-proof tests.

This is a lexical baseline, NOT embeddings/hybrid search -- swap a real
retriever behind the `Retriever` protocol and the diagnostics are unchanged.
"""

from __future__ import annotations

from typing import Protocol

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

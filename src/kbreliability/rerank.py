"""Reranking stage: reorder first-stage candidates by a sharper signal.

First-stage lexical retrieval optimises recall (get the right article into the
top-k) but not precision@1 -- a wordy distractor can sit at rank 1. A reranker
reorders the shortlist so the answerer (which cites rank 1) cites the right one.
`LexicalReranker` scores TITLE overlap (offline, deterministic); `LLMReranker`
asks a model to pick the most relevant (needs a key).
"""

from __future__ import annotations

from typing import Protocol

from .models import Article, Question
from .text import tokens as _tokens


class Reranker(Protocol):
    name: str

    def rerank(self, question: Question, candidates: list[Article]) -> list[Article]: ...


class LexicalReranker:
    """Reorder by title-term overlap with the query -- a precision signal the
    body-frequency first stage lacks. Stable on ties (keeps first-stage order)."""

    name = "rerank:lexical-title"

    def rerank(self, question: Question, candidates: list[Article]) -> list[Article]:
        q = set(_tokens(question.text))
        scored = [
            (-len(q & set(_tokens(a.title))), i, a) for i, a in enumerate(candidates)
        ]
        scored.sort(key=lambda s: (s[0], s[1]))
        return [a for _, _, a in scored]


class LLMReranker:
    """Ask a model to pick the single most relevant candidate (needs a key).
    Fail-open: on any error, keep the first-stage order."""

    def __init__(self, model: str = "gpt-4o", provider: str = "openai") -> None:
        from .answer import _make_client

        self.name = f"rerank:llm:{model}"
        self._model = model
        self._client = _make_client(provider, None, None)

    def rerank(self, question: Question, candidates: list[Article]) -> list[Article]:
        if not candidates:
            return candidates
        listing = "\n".join(f"[{i}] {a.title}: {a.body}" for i, a in enumerate(candidates))
        prompt = (
            f"Question: {question.text}\n\nCandidats:\n{listing}\n\n"
            "Réponds UNIQUEMENT par l'index du candidat le plus pertinent."
        )
        try:
            completion = self._client.chat.completions.create(
                model=self._model, messages=[{"role": "user", "content": prompt}]
            )
            content = (completion.choices[0].message.content or "").strip()
            idx = int("".join(c for c in content if c.isdigit()) or "-1")
        except Exception:  # noqa: BLE001 - fail open, keep first-stage order
            return candidates
        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            return [chosen, *[c for c in candidates if c is not chosen]]
        return candidates


def top1_correct(ranking: list[Article], gold_topic: str) -> bool:
    return bool(ranking) and ranking[0].topic == gold_topic


def demo_scenario() -> tuple[Question, list[Article], str]:
    """A crafted first-stage shortlist where a wordy distractor is ranked first
    and the correct article is rank 2 -- so reranking must move it to rank 1."""
    question = Question(
        question_id="q-rerank-demo",
        text="Comment activer ma nouvelle carte ?",
        gold_topic="card-activation",
    )
    # First-stage order as given: distractor first (it repeats "carte" a lot),
    # the on-topic article second.
    candidates = [
        Article(
            article_id="card-limit",
            topic="card-limit",
            title="Plafonds de la carte",
            body=(
                "La carte a des plafonds. Modifier les plafonds de la carte, la "
                "carte de paiement et la carte de retrait, se fait dans l'app."
            ),
        ),
        Article(
            article_id="card-activation",
            topic="card-activation",
            title="Activer une nouvelle carte",
            body="Activez votre nouvelle carte via l'app à sa réception.",
        ),
    ]
    return question, candidates, "card-activation"

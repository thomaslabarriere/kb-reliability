"""Evaluate one question and ATTRIBUTE any failure to a RAG layer.

The four signals are independent; the fault is the first failing layer in the
causal order retrieval -> permissions -> freshness -> generation, because a
retrieval miss also looks ungrounded downstream, and the root cause is what a
team needs in order to fix the right layer.
"""

from __future__ import annotations

from .judge import GroundednessJudge
from .kb import get_article, is_permitted
from .models import Answer, Article, Layer, Question, QuestionResult


def evaluate_question(
    question: Question,
    retrieved: list[Article],
    answer: Answer,
    judge: GroundednessJudge,
) -> QuestionResult:
    gold_topic = question.gold_topic
    retrieved_ids = {a.article_id for a in retrieved}

    retrieval_miss = not any(a.topic == gold_topic for a in retrieved)

    cited = get_article(answer.cited_article_id) if answer.cited_article_id else None

    # Permission leak covers both what retrieval surfaced AND what the answer
    # ended up citing. The latter matters for the LLM path: a model can cite an
    # article it was never given (hallucination / prompt injection), and citing
    # a forbidden document is a leak even if retrieval filtered correctly.
    permission_leak = any(not is_permitted(a, question.user_scopes) for a in retrieved)
    if cited is not None and not is_permitted(cited, question.user_scopes):
        permission_leak = True

    stale_answer = cited is not None and cited.topic == gold_topic and not cited.is_current

    if cited is None:
        ungrounded = True
    elif cited.article_id not in retrieved_ids:
        ungrounded = True  # cited a document that was never in the context
    elif cited.topic != gold_topic:
        ungrounded = True  # grounded in the wrong topic entirely
    else:
        ungrounded = not judge.is_grounded(answer.text, cited)

    fault: Layer | None = None
    if retrieval_miss:
        fault = Layer.RETRIEVAL
    elif permission_leak:
        fault = Layer.PERMISSIONS
    elif stale_answer:
        fault = Layer.FRESHNESS
    elif ungrounded:
        fault = Layer.GENERATION

    return QuestionResult(
        question_id=question.question_id,
        passed=fault is None,
        fault=fault,
        retrieval_miss=retrieval_miss,
        permission_leak=permission_leak,
        stale_answer=stale_answer,
        ungrounded=ungrounded,
        trace={
            "retrieved": ",".join(a.article_id for a in retrieved) or "(none)",
            "cited": answer.cited_article_id or "(none)",
            "gold_topic": gold_topic,
        },
    )

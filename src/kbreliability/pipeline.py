"""Run the full RAG pipeline (retrieve -> answer -> diagnose) per question."""

from __future__ import annotations

import time

from .answer import Answerer
from .evaluate import evaluate_question
from .judge import GroundednessJudge
from .models import Question, QuestionResult
from .rerank import Reranker
from .retrieve import Retriever


def run_question(
    retriever: Retriever,
    answerer: Answerer,
    judge: GroundednessJudge,
    question: Question,
    k: int,
    reranker: Reranker | None = None,
) -> QuestionResult:
    """Diagnose one question. A raising retriever/answerer is isolated into a
    generation-fault result rather than sinking the run.

    When a `reranker` is given, the first-stage shortlist is reordered before the
    answerer cites rank 1 -- so reranking genuinely changes the diagnosis, not a
    separate demo."""
    start = time.perf_counter()
    try:
        retrieved = retriever.retrieve(question, k)
        if reranker is not None:
            retrieved = reranker.rerank(question, retrieved)
        answer, usage = answerer.answer(question, retrieved)
        result = evaluate_question(question, retrieved, answer, judge)
    except Exception as exc:  # noqa: BLE001 - isolate per-question failures
        from .models import Layer

        result = QuestionResult(
            question_id=question.question_id,
            passed=False,
            fault=Layer.GENERATION,
            ungrounded=True,
            trace={"error": str(exc)},
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    result.latency_ms = (time.perf_counter() - start) * 1000
    result.usage = usage
    return result


def run_all(
    retriever: Retriever,
    answerer: Answerer,
    judge: GroundednessJudge,
    questions: list[Question],
    k: int,
    reranker: Reranker | None = None,
) -> list[QuestionResult]:
    return [run_question(retriever, answerer, judge, q, k, reranker) for q in questions]

"""Run the full RAG pipeline (retrieve -> answer -> diagnose) per question."""

from __future__ import annotations

import time

from .answer import Answerer
from .evaluate import evaluate_question
from .judge import GroundednessJudge
from .models import Layer, Question, QuestionResult
from .rerank import Reranker
from .retrieve import Retriever


def _crash_result(question: Question, layer: Layer, exc: Exception) -> QuestionResult:
    """A component raised. Attribute the crash to the STAGE that raised it, so
    the layer still points where to look -- a retriever that throws is a
    RETRIEVAL fault, never a generation one. Only a GENERATION crash is marked
    ungrounded (there is genuinely no verified answer from the model)."""
    return QuestionResult(
        question_id=question.question_id,
        passed=False,
        fault=layer,
        retrieval_miss=layer is Layer.RETRIEVAL,
        ungrounded=layer is Layer.GENERATION,
        trace={"error": str(exc), "crashed_stage": layer.value},
    )


def run_question(
    retriever: Retriever,
    answerer: Answerer,
    judge: GroundednessJudge,
    question: Question,
    k: int,
    reranker: Reranker | None = None,
) -> QuestionResult:
    """Diagnose one question. If a stage raises, the failure is attributed to
    THAT stage's layer -- a crashing retriever is a RETRIEVAL fault, a crashing
    answerer a GENERATION fault -- rather than dumping every exception on
    generation. Attributing a retrieval crash to `generation` would contradict
    the whole point of the tool ("a layer says *where* to fix it").

    When a `reranker` is given, the first-stage shortlist is reordered before the
    answerer cites rank 1 -- so reranking genuinely changes the diagnosis, not a
    separate demo."""
    start = time.perf_counter()
    usage = None
    try:
        try:
            retrieved = retriever.retrieve(question, k)
            if reranker is not None:
                retrieved = reranker.rerank(question, retrieved)
        except Exception as exc:  # noqa: BLE001 - retrieval-stage crash
            result = _crash_result(question, Layer.RETRIEVAL, exc)
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        try:
            answer, usage = answerer.answer(question, retrieved)
        except Exception as exc:  # noqa: BLE001 - generation-stage crash
            result = _crash_result(question, Layer.GENERATION, exc)
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        result = evaluate_question(question, retrieved, answer, judge)
    except Exception as exc:  # noqa: BLE001 - the diagnostic harness itself raised
        result = _crash_result(question, Layer.INFRA, exc)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    result.latency_ms = (time.perf_counter() - start) * 1000
    if usage is not None:
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

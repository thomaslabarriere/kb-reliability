"""Mutation proof: each layer's failure must be produced and attributed to the
right layer, and a correct system must pass cleanly."""

from __future__ import annotations

from kbreliability.answer import HallucinatingAnswerer, HeuristicAnswerer
from kbreliability.judge import StaticGroundednessJudge
from kbreliability.models import Layer
from kbreliability.pipeline import run_all, run_question
from kbreliability.questions import QUESTIONS, get_question
from kbreliability.report import build_report
from kbreliability.retrieve import BlindRetriever, KeywordRetriever, LeakyRetriever

_JUDGE = StaticGroundednessJudge()


def _report(retriever, answerer):  # type: ignore[no-untyped-def]
    results = run_all(retriever, answerer, _JUDGE, QUESTIONS, k=4)
    return build_report("test", results)


def test_baseline_only_failure_is_freshness() -> None:
    # Recall and groundedness are perfect; the ONLY failure is a stale answer
    # (the wordy outdated card article out-scores the current one). This is the
    # whole point of layer attribution: the failure is invisible on the two
    # obvious metrics and only shows on freshness.
    rep = _report(KeywordRetriever(freshness_aware=False), HeuristicAnswerer())
    assert rep.retrieval_recall == 1.0
    assert rep.groundedness_rate == 1.0
    assert rep.stale_answers == 1
    assert rep.fault_breakdown == {Layer.FRESHNESS: 1}
    assert rep.passed == rep.total - 1


def test_freshness_aware_retriever_fixes_it() -> None:
    rep = _report(KeywordRetriever(freshness_aware=True), HeuristicAnswerer())
    assert rep.passed == rep.total
    assert rep.stale_answers == 0
    assert rep.fault_breakdown == {}


def test_blind_retriever_is_attributed_to_retrieval() -> None:
    rep = _report(BlindRetriever(), HeuristicAnswerer())
    assert rep.retrieval_recall < 1.0
    assert rep.fault_breakdown.get(Layer.RETRIEVAL, 0) >= 1
    # No failure is mis-attributed to a downstream layer.
    assert set(rep.fault_breakdown) == {Layer.RETRIEVAL}


def test_leaky_retriever_is_attributed_to_permissions() -> None:
    rep = _report(LeakyRetriever(), HeuristicAnswerer())
    assert rep.permission_leaks >= 1
    assert rep.fault_breakdown.get(Layer.PERMISSIONS, 0) >= 1


def test_hallucinating_answerer_is_attributed_to_generation() -> None:
    rep = _report(KeywordRetriever(freshness_aware=True), HallucinatingAnswerer())
    assert rep.groundedness_rate < 1.0
    assert rep.fault_breakdown.get(Layer.GENERATION, 0) >= 1


def test_citing_a_forbidden_article_is_a_permission_leak() -> None:
    # A public user's answer must not cite the internal fraud playbook even if a
    # (permission-aware) retriever never surfaced it -- the LLM could still emit
    # the id by hallucination or injection. The leak is attributed to permissions.
    from kbreliability.evaluate import evaluate_question
    from kbreliability.kb import current_article
    from kbreliability.models import Answer

    question = get_question("q-card")  # public user, no scopes
    retrieved = [current_article("card-blocking")]
    answer = Answer(text="voir le playbook interne", cited_article_id="fraud-playbook")
    result = evaluate_question(question, retrieved, answer, _JUDGE)
    assert result.permission_leak is True
    assert result.fault is Layer.PERMISSIONS


def test_a_crashing_retriever_is_attributed_to_retrieval_not_generation() -> None:
    # The whole point of the tool is "a layer says *where* to fix it". A
    # retriever that raises must be a RETRIEVAL fault -- blaming generation for
    # an upstream crash would send the on-call to the wrong layer.
    class Boom:
        name = "boom"

        def retrieve(self, question, k):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

    r = run_question(Boom(), HeuristicAnswerer(), _JUDGE, get_question("q-card"), k=4)
    assert r.passed is False
    assert r.fault is Layer.RETRIEVAL
    assert r.retrieval_miss is True
    assert r.trace["crashed_stage"] == "retrieval"


def test_a_crashing_answerer_is_attributed_to_generation() -> None:
    from kbreliability.retrieve import KeywordRetriever

    class BoomAnswerer:
        name = "boom-answerer"

        def answer(self, question, retrieved):  # type: ignore[no-untyped-def]
            raise RuntimeError("model down")

    r = run_question(
        KeywordRetriever(), BoomAnswerer(), _JUDGE, get_question("q-card"), k=4
    )
    assert r.passed is False
    assert r.fault is Layer.GENERATION
    assert r.ungrounded is True
    assert r.trace["crashed_stage"] == "generation"

"""Retrieval-depth additions: RRF fusion, reranking, chunking (all offline)."""

from __future__ import annotations

from kbreliability.answer import HeuristicAnswerer
from kbreliability.chunking import analyze, chunk_text
from kbreliability.embeddings import HashingEmbedder
from kbreliability.judge import StaticGroundednessJudge
from kbreliability.measure import measure_recall
from kbreliability.pipeline import run_all
from kbreliability.questions import QUESTIONS, get_question
from kbreliability.rerank import LexicalReranker, demo_scenario, top1_correct
from kbreliability.retrieve import KeywordRetriever, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_rankings() -> None:
    # "a" and "b" are found by BOTH retrievers; "x"/"y" by only one. Agreement
    # across retrievers should rank the shared items above the single-list ones.
    lexical = ["a", "b", "x"]
    semantic = ["a", "b", "y"]
    fused = reciprocal_rank_fusion([lexical, semantic])
    assert fused[:2] == ["a", "b"]
    assert set(fused[2:]) == {"x", "y"}


def test_rrf_single_ranking_is_order_preserving() -> None:
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_lexical_reranker_promotes_the_on_topic_article() -> None:
    question, candidates, gold_topic = demo_scenario()
    # First stage ranks the wordy distractor first...
    assert top1_correct(candidates, gold_topic) is False
    reranked = LexicalReranker().rerank(question, candidates)
    # ...title-aware reranking moves the correct article to rank 1.
    assert top1_correct(reranked, gold_topic) is True


def test_chunking_pinpoints_the_relevant_section_and_cuts_context() -> None:
    a = analyze("Quels justificatifs et preuve d'achat joindre à un litige ?", max_chars=130)
    assert a.target_section_hit is True
    assert a.chunk_context_chars < a.whole_context_chars
    assert a.context_reduction > 0.5  # the section is a fraction of the whole doc


def test_chunk_figures_match_the_readme() -> None:
    # Pins the exact numbers quoted in the README so they can't silently drift.
    a = analyze("Quels justificatifs et preuve d'achat joindre à un litige ?", max_chars=130)
    assert a.whole_context_chars == 667
    assert a.chunk_context_chars == 138
    assert round(a.context_reduction * 100) == 79


def test_chunk_text_respects_the_size_bound() -> None:
    chunks = chunk_text("Un. Deux. Trois. Quatre. Cinq.", max_chars=12)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 12 or " " not in c.text for c in chunks)


def test_offline_recall_lexical_semantic_hybrid_all_saturate() -> None:
    # Pins the REAL measured recall reported in the README. On this toy gold
    # set every question's gold topic is trivially retrievable, so lexical,
    # semantic (hashed-BoW offline embeddings) and hybrid all reach 100% --
    # hybrid does NOT beat lexical here (recall is saturated, an honest finding).
    rows = measure_recall(HashingEmbedder(), k=4)
    assert [name for name, _ in rows] == ["keyword", "semantic", "hybrid"]
    assert all(recall == 1.0 for _, recall in rows)


def test_semantic_hybrid_run_fully_offline_with_hashed_embeddings() -> None:
    # The hashed-BoW embedder gives the semantic/hybrid retrievers a real vector
    # signal, so the whole per-layer pipeline runs with no API key.
    from kbreliability.retrieve import HybridRetriever, SemanticRetriever

    semantic = SemanticRetriever(HashingEmbedder())
    hybrid = HybridRetriever(KeywordRetriever(), SemanticRetriever(HashingEmbedder()))
    for retriever in (semantic, hybrid):
        results = run_all(
            retriever, HeuristicAnswerer(), StaticGroundednessJudge(), QUESTIONS, k=4
        )
        assert len(results) == len(QUESTIONS)


def test_reranker_wired_into_pipeline_reorders_before_citation() -> None:
    # With a reranker in the pipeline, the answerer cites the reranked rank 1,
    # not the first-stage rank 1 -- proving reranking is wired into diagnosis.
    # A fixed shortlist (real KB articles) puts an off-topic article first.
    from kbreliability.kb import get_article

    question = get_question("q-iban")
    shortlist = [get_article("refund-v1"), get_article("iban-v1")]

    class _FixedRetriever:
        name = "fixed"

        def retrieve(self, q: object, k: int) -> list:  # type: ignore[type-arg]
            return list(shortlist)

    plain = run_all(
        _FixedRetriever(), HeuristicAnswerer(), StaticGroundednessJudge(), [question], k=4
    )
    reranked = run_all(
        _FixedRetriever(),
        HeuristicAnswerer(),
        StaticGroundednessJudge(),
        [question],
        k=4,
        reranker=LexicalReranker(),
    )
    assert plain[0].trace["cited"] == "refund-v1"   # off-topic distractor first
    assert plain[0].passed is False                 # cites the wrong topic
    assert reranked[0].trace["cited"] == "iban-v1"  # reranker moved gold to rank 1
    assert reranked[0].passed is True

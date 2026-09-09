"""Retrieval-depth additions: RRF fusion, reranking, chunking (all offline)."""

from __future__ import annotations

from kbreliability.chunking import analyze, chunk_text
from kbreliability.rerank import LexicalReranker, demo_scenario, top1_correct
from kbreliability.retrieve import reciprocal_rank_fusion


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


def test_chunk_text_respects_the_size_bound() -> None:
    chunks = chunk_text("Un. Deux. Trois. Quatre. Cinq.", max_chars=12)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 12 or " " not in c.text for c in chunks)

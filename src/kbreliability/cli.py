"""kb-reliability CLI.

  kb-reliability diagnose   [--retriever keyword|fresh|semantic|hybrid] \
                            [--answerer heuristic|llm] [--reranker none|lexical|llm]
  kb-reliability calibrate  [--judge static|llm]
  kb-reliability retrievers                 # compare lexical vs semantic vs hybrid recall (offline)
  kb-reliability chunks     [--max-chars N] # chunk vs whole-article retrieval (offline)
  kb-reliability rerank     [--reranker lexical|llm]  # precision@1 before/after rerank

Offline by default -- every command runs with NO API key. The semantic/hybrid
retrievers fall back to a deterministic hashed-bag-of-words embedder offline and
use real embeddings when OPENAI_API_KEY (or OPENROUTER_API_KEY + --provider
openrouter) is set. The `llm` answerer/judge/reranker paths need a key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .answer import Answerer, HeuristicAnswerer, LLMAnswerer, LLMGroundednessJudge
from .goldset import GROUND_GOLD
from .judge import GroundednessJudge, StaticGroundednessJudge, calibrate_judge
from .pipeline import run_all
from .questions import QUESTIONS
from .report import build_report, render_report
from .rerank import Reranker
from .retrieve import HybridRetriever, KeywordRetriever, Retriever, SemanticRetriever


def _require_key(provider: str) -> None:
    var = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    if not os.environ.get(var):
        raise SystemExit(
            f"{var} is not set. The LLM/semantic/hybrid paths need it; set it "
            "and re-run (or use the offline defaults)."
        )


def _make_embedder(provider: str):  # type: ignore[no-untyped-def]
    """Real embeddings when a key is set, else the deterministic OFFLINE fallback
    (hashed bag-of-words) so semantic/hybrid retrieval runs with no key."""
    var = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    if os.environ.get(var):
        from .embeddings import EmbeddingClient

        return EmbeddingClient(provider=provider)
    from .embeddings import HashingEmbedder

    return HashingEmbedder()


def _build_retriever(kind: str, provider: str, model: str) -> Retriever:
    if kind == "fresh":
        return KeywordRetriever(freshness_aware=True)
    if kind in ("semantic", "hybrid"):
        semantic = SemanticRetriever(_make_embedder(provider))
        if kind == "semantic":
            return semantic
        return HybridRetriever(KeywordRetriever(), semantic)
    return KeywordRetriever()


def _build_reranker(kind: str, provider: str, model: str) -> Reranker | None:
    from .rerank import LexicalReranker, LLMReranker

    if kind == "lexical":
        return LexicalReranker()
    if kind == "llm":
        _require_key(provider)
        return LLMReranker(model=model, provider=provider)
    return None


def _diagnose(args: argparse.Namespace) -> int:
    retriever: Retriever = _build_retriever(args.retriever, args.provider, args.model)
    answerer: Answerer
    judge: GroundednessJudge
    if args.answerer == "llm":
        _require_key(args.provider)
        answerer = LLMAnswerer(model=args.model, provider=args.provider)
        judge = LLMGroundednessJudge(model=args.model, provider=args.provider)
    else:
        answerer = HeuristicAnswerer()
        judge = StaticGroundednessJudge()

    reranker = _build_reranker(args.reranker, args.provider, args.model)
    results = run_all(retriever, answerer, judge, QUESTIONS, args.k, reranker)
    system_name = f"{retriever.name} / {answerer.name}"
    if reranker is not None:
        system_name += f" + {reranker.name}"
    report = build_report(system_name, results)
    print(render_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)
        print(f"\nJSON écrit dans {args.json}")
    # Non-zero exit if any question fails -- drops into CI.
    return 0 if report.passed == report.total else 1


def _calibrate(args: argparse.Namespace) -> int:
    judge: GroundednessJudge
    if args.judge == "llm":
        _require_key(args.provider)
        judge = LLMGroundednessJudge(model=args.model, provider=args.provider)
    else:
        judge = StaticGroundednessJudge()
    cal = calibrate_judge(judge, GROUND_GOLD)
    print(f"Calibration du juge « {cal.judge_name} »")
    print(f"  Accord: {cal.agreement_rate * 100:.0f}% ({cal.agree}/{cal.total})")
    print(f"  Faux positifs: {cal.false_positive}   Faux négatifs: {cal.false_negative}")
    return 0


def _retrievers(args: argparse.Namespace) -> int:
    """Compare retrieval recall of lexical vs semantic vs hybrid.

    Runs OFFLINE by default via the deterministic hashed-bag-of-words embedder;
    set OPENAI_API_KEY (or OPENROUTER_API_KEY) to compare with real embeddings.
    """
    from .measure import measure_recall

    var = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "OPENAI_API_KEY"
    embed_mode = "réelles (clé)" if os.environ.get(var) else "hors-ligne (bag-of-words haché)"
    rows = measure_recall(_make_embedder(args.provider), k=args.k)
    print("Comparaison des retrievers (recall du bon article sur le jeu labellisé)")
    print(f"  Embeddings: {embed_mode}")
    print(f"  {'retriever':<12} recall")
    for name, recall in rows:
        print(f"  {name:<12} {recall * 100:>4.0f}%")
    return 0


def _chunks(args: argparse.Namespace) -> int:
    """Chunk-level vs whole-article retrieval on a long document (offline)."""
    from .chunking import analyze

    question = "Quels justificatifs et preuve d'achat joindre à un litige ?"
    a = analyze(question, max_chars=args.max_chars)
    print("Chunking: récupération ciblée sur un article long")
    print(f"  Question: {a.question}")
    print(f"  Article entier: {a.whole_context_chars} caractères de contexte")
    print(f"  Meilleur chunk: {a.chunk_context_chars} caractères")
    print(f"  Bonne section retrouvée: {'oui' if a.target_section_hit else 'non'}")
    print(f"  Réduction du contexte: {a.context_reduction * 100:.0f}%")
    print(f"  Chunk: « {a.best_chunk_text} »")
    return 0


def _rerank(args: argparse.Namespace) -> int:
    """Precision@1 before vs after reranking, on a crafted shortlist."""
    from .rerank import LexicalReranker, LLMReranker, demo_scenario, top1_correct

    question, candidates, gold_topic = demo_scenario()
    reranker: Reranker
    if args.reranker == "llm":
        _require_key(args.provider)  # guard BEFORE constructing the LLM client
        reranker = LLMReranker(model=args.model, provider=args.provider)
    else:
        reranker = LexicalReranker()
    reranked = reranker.rerank(question, candidates)
    print(f"Reranking ({reranker.name})")
    print(f"  Question: {question.text}")
    print(f"  Rang 1 avant: {candidates[0].article_id} "
          f"({'correct' if top1_correct(candidates, gold_topic) else 'incorrect'})")
    print(f"  Rang 1 après: {reranked[0].article_id} "
          f"({'correct' if top1_correct(reranked, gold_topic) else 'incorrect'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kb-reliability",
        description="Layer-attributed reliability diagnostics for a RAG knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="Run the RAG pipeline and attribute failures by layer.")
    diag.add_argument(
        "--retriever", choices=["keyword", "fresh", "semantic", "hybrid"], default="keyword"
    )
    diag.add_argument("--answerer", choices=["heuristic", "llm"], default="heuristic")
    diag.add_argument("--reranker", choices=["none", "lexical", "llm"], default="none")
    diag.add_argument("--k", type=int, default=4)
    diag.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    diag.add_argument("--model", default="gpt-4o")
    diag.add_argument("--json", help="Write the report as JSON to this path.")

    cal = sub.add_parser("calibrate", help="Measure the groundedness judge against the gold set.")
    cal.add_argument("--judge", choices=["static", "llm"], default="static")
    cal.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    cal.add_argument("--model", default="gpt-4o")

    ret = sub.add_parser("retrievers", help="Compare lexical vs semantic vs hybrid recall.")
    ret.add_argument("--k", type=int, default=4)
    ret.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    ret.add_argument("--model", default="gpt-4o")

    ch = sub.add_parser("chunks", help="Chunk vs whole-article retrieval on a long document.")
    ch.add_argument("--max-chars", type=int, default=130, dest="max_chars")

    rr = sub.add_parser("rerank", help="Precision@1 before/after reranking a shortlist.")
    rr.add_argument("--reranker", choices=["lexical", "llm"], default="lexical")
    rr.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    rr.add_argument("--model", default="gpt-4o")

    args = parser.parse_args(argv)
    dispatch = {
        "diagnose": _diagnose,
        "calibrate": _calibrate,
        "retrievers": _retrievers,
        "chunks": _chunks,
        "rerank": _rerank,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

"""kb-reliability CLI.

  kb-reliability diagnose  [--retriever keyword|fresh] [--answerer heuristic|llm] [--k N]
  kb-reliability calibrate [--judge static|llm]

Offline by default (keyword retriever + heuristic answerer + static judge).
The `llm` answerer/judge need OPENAI_API_KEY (or OPENROUTER_API_KEY +
--provider openrouter) -- the only thing required to run the real system.
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
from .retrieve import KeywordRetriever, Retriever


def _require_key(provider: str) -> None:
    var = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    if not os.environ.get(var):
        raise SystemExit(
            f"{var} is not set. The LLM answerer/judge need it; set it and "
            "re-run (or use the offline defaults)."
        )


def _diagnose(args: argparse.Namespace) -> int:
    retriever: Retriever = KeywordRetriever(freshness_aware=args.retriever == "fresh")
    answerer: Answerer
    judge: GroundednessJudge
    if args.answerer == "llm":
        _require_key(args.provider)
        answerer = LLMAnswerer(model=args.model, provider=args.provider)
        judge = LLMGroundednessJudge(model=args.model, provider=args.provider)
    else:
        answerer = HeuristicAnswerer()
        judge = StaticGroundednessJudge()

    results = run_all(retriever, answerer, judge, QUESTIONS, args.k)
    system_name = f"{retriever.name} / {answerer.name}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kb-reliability",
        description="Layer-attributed reliability diagnostics for a RAG knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="Run the RAG pipeline and attribute failures by layer.")
    diag.add_argument("--retriever", choices=["keyword", "fresh"], default="keyword")
    diag.add_argument("--answerer", choices=["heuristic", "llm"], default="heuristic")
    diag.add_argument("--k", type=int, default=4)
    diag.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    diag.add_argument("--model", default="gpt-4o")
    diag.add_argument("--json", help="Write the report as JSON to this path.")

    cal = sub.add_parser("calibrate", help="Measure the groundedness judge against the gold set.")
    cal.add_argument("--judge", choices=["static", "llm"], default="static")
    cal.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    cal.add_argument("--model", default="gpt-4o")

    args = parser.parse_args(argv)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command == "calibrate":
        return _calibrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

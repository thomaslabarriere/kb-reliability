"""Aggregate per-question results into a layer-attributed diagnostic report."""

from __future__ import annotations

from .models import DiagnosticReport, Layer, QuestionResult
from .pricing import estimate_usd, model_from_name

_LAYER_ORDER = [Layer.RETRIEVAL, Layer.PERMISSIONS, Layer.FRESHNESS, Layer.GENERATION]


def build_report(system_name: str, results: list[QuestionResult]) -> DiagnosticReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    fault_breakdown: dict[Layer, int] = {}
    for r in results:
        if r.fault is not None:
            fault_breakdown[r.fault] = fault_breakdown.get(r.fault, 0) + 1

    retrieval_hits = sum(1 for r in results if not r.retrieval_miss)
    answered = [r for r in results if r.trace.get("cited", "(none)") != "(none)"]
    # Groundedness is measured only over answers the judge could VERIFY. Answers
    # the judge could not verify (judge outage) are excluded from the rate and
    # counted separately, so a judge outage never silently inflates or deflates
    # the groundedness metric.
    verified = [r for r in answered if not r.judge_error]
    grounded = sum(1 for r in verified if not r.ungrounded)

    return DiagnosticReport(
        system_name=system_name,
        total=total,
        passed=passed,
        fault_breakdown=fault_breakdown,
        retrieval_recall=retrieval_hits / total if total else 1.0,
        groundedness_rate=grounded / len(verified) if verified else 1.0,
        groundedness_uncertain=sum(1 for r in answered if r.judge_error),
        stale_answers=sum(1 for r in results if r.stale_answer),
        permission_leaks=sum(1 for r in results if r.permission_leak),
        results=results,
        prompt_tokens=sum(r.usage.prompt_tokens for r in results),
        completion_tokens=sum(r.usage.completion_tokens for r in results),
        total_latency_ms=sum(r.latency_ms for r in results),
    )


def render_report(report: DiagnosticReport) -> str:
    bar = "─" * 64
    lines = [bar, f"kb-reliability: {report.system_name}", bar]
    lines.append(f"Réussite: {report.passed}/{report.total} questions")
    lines.append("")

    lines.append("Par question (faute attribuée à la couche)")
    for r in report.results:
        mark = "✓" if r.passed else "✗"
        fault = f"  [{r.fault.value}]" if r.fault else ""
        cited = r.trace.get("cited", "")
        lines.append(f"  {mark} {r.question_id}{fault}   cité: {cited}")
    lines.append("")

    lines.append("Métriques par couche")
    lines.append(f"  Retrieval recall:   {report.retrieval_recall * 100:.0f}%")
    lines.append(f"  Groundedness:       {report.groundedness_rate * 100:.0f}%")
    if report.groundedness_uncertain:
        lines.append(
            f"  Groundedness indéterminé / erreur juge: {report.groundedness_uncertain}"
        )
    lines.append(f"  Réponses périmées:  {report.stale_answers}")
    lines.append(f"  Fuites de permission: {report.permission_leaks}")
    lines.append("")

    lines.append("Attribution des fautes")
    if report.fault_breakdown:
        for layer in _LAYER_ORDER:
            count = report.fault_breakdown.get(layer, 0)
            if count:
                lines.append(f"  {layer.value:<12} {count}")
    else:
        lines.append("  (aucune faute)")

    tokens = report.prompt_tokens + report.completion_tokens
    if tokens > 0:
        lines.append("")
        lines.append("Coût & latence")
        usd = estimate_usd(
            model_from_name(report.system_name),
            report.prompt_tokens,
            report.completion_tokens,
        )
        cost = f"  Tokens: {tokens}"
        if usd is not None:
            cost += f"   Coût estimé: ${usd:.4f} (prix catalogue indicatif)"
        lines.append(cost)
        if report.total:
            lines.append(f"  Latence: {report.total_latency_ms / report.total:.0f} ms/question")

    lines.append(bar)
    return "\n".join(lines)

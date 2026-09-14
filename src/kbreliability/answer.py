"""Generation layer: turn a question + retrieved articles into a cited answer.

`HeuristicAnswerer` is the offline system: it grounds its answer in the
top-ranked retrieved article (so with the freshness-naive retriever it will
faithfully -- and wrongly -- answer from the stale article). `LLMAnswerer` is
the real system (needs a key). Fixtures drive the mutation-proof tests.
"""

from __future__ import annotations

import json
from typing import Protocol

from .llm_client import make_client
from .models import Answer, Article, GroundednessVerdict, Question, TokenUsage


class Answerer(Protocol):
    name: str

    def answer(self, question: Question, retrieved: list[Article]) -> tuple[Answer, TokenUsage]: ...


def _first_sentence(body: str) -> str:
    head, _, _ = body.partition(".")
    return (head + ".").strip()


class HeuristicAnswerer:
    """Cite the top-ranked retrieved article and answer from its body."""

    name = "heuristic"

    def answer(self, question: Question, retrieved: list[Article]) -> tuple[Answer, TokenUsage]:
        if not retrieved:
            answer = Answer(text="Je n'ai pas trouvé d'information.", cited_article_id=None)
            return answer, TokenUsage()
        top = retrieved[0]
        return Answer(text=_first_sentence(top.body), cited_article_id=top.article_id), TokenUsage()


class HallucinatingAnswerer:
    """Mutation fixture: cites the top article but answers with content not in
    it -> the groundedness judge must flag it (generation fault)."""

    name = "hallucinating"

    def answer(self, question: Question, retrieved: list[Article]) -> tuple[Answer, TokenUsage]:
        cited = retrieved[0].article_id if retrieved else None
        return (
            Answer(
                text="Redémarrez votre téléphone puis patientez soixante-douze heures.",
                cited_article_id=cited,
            ),
            TokenUsage(),
        )


_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "report_answer",
        "description": "Answer the support question grounded ONLY in the provided articles.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The concise answer."},
                "cited_article_id": {
                    "type": "string",
                    "description": "The id of the ONE article the answer is grounded in.",
                },
            },
            "required": ["answer", "cited_article_id"],
            "additionalProperties": False,
        },
    },
}

_ANSWER_SYSTEM = (
    "Tu es un assistant de support client. Réponds à la question EN T'APPUYANT "
    "UNIQUEMENT sur les articles fournis, et cite l'id de l'article utilisé. "
    "Si plusieurs versions existent, préfère l'information la plus à jour. "
    "N'invente rien qui ne soit pas dans les articles. Appelle report_answer."
)


def _render_articles(retrieved: list[Article]) -> str:
    if not retrieved:
        return "(aucun article)"
    return "\n".join(f"[{a.article_id}] {a.title}: {a.body}" for a in retrieved)


class LLMAnswerer:
    """The real system (needs OPENAI_API_KEY / OPENROUTER_API_KEY)."""

    def __init__(
        self, model: str = "gpt-4o", provider: str = "openai", api_key: str | None = None
    ) -> None:
        self.name = f"llm:{model}"
        self._model = model
        self._client = make_client(provider, api_key)

    def answer(self, question: Question, retrieved: list[Article]) -> tuple[Answer, TokenUsage]:
        user = f"Question: {question.text}\n\nArticles:\n{_render_articles(retrieved)}"
        try:
            completion = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self._model,
                tools=[_ANSWER_TOOL],
                tool_choice="required",
                messages=[
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface as an ungrounded non-answer
            return Answer(text=f"[erreur automation] {exc}", cited_article_id=None), TokenUsage()

        usage = TokenUsage(
            prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
        )
        message = completion.choices[0].message if completion.choices else None
        for call in getattr(message, "tool_calls", None) or []:
            if getattr(call, "type", None) == "function" and call.function.name == "report_answer":
                parsed = _parse_answer(call.function.arguments)
                if parsed is not None:
                    return parsed, usage
        return Answer(text="", cited_article_id=None), usage


def _parse_answer(args_json: str) -> Answer | None:
    try:
        data = json.loads(args_json)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    cited = data.get("cited_article_id")
    if not isinstance(answer, str):
        return None
    return Answer(
        text=answer,
        cited_article_id=cited if isinstance(cited, str) and cited else None,
    )


class LLMGroundednessJudge:
    """LLM judge for the real run (needs a key). On any judge fault (client
    raises, or an empty/unparseable reply) it returns UNCERTAIN -- NOT grounded
    and NOT ungrounded. A judge outage must neither be silently certified as a
    verified pass nor fabricated into a generation failure; it is surfaced as a
    distinct 'indéterminé' outcome in the report."""

    def __init__(
        self, model: str = "gpt-4o", provider: str = "openai", api_key: str | None = None
    ) -> None:
        self.name = f"llm-judge:{model}"
        self._model = model
        self._client = make_client(provider, api_key)

    def assess(self, answer_text: str, article: Article) -> GroundednessVerdict:
        prompt = (
            "L'affirmation est-elle entièrement soutenue par l'article ? "
            "Réponds par 'oui' ou 'non'.\n\n"
            f"Article: {article.title}. {article.body}\n\nAffirmation: {answer_text}"
        )
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (completion.choices[0].message.content or "").strip().lower()
        except Exception:  # noqa: BLE001 - judge fault -> UNCERTAIN, never a silent verdict
            return GroundednessVerdict.UNCERTAIN
        if not content:  # empty / unparseable reply is also a judge fault
            return GroundednessVerdict.UNCERTAIN
        if content.startswith("non"):
            return GroundednessVerdict.UNGROUNDED
        return GroundednessVerdict.GROUNDED

"""LLM path, tested with a stubbed client (no network, no API credits).

Proves the code that actually ships and breaks in production: the answer
tool-call parsing (valid AND malformed), the success mapping, and the fail-safe
paths -- an API error must never surface as a fabricated grounded answer, and
the LLM judge must never turn a broken call into a false 'grounded' verdict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kbreliability import answer as answer_mod
from kbreliability.answer import LLMAnswerer, LLMGroundednessJudge, _parse_answer
from kbreliability.kb import get_article
from kbreliability.models import GroundednessVerdict
from kbreliability.questions import get_question


def _fake_client(create_fn: Any) -> Any:
    """A stand-in for openai.OpenAI exposing chat.completions.create."""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def _tool_completion(args: dict[str, Any]) -> Any:
    call = SimpleNamespace(
        type="function",
        id="call_1",
        function=SimpleNamespace(name="report_answer", arguments=json.dumps(args)),
    )
    message = SimpleNamespace(tool_calls=[call], content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5),
    )


def _text_completion(content: str | None) -> Any:
    message = SimpleNamespace(tool_calls=[], content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )


# --- Pure parser: robust to valid and malformed tool-call payloads ----------

def test_parse_accepts_a_valid_payload() -> None:
    payload = '{"answer": "Bloquez la carte dans l\'app.", "cited_article_id": "card-block-v2"}'
    a = _parse_answer(payload)
    assert a is not None
    assert a.text.startswith("Bloquez")
    assert a.cited_article_id == "card-block-v2"


def test_parse_rejects_malformed_payloads() -> None:
    assert _parse_answer("not json at all") is None      # invalid JSON
    assert _parse_answer('["a", "b"]') is None            # not an object
    assert _parse_answer('{"cited_article_id": "x"}') is None  # missing answer
    assert _parse_answer('{"answer": 42}') is None        # answer not a string


def test_parse_drops_empty_or_nonstring_citation() -> None:
    a = _parse_answer('{"answer": "ok", "cited_article_id": ""}')
    assert a is not None and a.cited_article_id is None
    b = _parse_answer('{"answer": "ok", "cited_article_id": 5}')
    assert b is not None and b.cited_article_id is None


# --- LLMAnswerer over the stubbed client ------------------------------------

def test_answerer_maps_a_toolcall_to_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        answer_mod,
        "make_client",
        lambda *a, **k: _fake_client(
            lambda **kw: _tool_completion(
                {"answer": "Ouvrez l'app puis Bloquer.", "cited_article_id": "card-block-v2"}
            )
        ),
    )
    answerer = LLMAnswerer(model="gpt-4o")
    q = get_question("q-card")
    result, usage = answerer.answer(q, [get_article("card-block-v2")])
    assert result.cited_article_id == "card-block-v2"
    assert usage.prompt_tokens == 17 and usage.completion_tokens == 5


def test_answerer_fails_safe_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kw: Any) -> Any:
        raise RuntimeError("API is down")

    monkeypatch.setattr(answer_mod, "make_client", lambda *a, **k: _fake_client(boom))
    answerer = LLMAnswerer(model="gpt-4o")
    q = get_question("q-card")
    result, usage = answerer.answer(q, [get_article("card-block-v2")])
    # A failed call must never fabricate a grounded, cited answer.
    assert result.cited_article_id is None
    assert "erreur" in result.text.lower()
    assert usage.prompt_tokens == 0


def test_answerer_empty_when_no_toolcall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        answer_mod, "make_client", lambda *a, **k: _fake_client(lambda **kw: _text_completion(None))
    )
    answerer = LLMAnswerer(model="gpt-4o")
    result, _ = answerer.answer(get_question("q-card"), [get_article("card-block-v2")])
    # No tool call -> no citation, never a fabricated answer.
    assert result.cited_article_id is None
    assert result.text == ""


# --- LLMGroundednessJudge over the stubbed client ---------------------------

def test_judge_reads_a_clear_yes_and_no(monkeypatch: pytest.MonkeyPatch) -> None:
    article = get_article("card-block-v2")

    def _client_replying(content: str) -> Any:
        return _fake_client(lambda **kw: _text_completion(content))

    monkeypatch.setattr(answer_mod, "make_client", lambda *a, **k: _client_replying("Oui."))
    assert LLMGroundednessJudge(model="gpt-4o").assess("x", article) is GroundednessVerdict.GROUNDED
    monkeypatch.setattr(answer_mod, "make_client", lambda *a, **k: _client_replying("Non."))
    assert (
        LLMGroundednessJudge(model="gpt-4o").assess("x", article)
        is GroundednessVerdict.UNGROUNDED
    )


def test_judge_returns_uncertain_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kw: Any) -> Any:
        raise RuntimeError("API is down")

    monkeypatch.setattr(answer_mod, "make_client", lambda *a, **k: _fake_client(boom))
    judge = LLMGroundednessJudge(model="gpt-4o")
    # A judge fault must NOT fold into GROUNDED (pretending the answer was
    # verified) nor into UNGROUNDED (fabricating a generation failure). It is an
    # explicit UNCERTAIN outcome, surfaced separately in the report.
    verdict = judge.assess("anything", get_article("card-block-v2"))
    assert verdict is GroundednessVerdict.UNCERTAIN


def test_judge_returns_uncertain_on_empty_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        answer_mod, "make_client", lambda *a, **k: _fake_client(lambda **kw: _text_completion(None))
    )
    judge = LLMGroundednessJudge(model="gpt-4o")
    # An empty / unparseable reply is a judge fault, not a silent 'grounded'.
    assert judge.assess("anything", get_article("card-block-v2")) is GroundednessVerdict.UNCERTAIN

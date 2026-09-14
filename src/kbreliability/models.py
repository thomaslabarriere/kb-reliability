"""Shared contracts for kb-reliability. Every module imports from here.

SCOPE: this is NOT a real customer-service knowledge base and NOT production
advice. The articles, questions, and permissions are SYNTHETIC and illustrative
(a Qonto/Doctolib-style support KB, invented). No client data. The value is the
diagnostic instrument -- measuring, and attributing by layer, how a RAG system
serves the WRONG / STALE / FORBIDDEN / UNGROUNDED answer -- not the content.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Layer(StrEnum):
    """The RAG layer a failure is attributed to.

    The job posting names this exactly: "diagnostiquer la performance d'un RAG
    en séparant les problèmes de retrieval, de contexte, de modèle et de
    génération". A score says *that* it failed; a layer says *where*.
    """

    RETRIEVAL = "retrieval"
    PERMISSIONS = "permissions"
    FRESHNESS = "freshness"
    GENERATION = "generation"


class GroundednessVerdict(StrEnum):
    """A groundedness judge's outcome. UNCERTAIN is an explicit judge-error /
    could-not-verify state: on an API error or unparseable reply the judge must
    NOT fold the answer into GROUNDED (pretending it was verified) nor into
    UNGROUNDED (fabricating a generation failure). It is surfaced separately.
    """

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    UNCERTAIN = "uncertain"


class Article(BaseModel):
    """One KB article. Topics may have several versions; one is current."""

    article_id: str
    topic: str
    title: str
    body: str
    version: int = 1
    is_current: bool = True
    # Scopes a user must hold to be allowed to see this article ([] = public).
    required_scopes: list[str] = Field(default_factory=list)


class Question(BaseModel):
    """A support question with its ground truth."""

    question_id: str
    text: str
    # The topic whose CURRENT article correctly answers this question.
    gold_topic: str
    # Scopes held by the user asking (drives the permission-leak check).
    user_scopes: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    """What the system produced: a response grounded in ONE cited article."""

    text: str
    # The article the answer claims to be grounded in (None = cited nothing).
    cited_article_id: str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class QuestionResult(BaseModel):
    """Per-question diagnosis with layer attribution."""

    question_id: str
    passed: bool
    # The layer the failure is attributed to (None when passed).
    fault: Layer | None = None
    # Individual signals (all False on a clean pass).
    retrieval_miss: bool = False
    permission_leak: bool = False
    stale_answer: bool = False
    ungrounded: bool = False
    # The judge could not verify groundedness (API error / unparseable reply).
    # Distinct from `ungrounded`: the answer is NOT declared a generation fault,
    # but it is NOT counted as a verified pass either.
    judge_error: bool = False
    trace: dict[str, str] = Field(default_factory=dict)
    latency_ms: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)


class DiagnosticReport(BaseModel):
    """Aggregate diagnosis over the question set."""

    system_name: str
    total: int
    passed: int
    # Count of failures attributed to each layer.
    fault_breakdown: dict[Layer, int] = Field(default_factory=dict)
    # Retrieval recall: gold-topic article retrieved / total.
    retrieval_recall: float = 1.0
    # Grounded answers / VERIFIED answers (judge-error answers excluded, since
    # their groundedness is unknown -- see groundedness_uncertain).
    groundedness_rate: float = 1.0
    # Answers whose groundedness the judge could not verify (judge outage).
    groundedness_uncertain: int = 0
    stale_answers: int = 0
    permission_leaks: int = 0
    results: list[QuestionResult] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0


class GroundGoldItem(BaseModel):
    """Labelled example for calibrating the groundedness judge (who judges the
    judge?): is `answer` actually supported by the article body?"""

    question_id: str
    article_id: str
    answer_text: str
    grounded: bool


class JudgeCalibration(BaseModel):
    judge_name: str
    total: int
    agree: int
    false_positive: int  # judge said grounded, truth said not
    false_negative: int  # judge said ungrounded, truth said grounded

    @property
    def agreement_rate(self) -> float:
        return self.agree / self.total if self.total else 1.0

"""Groundedness judge: is an answer actually supported by the cited article?

The generation layer's verdict is subjective, so it is delegated to a judge --
and, as everywhere in this repo, the judge's OWN reliability is measured against
a labelled gold set (who judges the judge?). A deterministic static judge runs
offline and in tests; an LLM judge runs with a key.
"""

from __future__ import annotations

from typing import Protocol

from .models import Article, GroundednessVerdict, GroundGoldItem, JudgeCalibration
from .text import tokens as _tokens


class GroundednessJudge(Protocol):
    name: str

    def assess(self, answer_text: str, article: Article) -> GroundednessVerdict: ...


class StaticGroundednessJudge:
    """Deterministic: grounded iff the answer shares at least `min_overlap`
    content tokens with the article body. Offline, no key -- never UNCERTAIN."""

    def __init__(self, min_overlap: int = 3) -> None:
        self.name = "static-overlap"
        self._min = min_overlap

    def assess(self, answer_text: str, article: Article) -> GroundednessVerdict:
        answer_terms = set(_tokens(answer_text))
        body_terms = set(_tokens(f"{article.title} {article.body}"))
        if len(answer_terms & body_terms) >= self._min:
            return GroundednessVerdict.GROUNDED
        return GroundednessVerdict.UNGROUNDED


class AllGroundedJudge:
    """Deliberately bad judge (calls everything grounded) -- calibration must
    catch it with false positives."""

    name = "always-grounded"

    def assess(self, answer_text: str, article: Article) -> GroundednessVerdict:
        return GroundednessVerdict.GROUNDED


def calibrate_judge(
    judge: GroundednessJudge, gold: list[GroundGoldItem]
) -> JudgeCalibration:
    from .kb import get_article

    agree = 0
    false_positive = 0
    false_negative = 0
    for item in gold:
        verdict = judge.assess(item.answer_text, get_article(item.article_id))
        # Calibration scores the judge's decisiveness on labelled data; an
        # UNCERTAIN verdict counts as neither agreement nor a directional error.
        grounded = verdict is GroundednessVerdict.GROUNDED
        if verdict is GroundednessVerdict.UNCERTAIN:
            continue
        if grounded == item.grounded:
            agree += 1
        elif grounded and not item.grounded:
            false_positive += 1
        elif not grounded and item.grounded:
            false_negative += 1
    return JudgeCalibration(
        judge_name=judge.name,
        total=len(gold),
        agree=agree,
        false_positive=false_positive,
        false_negative=false_negative,
    )

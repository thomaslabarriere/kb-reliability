"""Groundedness judge: is an answer actually supported by the cited article?

The generation layer's verdict is subjective, so it is delegated to a judge --
and, as everywhere in this repo, the judge's OWN reliability is measured against
a labelled gold set (who judges the judge?). A deterministic static judge runs
offline and in tests; an LLM judge runs with a key.
"""

from __future__ import annotations

from typing import Protocol

from .models import Article, GroundGoldItem, JudgeCalibration
from .retrieve import _tokens


class GroundednessJudge(Protocol):
    name: str

    def is_grounded(self, answer_text: str, article: Article) -> bool: ...


class StaticGroundednessJudge:
    """Deterministic: grounded iff the answer shares at least `min_overlap`
    content tokens with the article body. Offline, no key."""

    def __init__(self, min_overlap: int = 3) -> None:
        self.name = "static-overlap"
        self._min = min_overlap

    def is_grounded(self, answer_text: str, article: Article) -> bool:
        answer_terms = set(_tokens(answer_text))
        body_terms = set(_tokens(f"{article.title} {article.body}"))
        return len(answer_terms & body_terms) >= self._min


class AllGroundedJudge:
    """Deliberately bad judge (calls everything grounded) -- calibration must
    catch it with false positives."""

    name = "always-grounded"

    def is_grounded(self, answer_text: str, article: Article) -> bool:
        return True


def calibrate_judge(
    judge: GroundednessJudge, gold: list[GroundGoldItem]
) -> JudgeCalibration:
    from .kb import get_article

    agree = 0
    false_positive = 0
    false_negative = 0
    for item in gold:
        verdict = judge.is_grounded(item.answer_text, get_article(item.article_id))
        if verdict == item.grounded:
            agree += 1
        elif verdict and not item.grounded:
            false_positive += 1
        elif not verdict and item.grounded:
            false_negative += 1
    return JudgeCalibration(
        judge_name=judge.name,
        total=len(gold),
        agree=agree,
        false_positive=false_positive,
        false_negative=false_negative,
    )

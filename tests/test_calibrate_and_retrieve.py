"""The groundedness judge is calibrated (who judges the judge?), and the
retriever's permission/freshness behaviour is what the diagnostics assume."""

from __future__ import annotations

from kbreliability.goldset import GROUND_GOLD
from kbreliability.judge import AllGroundedJudge, StaticGroundednessJudge, calibrate_judge
from kbreliability.questions import get_question
from kbreliability.retrieve import KeywordRetriever


def test_static_judge_is_well_calibrated() -> None:
    cal = calibrate_judge(StaticGroundednessJudge(), GROUND_GOLD)
    assert cal.false_positive == 0
    assert cal.agreement_rate == 1.0


def test_calibration_catches_a_judge_that_calls_everything_grounded() -> None:
    cal = calibrate_judge(AllGroundedJudge(), GROUND_GOLD)
    assert cal.false_positive > 0
    assert cal.agreement_rate < 1.0


def test_retriever_is_permission_aware() -> None:
    # A public user must never see the internal fraud playbook.
    retrieved = KeywordRetriever().retrieve(get_question("q-card"), k=4)
    assert all(a.article_id != "fraud-playbook" for a in retrieved)


def test_freshness_naive_retriever_ranks_the_stale_card_article_first() -> None:
    retrieved = KeywordRetriever(freshness_aware=False).retrieve(get_question("q-card"), k=4)
    assert retrieved[0].article_id == "card-block-v1"  # the outdated, wordier one


def test_freshness_aware_retriever_drops_the_stale_sibling() -> None:
    retrieved = KeywordRetriever(freshness_aware=True).retrieve(get_question("q-card"), k=4)
    ids = {a.article_id for a in retrieved}
    assert "card-block-v2" in ids
    assert "card-block-v1" not in ids

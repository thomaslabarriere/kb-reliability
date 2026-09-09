"""Synthetic support questions with ground truth (gold topic + asker scopes)."""

from __future__ import annotations

from .models import Question

QUESTIONS: list[Question] = [
    Question(
        question_id="q-card",
        text="J'ai perdu ma carte, comment la bloquer ?",
        gold_topic="card-blocking",
        user_scopes=[],
    ),
    Question(
        question_id="q-iban",
        text="Où puis-je trouver mon IBAN ?",
        gold_topic="iban",
        user_scopes=[],
    ),
    Question(
        question_id="q-refund",
        text="Sous combien de temps mon remboursement est-il crédité ?",
        gold_topic="refund-delay",
        user_scopes=[],
    ),
    Question(
        question_id="q-kyc",
        text="Quels documents sont nécessaires pour vérifier mon identité ?",
        gold_topic="kyc",
        user_scopes=[],
    ),
    # This asker is a support agent allowed to see the (restricted) chargeback
    # article -- so retrieving it is legitimate, not a leak.
    Question(
        question_id="q-chargeback",
        text="Comment contester un paiement frauduleux sur un compte ?",
        gold_topic="chargeback",
        user_scopes=["support:disputes"],
    ),
]


def get_question(question_id: str) -> Question:
    for question in QUESTIONS:
        if question.question_id == question_id:
            return question
    raise KeyError(f"unknown question {question_id!r}")

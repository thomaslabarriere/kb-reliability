"""Labelled examples for calibrating the groundedness judge (who judges the
judge?). Each says whether `answer_text` is truly supported by the article."""

from __future__ import annotations

from .models import GroundGoldItem

GROUND_GOLD: list[GroundGoldItem] = [
    GroundGoldItem(
        question_id="q-iban",
        article_id="iban-v1",
        answer_text="Votre IBAN se trouve dans l'application, section Compte.",
        grounded=True,
    ),
    GroundGoldItem(
        question_id="q-card",
        article_id="card-block-v2",
        answer_text="Ouvrez l'application, section Carte, puis Bloquer.",
        grounded=True,
    ),
    GroundGoldItem(
        question_id="q-refund",
        article_id="refund-v1",
        answer_text="Le remboursement est crédité sous 5 à 10 jours ouvrés.",
        grounded=True,
    ),
    GroundGoldItem(
        question_id="q-kyc",
        article_id="kyc-v1",
        answer_text="Il faut une pièce d'identité valide et un justificatif de domicile récent.",
        grounded=True,
    ),
    # Ungrounded: plausible-sounding but not supported by the article.
    GroundGoldItem(
        question_id="q-card",
        article_id="card-block-v2",
        answer_text="Redémarrez votre téléphone et patientez soixante-douze heures.",
        grounded=False,
    ),
    GroundGoldItem(
        question_id="q-refund",
        article_id="refund-v1",
        answer_text="Le remboursement est instantané et automatique.",
        grounded=False,
    ),
]

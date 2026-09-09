"""Synthetic customer-service knowledge base (a fintech support KB, invented).

Some topics carry two versions (one current, one outdated) so freshness can be
measured; some articles require a scope so permission leakage can be measured.
No real data.
"""

from __future__ import annotations

from .models import Article

ARTICLES: list[Article] = [
    # --- card-blocking: two versions. The OLD one is wordier and repeats the
    # query terms, so a freshness-naive retriever ranks it first -> a realistic
    # staleness trap.
    Article(
        article_id="card-block-v2",
        topic="card-blocking",
        title="Bloquer une carte perdue",
        body=(
            "Ouvrez l'application, section Carte, puis Bloquer. Le blocage est "
            "immédiat et réversible."
        ),
        version=2,
        is_current=True,
    ),
    Article(
        article_id="card-block-v1",
        topic="card-blocking",
        title="Bloquer une carte (ancienne procédure)",
        body=(
            "Pour bloquer une carte perdue, appelez le service client afin de "
            "bloquer la carte par téléphone. Le blocage de la carte par "
            "téléphone reste possible pour bloquer une carte perdue."
        ),
        version=1,
        is_current=False,
    ),
    Article(
        article_id="iban-v1",
        topic="iban",
        title="Trouver son IBAN",
        body=(
            "Votre IBAN se trouve dans l'application, section Compte, rubrique "
            "Coordonnées bancaires."
        ),
    ),
    Article(
        article_id="refund-v1",
        topic="refund-delay",
        title="Délai de remboursement",
        body=(
            "Un remboursement est crédité sous 5 à 10 jours ouvrés après "
            "validation par le marchand."
        ),
    ),
    Article(
        article_id="kyc-v1",
        topic="kyc",
        title="Vérification d'identité",
        body=(
            "La vérification d'identité demande une pièce d'identité en cours de "
            "validité et un justificatif de domicile de moins de trois mois."
        ),
    ),
    # chargeback: restricted to support agents handling disputes.
    Article(
        article_id="chargeback-v1",
        topic="chargeback",
        title="Contester un paiement frauduleux",
        body=(
            "Pour contester un paiement frauduleux, ouvrez un litige depuis la "
            "transaction concernée puis fournissez les justificatifs demandés."
        ),
        required_scopes=["support:disputes"],
    ),
    # internal playbook: must NEVER surface to an end user (permission trap).
    Article(
        article_id="fraud-playbook",
        topic="internal-fraud",
        title="Playbook interne anti-fraude",
        body=(
            "Procédure interne: seuils de déclenchement, règles de scoring et "
            "contacts de l'équipe fraude. Document strictement interne."
        ),
        required_scopes=["internal:fraud"],
    ),
]


def is_permitted(article: Article, user_scopes: list[str]) -> bool:
    """True iff a user holding `user_scopes` is allowed to see `article`.
    Single source of truth for the permission check (retrieval + evaluation)."""
    return all(scope in user_scopes for scope in article.required_scopes)


def get_article(article_id: str) -> Article:
    for article in ARTICLES:
        if article.article_id == article_id:
            return article
    raise KeyError(f"unknown article {article_id!r}")


def current_article(topic: str) -> Article:
    for article in ARTICLES:
        if article.topic == topic and article.is_current:
            return article
    raise KeyError(f"no current article for topic {topic!r}")

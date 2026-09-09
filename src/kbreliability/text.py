"""Shared text tokenization, used by both the retrieval and judge layers.

Kept in one place so the layers depend on it explicitly rather than reaching
into each other's internals.
"""

from __future__ import annotations

_STOPWORDS = {
    "je", "j", "mon", "ma", "mes", "le", "la", "les", "un", "une", "des", "du",
    "de", "d", "et", "ou", "a", "au", "aux", "en", "sur", "pour", "par", "que",
    "qui", "comment", "est", "il", "elle", "sous", "combien", "temps",
    "puis", "sont", "quels", "quelles", "necessaires", "concerne", "afin",
}

_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")


def _split(text: str) -> list[str]:
    out: list[str] = []
    current = ""
    for ch in text:
        if ch.isalnum():
            current += ch
        elif current:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def tokens(text: str) -> list[str]:
    """Lowercase, accent-stripped content tokens (stopwords + short words out)."""
    lowered = text.lower().translate(_ACCENTS)
    return [t for t in _split(lowered) if len(t) > 2 and t not in _STOPWORDS]

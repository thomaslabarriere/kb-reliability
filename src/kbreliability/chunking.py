"""Chunking dimension: retrieval granularity on a long article.

Whole-article retrieval feeds the entire document as context even when only one
section answers the question -- diluting groundedness and inflating cost.
Chunking retrieves the relevant SECTION instead. This module chunks a long
sample article and measures, for a section-specific question, whether chunking
pinpoints the right section and how much context it saves. Offline, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .text import tokens as _tokens

# A long, multi-section support article (synthetic). Whole-article retrieval
# would return all of this for any sub-question.
LONG_DOC = (
    "Ouvrir un litige. Pour contester un paiement, ouvrez la transaction "
    "concernée dans l'application puis sélectionnez Signaler un problème. "
    "Un conseiller est assigné sous vingt-quatre heures. "
    "Justificatifs à fournir. Pour instruire le litige, joignez une preuve "
    "d'achat, une capture de la transaction et, le cas échéant, un échange "
    "écrit avec le marchand. Sans ces justificatifs le dossier reste bloqué. "
    "Délais de traitement. Une fois le dossier complet, le traitement prend "
    "de quinze à quarante-cinq jours selon le réseau de paiement. "
    "Escalade. Si le litige n'aboutit pas, vous pouvez demander une révision "
    "auprès du service réclamations en citant le numéro de dossier."
)

# Signature word of the "documents to provide" section (distinct from the
# other sections), used to check the right section was pinpointed.
_TARGET_SECTION_MARKER = "preuve"


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    """Greedy sentence packing into chunks of at most `max_chars` characters."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks: list[Chunk] = []
    current = ""
    for sentence in sentences:
        piece = sentence if sentence.endswith(".") else sentence + "."
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(Chunk(len(chunks), current.strip()))
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(Chunk(len(chunks), current.strip()))
    return chunks


def _score(query: str, text: str) -> int:
    q = set(_tokens(query))
    doc = _tokens(text)
    return sum(doc.count(term) for term in q)


@dataclass(frozen=True)
class ChunkAnalysis:
    question: str
    whole_context_chars: int
    chunk_context_chars: int
    best_chunk_text: str
    target_section_hit: bool

    @property
    def context_reduction(self) -> float:
        if self.whole_context_chars == 0:
            return 0.0
        return 1 - self.chunk_context_chars / self.whole_context_chars


def analyze(question: str, max_chars: int = 130) -> ChunkAnalysis:
    """Compare whole-article vs chunk retrieval for a section-specific question."""
    chunks = chunk_text(LONG_DOC, max_chars)
    best = max(chunks, key=lambda c: (_score(question, c.text), -c.index))
    return ChunkAnalysis(
        question=question,
        whole_context_chars=len(LONG_DOC),
        chunk_context_chars=len(best.text),
        best_chunk_text=best.text,
        target_section_hit=_TARGET_SECTION_MARKER in best.text.lower(),
    )

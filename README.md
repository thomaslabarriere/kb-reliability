# kb-reliability

**Layer-attributed reliability diagnostics for a customer-service RAG knowledge base.**

A RAG support assistant is only trustworthy if, when it answers wrong, you can say **which layer failed**, the job it's built for is *"diagnostiquer la performance d'un RAG en séparant les problèmes de retrieval, de contexte, de modèle et de génération."* `kb-reliability` runs a knowledge-base RAG pipeline over a labelled question set and attributes every failure to one layer:

| Layer | The failure it catches |
|---|---|
| `retrieval` | the correct article was never retrieved (recall) |
| `permissions` | retrieval surfaced an article this user isn't allowed to see (leak) |
| `freshness` | the answer was built on an **outdated** version of an article (stale) |
| `generation` | the answer isn't supported by the cited article (groundedness) |

A score tells you *that* it failed; a layer tells you *where to fix it*.

> **Scope.** Not a real knowledge base and not production advice. The articles, questions, and permission scopes are **synthetic** (a fintech support KB, invented), no client data. The retrievers are compact, honest implementations, a lexical term-frequency baseline, an embeddings retriever, and reciprocal-rank-fusion hybrid, **not** a production vector DB or a cross-encoder reranker; a real stack drops in behind the `Retriever` / `Reranker` protocols unchanged. The value here is the **diagnostic instrument** (and how it attributes failures), not the retrieval sophistication. Plug in a real KB + retriever + eval set for real numbers.

## Quick start (no API key needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

kb-reliability diagnose      # offline: keyword retriever + heuristic answerer
kb-reliability calibrate     # how reliable is the groundedness judge itself?
```

## The point: a failure the obvious metrics miss

The offline baseline scores **100% retrieval recall and 100% groundedness**, and still ships a wrong answer. Verbatim `kb-reliability diagnose`:

```
────────────────────────────────────────────────────────────────
kb-reliability: keyword / heuristic
────────────────────────────────────────────────────────────────
Réussite: 4/5 questions

Par question (faute attribuée à la couche)
  ✗ q-card  [freshness]   cité: card-block-v1
  ✓ q-iban   cité: iban-v1
  ✓ q-refund   cité: refund-v1
  ✓ q-kyc   cité: kyc-v1
  ✓ q-chargeback   cité: chargeback-v1

Métriques par couche
  Retrieval recall:   100%
  Groundedness:       100%
  Réponses périmées:  1
  Fuites de permission: 0

Attribution des fautes
  freshness    1
────────────────────────────────────────────────────────────────
```

The card-blocking question retrieves the right topic and answers faithfully, but from the **outdated** article, because the older, wordier version out-scores the concise current one (a real term-frequency trap). Recall and groundedness both look perfect; the failure is only visible on the **freshness** axis. That is why per-layer attribution matters. A freshness-aware retriever fixes it:

```bash
kb-reliability diagnose --retriever fresh   # the stale sibling is dropped -> all pass
```

## Run the real system (LLM)

```bash
export OPENAI_API_KEY=sk-...                 # the only thing needed to go live
kb-reliability diagnose --answerer llm --model gpt-4o
# OpenRouter: --answerer llm --provider openrouter --model anthropic/claude-3.7-sonnet
```

With a key, an LLM answers each question grounded in the retrieved articles and an LLM judges groundedness; the report adds inference cost ($/question, illustrative) and latency, the *qualité / latence / coût* arbitrage the role calls for.

For a reliability tool, the direction the judge fails matters. When the judge's LLM call raises or returns an unparseable reply, the answer is **not** silently counted as grounded (that would certify an answer nobody verified) and **not** counted as a generation failure (that would blame the model for a judge outage). It becomes an explicit *indéterminé* outcome, excluded from the groundedness rate and surfaced on its own line (`Groundedness indéterminé / erreur juge: N`), so a judge outage can never be mistaken for either a clean pass or a generation fault.

## Retrieval depth (chunking · hybrid · reranking)

Two of the three retrieval bricks the role names are wired into the diagnostic pipeline, not bolted on as demos: `--retriever semantic|hybrid` and `--reranker lexical|llm` run the full per-layer diagnosis through those components. Chunking is the exception, and is labelled as such below: it ships as a standalone `chunks` analysis (context reduction), not a stage inside `diagnose`.

**Hybrid retrieval**, `SemanticRetriever` (embeddings + cosine) and `HybridRetriever` (reciprocal-rank fusion of the lexical and semantic rankings, no score normalization needed). Both run **offline** via a deterministic hashed-bag-of-words embedder, or with real embeddings when a key is set:

```bash
kb-reliability retrievers                     # lexical vs semantic vs hybrid recall (offline)
kb-reliability diagnose --retriever hybrid    # full per-layer diagnosis on the hybrid retriever
```

Measured recall of the gold topic on the labelled question set (offline embeddings), verbatim `kb-reliability retrievers`:

```
Comparaison des retrievers (recall du bon article sur le jeu labellisé)
  Embeddings: hors-ligne (bag-of-words haché)
  retriever    recall
  keyword       100%
  semantic      100%
  hybrid        100%
```

**Honest finding: on this toy KB, hybrid does _not_ beat lexical**, all three already reach 100% recall (the gold topic is trivially retrievable, even at `--k 1`). Recall is saturated, so it cannot discriminate here, which is exactly the repo's thesis: recall looks perfect while the real failure hides on the **freshness** axis (see above). Hybrid's payoff appears on hard corpora where lexical misses paraphrased queries; this synthetic set is too easy to show it. The number above is pinned by a regression test, and `diagnose --retriever hybrid` runs the identical per-layer attribution as the keyword baseline, so the instrument is retriever-agnostic. Plug in a real KB + embeddings for real separation.

**Chunking**, whole-article retrieval feeds the entire document as context even when one section answers the question. Chunking pinpoints the section (offline):

```
$ kb-reliability chunks
Chunking: récupération ciblée sur un article long
  Question: Quels justificatifs et preuve d'achat joindre à un litige ?
  Article entier: 667 caractères de contexte
  Meilleur chunk: 138 caractères
  Bonne section retrouvée: oui
  Réduction du contexte: 79%
  Chunk: « Pour instruire le litige, joignez une preuve d'achat, une capture de la transaction et, le cas échéant, un échange écrit avec le marchand. »
```

The right section is retrieved with **79% less context** (138 vs 667 characters) than feeding the whole article. That is a measured context reduction, pinned by a regression test; feeding a model fewer, on-target tokens is what lowers per-call cost and narrows what an answer can be (un)grounded in, but this repo measures only the context reduction, not a downstream groundedness or dollar delta on this toy doc.

**Reranking**, first-stage lexical retrieval optimises recall, not precision@1; a wordy distractor can sit at rank 1 (which the answerer cites). A reranker fixes the order. It is wired into the pipeline (`kb-reliability diagnose --reranker lexical|llm` reorders the shortlist before the answer step); `kb-reliability rerank` isolates the precision@1 flip on a crafted shortlist:

```
$ kb-reliability rerank
Reranking (rerank:lexical-title)
  Question: Comment activer ma nouvelle carte ?
  Rang 1 avant: card-limit (incorrect)
  Rang 1 après: card-activation (correct)
```

## Who judges the judge?

Groundedness is a judgement call, so its verdict is delegated to a judge, whose **own** reliability is measured against a labelled gold set:

```
Calibration du juge « static-overlap »
  Accord: 100% (6/6)
  Faux positifs: 0   Faux négatifs: 0
```

`kb-reliability calibrate` scores the judge; the tests prove a judge that rubber-stamps everything as grounded is caught by its false positives.

## Why you can trust the instrument (fault injection per layer)

Not mutation testing (no operator mutates the diagnostic source), this is **fault injection on the system under test**: `tests/` feeds the diagnostic a component broken in one specific way and asserts the failure is produced **and attributed to the right layer**, a blind retriever → `retrieval`; a permission-blind retriever → `permissions`; the stale-ranking baseline → `freshness`; a hallucinating answerer → `generation`. A crashing stage is attributed to the stage that raised, a retriever that throws is a `retrieval` fault, an answerer that throws a `generation` one, never dumped on generation wholesale. A correct system passes clean. Plus the judge-calibration guards.

```bash
ruff check src tests
mypy
pytest
kb-reliability diagnose   # exits non-zero if any question fails -> CI gate
```

## Layout

```
src/kbreliability/
  models.py      # shared contracts (pydantic) incl. the Layer enum
  kb.py          # synthetic KB: versioned articles + permission scopes
  questions.py   # support questions + ground truth (gold topic, asker scopes)
  text.py        # shared tokenizer (retrieval + judge)
  retrieve.py    # lexical / semantic / hybrid (RRF) retrievers + fixtures
  embeddings.py  # embedding client + OFFLINE hashed-BoW fallback + cosine
  measure.py     # recall of lexical/semantic/hybrid on the gold set
  rerank.py      # reranking stage (lexical + LLM)
  chunking.py    # chunk vs whole-article retrieval analysis
  answer.py      # heuristic + LLM answerer, LLM groundedness judge
  judge.py       # groundedness judge (static + calibration)
  goldset.py     # labelled examples to calibrate the judge
  evaluate.py    # per-question layer attribution
  pipeline.py    # retrieve -> answer -> diagnose
  report.py      # aggregate + render the layer-attributed report
  pricing.py     # illustrative token pricing for the cost line
  cli.py         # diagnose | calibrate | retrievers | chunks | rerank
tests/           # per-layer fault injection + calibration + retrieval-depth guards
```

## License

MIT

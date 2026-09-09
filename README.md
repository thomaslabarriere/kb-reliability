# kb-reliability

**Layer-attributed reliability diagnostics for a customer-service RAG knowledge base.**

A RAG support assistant is only trustworthy if, when it answers wrong, you can say **which layer failed** — the job it's built for is *"diagnostiquer la performance d'un RAG en séparant les problèmes de retrieval, de contexte, de modèle et de génération."* `kb-reliability` runs a knowledge-base RAG pipeline over a labelled question set and attributes every failure to one layer:

| Layer | The failure it catches |
|---|---|
| `retrieval` | the correct article was never retrieved (recall) |
| `permissions` | retrieval surfaced an article this user isn't allowed to see (leak) |
| `freshness` | the answer was built on an **outdated** version of an article (stale) |
| `generation` | the answer isn't supported by the cited article (groundedness) |

A score tells you *that* it failed; a layer tells you *where to fix it*.

> **Scope.** Not a real knowledge base and not production advice. The articles, questions, and permission scopes are **synthetic** (a fintech support KB, invented), no client data. The value is the diagnostic instrument, not the content. Plug in a real KB + eval set for real numbers.

## Quick start (no API key needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

kb-reliability diagnose      # offline: keyword retriever + heuristic answerer
kb-reliability calibrate     # how reliable is the groundedness judge itself?
```

## The point: a failure the obvious metrics miss

The offline baseline scores **100% retrieval recall and 100% groundedness** — and still ships a wrong answer:

```
Métriques par couche
  Retrieval recall:   100%
  Groundedness:       100%
  Réponses périmées:  1
  Fuites de permission: 0

Attribution des fautes
  freshness    1
```

The card-blocking question retrieves the right topic and answers faithfully — but from the **outdated** article, because the older, wordier version out-scores the concise current one (a real term-frequency trap). Recall and groundedness both look perfect; the failure is only visible on the **freshness** axis. That is why per-layer attribution matters. A freshness-aware retriever fixes it:

```bash
kb-reliability diagnose --retriever fresh   # the stale sibling is dropped -> all pass
```

## Run the real system (LLM)

```bash
export OPENAI_API_KEY=sk-...                 # the only thing needed to go live
kb-reliability diagnose --answerer llm --model gpt-4o
# OpenRouter: --answerer llm --provider openrouter --model anthropic/claude-3.7-sonnet
```

With a key, an LLM answers each question grounded in the retrieved articles and an LLM judges groundedness; the report adds inference cost ($/question, illustrative) and latency — the *qualité / latence / coût* arbitrage the role calls for.

## Who judges the judge?

Groundedness is a judgement call, so its verdict is delegated to a judge — whose **own** reliability is measured against a labelled gold set:

```
Calibration du juge « static-overlap »
  Accord: 100% (6/6)
  Faux positifs: 0   Faux négatifs: 0
```

`kb-reliability calibrate` scores the judge; the tests prove a judge that rubber-stamps everything as grounded is caught by its false positives.

## Why you can trust the instrument (mutation proof)

`tests/` asserts each layer's failure is produced **and attributed to the right layer**: a blind retriever → `retrieval`; a permission-blind retriever → `permissions`; the stale-ranking baseline → `freshness`; a hallucinating answerer → `generation`; a correct system passes clean; a raising component is isolated per question. Plus the judge-calibration guards.

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
  retrieve.py    # lexical retriever (permission-aware, freshness-naive) + fixtures
  answer.py      # heuristic + LLM answerer, LLM groundedness judge
  judge.py       # groundedness judge (static + calibration)
  goldset.py     # labelled examples to calibrate the judge
  evaluate.py    # per-question layer attribution
  pipeline.py    # retrieve -> answer -> diagnose
  report.py      # aggregate + render the layer-attributed report
  pricing.py     # illustrative token pricing for the cost line
  cli.py         # kb-reliability diagnose | calibrate
tests/           # mutation-proof (per layer) + calibration + retriever guards
```

## License

MIT

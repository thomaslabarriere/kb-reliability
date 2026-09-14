# Decisions

Why this instrument is shaped the way it is. Each entry is a fork I actually
hit, the options I weighed, and what I chose, plus what the choice still does
**not** prove. If you only read one file to judge the engineering, read this one:
the code is the *what*, this is the *why*.

---

## 1. The verdict comes from assigned state, never from the model's prose

**Fork.** Groundedness could be judged by asking the LLM "was this answer
correct?" and trusting its self-report, or by comparing the *cited article id*
and the *retrieved set* against ground truth.

**Chosen.** State, not prose. `evaluate.py` attributes a fault by looking at
which article the answer *cited* and which the retriever *surfaced*, against the
question's `gold_topic`. A model that says "I'm confident this is right" earns
nothing; only the citation it committed to is scored.

**Why.** A reliability tool that trusts the thing it is measuring measures
nothing. Self-report is the first thing to fail under distribution shift.

**Doesn't prove.** Groundedness of the *answer text itself* (does the prose
follow from the cited article?) still needs a judge, see decision 4.

---

## 2. The freshness trap is built, not scripted, and it is the whole point

**Fork.** To show that "100% recall + 100% groundedness" can still be wrong, I
could hard-code a `freshness` fault on `q-card`, or make it *emerge* from a
retriever ranking a stale article above the current one.

**Chosen.** Emerge. In `kb.py`, `card-block-v1` (the outdated version) is
deliberately wordier and repeats the query terms; the lexical term-frequency
score in `retrieve.py` genuinely ranks it above the concise current
`card-block-v2`. Nothing sets `fault = FRESHNESS` by hand, the stale answer is
retrieved, cited, faithfully summarised, and only the freshness axis catches it.

**Why.** A scripted failure proves the report can print "freshness". An emergent
one proves the *instrument* detects a failure the obvious metrics (recall,
groundedness) both rate perfect. That gap is the reason per-layer attribution
exists.

**Doesn't prove.** This is a *constructed demonstration* on a 7-article toy KB,
not a discovery on a real corpus. It shows the instrument attributes correctly;
it does not claim I found this phenomenon in the wild.

---

## 3. A crashing stage is attributed to the stage that crashed, not to generation

**Fork.** When a component raises (a retriever throws, the model API 500s), the
easy path is one `try/except` around the whole pipeline that labels every
exception a `generation` fault, one catch, done.

**Chosen.** Per-stage `try/except` in `pipeline.py`. A retriever that raises is a
`RETRIEVAL` fault; an answerer that raises is a `GENERATION` fault; a crash in the
diagnostic harness itself is `INFRA`. I added a `Layer.INFRA` rather than
overload an existing layer.

**Why.** The entire promise of the tool is *"a layer tells you where to fix it."*
Blaming generation for a retrieval crash sends the on-call engineer to the wrong
layer, it actively contradicts the thesis. This was originally the wrong
(single-catch) way; I changed it because a reviewer reading the test could see
the tool lying about its own core claim. The fix is small; shipping the
contradiction would have been fatal in a technical interview.

**Doesn't prove.** `INFRA` is a coarse bucket, it does not yet distinguish a
transient timeout (retry) from a bad deploy (page someone).

---

## 4. Judge outage becomes `UNCERTAIN`, never a silent pass and never a fake generation fault

**Fork.** When the groundedness judge's own LLM call fails or returns garbage,
the answer is either (a) counted grounded (fail-open), (b) counted ungrounded /
generation fault (fail-closed), or (c) surfaced as a third state.

**Chosen.** (c) `GroundednessVerdict.UNCERTAIN`: excluded from the groundedness
rate, reported on its own line. Commit `c980f8d` was exactly this fix, the judge
used to fail *open*.

**Why.** Fail-open certifies an answer nobody verified, the most dangerous
outcome for a health/finance-adjacent KB. Fail-closed fabricates a *generation*
failure out of a *judge* outage, blaming the model for the tool's blind spot and
polluting the very layer attribution the tool exists to get right. The honest
answer to "did the model ground this?" when the judge is down is "I don't know",
and a reliability tool has to be able to say that.

**Doesn't prove.** The *offline* judge (`static-overlap`, ≥3 shared content
tokens) is deliberately trivial, see decision 5.

---

## 5. The offline judge is trivial on purpose, and its "6/6" is a floor, not a headline

**Fork.** Ship a strong offline groundedness judge, or a deliberately weak one
plus the harness that measures *any* judge.

**Chosen.** Weak judge (`judge.py`, lexical overlap) + `calibrate` command that
scores the judge against a labelled gold set (`goldset.py`). The 6/6 agreement is
on 6 items constructed so grounded answers reuse the article's vocabulary.

**Why.** The deliverable is *"who judges the judge?"*, the calibration harness,
not a state-of-the-art judge. I would rather ship a weak judge I can *measure*
than a strong one I can't. Being explicit: 6/6 is an artefact of a small,
lexically-clean gold set; this judge will false-positive on a hallucination that
reuses the article's words and false-negative on a correct paraphrase. It is a
floor that proves the harness runs, not a reliability claim. The real judge is
the LLM one (decision 4); the harness is what makes *either* judge auditable.

**Doesn't prove.** Judge quality on real, paraphrase-heavy answers. That needs a
bigger, adversarial gold set, the clearest next piece of work.

---

## 6. Permission leak is checked on the article the answer CITED, not only what was retrieved

**Fork.** Flag a permission leak when a forbidden article is *retrieved*, or when
it is *cited by the answer*.

**Chosen.** Both, but the cited-article check is the one that matters
(`evaluate.py`). Even a permission-aware retriever that never surfaces the
internal article can't stop an LLM from emitting that article's id by
hallucination or prompt injection.

**Why.** For regulated clients (banks, insurers, the posting's own examples),
"the model cited a document this user may not see" is the failure that gets you a
compliance incident, and it can happen *downstream* of a perfectly scoped
retriever. Checking only retrieval would miss the leak that actually reaches the
user.

**Doesn't prove.** It checks *identifier* leakage, not whether the answer *text*
paraphrases forbidden content without citing it, a harder content-level check.

---

## 7. Offline-by-default, real system behind one env var

**Fork.** Require an API key to run anything, or make the full diagnosis run
offline with deterministic fixtures and light up the LLM path only when a key is
present.

**Chosen.** Offline by default (`keyword` retriever, `heuristic` answerer,
`static-overlap` judge, hashed-bag-of-words embedder); `OPENAI_API_KEY` swaps in
the real LLM answerer + LLM judge + real embeddings behind the same `Retriever` /
`Answerer` / `Judge` protocols.

**Why.** A reviewer must be able to `pip install` and see the whole thesis in two
minutes, zero credits, deterministic, CI-gateable. And the LLM path is exercised
by tests with a fake client, so the seam is proven without spend.

**Doesn't prove.** Offline "semantic"/"hybrid" recall is hashed bag-of-words, so
it re-plays lexical overlap, it cannot show *semantic* separation without real
embeddings. On this toy KB recall is saturated at 100% and hybrid does not beat
lexical (stated in the README); the payoff of hybrid/rerank appears only on a
hard corpus with paraphrased queries and distractors.

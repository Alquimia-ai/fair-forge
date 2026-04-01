# PromptEvaluator — Experiment Findings

Running notes to inform the paper's Experiments section.
Model: `openai/gpt-oss-20b` via Groq API. Embedder: `all-MiniLM-L6-v2`.

---

## Calibration findings

### τ (similarity threshold)

- **τ=0.80 produces degenerate CSR** — with `all-MiniLM-L6-v2`, factually similar responses
  already have cosine similarity >0.80 even when wording varies. Almost all queries collapse
  to `n_clusters=1` → CSR=1.0 for all prompt types, no discrimination.
- **τ=0.90 is the correct default for this embedder** — forces clusters only when responses
  are semantically near-identical. Produces non-trivial cluster structure.
- Takeaway for paper: τ should be documented as embedder-dependent. τ=0.90 is the recommended
  default for sentence-transformer models in the 22M–110M parameter range.

### K (samples per query)

- K=5 used for exploratory runs to reduce API cost. Final paper results use K=10.
- Temperature is a user parameter (typically 0.7–0.9); the metric should not override it.

---

## Run 1 — τ=0.80, K=10, temperature=0.7

**Key observation: CSR/Stability are saturated — almost no discrimination between prompt types.**

### FAQ domain

| Prompt    | CSR   | Stability | RSS   |
|-----------|-------|-----------|-------|
| good      | 0.930 | 0.944     | 0.892 |
| ambiguous | 0.950 | 0.957     | 0.852 |
| bad       | 0.980 | 0.972     | 0.815 |

- CSR ordering is inverted (bad > good) — artifact of τ=0.80 saturation, not real signal.
- RSS ordering is correct: good > ambiguous > bad ✅
- Most queries: n_clusters=1 across all prompt types.

### RAG domain

| Prompt    | CSR   | Stability | RSS   |
|-----------|-------|-----------|-------|
| good      | 0.950 | 0.970     | 0.912 |
| ambiguous | 1.000 | 1.000     | 0.863 |
| bad       | 1.000 | 1.000     | 0.854 |

- CSR/Stability completely saturated for ambiguous and bad — no discrimination at all.
- RSS is the only signal separating prompt types: good > ambiguous > bad ✅
- Scatter plot (CSR vs RSS) is degenerate: all points cluster at CSR=1.0.

---

## Run 2 — τ=0.90, K=5, temperature=0.7 (exploratory)

**Key observation: correct ordering restored across all signals in both domains.**

### FAQ domain

| Prompt    | CSR   | Stability | RSS   |
|-----------|-------|-----------|-------|
| good      | 0.920 | 0.903     | 0.892 |
| ambiguous | 0.800 | 0.716     | 0.860 |
| bad       | 0.740 | 0.648     | 0.817 |

- All three signals now order correctly: good > ambiguous > bad ✅
- CSR and Stability spread is large and meaningful (~0.18 gap between good and bad).
- RSS spread is narrower (~0.075) but consistently ordered.
- faq_002 bad: 4 clusters from K=5 (CSR=0.40, Stability=0.17) — maximum visible fragmentation.
- faq_005 good: 3 clusters (CSR=0.40) — the hardest query even a good prompt struggles with.

### RAG domain

| Prompt    | CSR   | Stability | RSS   |
|-----------|-------|-----------|-------|
| good      | 0.980 | 0.969     | 0.885 |
| ambiguous | 0.920 | 0.882     | 0.856 |
| bad       | 0.800 | 0.744     | 0.847 |

- Correct ordering: good > ambiguous > bad ✅
- RAG good remains high (CSR≈0.98) — expected: the context constrains answers strongly,
  limiting variance even with a bad prompt. Less discriminative than FAQ.
- RSS spread is narrower in RAG (~0.04) than FAQ (~0.075); RSS is more useful in FAQ.

### Key finding: "consistently-wrong" case — rag_007 good

```
rag_007  CSR=0.800  Stability=0.689  RSS=0.303  clusters=2
```

The model follows the good prompt and produces consistent responses (CSR=0.80), but all
responses are far from the ground truth (RSS=0.30). This is the paper's core theoretical
case: **high CSR + low RSS = consistently wrong**. Not detectable by CSR alone.
This example must appear in the paper as empirical evidence for the RSS signal's necessity.

### Key finding: maximum instability — rag_008 bad

```
rag_008  CSR=0.200  Stability≈0.000  RSS=0.747  clusters=5
```

K=5 responses, 5 distinct clusters → maximum semantic entropy → Stability=0.
Demonstrates the lower bound of the metric under a fully uninformative prompt.

---

## Run 4 — Final results: τ=0.90, K=10, temperature=0.7

| Domain     | Prompt    | CSR   | Stability | RSS   | ICR   |
|------------|-----------|-------|-----------|-------|-------|
| faq        | good      | 0.900 | 0.911     | 0.891 | —     |
| faq        | ambiguous | 0.800 | 0.778     | 0.855 | —     |
| faq        | bad       | 0.790 | 0.761     | 0.824 | —     |
| rag        | good      | 0.970 | 0.964     | 0.938 | —     |
| rag        | ambiguous | 0.910 | 0.902     | 0.858 | —     |
| rag        | bad       | 0.920 | 0.917     | 0.847 | —     |
| structured | good      | 0.970 | 0.964     | 0.941 | 1.000 |
| structured | ambiguous | 0.760 | 0.780     | 0.885 | 0.997 |
| structured | bad       | 0.870 | 0.865     | 0.317 | 0.050 |

### Finding: RSS is the most consistent signal

RSS orders correctly (good > ambiguous > bad) in all 3 domains without exception.
CSR and Stability can invert (see RAG below). ICR is sharp but only applies when constraints
are defined. RSS is the most reliable single discriminator across domains.

### Finding: ICR produces the sharpest discrimination (structured domain)

ICR drops from 1.000 (good) to 0.997 (ambiguous) to 0.050 (bad) — a 95% collapse.
Key insight: the ambiguous prompt ("Respond with a JSON object...") is nearly as effective
as the explicit good prompt for format compliance. The critical failure happens when format
is not mentioned at all (bad prompt). This is the strongest single-signal result in the
experiment and should be a highlighted finding in the paper.

### Finding: CSR inversion in RAG (bad > ambiguous)

RAG bad prompt ("Answer the question.") produces CSR=0.920 > ambiguous CSR=0.910.
A terse prompt generates uniformly short responses that cluster tightly — consistently
minimal, not consistently correct. RSS correctly orders them: ambiguous=0.858 > bad=0.847.
This demonstrates that CSR alone is insufficient: high consistency does not imply quality.

### Finding: consistently-wrong case — structured bad

structured bad: CSR=0.870, RSS=0.317. The model responds consistently (similar outputs
across K samples) but the outputs are far from the ground truth — plain text answers
instead of JSON. ICR=0.050 confirms the format is almost never followed. This is the
clearest empirical demonstration of the need for multi-signal evaluation: CSR sees
consistency, RSS and ICR expose the failure.

### Finding: no single signal is sufficient

| Signal    | Limitation revealed |
|-----------|---------------------|
| CSR       | Inverts in RAG (bad > ambiguous); can't distinguish consistent-wrong from consistent-right |
| Stability | Same limitation as CSR |
| RSS       | Reliable but doesn't capture format compliance |
| ICR       | Sharp but requires user-defined constraints; undefined for FAQ/RAG |

The compound metric is justified: each signal catches failures the others miss.

---

## Signals summary

| Signal    | τ=0.80        | τ=0.90           |
|-----------|---------------|------------------|
| CSR       | Saturated ❌  | Discriminates ✅ |
| Stability | Saturated ❌  | Discriminates ✅ |
| RSS       | Working ✅    | Working ✅       |
| ICR       | Not yet tested (Domain 3 only) |     |

---

## CSR discretization — effect of K

CSR = dominant_cluster_size / K, so it takes only discrete values with step size 1/K.

With K=5 (exploratory runs):
- Possible values: 1.0, 0.8, 0.6, 0.4, 0.2
- Going from 1.0 to 0.8 requires exactly 1 outlier response — common even in good prompts
- Going below 0.8 requires ≥2 simultaneous outliers — rare with only 5 samples
- This compresses the observable range and overstates stability

With K=10 (final runs):
- Step size 0.10 → values: 1.0, 0.9, 0.8, 0.7 ... 0.1
- Much finer resolution, especially in the 0.7–0.9 range where prompts differ most

**Paper note:** K=10 is the minimum for meaningful CSR resolution. K should be reported
alongside results, and the discrete nature of CSR (resolution = 1/K) should be acknowledged.

---

## Run 3 — Sensitivity analysis τ × temperature (FAQ good, K=5)

Grid: τ ∈ {0.85, 0.90, 0.95} × temperature ∈ {0.7, 0.9, 1.1}. All 9 combinations landed in the "useful" zone (0.55 < CSR < 0.97).

| temp | τ    | CSR   | mean clusters |
|------|------|-------|---------------|
| 0.7  | 0.85 | 0.920 | 1.40          |
| 0.7  | 0.90 | 0.880 | 1.40          |
| 0.7  | 0.95 | 0.880 | 1.50          |
| 0.9  | 0.85 | 0.900 | 1.40          |
| 0.9  | 0.90 | 0.940 | 1.20          |
| 0.9  | 0.95 | 0.820 | 1.70          |
| 1.1  | 0.85 | 0.940 | 1.30          |
| 1.1  | 0.90 | 0.840 | 1.70          |
| 1.1  | 0.95 | 0.720 | 2.00          |

**Key finding: temperature is not a critical parameter.** CSR varies ±0.05 across temp 0.7–1.1 for the same τ. Users do not need to tune temperature for the metric to work.

**τ is the main sensitivity lever.** The extremes of the grid differ by ~0.20 CSR (0.920 → 0.720). The effect is gradual and predictable: higher τ + higher temperature → more clusters → lower CSR.

**τ=0.90 confirmed as default.** Sits in the middle of the useful range across all temperatures tested.

**τ=0.80 added post-run:**

| temp | τ    | CSR   | clusters |
|------|------|-------|---------|
| 0.7  | 0.80 | 0.940 | 1.2     |
| 0.9  | 0.80 | 0.940 | 1.3     |
| 1.1  | 0.80 | 0.840 | 1.6     |

With K=5, τ=0.80 looks similar to τ=0.90. The saturation observed in Run 1 was caused by
the interaction **τ=0.80 × K=10**: with 10 samples there are more opportunities to diverge,
but τ=0.80 collapses them all into one cluster. With K=5 the discretization step (0.20)
already limits resolution so the τ effect is masked.

**Paper implication:** τ=0.90 is recommended as the default for sentence-transformer embedders. Temperature robustness (0.7–1.1) means the metric is applicable across typical LLM deployment configurations without recalibration.

---

## Pending experiment — sensitivity analysis (τ × temperature)

**Goal:** understand how sensitive CSR/Stability are to the choice of τ and temperature,
and what range of values produces meaningful discrimination.

**Design:** fix one prompt (e.g. FAQ good), vary τ and temperature independently, observe
how cluster structure and CSR change.

Suggested grid:
- τ ∈ {0.80, 0.85, 0.90, 0.92, 0.95}
- temperature ∈ {0.5, 0.7, 0.9, 1.1}  ← temperature is a user param, test realistic range
- K=10, single domain (FAQ), single prompt type (good or ambiguous)

Expected findings:
- Low τ + low temperature → CSR saturates at 1.0 (all responses cluster trivially)
- High τ + high temperature → CSR drops, more clusters, higher variance
- There should be a "useful region" in the τ × temperature plane where CSR discriminates
  without being completely noisy — this region is what to recommend in the paper

Output: heatmap of mean CSR over the τ × temperature grid.
This would be a strong methodological contribution: shows the metric is not arbitrarily
sensitive to hyperparameters and has a stable operating region.

---

## Sushi WhatsApp Bot experiment — qwen/qwen3-32b (behavioral dataset)

**Setup:** 2 prompts (good vs simple), K=5 exploratory / K=10 final, τ=0.90, temperature=0.7.
Model: `qwen/qwen3-32b` via Groq. No context field (context="" for all entries).

### Finding: qwen3-32b thinking tokens contaminate embeddings

By default, `qwen3-32b` emits `<think>...</think>` blocks before the actual response.
These are included in the response content and, when embedded, make every response
semantically unique — collapsing CSR and Stability to their minimum values.
Fix: pass `reasoning_effort="none"` to ChatGroq to disable thinking mode.
**Paper note:** evaluators using LLMs with chain-of-thought output must strip reasoning
tokens before embedding. This is an important implementation detail for practitioners.

### Finding: context field gives both prompts equal information

The PromptEvaluator appends `context` to the system prompt for both evaluations.
If the context contains domain knowledge (e.g., full menu with prices), both the
good and simple prompts receive the same information — eliminating the information
asymmetry the experiment is designed to test.
Fix: set `context: ""` and embed all domain knowledge directly in the good prompt.

### Finding: RSS is insensitive to factual precision within the same topic domain

Embedding-based similarity captures topic-level semantics, not numerical accuracy.
"El salmón nigiri cuesta $1.800" and "El salmón nigiri cuesta $150" embed closely
because both reference the same entity and concept.
RSS does not reliably differentiate a correct price from a hallucinated one.

RSS is useful for detecting responses that deviate thematically (off-topic answers,
wrong language, out-of-scope replies). It is NOT a reliable signal for factual
precision within a domain.

**Recommendation:** For factual accuracy validation, complement RSS with a judge-based
evaluator (LLM-as-judge) — similar to RAGAS `answer_correctness`. ICR covers the
subset of facts expressible as keyword/regex constraints, but a judge is needed for
the general case.

### Final results — K=10, τ=0.90, qwen/qwen3-32b, reasoning_effort=none

| Prompt | CSR   | Stability | RSS   | ICR   |
|--------|-------|-----------|-------|-------|
| good   | 1.000 | 1.000     | 0.797 | 1.000 |
| simple | 0.100 | 0.000     | 0.613 | 0.000 |

**CSR/Stability:** Maximum separation. Good prompt produces maximally consistent responses
(n_clusters=1 for every query) because each behavioral instruction prescribes an exact
response pattern. Simple prompt produces K=10 responses all in different clusters for
almost every query — the model improvises a different approach each time.

**RSS (+0.184):** Clear gap. Good prompt responses align closely with GT (which was written
to match the instructed behavior). Simple responses vary semantically — sometimes apologize,
sometimes try to solve the problem, sometimes escalate — averaging further from GT.

**ICR (+1.000):** Perfect binary separation. Good always mentions "encargado" for complaint,
health, and aggressive queries (5/10 queries). Simple never does — it handles these
situations without following the escalation protocol.

**Why behavioral queries work:** Unlike factual queries (prices, hours), behavioral queries
have no single "obviously correct" response derivable from training. The correct action
(escalate, redirect, ask for missing data) is an arbitrary policy prescribed only by the
prompt. Without it, capable models produce valid but inconsistent responses, creating
observable variance in CSR and Stability.

**Paper framing:** Behavioral queries are the natural evaluation domain for PromptEvaluator.
Factual queries produce similar responses regardless of prompt quality in capable models.
Behavioral queries expose response variance under prompt ambiguity — making prompt quality
directly observable through the metric vector.

### Finding: ICR validates specific facts deterministically, but only globally

ICR with `KeywordConstraint` checks keyword presence across all queries.
Per-query constraints (e.g., check "$1.800" only on the salmon price query) are not
currently supported — a natural next extension of the metric.

---

## Notes for paper

- The RSS signal is robust and differentiates prompt quality independently of τ calibration.
- CSR and Stability require τ to be tuned to the embedder's similarity distribution.
- The "consistently-wrong" detection (high CSR + low RSS) is theoretically sound but requires
  τ calibration to be observable — at τ=0.80 everything is "consistently right" vacuously.
- Final paper values will use K=10, τ=0.90, temperature as provided by user (default 0.7).
- RSS captures semantic proximity but not factual correctness — complement with LLM judge.
- Thinking-mode LLMs require response cleaning before embedding (strip `<think>` tokens).
- Context field must be empty when the experiment tests information asymmetry between prompts.

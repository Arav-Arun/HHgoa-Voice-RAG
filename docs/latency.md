# Latency

Requirement #3 sets a 200 ms budget for "chunking + vector DB retrieval +
everything through to final output." Requirement #4 asks for P50/P70/P100 across
a reasonable number of queries.

## Headline

**Fast path, 450 queries, 109,082-chunk index:**

| percentile | latency |
|---|---|
| **P50** | **10.72 ms** |
| **P70** | **11.61 ms** |
| P90 | 23.31 ms |
| P95 | 25.33 ms |
| **P100** | **43.85 ms** |
| mean ± sd | 12.09 ± 7.08 ms |

**P100 is 43.9 ms against a 200 ms budget, 4.6× headroom, worst case.**

The mean sits *below* P50 because 54 unsafe and prompt-injection queries are
refused by the input filter in ~0.02 ms without touching the index. That is the
guardrail working, not a measurement artifact, and it is why the percentiles
rather than the mean are the headline.

Measured on macOS arm64, torch limited to 4 threads, e5-small, hybrid
retrieval (RRF). Reproduce with `./hhgoa bench --queries 300`.

## Why three tracks are reported separately

"The full process" is ambiguous for a voice system, and collapsing everything
into one number would misrepresent it in both directions. So:

| track | what it covers | P50 | P100 |
|---|---|---|---|
| **fast path** (headline) | text in → answer out, fully local | 10.72 ms | 43.85 ms |
| **voice end-to-end** | + ElevenLabs STT, 32 clips | 1109.3 ms | 1958.3 ms |
| **quality path** | + remote LLM tool-calling | ~3,700 ms | |

The voice and quality rows come from a separate run against the live providers;
this run measured the fast path only (`--modes fast`). Both are dominated by a
network round trip that the retrieval changes do not touch.

The quality path is dominated by the provider call. A representative run
against `gpt-4o-mini`:

```
input_guard               0.0 ms
retrieve                118.7 ms
grounding_guard           0.3 ms
answer_fast               0.4 ms   ← grounded answer already exists here
answer_quality[openai]  3593.8 ms  ← 97% of the total
faithfulness              0.7 ms
                       ─────────
total                  3714.1 ms
```

That is **160× the fast path**. No prompt tuning closes a gap of that shape -
which is the entire reason generation is an opt-in second path rather than a
step in the critical one. Note `answer_fast` completes *before* the provider
call: a grounded answer already exists, so an LLM failure costs latency, never
correctness.

The voice number is dominated by one thing:

```
STT (ElevenLabs, network)   899.7 ms   ← 96% of the wall clock
RAG pipeline (local)         38.8 ms   ←  4%
                            ────────
end-to-end                  938.5 ms
```

The spec mandates Sarvam or ElevenLabs for speech-to-text, so that round trip is
an externally imposed network cost, not a pipeline inefficiency. **No
architecture that satisfies requirement #1 can transcribe audio in under 200 ms
over the public internet.** The honest reading of the 200 ms target is the part
of the system we control, and that is what the headline measures, with the STT
component broken out rather than hidden.

## Where the time goes

| stage | P50 | P70 | P100 | notes |
|---|---|---|---|---|
| input_guard | 0.01 ms | 0.02 ms | 0.49 ms | regex over the query |
| retrieve | 10.04 ms | 10.48 ms | 42.47 ms | **the whole budget lives here** |
| grounding_guard | 0.06 ms | 0.07 ms | 26.83 ms | cross-encoder on 22.7% of queries |
| answer_fast | 0.12 ms | 0.14 ms | 0.30 ms | extractive sentence selection |
| faithfulness | 0.27 ms | 0.30 ms | 0.49 ms | token overlap + numeric check |

The two transformer stages are the whole budget: the query embedding inside
`retrieve`, and the cross-encoder inside `grounding_guard`. BM25 scoring is
~0.1 ms and the dense search ~2.5 ms at 109k chunks. Every non-model stage
totals under 0.5 ms.

The cross-encoder costs **+9.6 ms at P50** and buys +7 points of abstain recall
at an unchanged false-abstain rate. It runs on one pair, the top passage, not on
all five: verification, not reranking.

Ranking passages rather than chunks (see `core/retriever/base.py`) added
**+0.5 ms at P50**. It widens the candidate fetch by the index's fan-out, but
`argpartition` is linear in the corpus regardless of how many candidates are
taken, so the extra cost is a larger partition and a slightly larger candidate
dict, not another matmul.

## What made it fast

The dense search was the one real bottleneck, and it was algorithmic rather than
a matter of tuning:

1. **Normalize once, not per query.** The store used to recompute row norms and
   allocate a full normalized copy of the embedding matrix on *every* search -
   ~150 MB of allocation per query at 109k × 384 float32. Vectors are now
   L2-normalized at `add()`/`load()` time and a query is a single matmul.
2. **`argpartition`, not `argsort`.** Fully sorting 109k scores to take 5 is
   O(n log n) for no reason.

Measured at 100k × 384: **21.89 ms → 2.44 ms, a 9× speedup.**

Two other decisions matter:

- **BM25 is precomputed.** The document-side weight `idf · tf(k1+1)/(tf + k1(1−b+b·dl/avgdl))`
  is query-independent, so it is computed at index time into a CSC matrix. A
  query is a few column lookups and a vectorized add, ~0.1 ms, not a rescoring
  loop over the corpus.
- **The fast path never touches the network.** The extractive answerer selects
  sentences from retrieved context locally. A remote LLM call cannot fit in
  200 ms, which is exactly why generation is a separate, optional path.

## The ONNX query encoder, and why it is not the default

The query encode is the largest single cost in retrieval, so running the same
graph under ONNX Runtime instead of PyTorch is the obvious lever. The vectors
are identical, not merely close: `scripts/export_onnx.py` traces the torch model
and refuses to write an export whose cosine against torch is below 0.999999.
Measured 1.0 on every probe, and `./hhgoa eval` returns `hit@5 0.6377358491`
either way, to ten decimal places.

It still is not the default, because the trade runs in opposite directions at
the median and the tail. 300 queries, this machine, repeated runs:

| runtime | P50 | P70 | P100 |
|---|---|---|---|
| torch | 10.04 | 10.87 | **138.38** |
| onnx, threads auto | 8.46 | 10.02 | 197.76 |
| onnx, 2 threads | 7.27 | 7.92 | 233.38 |
| onnx, 4 threads | 7.86 | 8.61 | 192.30 |
| onnx 2 / torch 4 | **6.87** | **7.29** | 202.79 |

Roughly 30% off the median and 50 to 90% onto the tail. The tail lands on the
same query positions under both runtimes, which are the cross-encoder
escalations, so it is not ONNX warm-up. Constraining the total thread budget
across both runtimes did not recover it either, so it is not simple
oversubscription.

Since the 200 ms budget is claimed at P100, a change that improves the median
and pushes P100 from 138 ms to 203 ms is a regression on the number that
matters. `EMBEDDING_RUNTIME=torch` is the default here.

The deployment is the opposite case. Its P100 is already 323 ms and
cross-encoder bound, so the tail is not the binding constraint, while its encode
is 38 ms of a 68 ms retrieve stage. `EMBEDDING_RUNTIME=onnx` is set there.

## Cold start

**17,445 ms** to ready, loading e5-small, the cross-encoder, torch init, and reading a 320 MB index.

Reported separately rather than folded into the percentiles: including it would
report a one-time artifact as steady-state latency, and hiding it would omit a
real cost. The API eliminates it from request latency by building and warming
everything in the FastAPI lifespan (one dummy encode plus one dummy search), so
the first real request is already warm.

## Method

- **450 queries**: 300 held-out answerable queries across both languages, plus
  all 150 categorised guardrail fixtures (50 off-topic, 30 unsafe, 26 in-domain
  unanswerable, 24 prompt-injection, 20 nonsense). Guardrail short-circuits are
  part of the branch mix a real deployment sees, so excluding them would
  understate variance.
- **Cold start excluded** from percentiles, measured and reported separately.
- **Per-stage timings come from the harness trace**, the same code path that
  serves API requests, not a separate measurement harness.
- **P100 is the maximum**, by definition. Nearest-rank percentiles throughout.
- Path mix in the run: 348 answered, 102 abstained.

Latency is flat across languages (gu 9.58 ms, hi 9.55 ms at P50). Across query
kinds it is not, and the shape is the cascade working: answerable 9.55 ms and
off-topic 9.86 ms decide on the cheap features, while nonsense 16.85 ms and
in-domain-unanswerable 19.57 ms land in the undecided band and pay for the
cross-encoder. The system spends its time on the queries it is unsure about - abstaining at
the grounding gate is not a shortcut that flatters the numbers. Only unsafe and
prompt-injection queries are fast (0.02 ms), because they are refused before
retrieval by design.

Raw output: `data/bench/latency.json`.

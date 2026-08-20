# Latency

Requirement #3 sets a 200 ms budget for "chunking + vector DB retrieval +
everything through to final output." Requirement #4 asks for P50/P70/P100 across
a reasonable number of queries.

## Headline

**Fast path, 450 queries, 109,082-chunk index:**

| percentile | latency |
|---|---|
| **P50** | **12.19 ms** |
| **P70** | **12.75 ms** |
| P90 | 15.96 ms |
| P95 | 18.87 ms |
| **P100** | **32.86 ms** |
| mean ± sd | 11.57 ± 4.95 ms |

**P100 is 32.9 ms against a 200 ms budget, 6× headroom, worst case.**

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
| **fast path** (headline) | text in → answer out, fully local | 12.19 ms | 32.86 ms |
| **voice end-to-end** | + ElevenLabs STT round trip | 938.5 ms | 1021.0 ms |
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
| input_guard | 0.01 ms | 0.01 ms | 0.47 ms | regex over the query |
| retrieve | 11.91 ms | 12.38 ms | 31.53 ms | **the whole budget lives here** |
| grounding_guard | 0.05 ms | 0.06 ms | 0.18 ms | 3 features + a sigmoid |
| answer_fast | 0.12 ms | 0.13 ms | 0.26 ms | extractive sentence selection |
| faithfulness | 0.27 ms | 0.30 ms | 0.46 ms | token overlap + numeric check |

Retrieval is ~97% of the total, and within it the query embedding dominates -
BM25 scoring is ~0.1 ms and the dense search ~2.5 ms at 109k chunks. Everything
after retrieval costs under 0.5 ms combined.

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

## Cold start

**8,643 ms**, loading e5-small, torch init, and reading a 288 MB index.

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
- Path mix in the run: 346 answered, 104 abstained.

Latency is flat across languages (gu 11.77 ms, hi 12.35 ms at P50) and across
the query kinds that actually reach retrieval (answerable 12.51 ms, off-topic
12.16 ms, in-domain unanswerable 12.13 ms, nonsense 11.62 ms) - abstaining at
the grounding gate is not a shortcut that flatters the numbers. Only unsafe and
prompt-injection queries are fast (0.02 ms), because they are refused before
retrieval by design.

Raw output: `data/bench/latency.json`.

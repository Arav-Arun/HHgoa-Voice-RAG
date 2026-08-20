# RAG architecture

Modular pipeline scoped to **Hindi** and **Gujarati**. Each layer has a base
class, swap one piece without rewriting the rest.

## Flow

```
ingest/loaders → core/chunking → core/embeddings ─┬─▶ core/vectorstore (dense)
                                                  └─▶ core/retriever/sparse (BM25)
                                                              │
core/stt (voice) ──▶ core/harness/orchestrator ◀──────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  core/guardrails    core/retriever/hybrid   core/llm
  (intent, gate,      (RRF fusion)      (extractive | chat)
   faithfulness)
                            │
                            ▼
                     answer + trace
```

Evaluation and benchmarking (`eval/`, `bench/`) drive the same orchestrator the
API does, so published numbers come from shipping code.

## Two paths

- **fast** (default), input guard → hybrid retrieve → grounding gate →
  extractive answer → faithfulness. Fully local, P100 49.7 ms.
- **quality** (`mode=quality`), the fast path, then LLM tool-calling with
  structured output, retries, and provider failover. Degrades to the fast
  answer on any failure.

See [harness.md](harness.md).

## Where to tweak what

| Goal | Edit |
|------|------|
| Chunk size / overlap | `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) or a new class in `core/chunking/` |
| Chunking strategy | `.env` `CHUNKING_PROVIDER=fixed\|semantic\|metadata\|recursive\|parent_child\|token_window` |
| Embedding model | `.env` `EMBEDDING_PRESET=e5-small\|indic-sbert\|bge-m3`, or add a provider in `core/embeddings/` + `core/factory.py` |
| Retrieval strategy | `.env` `RETRIEVER=dense\|sparse\|hybrid` |
| Fusion behaviour | `.env` `FUSION`, `RRF_K`, `FUSION_DENSE_WEIGHT_HI/GU` |
| Vector DB | Implement `BaseVectorStore` in `core/vectorstore/`, register in `core/factory.py` |
| Tokenization (all layers) | `core/text.py` |
| Prompts / language tone | `core/rag/prompts.py` |
| Guardrails | `core/guardrails/` + `core/factory.py` |
| Orchestration, retries, tools | `core/harness/` |
| LLM provider | `.env` `LLM_PROVIDER`, `LLM_API_KEY_SECONDARY` for failover |
| Speech-to-text | `core/stt/` + `core/factory.py` |
| Data sources | `ingest/loaders.py` |
| Metrics | `eval/metrics.py`, `eval/significance.py` |
| HTTP API | `api/app.py`, `api/schemas.py` |
| Demo UI | `api/static/` (no build step; pure client of the API) |
| Config defaults | `core/config.py`, `.env.example` |

## Components (defaults)

- **Chunker**, `fixed` (character windows). Five alternatives registered; see
  the README table for what distinguishes each.
- **Embedder**, `SentenceTransformerEmbedder` with `e5-small`, applying
  `query:` / `passage:` prefixes.
- **Store**, `MemoryVectorStore`: numpy cosine search over an L2-normalized
  matrix, `argpartition` top-k, persisted under `data/index/`.
- **Sparse index**, `BM25Index`: query-independent document weights
  precomputed into a scipy CSC matrix at ingest time.
- **Retriever**, `HybridRetriever`, weighted Reciprocal Rank Fusion with
  per-language dense weights tuned on the dev slice.
- **Guardrails**, `CompositeGuardrail`: `InputIntentFilter` (unsafe + injection,
  pre-retrieval) → `ConfidenceGate` (multi-feature grounding) →
  `HallucinationChecker` (token overlap + numeric grounding). `GUARDRAIL_PROVIDER=stub`
  disables.
- **Fast answerer**, `ExtractiveAnswerer`: IDF-weighted sentence selection from
  retrieved context. Grounded by construction, it cannot invent.
- **Quality LLM**, `ChatClient` against any OpenAI-compatible endpoint
  (OpenAI, Groq, local vLLM); only `base_url` and `model` differ.
- **STT**, `ElevenLabsSTT` (`scribe_v2`); `StubSTT` without a key.

## Key invariants

- **`ScoredChunk.score` is whatever the active retriever ranks by.** Under
  hybrid retrieval it is an RRF value with no absolute meaning. Anything needing
  a *calibrated* signal must read `ScoredChunk.dense_score`. The grounding gate
  does; thresholding the fusion score against a cosine-calibrated value would
  abstain on every query.
- **BM25 row *i* describes the same chunk as embedding row *i*.** The sparse
  index is built from `store.chunks` after ingest, never from a separately
  accumulated list, because any drift would silently corrupt fusion.
- **`document_id` is the parent passage**, for every chunker. `parent_child`
  emits several chunks per passage but keeps `document_id` on the parent, so
  eval labels, citations, and dedup are unaffected by chunking choice.
- **Retrieval ranks passages, not chunks.** Candidates collapse to one best
  chunk per `document_id` *before* fusion ranks them, and `candidate_k` counts
  passages. Without this, a chunker's fan-out silently penalises it: sibling
  chunks consume RRF ranks and candidate slots that distinct passages need. See
  `core/retriever/base.py`.
- **Everything tuned is tuned on dev.** Eval is touched only to report.

## Adding a new component

1. Subclass the relevant base (`BaseChunker`, `BaseEmbedder`, `BaseRetriever`,
   `BaseVectorStore`, `BaseGuardrail`, `BaseLLM`, `BaseSTT`).
2. Register it in the matching `core/factory.py` builder.
3. Add the provider string to `core/config.py` and `.env.example`.
4. If it changes retrieval quality, add it to the relevant `*-compare` command
   so the claim comes with a bootstrap p-value.

## CLI

```bash
./hhgoa ingest [path|msmarco]     # build vector + BM25 index
./hhgoa query "question"          # --mode fast|quality
./hhgoa voice-query <audio.wav>   # STT + RAG
./hhgoa transcribe <audio.wav>    # STT only
./hhgoa eval                      # hit@5 / recall@5 / MRR
./hhgoa retriever-compare         # dense vs sparse vs hybrid + bootstrap
./hhgoa chunk-compare             # six chunking strategies + bootstrap
./hhgoa guardrail-calibrate       # fit + report the grounding gate
./hhgoa bench                     # P50/P70/P100
./hhgoa serve                     # HTTP API
```

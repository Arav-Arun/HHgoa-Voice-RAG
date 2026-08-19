# hhgoa

Modular **RAG framework** for **Hindi** and **Gujarati** (MS MARCO-XI scope).

Each layer is swappable — chunking, embeddings, vector store, retrieval, LLM, prompts, ingest, eval, API. See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
cp .env.example .env          # optional: add LLM_API_KEY for real answers
uv sync
./hhgoa eval-build              # held-out queries.jsonl + split.json
./hhgoa ingest msmarco          # corpus slice from split.json
./hhgoa eval-validate           # labels present in index
./hhgoa eval                    # metrics on held-out queries (not dev slice)
```

Equivalent: `uv run python -m core.cli ingest`

With an LLM key set, drop `--template-llm` for generated answers.

## Layout

| Path | Tweak here for… |
|------|-----------------|
| `core/chunking/` | How documents are split |
| `core/embeddings/` | Embedding models |
| `core/vectorstore/` | Index storage (FAISS, Qdrant, …) |
| `core/retriever/` | Search / reranking |
| `core/llm/` | Answer generation |
| `core/rag/prompts.py` | Prompt templates |
| `core/factory.py` | Wire components from config |
| `ingest/` | Load MS MARCO-XI or custom data |
| `eval/` | Retrieval metrics |
| `bench/` | Latency benchmarks |
| `api/` | HTTP service |
| `data/samples/` | Local ingest examples (MS MARCO-XI loaded from HF by default) |
| `data/index/` | Built index (gitignored) |

## Config (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Local ST models (default) |
| `EMBEDDING_PRESET` | `e5-small` | `e5-small`, `indic-sbert`, `bge-m3` |
| `EMBEDDING_MODEL` | — | Override HF model id (optional) |
| `VECTOR_STORE` | `memory` | In-memory numpy store |
| `CHUNKING_PROVIDER` | `fixed` | `fixed`, `semantic`, `metadata` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | Chunking limits |
| `TOP_K` | 5 | Retrieval depth |
| `LLM_API_KEY` | — | OpenAI-compatible chat API |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model |

Full list in `.env.example`.

## Embedding model (locked in)

**Default: `e5-small`** (`intfloat/multilingual-e5-small`) — local, no API key, query/passage prefixes applied in `core/embeddings/`.

Compared against `indic-sbert` on the **dev** slice (first 500 rows after seeded shuffle, used for embedder selection — not for chunking eval):

| Preset | hi hit@5 | gu hit@5 | overall hit@5 |
|--------|----------|----------|---------------|
| **e5-small** | **0.76** | **0.52** | **0.64** |
| indic-sbert | 0.41 | 0.39 | 0.40 |

Bootstrap significance (e5-small baseline): Hindi p≈0, Gujarati p=0.0004 on hit@5 — both significant. `bge-m3` was not evaluated; we proceeded with e5-small for speed and strong numbers on both languages.

**Non-obvious finding:** the Indic-specialized `indic-sbert` did not beat general-purpose e5-small on this slice — worth reporting honestly, not assuming Indic-native always wins.

To try another preset later: set `EMBEDDING_PRESET` in `.env`, then re-ingest.

**Held-out eval** for chunking/retrieval work: validation rows are **shuffled with seed 42** before slicing, so dev and eval are comparable random bags (not contiguous parquet blocks). Eval uses shuffled positions `[500, 1000)`; dev/embedder work used `[0, 500)`. See [data/eval/README.md](data/eval/README.md).

Compare chunking strategies with bootstrap significance:

```bash
./hhgoa chunk-compare --strategies fixed semantic metadata
```

## Chunking (locked in)

| Strategy | hi hit@5 | gu hit@5 | overall hit@5 |
|----------|----------|----------|---------------|
| **fixed** (default) | 0.74 | 0.54 | 0.64 |
| semantic | 0.75 | 0.55 | 0.65 |
| metadata | 0.75 | 0.55 | 0.65 |

Held-out eval (530 queries, e5-small). Semantic and metadata tie — most MS MARCO passages fit in one chunk, so metadata-aware atomic splitting collapses to the same result as sentence packing. Both edge fixed by ~+0.9pp hit@5, but **bootstrap p=0.27 (not significant)**. Default stays **`fixed`** (simpler; no proven loss). Set `CHUNKING_PROVIDER=semantic` if you prefer sentence boundaries on principle.

## API

```bash
./hhgoa serve
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"भारत की राजधानी क्या है?","language":"hi"}'

# Voice (set STT_PROVIDER=elevenlabs + STT_API_KEY in .env)
curl -X POST http://127.0.0.1:8000/transcribe \
  -F language=hi -F audio=@sample.wav
curl -X POST http://127.0.0.1:8000/voice-query \
  -F language=hi -F audio=@sample.wav
```

CLI: `./hhgoa transcribe sample.wav --language hi` · `./hhgoa voice-query sample.wav`

## Scope

Languages: Hindi (`hi`) + Gujarati (`gu`) only — [docs/scope.md](docs/scope.md).

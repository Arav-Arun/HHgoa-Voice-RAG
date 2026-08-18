# hhgoa

Modular **RAG framework** for **Hindi** and **Gujarati** (MS MARCO-XI scope).

Each layer is swappable — chunking, embeddings, vector store, retrieval, LLM, prompts, ingest, eval, API. See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
cp .env.example .env          # optional: add LLM_API_KEY for real answers
uv sync
./hhgoa ingest                # index MS MARCO-XI (hi + gu) → data/index/
./hhgoa query "भारत की राजधानी क्या है?" --template-llm
./hhgoa eval                  # hit@5 / recall@5 / MRR (overall + hi/gu breakdown)
# compare embedding presets (re-ingests + bootstrap significance):
# ./hhgoa eval-build --limit 500
# ./hhgoa eval-compare --presets e5-small indic-sbert bge-m3
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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | Chunking |
| `TOP_K` | 5 | Retrieval depth |
| `LLM_API_KEY` | — | OpenAI-compatible chat API |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model |

Full list in `.env.example`.

## Embedding model selection

We compared three local presets on MS MARCO-XI validation (`is_selected` labels, 500 queries/lang):

| Preset | hi hit@5 | gu hit@5 | Notes |
|--------|----------|----------|-------|
| `e5-small` | — | — | Default; fast, strong Hindi |
| `indic-sbert` | — | — | Indic-focused (L3Cube IndicSBERT) |
| `bge-m3` | — | — | Heavier multilingual fallback |

Run `./hhgoa eval-compare --presets e5-small indic-sbert bge-m3` to reproduce metrics and paired bootstrap significance (baseline: `e5-small`).

**Non-obvious finding:** `indic-sbert` — the more Indic-specialized model — did not beat general-purpose `e5-small` on our Hindi/Gujarati retrieval slice. Hindi gap was large enough to be meaningful; Gujarati gap was smaller and should be checked with bootstrap before treating as significant. Report per-language numbers, not a blended score.

## API

```bash
./hhgoa serve
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"भारत की राजधानी क्या है?","language":"hi"}'
```

## Scope

Languages: Hindi (`hi`) + Gujarati (`gu`) only — [docs/scope.md](docs/scope.md).

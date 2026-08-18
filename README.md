# hhgoa

Modular **RAG framework** for **Hindi** and **Gujarati** (MS MARCO-XI scope).

Each layer is swappable — chunking, embeddings, vector store, retrieval, LLM, prompts, ingest, eval, API. See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
cp .env.example .env          # optional: add LLM_API_KEY for real answers
uv sync
./hhgoa ingest                # index sample corpus → data/index/
./hhgoa query "भारत की राजधानी क्या है?" --template-llm
./hhgoa eval                  # hit@5 / MRR on sample queries
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
| `data/samples/` | Starter corpus (committed) |
| `data/index/` | Built index (gitignored) |

## Config (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `hash` | `hash` (local) or `openai` |
| `VECTOR_STORE` | `memory` | In-memory numpy store |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | Chunking |
| `TOP_K` | 5 | Retrieval depth |
| `LLM_API_KEY` | — | OpenAI-compatible chat API |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model |

Full list in `.env.example`.

## API

```bash
./hhgoa serve
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"भारत की राजधानी क्या है?","language":"hi"}'
```

## Scope

Languages: Hindi (`hi`) + Gujarati (`gu`) only — [docs/scope.md](docs/scope.md).

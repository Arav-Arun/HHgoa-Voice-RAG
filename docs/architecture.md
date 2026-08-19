# RAG architecture

Modular pipeline scoped to **Hindi** and **Gujarati**. Each layer has a base class — swap one piece without rewriting the rest.

## Flow

```
ingest/loaders  →  core/chunking  →  core/embeddings  →  core/vectorstore
                                                              ↓
query / api  →  core/retriever  →  core/llm  →  answer
                     ↑
              eval/metrics, bench/runner
```

## Where to tweak what

| Goal | Edit |
|------|------|
| Chunk size / overlap | `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) or new class in `core/chunking/` |
| Embedding model | `.env` `EMBEDDING_PROVIDER=hash\|openai` or add provider in `core/embeddings/` + `core/factory.py` |
| Vector DB | Implement `BaseVectorStore` in `core/vectorstore/`, register in `core/factory.py` |
| Retrieval logic | Subclass `BaseRetriever` in `core/retriever/` (e.g. hybrid BM25 + dense) |
| Prompts / language tone | `core/rag/prompts.py` |
| LLM provider | `core/llm/` + `core/factory.py` |
| Data sources (MS MARCO-XI, etc.) | `ingest/loaders.py` |
| Metrics | `eval/metrics.py` |
| HTTP API | `api/app.py`, `api/schemas.py` |
| Config defaults | `core/config.py`, `.env` |

## Components (defaults)

- **Chunker:** swappable via `CHUNKING_PROVIDER` — `fixed` (char windows), `semantic` (sentence boundaries), `metadata` (atomic passages + semantic fallback)
- **Embedder:** `SentenceTransformerEmbedder` — local models via `EMBEDDING_PRESET` (`e5-small` default; applies `query:` / `passage:` prefixes for E5)
- **Store:** `MemoryVectorStore` — numpy cosine search, persists under `data/index/`
- **Retriever:** `DenseRetriever`
- **LLM:** `TemplateLLM` if no key; `OpenAICompatibleLLM` when `LLM_API_KEY` is set

## CLI entry points

```bash
./hhgoa ingest [path]          # build index
./hhgoa query "question"       # run RAG query
./hhgoa eval [file]            # retrieval metrics
./hhgoa bench                  # latency benchmark
./hhgoa serve                  # POST /query
```

## Adding a new embedding provider

1. Subclass `BaseEmbedder` in `core/embeddings/your_provider.py`
2. Register it in `core/factory.build_embedder()`
3. Set `EMBEDDING_PROVIDER=your_provider` in `.env`

Same pattern for chunkers, stores, retrievers, and LLMs.

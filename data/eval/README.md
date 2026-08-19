# Held-out retrieval eval

Query → relevant-passage labels for measuring chunking and retrieval strategies.

## Partition (`split.json`)

MS MARCO-XI **validation** rows are split per language:

| Slice | Rows | Purpose |
|-------|------|---------|
| **corpus** | `[0, 1000)` | Passages indexed for retrieval (includes eval-relevant docs) |
| **dev** | `[0, 500)` | Queries used for embedder comparison — **do not** use for chunking eval |
| **eval** | `[500, 1000)` | **Held-out** queries + `is_selected` passage labels |

Eval queries are disjoint from the dev slice used to pick `e5-small`.

## Files

- `split.json` — machine-readable partition (source of truth for ingest + eval-build)
- `queries.jsonl` — held-out eval set (`query`, `query_id`, `language`, `expected_doc_ids`)

Each `expected_doc_id` is `{lang}_{query_id}_p{idx}` — same IDs as ingest.

## Rebuild

```bash
./hhgoa eval-build              # writes split.json + queries.jsonl from HF
./hhgoa ingest msmarco          # indexes corpus slice from split.json
./hhgoa eval-validate           # confirm labels exist in index
./hhgoa eval                    # hit@5 / recall@5 / MRR on held-out queries
```

## Labels

Relevance comes from MSMARCO-XI `is_selected` flags on translated passages. Examples with no selected passage are skipped.

# Held-out retrieval eval

Query → relevant-passage labels for measuring chunking, retrieval, and guardrail
strategies.

## Partition (`split.json`)

MS MARCO-XI **validation** rows are **shuffled with a fixed seed**
(`shuffle_seed: 42`) before slicing, not raw parquet order. This avoids
positional bias (topic and type cluster in the source file) between dev and eval.

| slice | shuffled positions | purpose |
|-------|-------------------|---------|
| **corpus** | `[0, 5000)` | passages indexed for retrieval, 109,082 chunks |
| **dev** | `[0, 500)` | **all tuning**: embedder, fusion weights, guardrail |
| **eval** | `[500, 1000)` | **held out**, every reported metric |

Eval labels sit inside the corpus slice, so widening the corpus adds distractors
without making labels unreachable. It was widened from 1,000 to 5,000
rows/language deliberately: dense hit@5 fell 0.638 → 0.568, which is the point.
A benchmark you can saturate is not measuring anything.

The same shuffled row indices are applied to Hindi and Gujarati (parallel
MS MARCO-XI parquets).

## Files

| file | contents |
|---|---|
| `split.json` | shuffle seed, validation row count, slice specs (source of truth) |
| `queries.jsonl` | **held-out** eval set, 530 queries (265 hi + 265 gu) |
| `dev_queries.jsonl` | **dev** set, 590 queries, for tuning only |
| `abstain_queries.jsonl` | 150 categorised queries that should be refused |
| `retriever-compare.json` | dense vs sparse vs hybrid + bootstrap |
| `chunk-compare.json` | chunking strategies + bootstrap |
| `fusion-sweep.json` | per-language dense-weight sweep (dev slice) |
| `child-size-sweep.json` | parent_child window size, selected on dev |
| `guardrail-calibration.json` | gate comparison report |
| `guardrail-model.json` | fitted gate coefficients + standardization, loaded at serving |
| `stage1-e5-vs-indic.json` | embedder selection (dev slice) |

`dev_queries.jsonl` and `queries.jsonl` share **zero** query ids. Each
`expected_doc_id` is `{lang}_{query_id}_p{idx}`, the same ids ingest produces.

## Rebuild

```bash
./hhgoa eval-build          # shuffle + write split.json, queries.jsonl, dev_queries.jsonl
./hhgoa ingest msmarco      # index the corpus slice (~24 min, 109k chunks)
./hhgoa eval-validate       # confirm labels exist in the index -> ok: true
./hhgoa eval                # hit@5 / recall@5 / MRR on held-out queries
```

## Labels

Relevance comes from MS MARCO-XI `is_selected` flags on translated passages.
Examples with no selected passage are skipped. Mean 1.05 labels per query, so
hit@5 and recall@5 track each other closely.

## Comparisons

```bash
./hhgoa retriever-compare --retrievers dense sparse hybrid
./hhgoa chunk-compare --strategies fixed semantic metadata recursive parent_child token_window
```

Both re-score the held-out set and run a paired bootstrap (10,000 resamples)
against the baseline. `chunk-compare` re-ingests per strategy into
`data/index_chunkcmp/<strategy>/` so it never clobbers the working index.

## Guardrail calibration

`abstain_queries.jsonl`, 150 queries that should be refused, in five categories:

| category | n | refused by |
|---|---|---|
| `off_topic` | 50 | grounding gate |
| `nonsense` | 20 | grounding gate |
| `in_domain_unanswerable` | 26 | grounding gate |
| `unsafe` | 30 | **input filter, pre-retrieval** |
| `prompt_injection` | 24 | **input filter, pre-retrieval** |

```bash
./hhgoa guardrail-calibrate
```

**Only the three gate-reaching categories are used to calibrate the grounding
gate.** Unsafe and injection queries are refused before retrieval ever happens,
so counting them would credit the confidence gate with refusals it never made.
Both are covered by `tests/test_guardrails_v2.py` instead, currently 54/54
caught, 0 false positives on all 530 real eval queries.

Calibration fits on one stratified half and reports on the other. Current
held-out result:

| gate | answerable recall | false abstain | abstain recall | balanced acc |
|---|---|---|---|---|
| cosine threshold 0.86 | 0.838 | 16.2% | 0.417 | 0.627 |
| cosine, swept to >=95% recall (0.843) | 0.977 | 2.3% | 0.063 | 0.520 |
| **multi-feature logistic** | **0.955** | **4.5%** | **0.333** | **0.644** |

Top-1 cosine alone cannot separate the classes (answerable mean 0.884 vs
abstain 0.865, they genuinely overlap). Lexical overlap between query and top
passage can (0.62 vs 0.31), and carries a fitted coefficient of 4.35 against
0.89 for cosine.

> **Note on an earlier revision:** the metric reported as `balanced_accuracy`
> was computed as `min(recall_answerable, recall_abstain)`, which is worst-class
> recall and always the lower of the two. Both are now reported, under their
> correct names.

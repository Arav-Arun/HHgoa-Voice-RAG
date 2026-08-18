# Language scope

## Decision

We go deep on **Hindi** and **Gujarati** from [MS MARCO-XI](https://huggingface.co/datasets/unicamp-dl/mmarco), not the full multilingual set.

| Language  | ISO 639-1 | Script   | Role                          |
|-----------|-----------|----------|-------------------------------|
| Hindi     | `hi`      | Devanagari | Primary Indic target        |
| Gujarati  | `gu`      | Gujarati   | Second Indic target         |

All ingest, eval, and bench work in this repo should assume these two languages unless a change explicitly revises this document.

## Why these two (not all of MS MARCO-XI)

MS MARCO-XI covers many languages. Spreading effort across all of them dilutes quality on:

- STT / ASR behavior under Indic phonology and code-mixing
- Retrieval and ranking for Devanagari and Gujarati script
- Evaluation design (queries, judgments, failure analysis)

Hindi is the largest Indic language in the set and a strong baseline for Devanagari pipelines. Gujarati adds a second Indic script and different morphology/orthography without exploding the matrix of languages, models, and eval sets.

## Out of scope (for now)

- Other MS MARCO-XI languages (e.g. Arabic, Chinese, Spanish, …) as first-class targets
- Cross-lingual transfer experiments that treat additional languages as equal peers

Exploratory smoke tests on other languages are allowed only as secondary checks and must not drive architecture or milestone criteria.

## Revising scope

If we add or drop a language, update this file first and note the date and rationale in the same PR.

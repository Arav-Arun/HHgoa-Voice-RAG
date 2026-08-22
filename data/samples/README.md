# Samples

The toy `corpus.jsonl` has been replaced by the real [MS MARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) Hindi + Gujarati subset.

Ingest from Hugging Face (default):

```bash
uv run hhgoa ingest                  # validation, 100 examples/lang (~2k passages)
uv run hhgoa ingest msmarco --all    # full validation split
uv run hhgoa ingest msmarco --split train --limit 500
```

You can still ingest local `.jsonl` / `.txt` files or directories via `uv run hhgoa ingest path/to/data`.

# Samples

The toy `corpus.jsonl` has been replaced by the real [MS MARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) Hindi + Gujarati subset.

Ingest from Hugging Face (default):

```bash
./hhgoa ingest                  # validation, 100 examples/lang (~2k passages)
./hhgoa ingest msmarco --all    # full validation split
./hhgoa ingest msmarco --split train --limit 500
```

You can still ingest local `.jsonl` / `.txt` files or directories via `./hhgoa ingest path/to/data`.

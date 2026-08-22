"""Trace the query encoder to ONNX so serving can run it without PyTorch's overhead.

    uv run python scripts/export_onnx.py
    # -> data/onnx/multilingual-e5-small-<hash>/model.onnx

The export is the torch model traced, not a re-implementation, so the vectors it
produces are the ones the index was built with. The script verifies that before
writing anything: if cosine similarity against the torch encoder is not 1.0 on
every probe, it refuses, because a silently different encoder degrades recall
instead of failing.

Run once. The Dockerfile runs it at build time so the first request does not pay
for it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import get_settings
from core.embeddings.onnx import DEFAULT_CACHE, export_path
from core.embeddings.presets import EMBEDDING_PRESETS

# Below this the exported graph is not the same encoder and must not be shipped.
MIN_COSINE = 0.999999

PROBES = [
    "query: लाल मिर्च में कौन सा विटामिन होता है",
    "query: સૌથી વધુ રોકડ પુરસ્કાર ક્રેડિટ કાર્ડ્સ",
    "passage: बोनस के रूप में, अपने आहार में कच्चे हरे बेल मिर्च को शामिल करने से",
]


def export(model_name: str, cache_dir: Path) -> Path:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    directory = export_path(model_name, cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "model.onnx"

    encoder = SentenceTransformer(model_name, device="cpu")
    # The first module is the transformer; the rest is pooling and normalizing,
    # which core/embeddings/onnx.py reproduces in numpy.
    transformer = encoder[0].auto_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample = tokenizer(["export trace"], return_tensors="pt", truncation=True, max_length=512)
    torch.onnx.export(
        transformer,
        (sample["input_ids"], sample["attention_mask"]),
        str(target),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )
    tokenizer.save_pretrained(str(directory))

    # Verify before trusting it. An export that disagrees with torch is worse
    # than no export: retrieval would quietly rank against different vectors.
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    worst = 1.0
    for probe in PROBES:
        reference = encoder.encode([probe], normalize_embeddings=True)[0]
        encoded = tokenizer([probe], return_tensors="np", truncation=True, max_length=512)
        feed = {
            n: (encoded[n] if n in encoded else np.zeros_like(encoded["input_ids"]))
            for n in names
        }
        hidden = session.run(None, feed)[0]
        mask = encoded["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        got = (pooled / np.linalg.norm(pooled, axis=1, keepdims=True))[0]
        worst = min(worst, float(np.dot(reference, got)))

    if worst < MIN_COSINE:
        target.unlink(missing_ok=True)
        raise SystemExit(
            f"export disagrees with the torch encoder (worst cosine {worst:.8f}); "
            "refusing to ship it, retrieval would rank against different vectors"
        )

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"wrote {target} ({size_mb:.0f} MB), worst cosine vs torch {worst:.8f}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="defaults to the configured preset")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    model = args.model
    if not model:
        settings = get_settings()
        preset = EMBEDDING_PRESETS.get(settings.embedding_preset)
        model = settings.embedding_model or (preset.model if preset else "")
    if not model:
        print("no embedding model configured", file=sys.stderr)
        return 1

    export(model, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

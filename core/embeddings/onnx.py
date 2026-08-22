"""ONNX Runtime query encoder: the same vectors, produced faster.

The query encode is the largest single cost in retrieval. Measured on the
deployment VM it is 38 ms of a 68 ms retrieve stage; on Apple Silicon, 5.47 ms
of 9.6 ms end to end. Running the identical graph under ONNX Runtime instead of
PyTorch cuts that by roughly half with no change to the output.

"No change" is meant literally, and is the reason this is safe to adopt without
re-running the evaluation: the exported graph is the torch model traced, so the
vectors agree to float precision. `cosine(torch, onnx) == 1.0` on every query
tested, and a test asserts it. Because the corpus vectors were built with torch,
anything less than exact agreement would be train/serve skew, silently
degrading recall rather than failing loudly.

Falls back to the torch encoder whenever the export or onnxruntime is missing,
so a deployment without them keeps working at the original speed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

from core.embeddings.base import BaseEmbedder
from core.embeddings.sentence_transformers import SentenceTransformerEmbedder

# Exports live beside the Hugging Face cache so a container that bakes models in
# bakes this too, rather than paying the export on the first request.
DEFAULT_CACHE = Path("data/onnx")


def export_path(model_name: str, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """One directory per model, named by a hash so slashes cannot escape it."""
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{model_name.split('/')[-1]}-{digest}"


class OnnxEmbedder(BaseEmbedder):
    """Drop-in for SentenceTransformerEmbedder, backed by ONNX Runtime.

    Holds a torch encoder alongside it. That is not redundancy: it produces the
    export, answers `dimension`, and takes over whenever the ONNX session cannot
    be built.
    """

    _sessions: ClassVar[dict[tuple[str, int], object]] = {}

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        device: str = "cpu",
        threads: int = 0,
        cache_dir: Path | None = None,
    ) -> None:
        self.torch_embedder = SentenceTransformerEmbedder(
            model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
            device=device,
        )
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # 0 lets onnxruntime match the machine, which is right locally. The
        # container pins it, because oversubscribing two cores costs more in
        # coordination than it wins: measured 112 ms against 97 ms.
        self.threads = threads
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self._unavailable = False

    @property
    def dimension(self) -> int:
        return self.torch_embedder.dimension

    def _session(self):
        """ONNX session and tokenizer, or None to mean "use torch"."""
        if self._unavailable:
            return None
        key = (self.model_name, self.threads)
        if key in self._sessions:
            return self._sessions[key]
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            directory = export_path(self.model_name, self.cache_dir)
            model_file = directory / "model.onnx"
            if not model_file.exists():
                self._unavailable = True
                return None

            options = ort.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = self.threads
            session = ort.InferenceSession(
                str(model_file), options, providers=["CPUExecutionProvider"]
            )
            tokenizer = AutoTokenizer.from_pretrained(str(directory))
            self._sessions[key] = (session, tokenizer, [i.name for i in session.get_inputs()])
        except Exception:  # noqa: BLE001 - any failure here means "use torch"
            self._unavailable = True
            return None
        return self._sessions[key]

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        loaded = self._session()
        if loaded is None:
            return self.torch_embedder.embed_texts(texts, is_query=is_query)

        import numpy as np

        session, tokenizer, input_names = loaded
        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed = [f"{prefix}{text}" if prefix else text for text in texts]
        encoded = tokenizer(prefixed, return_tensors="np", padding=True, truncation=True,
                            max_length=512)
        feed = {
            name: (encoded[name] if name in encoded else np.zeros_like(encoded["input_ids"]))
            for name in input_names
        }
        hidden = session.run(None, feed)[0]
        # e5 pools by mean over unmasked tokens, then L2-normalizes. Applied
        # here rather than inherited, because the export is the bare encoder.
        mask = encoded["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], is_query=True)[0]

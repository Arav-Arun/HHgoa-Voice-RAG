"""Held-out train/dev/eval partition for MS MARCO-XI retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ingest.loaders import MSMARCO_XI_DATASET, MSMARCO_XI_SCOPE_LANGUAGES

DEFAULT_SPLIT_PATH = Path("data/eval/split.json")


@dataclass(frozen=True)
class SliceSpec:
    """Half-open row range in the HF split: [offset, offset + limit)."""

    offset: int = 0
    limit: int = 500

    def hf_slice(self, split: str) -> str:
        end = self.offset + self.limit
        return f"{split}[{self.offset}:{end}]"


@dataclass(frozen=True)
class EvalSplitConfig:
    """Partition MSMARCO-XI validation into corpus / dev / held-out eval."""

    dataset: str = MSMARCO_XI_DATASET
    split: str = "validation"
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES
    # Passages indexed for retrieval (must cover eval relevant docs).
    corpus: SliceSpec = SliceSpec(offset=0, limit=1000)
    # Queries used for embedder selection (do not reuse for chunking eval).
    dev: SliceSpec = SliceSpec(offset=0, limit=500)
    # Held-out query → relevant-passage labels for strategy evaluation.
    eval: SliceSpec = SliceSpec(offset=500, limit=500)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "languages": list(self.languages),
            "corpus": asdict(self.corpus),
            "dev": asdict(self.dev),
            "eval": asdict(self.eval),
            "notes": (
                "corpus: passages indexed for retrieval; "
                "dev: queries used for embedder comparison; "
                "eval: held-out queries (disjoint from dev) with is_selected labels"
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalSplitConfig:
        return cls(
            dataset=data.get("dataset", MSMARCO_XI_DATASET),
            split=data.get("split", "validation"),
            languages=tuple(data.get("languages", MSMARCO_XI_SCOPE_LANGUAGES)),
            corpus=SliceSpec(**data.get("corpus", {"offset": 0, "limit": 1000})),
            dev=SliceSpec(**data.get("dev", {"offset": 0, "limit": 500})),
            eval=SliceSpec(**data.get("eval", {"offset": 500, "limit": 500})),
        )


DEFAULT_SPLIT = EvalSplitConfig()


def load_split_config(path: Path = DEFAULT_SPLIT_PATH) -> EvalSplitConfig:
    if path.exists():
        return EvalSplitConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return DEFAULT_SPLIT


def write_split_config(path: Path, config: EvalSplitConfig = DEFAULT_SPLIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

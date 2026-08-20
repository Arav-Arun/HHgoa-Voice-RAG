"""Held-out train/dev/eval partition for MS MARCO-XI retrieval."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ingest.loaders import (
    MSMARCO_XI_DATASET,
    MSMARCO_XI_SCOPE_LANGUAGES,
    msmarco_validation_row_count,
)

DEFAULT_SPLIT_PATH = Path("data/eval/split.json")
DEFAULT_SHUFFLE_SEED = 42


@dataclass(frozen=True)
class SliceSpec:
    """Half-open range into the seeded shuffle permutation: [shuffle_start, shuffle_start + limit)."""

    shuffle_start: int = 0
    limit: int = 500


@dataclass(frozen=True)
class EvalSplitConfig:
    """Partition MSMARCO-XI validation into corpus / dev / held-out eval."""

    dataset: str = MSMARCO_XI_DATASET
    split: str = "validation"
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    validation_rows: int | None = None
    # Passages indexed for retrieval (must cover eval relevant docs).
    corpus: SliceSpec = SliceSpec(shuffle_start=0, limit=1000)
    # Queries used for embedder selection (do not reuse for chunking eval).
    dev: SliceSpec = SliceSpec(shuffle_start=0, limit=500)
    # Held-out query → relevant-passage labels for strategy evaluation.
    eval: SliceSpec = SliceSpec(shuffle_start=500, limit=500)

    def to_dict(self) -> dict:
        payload = {
            "dataset": self.dataset,
            "split": self.split,
            "languages": list(self.languages),
            "shuffle_seed": self.shuffle_seed,
            "corpus": asdict(self.corpus),
            "dev": asdict(self.dev),
            "eval": asdict(self.eval),
            "notes": (
                "Rows are shuffled with shuffle_seed before slicing (not raw parquet order). "
                "corpus: passages indexed for retrieval; "
                "dev: queries used for embedder comparison; "
                "eval: held-out queries (disjoint from dev) with is_selected labels"
            ),
        }
        if self.validation_rows is not None:
            payload["validation_rows"] = self.validation_rows
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> EvalSplitConfig:
        return cls(
            dataset=data.get("dataset", MSMARCO_XI_DATASET),
            split=data.get("split", "validation"),
            languages=tuple(data.get("languages", MSMARCO_XI_SCOPE_LANGUAGES)),
            shuffle_seed=int(data.get("shuffle_seed", DEFAULT_SHUFFLE_SEED)),
            validation_rows=data.get("validation_rows"),
            corpus=_slice_from_dict(data.get("corpus", {}), shuffle_start=0, limit=1000),
            dev=_slice_from_dict(data.get("dev", {}), shuffle_start=0, limit=500),
            eval=_slice_from_dict(data.get("eval", {}), shuffle_start=500, limit=500),
        )


def _slice_from_dict(data: dict, *, shuffle_start: int, limit: int) -> SliceSpec:
    if "shuffle_start" in data:
        return SliceSpec(shuffle_start=int(data["shuffle_start"]), limit=int(data["limit"]))
    # Legacy raw-offset splits (pre-shuffle).
    return SliceSpec(shuffle_start=int(data.get("offset", shuffle_start)), limit=int(data.get("limit", limit)))


DEFAULT_SPLIT = EvalSplitConfig()


def build_shuffle_permutation(n_rows: int, seed: int = DEFAULT_SHUFFLE_SEED) -> list[int]:
    """Deterministic shuffle of validation row indices."""
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    return indices


def resolve_slice_row_indices(permutation: list[int], spec: SliceSpec) -> list[int]:
    return permutation[spec.shuffle_start : spec.shuffle_start + spec.limit]


def get_validation_rows(config: EvalSplitConfig) -> int:
    if config.validation_rows is not None:
        return config.validation_rows
    return msmarco_validation_row_count(config.languages[0], config.split)


def get_shuffle_permutation(config: EvalSplitConfig) -> list[int]:
    return build_shuffle_permutation(get_validation_rows(config), config.shuffle_seed)


def get_corpus_row_indices(config: EvalSplitConfig) -> list[int]:
    return resolve_slice_row_indices(get_shuffle_permutation(config), config.corpus)


def get_dev_row_indices(config: EvalSplitConfig) -> list[int]:
    return resolve_slice_row_indices(get_shuffle_permutation(config), config.dev)


def get_eval_row_indices(config: EvalSplitConfig) -> list[int]:
    return resolve_slice_row_indices(get_shuffle_permutation(config), config.eval)


def load_split_config(path: Path = DEFAULT_SPLIT_PATH) -> EvalSplitConfig:
    if path.exists():
        return EvalSplitConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return DEFAULT_SPLIT


def write_split_config(path: Path, config: EvalSplitConfig = DEFAULT_SPLIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

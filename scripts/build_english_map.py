"""Map each indexed passage to its original English text, for display.

MS MARCO-XI is a translation of English MS MARCO: every row carries
``English_passages`` index-aligned with ``Translated_passages``. So the English
shown next to an answer is the **source text**, not a machine translation of the
Hindi or Gujarati back into English. Nothing is generated, so nothing can drift
from what the corpus actually says.

Written as a side file rather than into the index. The indexer deliberately does
not persist English text (it would roughly double a 105 MB chunks.json and the
RAM behind it), and retrieval never reads this: it is display only, loaded
lazily, and the pipeline works identically without it.

    uv run python scripts/build_english_map.py
    # -> data/index/passages_en.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import get_settings
from eval.split import get_corpus_row_indices, load_split_config
from ingest.loaders import load_msmarco_xi_rows_by_indices, msmarco_passage_id


def build(languages: tuple[str, ...], split: str, row_indices: list[int]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for language in languages:
        rows = load_msmarco_xi_rows_by_indices(language, split, row_indices)
        for row in rows:
            passages = row["passages"]
            english = passages.get("English_passages") or []
            for idx, text in enumerate(english):
                text = str(text).strip()
                if not text:
                    continue
                mapping[msmarco_passage_id(language, row["query_id"], idx)] = text
        print(f"  {language}: {len(mapping)} passages so far")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    config = load_split_config()
    output = args.output or settings.index_dir / "passages_en.json"

    mapping = build(config.languages, config.split, get_corpus_row_indices(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output} ({len(mapping)} passages, {size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

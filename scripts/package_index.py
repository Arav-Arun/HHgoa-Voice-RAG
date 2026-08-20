"""Package the built index into a tarball for INDEX_URL.

The index is gitignored (288 MB) and container disks are ephemeral, so a
deployment fetches it at boot instead of rebuilding it, which would take ~24
minutes on a shared CPU. Upload the tarball anywhere that serves a plain HTTPS
GET (a GitHub release asset, S3, a Hugging Face dataset repo) and point
INDEX_URL at it.

    uv run python scripts/package_index.py
    # -> data/index.tar.gz, extracts to data/index/
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REQUIRED = ("chunks.json", "embeddings.npy", "bm25.npz", "bm25_vocab.json", "bm25_idf.npy")


def package(index_dir: Path, output: Path) -> Path:
    missing = [name for name in REQUIRED if not (index_dir / name).exists()]
    if missing:
        raise SystemExit(f"{index_dir} is missing {', '.join(missing)}. Run './hhgoa ingest msmarco' first.")

    output.parent.mkdir(parents=True, exist_ok=True)
    # arcname is the directory name so the tarball extracts to data/index/,
    # matching what the entrypoint expects.
    with tarfile.open(output, "w:gz") as tar:
        for name in REQUIRED:
            tar.add(index_dir / name, arcname=f"{index_dir.name}/{name}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--output", type=Path, default=Path("data/index.tar.gz"))
    args = parser.parse_args()

    out = package(args.index_dir, args.output)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({size_mb:.0f} MB)")
    print("Upload it, then set INDEX_URL to its download URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""hhgoa — run `uv run hhgoa-ingest` then `uv run hhgoa-query "..."`."""

from __future__ import annotations

import sys


def main() -> None:
    print("hhgoa RAG framework (Hindi + Gujarati)\n")
    print("Commands:")
    print("  ./hhgoa ingest [path]            Build vector index")
    print('  ./hhgoa query "question"         Run RAG query')
    print("  ./hhgoa eval [file]              Retrieval metrics")
    print("  ./hhgoa bench                    Latency benchmark")
    print("  ./hhgoa transcribe <audio>       Speech-to-text (hi/gu)")
    print("  ./hhgoa serve                    Start API server")
    print("\nSee docs/architecture.md for where to tweak each layer.")
    if len(sys.argv) > 1:
        print(f"\n(Unknown args ignored: {' '.join(sys.argv[1:])})")


if __name__ == "__main__":
    main()

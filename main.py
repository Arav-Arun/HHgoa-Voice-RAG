"""hhgoa, voice-enabled RAG for Hindi and Gujarati.

Run `./hhgoa <command>` (or `uv run python -m core.cli <command>`).
"""

from __future__ import annotations

import sys

COMMANDS = [
    ("ingest [path|msmarco]", "Build the vector + BM25 index"),
    ('query "question" [--mode fast|quality]', "Run a RAG query"),
    ("voice-query <audio.wav>", "Transcribe then answer"),
    ("transcribe <audio.wav>", "Speech-to-text only (hi/gu)"),
    ("eval [file]", "Retrieval metrics on the held-out set"),
    ("eval-build / eval-validate", "Build and check eval fixtures"),
    ("retriever-compare", "dense vs sparse vs hybrid, with bootstrap"),
    ("chunk-compare", "Compare chunking strategies"),
    ("guardrail-calibrate", "Fit and report the grounding gate"),
    ("bench", "Latency benchmark (P50/P70/P100)"),
    ("serve", "Start the HTTP API"),
]


def main() -> None:
    print("hhgoa, voice RAG (Hindi + Gujarati)\n")
    print("Commands:")
    width = max(len(name) for name, _ in COMMANDS)
    for name, description in COMMANDS:
        print(f"  ./hhgoa {name:<{width}}  {description}")
    print("\nSee docs/architecture.md for where to tweak each layer.")
    if len(sys.argv) > 1:
        print(f"\n(Unknown args ignored: {' '.join(sys.argv[1:])})")


if __name__ == "__main__":
    main()

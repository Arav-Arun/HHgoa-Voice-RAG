"""Generate voice fixtures from real held-out eval queries.

Run with the gtts extra, which is dev-only and never imported at runtime:

    uv run --with gtts python scripts/make_audio_fixtures.py

Why Google TTS rather than ElevenLabs, whose key this repo already holds: the
free tier exposes only English voices. Synthesizing Gujarati with an English
voice produces an English speaker reading the script phonetically, and Scribe
then transcribes exactly that. The first attempt returned
"Vii Mari Holjaushin Shunche" for "વીમા રિઝોલ્યુશન શું છે?", which is the STT
correctly reporting bad input, not an STT failure. macOS `say` has Hindi
(Lekha) but no Gujarati voice at all.

These are synthetic clips. They exercise the real network path end to end but
are cleaner than human speech, so treat the transcripts as a smoke test rather
than a WER benchmark.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("data/samples/audio")
QUERIES = Path("data/eval/queries.jsonl")

# query_id -> output stem. Both ids exist in both languages in MS MARCO-XI.
FIXTURES = {
    "hi": {1053197: "hi_01", 202006: "hi_02"},
    "gu": {1053197: "gu_01", 202006: "gu_02"},
}


def load_queries() -> dict[tuple[str, int], str]:
    rows = {}
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["language"], row["query_id"])] = row["query"]
    return rows


def synthesize(text: str, language: str, stem: str) -> Path:
    from gtts import gTTS

    mp3 = OUT_DIR / f"{stem}.mp3"
    wav = OUT_DIR / f"{stem}.wav"
    gTTS(text, lang=language).save(str(mp3))
    # 16 kHz mono matches what the STT provider expects and keeps clips small.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)],
        check=True,
    )
    mp3.unlink()
    return wav


def main() -> None:
    if not QUERIES.exists():
        sys.exit(f"{QUERIES} not found. Run ./hhgoa eval-build first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries()

    for language, wanted in FIXTURES.items():
        for query_id, stem in wanted.items():
            text = queries.get((language, query_id))
            if text is None:
                print(f"skip {stem}: no {language} query {query_id} in the eval set")
                continue
            wav = synthesize(text, language, stem)
            print(f"{wav}  <-  {text}")


if __name__ == "__main__":
    main()

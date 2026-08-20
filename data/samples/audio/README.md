# Voice fixtures

Synthesized clips of real held-out queries from `data/eval/queries.jsonl`, used
to exercise the speech-to-text path and the voice latency track.

| file | language | spoken text | query id |
|---|---|---|---|
| `hi_01.wav` | hi | बीमा समाधान क्या है | 1053197 |
| `hi_02.wav` | hi | उच्चतम नकद पुरस्कार क्रेडिट कार्ड | 202006 |
| `gu_01.wav` | gu | વીમા રિઝોલ્યુશન શું છે? | 1053197 |
| `gu_02.wav` | gu | સૌથી વધુ રોકડ પુરસ્કાર ક્રેડિટ કાર્ડ્સ | 202006 |

Regenerate:

```bash
uv run --with gtts python scripts/make_audio_fixtures.py
```

## Measured transcription

| file | transcript | note |
|---|---|---|
| `hi_01.wav` | बीमा समाधान क्या है? | exact |
| `gu_01.wav` | વીમા રિઝોલ્યુશન શું છે? | exact |
| `gu_02.wav` | સૌથી વધુ રોકડ પુરસ્કાર credit cards | loanword returned in Latin script |

`gu_02` is the interesting one. Scribe returns the loanword "ક્રેડિટ કાર્ડ્સ"
as Latin "credit cards", which is realistic code-mixed ASR output. Retrieval
still lands on the correct passage family (`gu_202006_p4`), because the hybrid
retriever's BM25 half matches the Latin token while the dense half carries the
Gujarati context.

## Why not ElevenLabs TTS

The key in this repo can synthesize speech, but the free tier exposes only
English voices. Rendering Gujarati through an English voice produces an English
speaker reading the script phonetically, and Scribe then transcribes exactly
that: the first attempt returned "Vii Mari Holjaushin Shunche" for
`વીમા રિઝોલ્યુશન શું છે?`. That was the STT faithfully reporting bad input, not
an STT failure. macOS `say` ships Hindi (Lekha) but no Gujarati voice.

These clips are synthetic and therefore cleaner than human speech. They verify
the pipeline end to end; they are not a WER benchmark.

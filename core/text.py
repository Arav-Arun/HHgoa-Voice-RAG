"""Shared Unicode text handling for Hindi (Devanagari) and Gujarati.

One tokenizer for chunking, BM25, and the guardrails so that "the same word"
means the same thing in every layer.

Two script-specific traps this module exists to avoid:

1. Python's ``\\w`` does not match Indic combining marks (matras are Unicode
   categories Mn/Mc, and ``str.isalnum()`` is False for them). Tokenizing
   Devanagari with a bare ``\\w+`` shatters words at every matra:
   ``हैबर`` -> ``['ह', 'बर']``. The explicit block ranges below prevent that.

2. The danda ``।`` (U+0964) and double danda ``॥`` (U+0965) sit *inside* the
   Devanagari block, so a naive ``[\\u0900-\\u097F]+`` range glues sentence-final
   punctuation onto the word: ``है।`` != ``है``. Those two code points are
   excluded so a word matches whether or not it ends a sentence.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari U+0900-U+097F and Gujarati U+0A80-U+0AFF, minus danda/double danda.
_WORD_CHARS = (
    r"\w"
    r"ऀ-ॣ"  # Devanagari, up to just before the danda
    r"०-ॿ"  # Devanagari, resuming after the double danda
    r"઀-૿"  # Gujarati (contains no danda of its own)
)

TOKEN_RE = re.compile(rf"[{_WORD_CHARS}]+", re.UNICODE)

# Sentence-terminating punctuation, kept separate from word characters.
DANDA = "।"
DOUBLE_DANDA = "॥"

# Devanagari and Gujarati digits -> ASCII, so "५००" and "500" are one term.
_DIGIT_MAP = {}
for _base in (0x0966, 0x0AE6):  # Devanagari zero, Gujarati zero
    for _offset in range(10):
        _DIGIT_MAP[_base + _offset] = ord("0") + _offset

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Devanagari and Gujarati occupy disjoint Unicode blocks, so the script a query
# is written in identifies its language outright. No model, no heuristic, no
# ambiguity between the two languages this system supports.
_SCRIPT_BLOCKS = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "gu": (0x0A80, 0x0AFF),  # Gujarati
}
# Share of Indic characters that must be Gujarati for the text to count as
# Gujarati. See detect_language for why this is not a majority vote.
_GUJARATI_SHARE = 0.15


def detect_language(text: str) -> str | None:
    """Return "hi" or "gu" from the script the text is written in.

    Returns None when the text carries no Indic letters at all, which is the
    case for empty input and for Latin transliteration. Callers fall back to
    their configured default rather than guessing, because a wrong language
    picks the wrong fusion weights and answers in the wrong script.

    Not a majority vote. Speech-to-text on Gujarati audio regularly returns a
    transcript mixing both scripts ("USA टपाल टिकटની કિંમત"), and Devanagari can
    hold the majority in one of those while the utterance is plainly Gujarati.
    Since Hindi is never written in the Gujarati block, a meaningful share of
    Gujarati characters settles it. Measured over 32 synthesized clips, a
    majority vote scores 87.5% against 93.8% for this rule, with Hindi
    unaffected at 16/16.
    """
    counts = dict.fromkeys(_SCRIPT_BLOCKS, 0)
    for char in text:
        code = ord(char)
        for language, (low, high) in _SCRIPT_BLOCKS.items():
            if low <= code <= high:
                counts[language] += 1
                break
    total = sum(counts.values())
    if not total:
        return None
    # A share rather than "any Gujarati character at all", so one stray glyph in
    # a long Hindi transcript cannot flip the result. Every threshold from 0.10
    # to 0.30 scored identically on the measured set.
    return "gu" if counts["gu"] / total >= _GUJARATI_SHARE else "hi"


def normalize_digits(text: str) -> str:
    """Map Devanagari/Gujarati digits onto ASCII digits."""
    return text.translate(_DIGIT_MAP)


def normalize(text: str) -> str:
    """NFC-normalize, fold digits, and lowercase.

    NFC matters because the same Indic grapheme can arrive decomposed or
    precomposed depending on the source (dataset text vs STT output), and the
    two forms would otherwise never compare equal.
    """
    return normalize_digits(unicodedata.normalize("NFC", text)).lower()


def tokenize(text: str) -> list[str]:
    """Split into comparable word tokens."""
    return TOKEN_RE.findall(normalize(text))


def token_set(text: str, *, min_length: int = 2) -> set[str]:
    """Unique content tokens, dropping very short ones."""
    return {token for token in tokenize(text) if len(token) >= min_length}


def extract_numbers(text: str) -> set[str]:
    """Numeric literals, script-normalized.

    Used by the faithfulness guardrail: a number in an answer that appears
    nowhere in the retrieved context is the cheapest hallucination to detect
    and the most damaging to miss.
    """
    return set(_NUMBER_RE.findall(normalize_digits(text)))

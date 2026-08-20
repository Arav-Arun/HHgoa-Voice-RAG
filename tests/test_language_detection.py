"""Script detection, including the mixed-script transcripts STT actually returns.

Speech-to-text on Gujarati audio often emits both scripts in one string. A
majority vote gets those wrong; these pin the rule that replaced it.
"""

from __future__ import annotations

from core.text import detect_language


def test_pure_scripts():
    assert detect_language("बीमा समाधान क्या है") == "hi"
    assert detect_language("વીમા સમાધાન શું છે") == "gu"


def test_latin_and_empty_yield_no_opinion():
    # Callers fall back to their configured default rather than guessing.
    assert detect_language("bima samadhan kya hai") is None
    assert detect_language("") is None
    assert detect_language("500?") is None


def test_latin_loanwords_do_not_change_the_verdict():
    assert detect_language("2026 में World Cup किधर हो रहा है?") == "hi"
    assert detect_language("સૌથી વધુ રોકડ પુરસ્કાર credit cards") == "gu"


def test_mixed_script_transcripts_resolve_to_gujarati():
    """Real Scribe output on Gujarati audio, where Devanagari can dominate."""
    # dev=8 guj=7: a majority vote would call this Hindi.
    assert detect_language("USA टपाल टिकटની કિંમત") == "gu"
    # dev=3 guj=2: likewise.
    assert detect_language("Roblox cards काय છે?") == "gu"
    assert detect_language("कीबोर्ड અને કીપેડ વચ્ચેનો તફાવત") == "gu"


def test_a_stray_gujarati_glyph_does_not_flip_a_hindi_sentence():
    """The rule is a share, not "any Gujarati character"."""
    hindi = "यह एक लंबा हिन्दी वाक्य है जिसमें बहुत सारे देवनागरी अक्षर हैं और अर्थ स्पष्ट है"
    assert detect_language(hindi + "ક") == "hi"

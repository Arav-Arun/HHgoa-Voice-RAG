"""User-facing guardrail messages in Hindi and Gujarati."""

from __future__ import annotations

INPUT_REJECTED: dict[str, str] = {
    "hi": "कृपया एक वैध प्रश्न पूछें।",
    "gu": "કૃપા કરીને માન્ય પ્રશ્ન પૂછો.",
}

ABSTAIN_NO_CONTEXT: dict[str, str] = {
    "hi": "दिए गए संदर्भ में इस प्रश्न का उत्तर नहीं मिला। कृपया प्रश्न को दूसरे शब्दों में पूछें।",
    "gu": "આપેલ સંદર્ભમાં આ પ્રશ્નનો જવાબ મળ્યો નથી. કૃપા કરીને પ્રશ્ન બીજા શબ્દોમાં પૂછો.",
}

ABSTAIN_LOW_CONFIDENCE: dict[str, str] = {
    "hi": "मुझे इस प्रश्न के लिए पर्याप्त विश्वसनीय संदर्भ नहीं मिला।",
    "gu": "મને આ પ્રશ્ન માટે પૂરતો વિશ્વસનીય સંદર્ભ મળ્યો નથી.",
}

ABSTAIN_HALLUCINATION: dict[str, str] = {
    "hi": "मुझे दिए गए संदर्भ के आधार पर इस प्रश्न का विश्वसनीय उत्तर नहीं दे सकता।",
    "gu": "આપેલ સંદર્ભના આધારે હું આ પ્રશ્નનો વિશ્વસનીય જવાબ આપી શકતો નથી.",
}


def message_for(language: str, table: dict[str, str]) -> str:
    return table.get(language, table["hi"])

"""Detecting refusals in model output.

v3+ prompts use an explicit `[REFUSE]` marker. Earlier prompts have no protocol, so
common phrasings are matched as a fallback; this keeps refusal metrics comparable across
all prompt versions.
"""

from __future__ import annotations

import re

REFUSE_MARKER = "[REFUSE]"
DEFAULT_REFUSAL = (
    "I can't answer this from the clinical guidelines I have access to. "
    "The question may be outside the guideline corpus or need a clinician's judgement."
)

_REFUSAL_PATTERNS = re.compile(
    r"|".join(
        [
            r"\b(?:i\s+)?(?:can(?:'|no)?t|cannot|am unable to|'m unable to)\s+"
            r"(?:find|answer|provide|locate|determine)\b",
            r"\b(?:do(?:es)?\s+not|don't|doesn't)\s+(?:contain|cover|include|address|mention)\b",
            r"\bno (?:relevant |such )?(?:information|evidence|excerpts?|guidance)\b",
            r"\bnot (?:covered|addressed|mentioned|available) in\b",
            r"\boutside (?:the |my )?(?:scope|corpus)\b",
            r"\binsufficient (?:information|evidence)\b",
        ]
    ),
    re.IGNORECASE,
)


def is_refusal(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    head = text.strip()
    if head.startswith(REFUSE_MARKER):
        return True
    # Fallback for prompts without a marker: only short replies count, so an answer that
    # merely notes a gap ("...the guideline does not cover children") is not a refusal.
    return len(head.split()) <= 45 and bool(_REFUSAL_PATTERNS.search(head))


def strip_marker(text: str) -> str:
    text = text.strip()
    if text.startswith(REFUSE_MARKER):
        text = text[len(REFUSE_MARKER) :].strip(" :-")
    return text or DEFAULT_REFUSAL

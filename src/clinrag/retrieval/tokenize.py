"""Tokenizer for BM25, tuned for clinical text.

Keeps numbers, units and compound terms ("covid-19", "hba1c", "mg/kg", "ace-i") intact and
also indexes their parts, so "type-2 diabetes" matches "type 2 diabetes".
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/+][a-z0-9]+)*")

# fmt: off
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had has have
    having he her here hers him his how i if in into is it its itself just me more most my no
    nor not of off on once only or other our ours out over own same she should so some such
    than that the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you
    your yours about above after again against all am any because before below between both
    during each few further
    """.split()
)
# fmt: on
# Negations and comparatives matter clinically ("not recommended", "less than").
STOPWORDS = STOPWORDS - {"not", "no", "nor", "more", "most", "against", "before", "after"}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if any(sep in tok for sep in "-/+"):
            tokens.extend(p for p in re.split(r"[\-/+]", tok) if p and p not in STOPWORDS)
    return tokens

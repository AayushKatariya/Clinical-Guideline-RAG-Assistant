"""Ranking metrics.

`ranked` may contain duplicates (several chunks from the same guideline when evaluating at
document level). A relevant id counts once, at its first occurrence, so recall@5 means
"fraction of relevant guidelines present among the 5 chunks the model would see".
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(r in relevant for r in ranked[:k]))


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(r in relevant for r in top) / k if k else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant and r not in seen:
            dcg += 1.0 / math.log2(i + 1)
            seen.add(r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0

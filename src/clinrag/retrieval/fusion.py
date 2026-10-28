"""Rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    RRF only uses ranks, so BM25 scores (unbounded) and cosine similarities (bounded)
    can be fused without any score normalisation.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    fused: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] += w / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    # Ties broken by first appearance, which keeps results deterministic.
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))

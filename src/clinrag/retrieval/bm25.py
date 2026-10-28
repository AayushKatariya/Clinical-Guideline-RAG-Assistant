"""Sparse retrieval with BM25 (Okapi)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from clinrag.retrieval.tokenize import tokenize

BM25_FILENAME = "bm25.pkl"


class BM25Index:
    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]):
        self._bm25 = bm25
        self.chunk_ids = chunk_ids

    @classmethod
    def build(
        cls, texts: list[str], chunk_ids: list[str], k1: float = 1.2, b: float = 0.75
    ) -> BM25Index:
        if len(texts) != len(chunk_ids):
            raise ValueError("texts and chunk_ids must have the same length")
        corpus = [tokenize(t) or ["<empty>"] for t in texts]
        return cls(BM25Okapi(corpus, k1=k1, b=b), list(chunk_ids))

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        if not q:
            return np.zeros(len(self.chunk_ids), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(q), dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (row index, score) pairs for the top-k chunks with a positive score."""
        scores = self.scores(query)
        k = min(k, len(scores))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / BM25_FILENAME).open("wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, index_dir: Path) -> BM25Index:
        # Only load index files you built yourself: pickle executes code on load.
        with (index_dir / BM25_FILENAME).open("rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["chunk_ids"])

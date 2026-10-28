"""Cross-encoder reranking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance of each passage to the query, in [0, 1]."""
        ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (bge-reranker, ms-marco-MiniLM, MedCPT-Cross-Encoder...).

    Scores are mapped to [0, 1] so a single refusal threshold works across models: some
    checkpoints already apply a sigmoid, others return raw logits.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        raw = np.asarray(
            self._model.predict(
                [(query, p) for p in passages],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if raw.ndim > 1:  # 2-class heads: take the "relevant" logit
            raw = raw[:, -1]
        if raw.min() < 0.0 or raw.max() > 1.0:
            raw = _sigmoid(raw)
        return raw

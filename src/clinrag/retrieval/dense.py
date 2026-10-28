"""Dense retrieval: bi-encoder embeddings in a FAISS inner-product index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

FAISS_FILENAME = "dense.faiss"
META_FILENAME = "dense_meta.json"


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Any sentence-transformers bi-encoder (BGE, E5, MedCPT, ...).

    BGE / E5 style models expect an instruction prefix on queries only; pass it as
    `query_prefix` (empty string for models that don't use one).
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        device: str | None = None,
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec[0].astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class DenseIndex:
    """Exact cosine search (IndexFlatIP over L2-normalised vectors).

    Exact search over ~30k chunks takes a few ms on CPU; swap in `faiss.IndexHNSWFlat`
    if the corpus grows by orders of magnitude.
    """

    def __init__(self, index: faiss.Index, chunk_ids: list[str], model_name: str):
        self._index = index
        self.chunk_ids = chunk_ids
        self.model_name = model_name

    @classmethod
    def build(cls, embedder: Embedder, texts: list[str], chunk_ids: list[str]) -> DenseIndex:
        vecs = _normalize(embedder.embed_documents(texts))
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(index, list(chunk_ids), embedder.name)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self._index.ntotal)
        if k == 0:
            return []
        q = _normalize(query_vec.reshape(1, -1))
        scores, idx = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0], strict=True) if i >= 0]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / FAISS_FILENAME))
        (index_dir / META_FILENAME).write_text(
            json.dumps({"model_name": self.model_name, "chunk_ids": self.chunk_ids}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> DenseIndex:
        index = faiss.read_index(str(index_dir / FAISS_FILENAME))
        meta = json.loads((index_dir / META_FILENAME).read_text(encoding="utf-8"))
        return cls(index, meta["chunk_ids"], meta["model_name"])

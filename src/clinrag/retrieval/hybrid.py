"""Hybrid retriever: BM25 + dense -> RRF -> cross-encoder rerank.

Each stage is exposed as a `mode` so the retrieval ablation (bm25 / dense / hybrid /
hybrid_rerank) runs the exact same code path as production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langsmith import traceable

from clinrag.config import Settings
from clinrag.ingestion.pipeline import load_chunks
from clinrag.retrieval.bm25 import BM25Index
from clinrag.retrieval.dense import DenseIndex, Embedder, SentenceTransformerEmbedder
from clinrag.retrieval.fusion import reciprocal_rank_fusion
from clinrag.retrieval.reranker import CrossEncoderReranker, Reranker
from clinrag.schemas import Chunk, RetrievalMode, RetrievedChunk

log = logging.getLogger(__name__)

MODES: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")
MANIFEST_FILENAME = "manifest.json"


@dataclass
class RetrievalConfig:
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 40
    final_top_k: int = 5

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            bm25_top_k=s.bm25_top_k,
            dense_top_k=s.dense_top_k,
            rrf_k=s.rrf_k,
            rerank_candidates=s.rerank_candidates,
            final_top_k=s.final_top_k,
        )


def _passes(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    org = filters.get("organization")
    if org and (chunk.organization or "").lower() != str(org).lower():
        return False
    year_min = filters.get("year_min")
    if year_min and (chunk.year is None or chunk.year < int(year_min)):
        return False
    doc_ids = filters.get("doc_ids")
    return not (doc_ids and chunk.doc_id not in set(doc_ids))


def _trace_inputs(inputs: dict) -> dict:
    return {k: inputs.get(k) for k in ("query", "k", "mode", "filters")}


def _trace_outputs(outputs: Any) -> dict:
    docs = outputs if isinstance(outputs, list) else []
    return {
        "documents": [
            {
                "type": "Document",
                "page_content": r.chunk.text,
                "metadata": {"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id, **r.scores},
            }
            for r in docs
            if isinstance(r, RetrievedChunk)
        ]
    }


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        dense: DenseIndex,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ):
        if bm25.chunk_ids != [c.chunk_id for c in chunks]:
            raise ValueError("BM25 index is out of sync with chunks.jsonl; rebuild the indexes.")
        if dense.chunk_ids != bm25.chunk_ids:
            raise ValueError("Dense index is out of sync with chunks.jsonl; rebuild the indexes.")
        if dense.model_name != embedder.name:
            raise ValueError(
                f"Dense index was built with {dense.model_name!r} but the query embedder is "
                f"{embedder.name!r}. Rebuild with `clinrag index` or set EMBEDDING_MODEL."
            )
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    # ------------------------------------------------------------------ build / load

    @staticmethod
    def build_indexes(chunks: list[Chunk], embedder: Embedder, index_dir: Path) -> None:
        texts = [c.index_text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        log.info("Building BM25 index over %d chunks", len(chunks))
        BM25Index.build(texts, ids).save(index_dir)
        log.info("Embedding %d chunks with %s", len(chunks), embedder.name)
        DenseIndex.build(embedder, texts, ids).save(index_dir)
        (index_dir / MANIFEST_FILENAME).write_text(
            json.dumps({"n_chunks": len(chunks), "embedding_model": embedder.name}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        load_reranker: bool = True,
    ) -> HybridRetriever:
        chunks = load_chunks(settings.chunks_path)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                query_prefix=settings.embedding_query_prefix,
                device=settings.device,
                batch_size=settings.embedding_batch_size,
            )
        if reranker is None and load_reranker:
            reranker = CrossEncoderReranker(
                settings.reranker_model,
                device=settings.device,
                batch_size=settings.reranker_batch_size,
            )
        return cls(
            chunks=chunks,
            bm25=BM25Index.load(settings.index_dir),
            dense=DenseIndex.load(settings.index_dir),
            embedder=embedder,
            reranker=reranker,
            config=RetrievalConfig.from_settings(settings),
        )

    # ------------------------------------------------------------------ retrieval

    def _first_stage(
        self, query: str, depth_mult: int
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        sparse = self.bm25.search(query, self.config.bm25_top_k * depth_mult)
        dense = self.dense.search(
            self.embedder.embed_query(query), self.config.dense_top_k * depth_mult
        )
        return sparse, dense

    @traceable(
        run_type="retriever",
        name="hybrid_retrieve",
        process_inputs=_trace_inputs,
        process_outputs=_trace_outputs,
    )
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: RetrievalMode = "hybrid_rerank",
        filters: dict[str, Any] | None = None,
        max_per_doc: int | None = None,
    ) -> list[RetrievedChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        k = k or self.config.final_top_k
        depth_mult = 4 if filters else 1
        sparse, dense = self._first_stage(query, depth_mult)
        bm25_scores = dict(sparse)
        dense_scores = dict(dense)

        if mode == "bm25":
            ranked = [(i, s, {"bm25": s}) for i, s in sparse]
        elif mode == "dense":
            ranked = [(i, s, {"dense": s}) for i, s in dense]
        else:
            fused = reciprocal_rank_fusion(
                [[i for i, _ in sparse], [i for i, _ in dense]], k=self.config.rrf_k
            )
            candidates = [
                (int(i), float(s)) for i, s in fused if _passes(self.chunks[int(i)], filters)
            ]
            if mode == "hybrid":
                ranked = [(i, s, {"rrf": s}) for i, s in candidates]
            else:
                pool = candidates[: self.config.rerank_candidates]
                rr = self.reranker.score(query, [self.chunks[i].index_text for i, _ in pool])
                rescored = [
                    (i, float(r), {"rrf": s, "rerank": float(r)})
                    for (i, s), r in zip(pool, rr, strict=True)
                ]
                ranked = sorted(rescored, key=lambda t: -t[1])

        results: list[RetrievedChunk] = []
        per_doc: dict[str, int] = {}
        for i, score, stage_scores in ranked:
            chunk = self.chunks[i]
            if not _passes(chunk, filters):
                continue
            if max_per_doc is not None:
                if per_doc.get(chunk.doc_id, 0) >= max_per_doc:
                    continue
                per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            stage_scores = dict(stage_scores)
            if i in bm25_scores:
                stage_scores.setdefault("bm25", bm25_scores[i])
            if i in dense_scores:
                stage_scores.setdefault("dense", dense_scores[i])
            results.append(
                RetrievedChunk(chunk=chunk, score=score, rank=len(results) + 1, scores=stage_scores)
            )
            if len(results) >= k:
                break
        return results

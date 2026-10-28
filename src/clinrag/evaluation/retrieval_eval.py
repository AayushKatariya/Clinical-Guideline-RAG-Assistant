"""Retrieval ablation: the same eval questions through bm25 / dense / hybrid / hybrid_rerank."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from clinrag.evaluation.dataset import EvalExample
from clinrag.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from clinrag.retrieval.hybrid import HybridRetriever
from clinrag.schemas import RetrievalMode

log = logging.getLogger(__name__)

Level = Literal["doc", "chunk"]


@dataclass
class RetrievalReport:
    level: Level
    ks: tuple[int, ...]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # mode -> name -> value
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def table(self) -> list[dict[str, Any]]:
        return [{"mode": mode, **vals} for mode, vals in self.metrics.items()]


def _labels(ex: EvalExample, level: Level) -> set[str]:
    return set(ex.relevant_chunk_ids if level == "chunk" else ex.relevant_doc_ids)


def evaluate_retrieval(
    retriever: HybridRetriever,
    examples: list[EvalExample],
    modes: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank"),
    ks: tuple[int, ...] = (1, 3, 5, 10),
    level: Level = "doc",
) -> RetrievalReport:
    # In-corpus questions only: adversarial rows may carry labels too, but they test the
    # assistant's behaviour, not retrieval.
    labelled = [e for e in examples if e.category == "in_corpus" and _labels(e, level)]
    if not labelled:
        raise ValueError(
            "No in_corpus examples have "
            f"{'relevant_chunk_ids' if level == 'chunk' else 'relevant_doc_ids'}."
        )
    depth = max(ks)
    report = RetrievalReport(level=level, ks=ks)
    for mode in modes:
        rows = []
        for ex in labelled:
            t0 = time.perf_counter()
            hits = retriever.retrieve(ex.question, k=depth, mode=mode)
            latency = time.perf_counter() - t0
            ranked = [h.chunk.chunk_id if level == "chunk" else h.chunk.doc_id for h in hits]
            rel = _labels(ex, level)
            row: dict[str, Any] = {
                "id": ex.id,
                "mode": mode,
                "question": ex.question,
                "relevant": sorted(rel),
                "retrieved": ranked,
                "mrr": reciprocal_rank(ranked, rel),
                "latency_ms": latency * 1000,
            }
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
                row[f"hit@{k}"] = hit_at_k(ranked, rel, k)
                row[f"precision@{k}"] = precision_at_k(ranked, rel, k)
                row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
            rows.append(row)
        metric_names = [m for m in rows[0] if "@" in m] + ["mrr"]
        agg = {m: statistics.fmean(r[m] for r in rows) for m in metric_names}
        lat = sorted(r["latency_ms"] for r in rows)
        agg["latency_p50_ms"] = lat[len(lat) // 2]
        agg["latency_p95_ms"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        agg["n_questions"] = float(len(rows))
        report.metrics[mode] = agg
        report.per_question.extend(rows)
        log.info("%s: recall@5=%.3f mrr=%.3f", mode, agg.get("recall@5", float("nan")), agg["mrr"])
    return report

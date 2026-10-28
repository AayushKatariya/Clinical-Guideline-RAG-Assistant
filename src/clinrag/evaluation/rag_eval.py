"""End-to-end evaluation of the assistant for one prompt version.

1. run every eval question through the assistant (each run traced to LangSmith with a
   known run id);
2. score behaviour (answer vs refuse) against the label;
3. score answered responses with RAGAS;
4. log params / metrics / per-question table to MLflow and push scores back to LangSmith
   as feedback on the traced runs.
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from clinrag.agent.assistant import ClinicalAssistant
from clinrag.evaluation.dataset import EvalExample
from clinrag.evaluation.ragas_eval import ScoringItem, score_items

log = logging.getLogger(__name__)


@dataclass
class RagEvalResult:
    prompt_version: str
    metrics: dict[str, float] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None
    mlflow_run_id: str | None = None


# ---------------------------------------------------------------------------
# 1. Generation
# ---------------------------------------------------------------------------


def _run_one(assistant: ClinicalAssistant, ex: EvalExample, tags: list[str]) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "id": ex.id,
        "question": ex.question,
        "category": ex.category,
        "subtype": ex.subtype,
        "expected_behavior": ex.expected_behavior,
        "reference_answer": ex.reference_answer,
        "relevant_doc_ids": ex.relevant_doc_ids,
        "run_id": run_id,
    }
    try:
        resp = assistant.answer(
            ex.question,
            langsmith_extra={
                "run_id": run_id,
                "tags": tags,
                "metadata": {
                    "eval_id": ex.id,
                    "category": ex.category,
                    "prompt_version": assistant.prompt.version,
                },
            },
        )
    except Exception as exc:
        log.exception("assistant failed on %s", ex.id)
        record.update(error=f"{type(exc).__name__}: {exc}", refused=None)
        return record
    record.update(
        answer=resp.answer,
        refused=resp.refused,
        refusal_reason=resp.refusal_reason,
        contexts=resp.contexts,
        citations=[c.source_id for c in resp.citations],
        retrieved_doc_ids=resp.retrieved_doc_ids,
        max_rerank_score=resp.max_rerank_score,
        sql_queries=resp.sql_queries,
        tools_used=[t.name for t in resp.tool_calls],
        latency_s=resp.latency_s,
    )
    return record


def generate(
    assistant: ClinicalAssistant, examples: list[EvalExample], workers: int = 4
) -> list[dict[str, Any]]:
    tags = ["eval", f"prompt:{assistant.prompt.version}"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda ex: _run_one(assistant, ex, tags), examples))


# ---------------------------------------------------------------------------
# 2. Behaviour metrics
# ---------------------------------------------------------------------------


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def behaviour_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    ok = [r for r in records if r.get("refused") is not None]
    should_refuse = [r for r in ok if r["expected_behavior"] == "refuse"]
    should_answer = [r for r in ok if r["expected_behavior"] == "answer"]
    oos = [r for r in ok if r["category"] == "out_of_scope"]
    adversarial = [r for r in ok if r["category"] == "adversarial"]
    answered = [r for r in ok if not r["refused"]]
    for r in ok:
        r["behavior_correct"] = r["refused"] == (r["expected_behavior"] == "refuse")
    lat = sorted(r["latency_s"] for r in ok)
    return {
        "n_examples": float(len(records)),
        "n_errors": float(len(records) - len(ok)),
        # Headline: share of out-of-corpus questions the assistant declines.
        "refusal_rate_out_of_corpus": _rate([r["refused"] for r in oos]),
        "refusal_recall": _rate([r["refused"] for r in should_refuse]),
        "false_refusal_rate": _rate([r["refused"] for r in should_answer]),
        "adversarial_pass_rate": _rate([r["behavior_correct"] for r in adversarial]),
        "behavior_accuracy": _rate([r["behavior_correct"] for r in ok]),
        "gate_refusal_share": _rate(
            [r.get("refusal_reason") == "no_evidence" for r in ok if r["refused"]]
        ),
        "uncited_answer_rate": _rate([not r.get("citations") for r in answered]),
        "tool_use_rate": _rate([bool(r.get("tools_used")) for r in ok]),
        "sql_tool_rate": _rate(
            ["query_clinical_database" in (r.get("tools_used") or []) for r in ok]
        ),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }


# ---------------------------------------------------------------------------
# 3. RAGAS
# ---------------------------------------------------------------------------


def ragas_items(records: list[dict[str, Any]]) -> list[ScoringItem]:
    """Every *answered* response is judged - including ones that should have been refused,
    so answering an out-of-corpus question from background knowledge costs faithfulness."""
    return [
        ScoringItem(
            key=r["id"],
            question=r["question"],
            answer=r["answer"],
            contexts=r.get("contexts") or [],
            reference=r.get("reference_answer") if r["expected_behavior"] == "answer" else None,
        )
        for r in records
        if r.get("refused") is False and r.get("answer")
    ]


def aggregate_scores(records: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        vals = [r["scores"][name] for r in records if r.get("scores", {}).get(name) is not None]
        if vals:
            out[name] = statistics.fmean(vals)
            out[f"{name}_n"] = float(len(vals))
    # Faithfulness restricted to questions that *should* be answered (the in-scope view).
    in_scope = [
        r["scores"]["faithfulness"]
        for r in records
        if r["expected_behavior"] == "answer"
        and r.get("scores", {}).get("faithfulness") is not None
    ]
    if in_scope:
        out["faithfulness_in_scope"] = statistics.fmean(in_scope)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_rag_eval(
    assistant: ClinicalAssistant,
    examples: list[EvalExample],
    *,
    ragas_metrics: dict[str, Any] | None,
    output_root: Path,
    workers: int = 4,
    ragas_concurrency: int = 8,
) -> RagEvalResult:
    version = assistant.prompt.version
    log.info("Generating answers for %d questions with prompt %s", len(examples), version)
    records = generate(assistant, examples, workers=workers)
    metrics = behaviour_metrics(records)

    if ragas_metrics:
        items = ragas_items(records)
        log.info("Scoring %d answered responses with RAGAS", len(items))
        scores = score_items(items, ragas_metrics, concurrency=ragas_concurrency)
        for r in records:
            r["scores"] = scores.get(r["id"], {})
        metrics.update(aggregate_scores(records, tuple(ragas_metrics)))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / f"{stamp}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if v is not None}, indent=2),
        encoding="utf-8",
    )
    clean = {k: float(v) for k, v in metrics.items() if v is not None}
    return RagEvalResult(prompt_version=version, metrics=clean, records=records, output_dir=out_dir)


def flat_table(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Column-oriented per-question table for `mlflow.log_table`."""
    cols = [
        "id",
        "category",
        "subtype",
        "expected_behavior",
        "refused",
        "behavior_correct",
        "question",
        "answer",
        "citations",
        "max_rerank_score",
        "latency_s",
    ]
    table: dict[str, list[Any]] = {c: [] for c in cols}
    score_names = sorted({k for r in records for k in r.get("scores", {})})
    for s in score_names:
        table[s] = []
    for r in records:
        for c in cols:
            v = r.get(c)
            table[c].append(", ".join(v) if isinstance(v, list) else v)
        for s in score_names:
            table[s].append(r.get("scores", {}).get(s))
    return table

"""High-level eval entry points used by the CLI: wire settings, models, MLflow, LangSmith."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from clinrag.config import PROJECT_ROOT, Settings
from clinrag.evaluation.dataset import EvalExample, file_fingerprint, load_eval_set, summarize
from clinrag.evaluation.rag_eval import RagEvalResult, flat_table, run_rag_eval
from clinrag.evaluation.retrieval_eval import RetrievalReport, evaluate_retrieval
from clinrag.evaluation.tracking import (
    langsmith_enabled,
    log_metrics,
    mlflow_run,
    push_feedback,
    search_runs,
)
from clinrag.schemas import RetrievalMode

log = logging.getLogger(__name__)

RETRIEVAL_EXPERIMENT = "clinrag-retrieval"
RAG_EXPERIMENT = "clinrag-rag-eval"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def _retrieval_params(settings: Settings) -> dict[str, Any]:
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "bm25_top_k": settings.bm25_top_k,
        "dense_top_k": settings.dense_top_k,
        "rrf_k": settings.rrf_k,
        "rerank_candidates": settings.rerank_candidates,
    }


# ---------------------------------------------------------------------------
# Retrieval ablation
# ---------------------------------------------------------------------------


def run_retrieval_ablation(
    settings: Settings,
    dataset: Path,
    modes: tuple[RetrievalMode, ...],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    level: str = "doc",
    use_mlflow: bool = True,
    retriever=None,
) -> RetrievalReport:
    from clinrag.retrieval.hybrid import HybridRetriever

    examples = load_eval_set(dataset)
    if retriever is None:
        retriever = HybridRetriever.load(settings, load_reranker="hybrid_rerank" in modes)
    report = evaluate_retrieval(retriever, examples, modes=modes, ks=ks, level=level)

    out_dir = OUTPUT_ROOT / "retrieval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report.metrics, indent=2), encoding="utf-8")
    with (out_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for row in report.per_question:
            f.write(json.dumps(row) + "\n")

    if use_mlflow:
        fingerprint = file_fingerprint(dataset)
        for mode, metrics in report.metrics.items():
            params = {
                "mode": mode,
                "level": level,
                "dataset": dataset.name,
                "dataset_sha": fingerprint,
                **_retrieval_params(settings),
            }
            with mlflow_run(settings.mlflow_tracking_uri, RETRIEVAL_EXPERIMENT, mode, params):
                import mlflow

                log_metrics(metrics)
                mode_rows = out_dir / f"per_question_{mode}.jsonl"
                with mode_rows.open("w", encoding="utf-8") as f:
                    for row in report.per_question:
                        if row["mode"] == mode:
                            f.write(json.dumps(row) + "\n")
                mlflow.log_artifact(str(mode_rows))
    log.info("Retrieval results written to %s", out_dir)
    return report


# ---------------------------------------------------------------------------
# Prompt-version RAG evals
# ---------------------------------------------------------------------------


def _filter(examples: list[EvalExample], categories: list[str] | None, limit: int | None):
    if categories:
        examples = [e for e in examples if e.category in categories]
    return examples[:limit] if limit else examples


def run_prompt_evals(
    settings: Settings,
    dataset: Path,
    versions: list[str],
    judge: bool = True,
    use_mlflow: bool = True,
    workers: int = 4,
    categories: list[str] | None = None,
    limit: int | None = None,
    metric_names: list[str] | None = None,
) -> list[RagEvalResult]:
    """`metric_names` picks the RAGAS metrics (default: all four). On a free tier,
    `["faithfulness"]` needs about a quarter of the judge calls."""
    from clinrag.agent.assistant import ClinicalAssistant, build_assistant
    from clinrag.agent.prompts import load_prompt
    from clinrag.evaluation.ragas_eval import METRICS, build_metrics

    examples = _filter(load_eval_set(dataset), categories, limit)
    log.info("Eval set: %s", summarize(examples))
    base: ClinicalAssistant = build_assistant(settings, prompt_version=versions[0])
    metrics = None
    if judge:
        metrics = build_metrics(
            settings.eval_judge_model,
            settings.eval_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            names=tuple(metric_names) if metric_names else METRICS,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.eval_judge_max_tokens,
            temperature=settings.eval_judge_temperature,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    feedback = langsmith_enabled()
    fingerprint = file_fingerprint(dataset)

    results = []
    for version in versions:
        prompt = load_prompt(settings.prompts_dir, version)
        assistant = ClinicalAssistant(
            llm=base.llm,
            retriever=base.retriever,
            prompt=prompt,
            text_to_sql=base.text_to_sql,
            retrieval_mode=base.retrieval_mode,
            top_k=base.top_k,
            refusal_threshold=base.refusal_threshold,
            max_steps=base.max_steps,
        )
        result = run_rag_eval(
            assistant,
            examples,
            ragas_metrics=metrics,
            output_root=OUTPUT_ROOT / "rag",
            workers=workers,
            ragas_concurrency=settings.eval_concurrency,
        )
        if use_mlflow:
            result.mlflow_run_id = _log_rag_run(
                settings, assistant, prompt, result, dataset, fingerprint
            )
        if feedback:
            _push_feedback(result)
        _print_summary(result)
        results.append(result)
    return results


def _log_rag_run(
    settings, assistant, prompt, result: RagEvalResult, dataset: Path, sha: str
) -> str:
    import mlflow

    from clinrag.evaluation.ragas_eval import METRICS

    params = {
        "prompt_version": prompt.version,
        "prompt_name": prompt.name,
        "prompt_sha": prompt.sha,
        "llm_model": assistant.llm.model,
        "judge_model": settings.eval_judge_model,
        "ragas_metrics": ",".join(m for m in METRICS if m in result.metrics) or "none",
        "retrieval_mode": assistant.retrieval_mode,
        "top_k": assistant.top_k,
        "refusal_threshold": assistant.refusal_threshold,
        "sql_tool": assistant.text_to_sql is not None,
        "dataset": dataset.name,
        "dataset_sha": sha,
        **_retrieval_params(settings),
    }
    with mlflow_run(
        settings.mlflow_tracking_uri,
        RAG_EXPERIMENT,
        run_name=f"{prompt.version}-{prompt.name}",
        params=params,
        tags={"prompt_version": prompt.version},
    ) as run:
        log_metrics(result.metrics)
        if result.output_dir:
            mlflow.log_artifacts(str(result.output_dir), artifact_path="eval")
        if prompt.path:
            mlflow.log_artifact(prompt.path, artifact_path="prompt")
        try:
            mlflow.log_table(flat_table(result.records), artifact_file="per_question.json")
        except Exception as exc:  # log_table needs pandas; never fail the run on it
            log.warning("mlflow.log_table failed: %s", exc)
        return run.info.run_id


def _push_feedback(result: RagEvalResult) -> None:
    for r in result.records:
        if not r.get("run_id") or r.get("refused") is None:
            continue
        scores = dict(r.get("scores", {}))
        scores["behavior_correct"] = float(r.get("behavior_correct", False))
        try:
            push_feedback(r["run_id"], scores)
        except Exception as exc:
            log.warning("LangSmith feedback failed for %s: %s", r["id"], exc)
            return


def _print_summary(result: RagEvalResult) -> None:
    keys = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "refusal_rate_out_of_corpus",
        "false_refusal_rate",
        "behavior_accuracy",
    ]
    shown = ", ".join(f"{k}={result.metrics[k]:.3f}" for k in keys if k in result.metrics)
    log.info("[%s] %s", result.prompt_version, shown)


def compare_runs(settings: Settings, experiment: str = RAG_EXPERIMENT) -> list[dict[str, Any]]:
    rows = search_runs(settings.mlflow_tracking_uri, experiment)
    return [
        {
            "run": r.get("tags.mlflow.runName"),
            "start": str(r.get("start_time"))[:19],
            **{
                k.removeprefix("metrics."): v
                for k, v in r.items()
                if k.startswith("metrics.") and v == v and v is not None
            },
        }
        for r in rows
    ]

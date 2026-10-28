"""MLflow and LangSmith helpers. Both are optional at runtime: the harness still writes
results to `outputs/` when neither is configured."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


@contextmanager
def mlflow_run(
    tracking_uri: str,
    experiment: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: _param(v) for k, v in params.items()})
        if tags:
            mlflow.set_tags(tags)
        yield run


def _param(value: Any) -> str:
    s = str(value)
    return s if len(s) <= 500 else s[:497] + "..."


def log_metrics(metrics: dict[str, float | None]) -> None:
    import mlflow

    clean = {
        k.replace("@", "_at_"): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and v == v  # drop None / NaN
    }
    mlflow.log_metrics(clean)


def search_runs(tracking_uri: str, experiment: str, max_results: int = 50) -> list[dict[str, Any]]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return []
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=max_results
    )
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


def langsmith_enabled() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return tracing.lower() == "true" and bool(key)


def push_feedback(run_id: str, scores: dict[str, float | None], comment: str | None = None) -> None:
    """Attach eval scores to the traced run, so failures can be inspected in LangSmith."""
    from langsmith import Client

    client = Client()
    for key, score in scores.items():
        if score is None or score != score:
            continue
        client.create_feedback(run_id, key=key, score=float(score), comment=comment)


def sync_dataset(name: str, examples: list[dict[str, Any]], description: str = "") -> str:
    """Create (or replace the examples of) a LangSmith dataset. Returns the dataset id."""
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = [ex.id for ex in client.list_examples(dataset_id=dataset.id)]
        if existing:
            client.delete_examples(existing)
    else:
        dataset = client.create_dataset(name, description=description)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": ex["question"]},
                "outputs": {
                    "reference_answer": ex.get("reference_answer"),
                    "expected_behavior": ex["expected_behavior"],
                    "relevant_doc_ids": ex.get("relevant_doc_ids", []),
                },
                "metadata": {
                    "id": ex["id"],
                    "category": ex["category"],
                    "subtype": ex.get("subtype"),
                },
            }
            for ex in examples
        ],
    )
    return str(dataset.id)

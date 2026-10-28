"""RAGAS scoring (ragas 0.4 "collections" metrics, LLM-as-judge).

Metrics
- faithfulness       share of answer claims supported by the evidence shown to the model
- answer_relevancy   how directly the answer addresses the question
- context_precision  are the relevant passages ranked high? (needs a reference answer)
- context_recall     does the evidence cover the reference answer? (needs a reference answer)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class ScoringItem:
    key: str
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None


def build_metrics(
    judge_model: str,
    embedding_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    names: tuple[str, ...] = METRICS,
    max_retries: int = 4,
    **judge_args: Any,
) -> dict[str, Any]:
    """RAGAS metric objects sharing one judge LLM.

    `judge_args` (e.g. max_tokens, temperature, reasoning_effort) are forwarded to every
    judge call; None values are dropped so the provider defaults apply.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    unknown = set(names) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown RAGAS metrics: {sorted(unknown)}; choose from {METRICS}")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
    llm = llm_factory(
        judge_model, client=client, **{k: v for k, v in judge_args.items() if v is not None}
    )
    factories = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=llm, embeddings=OpenAIEmbeddings(client=client, model=embedding_model)
        ),
        "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return {name: factories[name]() for name in names}


def _kwargs(name: str, item: ScoringItem) -> dict[str, Any] | None:
    """Arguments for each metric, or None when the item can't be scored on it."""
    if name == "faithfulness":
        if not item.contexts:
            return None
        return {
            "user_input": item.question,
            "response": item.answer,
            "retrieved_contexts": item.contexts,
        }
    if name == "answer_relevancy":
        return {"user_input": item.question, "response": item.answer}
    if not item.reference or not item.contexts:
        return None
    if name == "context_precision":
        return {
            "user_input": item.question,
            "reference": item.reference,
            "retrieved_contexts": item.contexts,
        }
    if name == "context_recall":
        return {
            "user_input": item.question,
            "retrieved_contexts": item.contexts,
            "reference": item.reference,
        }
    raise ValueError(name)


async def _score_all(
    metrics: dict[str, Any], items: list[ScoringItem], concurrency: int
) -> dict[str, dict[str, float | None]]:
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict[str, float | None]] = {it.key: {} for it in items}

    async def one(item: ScoringItem, name: str, metric: Any) -> None:
        kwargs = _kwargs(name, item)
        if kwargs is None:
            return
        async with sem:
            try:
                result = await metric.ascore(**kwargs)
                value = result.value if hasattr(result, "value") else result
                out[item.key][name] = None if value is None else float(value)
            except Exception as exc:  # judge parse failures etc. -> missing, not zero
                log.warning("RAGAS %s failed for %s: %s", name, item.key, exc)
                out[item.key][name] = None

    await asyncio.gather(*(one(it, name, m) for it in items for name, m in metrics.items()))
    return out


def score_items(
    items: list[ScoringItem], metrics: dict[str, Any], concurrency: int = 8
) -> dict[str, dict[str, float | None]]:
    """Returns {item.key: {metric_name: score}}; metrics that don't apply are absent."""
    if not items:
        return {}
    return asyncio.run(_score_all(metrics, items, concurrency))

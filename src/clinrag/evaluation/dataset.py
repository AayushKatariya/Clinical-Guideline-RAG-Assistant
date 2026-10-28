"""Evaluation set schema and loading.

One JSON object per line:

    {"id": "ic-001", "question": "...", "category": "in_corpus", "expected_behavior": "answer",
     "reference_answer": "...", "relevant_doc_ids": ["who-tb-2022"], "relevant_chunk_ids": []}

Categories
- in_corpus     answerable from the guideline text      -> answer
- sql           answerable from the structured tables   -> answer
- adversarial   false premise / prompt injection / ...  -> answer (correct the premise) or refuse
- out_of_scope  not covered by the corpus or off-topic  -> refuse
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from clinrag.schemas import iter_jsonl

Category = Literal["in_corpus", "sql", "adversarial", "out_of_scope"]
Behavior = Literal["answer", "refuse"]


class EvalExample(BaseModel):
    id: str
    question: str
    category: Category
    expected_behavior: Behavior
    subtype: str | None = None
    reference_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EvalExample:
        if self.category == "out_of_scope" and self.expected_behavior != "refuse":
            raise ValueError(f"{self.id}: out_of_scope examples must expect 'refuse'")
        if self.category in ("in_corpus", "sql") and self.expected_behavior != "answer":
            raise ValueError(f"{self.id}: {self.category} examples must expect 'answer'")
        return self

    @property
    def has_retrieval_labels(self) -> bool:
        return bool(self.relevant_doc_ids or self.relevant_chunk_ids)


def load_eval_set(path: Path) -> list[EvalExample]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build a draft with `clinrag eval build-testset`, review it, "
            "and save it as data/eval/eval_set.jsonl (see data/README.md)."
        )
    examples = [EvalExample(**row) for row in iter_jsonl(path)]
    dupes = [k for k, v in Counter(e.id for e in examples).items() if v > 1]
    if dupes:
        raise ValueError(f"duplicate example ids: {dupes[:5]}")
    return examples


def summarize(examples: list[EvalExample]) -> dict[str, int]:
    out: dict[str, int] = {"total": len(examples)}
    for cat, n in Counter(e.category for e in examples).items():
        out[f"category/{cat}"] = n
    for beh, n in Counter(e.expected_behavior for e in examples).items():
        out[f"expected/{beh}"] = n
    out["with_retrieval_labels"] = sum(e.has_retrieval_labels for e in examples)
    out["with_reference_answer"] = sum(bool(e.reference_answer) for e in examples)
    return out


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

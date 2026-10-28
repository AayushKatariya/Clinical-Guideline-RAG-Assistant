"""Shared data models."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One clinical guideline, as loaded from the raw corpus."""

    doc_id: str
    title: str
    text: str
    organization: str | None = None
    year: int | None = None
    specialty: str | None = None
    url: str | None = None
    source_path: str | None = None
    # Character offsets where each PDF page starts (empty for non-paginated sources).
    page_starts: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of a guideline."""

    chunk_id: str
    doc_id: str
    title: str
    section: str = ""
    text: str
    position: int = 0
    page_start: int | None = None
    page_end: int | None = None
    organization: str | None = None
    year: int | None = None
    specialty: str | None = None

    @property
    def index_text(self) -> str:
        """Text that is indexed: a contextual header + the body.

        Prepending the guideline title and section path lets both BM25 and the dense
        encoder match queries that name the guideline or topic but not the passage words.
        """
        header = self.title if not self.section else f"{self.title} > {self.section}"
        return f"{header}\n{self.text}"

    def citation(self) -> str:
        parts = [self.title]
        if self.section:
            parts.append(f"§ {self.section}")
        if self.page_start is not None:
            pages = (
                f"p. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"pp. {self.page_start}-{self.page_end}"
            )
            parts.append(pages)
        return ", ".join(parts)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    # Per-stage scores, e.g. {"bm25": 12.3, "dense": 0.71, "rrf": 0.03, "rerank": 0.88}
    scores: dict[str, float] = Field(default_factory=dict)


RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class SQLResult(BaseModel):
    question: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    error: str | None = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.sql is not None

    def to_markdown(self, max_rows: int = 25) -> str:
        if not self.ok:
            return f"Query failed: {self.error}"
        if not self.rows:
            return "The query returned no rows."
        header = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
            for row in self.rows[:max_rows]
        ]
        extra = (
            [f"... ({len(self.rows) - max_rows} more rows)"] if len(self.rows) > max_rows else []
        )
        return "\n".join([header, sep, *body, *extra])


class Citation(BaseModel):
    source_id: str  # e.g. "S1" or "DB1"
    kind: Literal["guideline", "database"]
    doc_id: str | None = None
    chunk_id: str | None = None
    label: str = ""


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    output_preview: str = ""


class AssistantResponse(BaseModel):
    question: str
    answer: str
    refused: bool = False
    refusal_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    # Exactly the evidence text shown to the model (what RAGAS faithfulness is judged against).
    contexts: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    max_rerank_score: float | None = None
    sql_queries: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    prompt_version: str = ""
    latency_s: float = 0.0
    run_id: str | None = None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc

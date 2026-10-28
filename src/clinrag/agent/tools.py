"""Tools exposed to the LLM, plus the evidence registry they write into."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from clinrag.schemas import Chunk, Citation, RetrievedChunk, SQLResult

SEARCH_TOOL = "search_guidelines"
SQL_TOOL = "query_clinical_database"

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL,
            "description": (
                "Search the clinical guideline corpus (hybrid BM25 + dense retrieval with "
                "cross-encoder reranking). Use a focused query naming the condition, "
                "population and intervention. Returns excerpts with IDs like [S3]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "organization": {
                        "type": "string",
                        "description": "Optional issuing body filter, e.g. 'WHO' or 'NICE'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SQL_TOOL,
            "description": (
                "Answer a question from the structured guideline database (tables: "
                "guidelines, recommendations, drug_dosing, diagnostic_thresholds). Good for "
                "counts, lists, filtering by organisation/year, drug doses and numeric "
                "thresholds. Pass the question in plain English; SQL is written for you. "
                "Returns a table with an ID like [DB1]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question in plain English."}
                },
                "required": ["question"],
            },
        },
    },
]

_CITATION_GROUP = re.compile(r"\[([^\[\]]{1,60})\]")
_SOURCE_ID = re.compile(r"^(S|DB)(\d+)$")


@dataclass
class EvidenceStore:
    """Numbers every piece of evidence shown to the model and renders it consistently.

    Its contents are exactly what the answer must be faithful to, so they are also what
    RAGAS faithfulness is computed against.
    """

    chunks: dict[str, Chunk] = field(default_factory=dict)  # "S1" -> chunk
    sql: dict[str, SQLResult] = field(default_factory=dict)  # "DB1" -> result
    _by_chunk_id: dict[str, str] = field(default_factory=dict)
    _rendered: dict[str, str] = field(default_factory=dict)

    def add_chunks(self, hits: list[RetrievedChunk]) -> list[str]:
        ids = []
        for hit in hits:
            cid = hit.chunk.chunk_id
            if cid not in self._by_chunk_id:
                sid = f"S{len(self.chunks) + 1}"
                self.chunks[sid] = hit.chunk
                self._by_chunk_id[cid] = sid
                self._rendered[sid] = self._render_chunk(sid, hit.chunk)
            ids.append(self._by_chunk_id[cid])
        return ids

    def add_sql(self, result: SQLResult) -> str:
        sid = f"DB{len(self.sql) + 1}"
        self.sql[sid] = result
        self._rendered[sid] = f"[{sid}] Database query: {result.sql}\n{result.to_markdown()}"
        return sid

    @staticmethod
    def _render_chunk(sid: str, chunk: Chunk) -> str:
        meta = " · ".join(str(x) for x in (chunk.organization, chunk.year) if x not in (None, ""))
        header = f"[{sid}] {chunk.citation()}" + (f" ({meta})" if meta else "")
        return f"{header}\n{chunk.text}"

    def render(self, ids: list[str]) -> str:
        return "\n\n".join(self._rendered[i] for i in dict.fromkeys(ids))

    def render_all(self) -> str:
        return "\n\n".join(self._rendered.values())

    def contexts(self) -> list[str]:
        return list(self._rendered.values())

    @property
    def has_support(self) -> bool:
        return bool(self.chunks) or any(r.ok and r.rows for r in self.sql.values())

    def citations_in(self, answer: str) -> list[Citation]:
        found: dict[str, Citation] = {}
        for group in _CITATION_GROUP.findall(answer):
            for token in re.split(r"[,;\s]+", group.strip()):
                if not _SOURCE_ID.match(token) or token in found:
                    continue
                if token in self.chunks:
                    c = self.chunks[token]
                    found[token] = Citation(
                        source_id=token,
                        kind="guideline",
                        doc_id=c.doc_id,
                        chunk_id=c.chunk_id,
                        label=c.citation(),
                    )
                elif token in self.sql:
                    found[token] = Citation(
                        source_id=token, kind="database", label=self.sql[token].sql or ""
                    )
        return list(found.values())

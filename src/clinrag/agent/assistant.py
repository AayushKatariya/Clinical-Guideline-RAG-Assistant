"""The clinical guideline assistant: retrieve -> (tool-calling loop) -> grounded answer.

Flow for one question:

1. Hybrid retrieval on the question. Passages whose cross-encoder score is below
   `refusal_threshold` are *not* shown to the model (they are not evidence).
2. Tool-calling loop: the model may search again with a reformulated query or ask the
   text-to-SQL tool for structured facts. Each tool result is numbered ([S#], [DB#]).
3. Refusal: the model refuses via the prompt protocol, or - regardless of prompt - the
   evidence gate refuses when nothing relevant was ever retrieved. This gate is what
   makes out-of-corpus refusals robust to prompt changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langsmith import traceable

from clinrag.agent.prompts import PromptTemplate
from clinrag.agent.refusal import DEFAULT_REFUSAL, is_refusal, strip_marker
from clinrag.agent.tools import SEARCH_TOOL, SQL_TOOL, TOOL_SPECS, EvidenceStore
from clinrag.llm.client import LLMClient
from clinrag.retrieval.hybrid import HybridRetriever
from clinrag.schemas import AssistantResponse, RetrievalMode, RetrievedChunk, ToolCallRecord
from clinrag.sql.text_to_sql import TextToSQL

log = logging.getLogger(__name__)


class ClinicalAssistant:
    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        prompt: PromptTemplate,
        text_to_sql: TextToSQL | None = None,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
        top_k: int = 5,
        refusal_threshold: float = 0.2,
        max_steps: int = 4,
        evidence_gate: bool = True,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.text_to_sql = text_to_sql
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.refusal_threshold = refusal_threshold
        self.max_steps = max_steps
        self.evidence_gate = evidence_gate
        self.tools = [
            t for t in TOOL_SPECS if t["function"]["name"] != SQL_TOOL or text_to_sql is not None
        ]

    # ------------------------------------------------------------------ helpers

    def _relevant(self, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Only reranked scores are calibrated enough to threshold."""
        if self.retrieval_mode != "hybrid_rerank":
            return hits
        return [h for h in hits if h.score >= self.refusal_threshold]

    def _run_tool(
        self, name: str, args: dict[str, Any], evidence: EvidenceStore, seen: list[RetrievedChunk]
    ) -> str:
        if name == SEARCH_TOOL:
            query = str(args.get("query") or "").strip()
            if not query:
                return "Error: 'query' is required."
            filters = {"organization": args["organization"]} if args.get("organization") else None
            hits = self.retriever.retrieve(
                query, k=self.top_k, mode=self.retrieval_mode, filters=filters
            )
            seen.extend(hits)
            relevant = self._relevant(hits)
            if not relevant:
                return "No guideline passages were relevant enough to this query."
            return evidence.render(evidence.add_chunks(relevant))
        if name == SQL_TOOL and self.text_to_sql is not None:
            question = str(args.get("question") or "").strip()
            if not question:
                return "Error: 'question' is required."
            result = self.text_to_sql.run(question)
            if not result.ok:
                return f"The database could not answer this: {result.error}"
            return evidence.render([evidence.add_sql(result)])
        return f"Error: unknown tool {name!r}."

    # ------------------------------------------------------------------ main entry

    @traceable(run_type="chain", name="clinical_assistant")
    def answer(self, question: str) -> AssistantResponse:
        t0 = time.perf_counter()
        evidence = EvidenceStore()
        seen: list[RetrievedChunk] = []
        tool_log: list[ToolCallRecord] = []

        initial = self.retriever.retrieve(question, k=self.top_k, mode=self.retrieval_mode)
        seen.extend(initial)
        evidence.add_chunks(self._relevant(initial))

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(question, evidence.render_all())},
        ]

        final_text: str | None = None
        for step in range(self.max_steps + 1):
            tools = self.tools if step < self.max_steps else None  # last turn must answer
            result = self.llm.chat(messages, tools=tools)
            if not result.tool_calls:
                final_text = result.content or ""
                break
            messages.append(result.as_message())
            for call in result.tool_calls:
                args = call.parsed_arguments()
                output = self._run_tool(call.name, args, evidence, seen)
                tool_log.append(
                    ToolCallRecord(name=call.name, arguments=args, output_preview=output[:300])
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        final_text = final_text or ""
        refused = is_refusal(final_text)
        refusal_reason = "model" if refused else None
        if not refused and self.evidence_gate and not evidence.has_support:
            refused, refusal_reason = True, "no_evidence"
            final_text = DEFAULT_REFUSAL

        answer = strip_marker(final_text) if refused else final_text.strip()
        scores = [h.scores.get("rerank") for h in seen if "rerank" in h.scores]
        return AssistantResponse(
            question=question,
            answer=answer,
            refused=refused,
            refusal_reason=refusal_reason,
            citations=[] if refused else evidence.citations_in(answer),
            contexts=evidence.contexts(),
            retrieved_doc_ids=list(dict.fromkeys(h.chunk.doc_id for h in initial)),
            retrieved_chunk_ids=[h.chunk.chunk_id for h in initial],
            max_rerank_score=max(scores) if scores else None,
            sql_queries=[r.sql for r in evidence.sql.values() if r.sql],
            tool_calls=tool_log,
            prompt_version=self.prompt.version,
            latency_s=round(time.perf_counter() - t0, 3),
        )


def build_assistant(settings, prompt_version: str | None = None, retriever=None, llm=None):
    """Wire up the production assistant from settings (models, Postgres, prompt)."""
    from clinrag.agent.prompts import load_prompt
    from clinrag.llm.client import make_llm
    from clinrag.sql.db import make_engine

    llm = llm or make_llm(settings)
    retriever = retriever or HybridRetriever.load(settings)
    prompt = load_prompt(settings.prompts_dir, prompt_version or settings.prompt_version)
    text_to_sql = None
    try:
        engine = make_engine(settings.database_url)
        with engine.connect():
            pass
        text_to_sql = TextToSQL(
            llm,
            engine,
            row_limit=settings.sql_row_limit,
            timeout_ms=settings.sql_timeout_ms,
            max_attempts=settings.sql_max_attempts,
        )
    except Exception as exc:  # the assistant still works text-only without Postgres
        log.warning("Postgres unavailable (%s); text-to-SQL tool disabled.", exc)
    return ClinicalAssistant(
        llm=llm,
        retriever=retriever,
        prompt=prompt,
        text_to_sql=text_to_sql,
        top_k=settings.final_top_k,
        refusal_threshold=settings.refusal_score_threshold,
        max_steps=settings.agent_max_steps,
    )

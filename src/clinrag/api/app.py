"""FastAPI service: chat page at GET /, plus POST /ask, POST /search, GET /prompts, GET /health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from clinrag import __version__
from clinrag.agent.assistant import ClinicalAssistant
from clinrag.agent.prompts import list_prompts, load_prompt
from clinrag.config import get_settings
from clinrag.schemas import AssistantResponse, RetrievalMode, RetrievedChunk

CHAT_PAGE = Path(__file__).with_name("static") / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    prompt_version: str | None = Field(default=None, description="e.g. v1..v4; default from config")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode = "hybrid_rerank"
    organization: str | None = None


def create_app(assistant: ClinicalAssistant | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "assistant", None) is None:
            from clinrag.agent.assistant import build_assistant

            app.state.assistant = build_assistant(get_settings())
        yield

    app = FastAPI(title="Clinical Guideline RAG Assistant", version=__version__, lifespan=lifespan)
    app.state.assistant = assistant
    variants: dict[str, ClinicalAssistant] = {}  # prompt version -> assistant

    def get_assistant(request: Request) -> ClinicalAssistant:
        a = request.app.state.assistant
        if a is None:
            raise HTTPException(503, "assistant not loaded")
        return a

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_page() -> HTMLResponse:
        """A single-page chat UI over /ask (no build step, no external assets)."""
        return HTMLResponse(CHAT_PAGE.read_text(encoding="utf-8"))

    @app.get("/health")
    def health(a: ClinicalAssistant = Depends(get_assistant)) -> dict:
        return {
            "status": "ok",
            "prompt_version": a.prompt.version,
            "n_chunks": len(a.retriever.chunks),
            "sql_tool": a.text_to_sql is not None,
        }

    @app.get("/prompts")
    def prompts() -> list[dict]:
        return [
            {"version": p.version, "name": p.name, "description": p.description, "sha": p.sha}
            for p in list_prompts(get_settings().prompts_dir)
        ]

    @app.post("/ask", response_model=AssistantResponse)
    def ask(req: AskRequest, a: ClinicalAssistant = Depends(get_assistant)) -> AssistantResponse:
        if req.prompt_version and req.prompt_version != a.prompt.version:
            if req.prompt_version not in variants:
                try:
                    prompt = load_prompt(get_settings().prompts_dir, req.prompt_version)
                except FileNotFoundError as exc:
                    raise HTTPException(404, str(exc)) from exc
                variants[req.prompt_version] = _with_prompt(a, prompt)
            a = variants[req.prompt_version]
        return a.answer(req.question)

    @app.post("/search", response_model=list[RetrievedChunk])
    def search(
        req: SearchRequest, a: ClinicalAssistant = Depends(get_assistant)
    ) -> list[RetrievedChunk]:
        filters = {"organization": req.organization} if req.organization else None
        return a.retriever.retrieve(req.query, k=req.k, mode=req.mode, filters=filters)

    return app


def _with_prompt(a: ClinicalAssistant, prompt) -> ClinicalAssistant:
    return ClinicalAssistant(
        llm=a.llm,
        retriever=a.retriever,
        prompt=prompt,
        text_to_sql=a.text_to_sql,
        retrieval_mode=a.retrieval_mode,
        top_k=a.top_k,
        refusal_threshold=a.refusal_threshold,
        max_steps=a.max_steps,
        evidence_gate=a.evidence_gate,
    )


app = create_app()

__all__ = ["app", "create_app"]

"""Central configuration, read from environment variables / `.env`."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root when installed editable; override with CLINRAG_HOME (e.g. in Docker).
PROJECT_ROOT = Path(os.getenv("CLINRAG_HOME") or Path(__file__).resolve().parents[2])
ENV_FILE = ".env"


class Settings(BaseSettings):
    # env_ignore_empty: a line like `FOO=` in .env means "use the default".
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", env_ignore_empty=True)

    # ---- LLM ----
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800  # thinking models (Gemini 3+): thinking + answer, use ~4096
    llm_reasoning_effort: str | None = None  # sent only when set, e.g. "low"
    llm_max_retries: int = 4  # back-off retries on 429 / 5xx / connection errors
    agent_max_steps: int = 4

    # ---- Paths ----
    data_dir: Path = PROJECT_ROOT / "data"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    sql_dir: Path = PROJECT_ROOT / "sql"

    # ---- Chunking ----
    chunk_size_words: int = 350
    chunk_overlap_words: int = 60
    min_chunk_words: int = 40

    # ---- Retrieval models ----
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 64
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 32
    device: str | None = None

    # ---- Retrieval knobs ----
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 40
    final_top_k: int = 5
    refusal_score_threshold: float = 0.20

    # ---- Postgres ----
    database_url: str = "postgresql+psycopg://clinrag_readonly:readonly@localhost:5432/clinical"
    database_admin_url: str = "postgresql+psycopg://clinrag:clinrag@localhost:5432/clinical"
    sql_row_limit: int = 50
    sql_timeout_ms: int = 5000
    sql_max_attempts: int = 3

    # ---- Prompting ----
    prompt_version: str = "v4"

    # ---- Evaluation ----
    eval_judge_model: str = "gpt-4o-mini"
    eval_judge_max_tokens: int = 4096
    eval_judge_temperature: float | None = None  # None: the provider's default
    eval_embedding_model: str = "text-embedding-3-small"
    eval_concurrency: int = 8
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    langsmith_project: str = "clinical-guideline-rag"

    # ---- Derived paths ----
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def structured_dir(self) -> Path:
        return self.data_dir / "structured"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def metadata_dir(self) -> Path:
        # Outside raw/: every *.jsonl in raw/ is ingested as corpus.
        return self.data_dir / "metadata"

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"


def export_langsmith_env(settings: Settings, env_file: str | Path = ENV_FILE) -> None:
    """Copy the LangSmith variables from `.env` into os.environ.

    pydantic-settings reads `.env` without exporting it, but the langsmith SDK only looks at
    os.environ (and caches what it finds), so without this tracing stays off unless the
    variables are set in the shell. Variables already set in the environment win, as they
    do for Settings, and nothing else in `.env` is exported.
    """
    for key, value in dotenv_values(env_file).items():
        if key.upper().startswith(("LANGSMITH_", "LANGCHAIN_")) and value and key not in os.environ:
            os.environ[key] = value
    if not (os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")):
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    export_langsmith_env(settings)
    return settings

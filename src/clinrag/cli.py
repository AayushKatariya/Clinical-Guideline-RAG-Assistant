"""Command-line interface: `clinrag --help`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clinrag.config import get_settings

app = typer.Typer(help="Clinical Guideline RAG Assistant", no_args_is_help=True)
db_app = typer.Typer(help="Postgres: schema, loading, structured extraction", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness", no_args_is_help=True)
metadata_app = typer.Typer(
    help="Descriptive titles for untitled corpus documents (LLM calls)",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")
app.add_typer(eval_app, name="eval")
app.add_typer(metadata_app, name="metadata")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Hub auth hints and torch's "redirects not supported on Windows" note are just noise.
    for noisy in ("huggingface_hub", "torch.distributed.elastic"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _print_table(rows: list[dict], title: str | None = None, digits: int = 3) -> None:
    if not rows:
        console.print("[yellow]nothing to show[/yellow]")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    table = Table(title=title)
    for c in cols:
        table.add_column(str(c))
    for r in rows:
        table.add_row(
            *[
                f"{v:.{digits}f}" if isinstance(v, float) else str(v if v is not None else "")
                for v in (r.get(c) for c in cols)
            ]
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Corpus and indexes
# ---------------------------------------------------------------------------


@app.command()
def ingest(raw_dir: Path = typer.Option(None, help="Defaults to data/raw")) -> None:
    """Load raw guidelines and write documents.jsonl + chunks.jsonl."""
    from clinrag.ingestion.pipeline import ingest_corpus

    s = get_settings()
    n_docs, n_chunks = ingest_corpus(s, raw_dir)
    console.print(f"[green]{n_docs} documents -> {n_chunks} chunks[/green] in {s.processed_dir}")


@app.command()
def index() -> None:
    """Build the BM25 and dense (FAISS) indexes from chunks.jsonl."""
    from clinrag.ingestion.pipeline import load_chunks
    from clinrag.retrieval.dense import SentenceTransformerEmbedder
    from clinrag.retrieval.hybrid import HybridRetriever

    s = get_settings()
    chunks = load_chunks(s.chunks_path)
    embedder = SentenceTransformerEmbedder(
        s.embedding_model, s.embedding_query_prefix, s.device, s.embedding_batch_size
    )
    HybridRetriever.build_indexes(chunks, embedder, s.index_dir)
    console.print(f"[green]Indexed {len(chunks)} chunks[/green] into {s.index_dir}")


@app.command()
def search(
    query: str,
    mode: str = typer.Option("hybrid_rerank", help="bm25 | dense | hybrid | hybrid_rerank"),
    k: int = 5,
    organization: str = typer.Option(None, "--org"),
) -> None:
    """Run the retriever only."""
    from clinrag.retrieval.hybrid import HybridRetriever

    s = get_settings()
    retriever = HybridRetriever.load(s, load_reranker=mode == "hybrid_rerank")
    filters = {"organization": organization} if organization else None
    hits = retriever.retrieve(query, k=k, mode=mode, filters=filters)
    for h in hits:
        scores = " ".join(f"{n}={v:.3f}" for n, v in h.scores.items())
        console.print(f"[bold]{h.rank}. {h.chunk.citation()}[/bold]  [dim]{scores}[/dim]")
        console.print(h.chunk.text[:400].replace("\n", " ") + "...\n")


@app.command()
def ask(
    question: str,
    prompt: str = typer.Option(None, "--prompt", help="Prompt version, e.g. v4"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Ask the assistant a question."""
    from clinrag.agent.assistant import build_assistant

    resp = build_assistant(get_settings(), prompt_version=prompt).answer(question)
    if as_json:
        console.print_json(resp.model_dump_json())
        return
    console.print(resp.answer)
    if resp.citations:
        console.print("\n[bold]Sources[/bold]")
        for c in resp.citations:
            console.print(f"  [{c.source_id}] {c.label}")
    console.print(
        f"\n[dim]prompt={resp.prompt_version} refused={resp.refused} "
        f"max_rerank={resp.max_rerank_score} latency={resp.latency_s}s "
        f"tools={[t.name for t in resp.tool_calls]}[/dim]"
    )


@app.command()
def prompts() -> None:
    """List prompt versions."""
    from clinrag.agent.prompts import list_prompts

    rows = [
        {"version": p.version, "name": p.name, "sha": p.sha, "description": p.description}
        for p in list_prompts(get_settings().prompts_dir)
    ]
    _print_table(rows, "Prompt versions")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Use 0.0.0.0 to accept other machines."),
    port: int = 8000,
) -> None:
    """Run the API and the chat page (http://127.0.0.1:8000)."""
    import uvicorn

    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    console.print(
        f"Loading models, then serving the chat page at [bold]http://{shown}:{port}[/bold] "
        f"(API docs at /docs). Ctrl+C to stop."
    )
    uvicorn.run("clinrag.api.app:app", host=host, port=port)


# ---------------------------------------------------------------------------
# Corpus titles (for documents that have none)
# ---------------------------------------------------------------------------


def _metadata_paths(s) -> tuple[Path, Path, Path]:
    """(generated-titles cache, backup of the original corpus file, review CSV)."""
    d = s.metadata_dir
    return d / "generated_titles.jsonl", d / "guidelines.orig.jsonl", d / "titles_review.csv"


def _corpus_source(s, raw: Path | None) -> tuple[Path, Path]:
    """(corpus file to rewrite, file to read documents from). After `apply`, documents
    are read from the untouched backup."""
    raw = raw or s.raw_dir / "guidelines.jsonl"
    _, backup, _ = _metadata_paths(s)
    return raw, backup if backup.exists() else raw


@metadata_app.command("titles")
def metadata_titles(
    limit: int = typer.Option(None, help="At most N documents (pilot)."),
    batch_size: int = typer.Option(20, help="Documents per LLM call."),
    retry: bool = typer.Option(False, "--retry", help="Redo rejected or missing titles."),
    model: str = typer.Option(None, help="Model to use (defaults to LLM_MODEL)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the first prompt; no calls."),
    raw: Path = typer.Option(None, help="Corpus file (default data/raw/guidelines.jsonl)."),
) -> None:
    """Have the LLM write descriptive titles for documents without one. Resumable."""
    from clinrag.ingestion.titles import (
        batches,
        build_prompt,
        generate_titles,
        load_cache,
        select_documents,
        summarize,
    )
    from clinrag.llm.client import make_llm

    s = get_settings()
    cache_path, _, _ = _metadata_paths(s)
    _, source = _corpus_source(s, raw)
    docs = select_documents(source, load_cache(cache_path), limit=limit, retry=retry)
    n_calls = len(batches(docs, batch_size))
    if not docs:
        console.print("[yellow]Nothing to do: every untitled document has a title.[/yellow]")
        return
    if dry_run:
        console.print(build_prompt(docs[:batch_size]), markup=False, highlight=False)
        console.print(f"\n[dim]{len(docs)} documents -> {n_calls} LLM calls[/dim]")
        return
    llm = make_llm(s, model=model)
    console.print(f"Titling {len(docs)} documents with {llm.model}: {n_calls} LLM calls.")

    def progress(n: int, total: int) -> None:
        console.print(f"[dim]call {n}/{total} done[/dim]")

    try:
        from dataclasses import asdict

        records = generate_titles(docs, llm, cache_path, batch_size, on_batch=progress)
    except Exception as exc:  # quota, network: finished batches are already cached
        console.print(f"[red]Stopped: {exc}[/red]\nRe-run the same command to continue.")
        raise typer.Exit(1) from exc
    _print_table(summarize(asdict(r) for r in records), "This run")
    console.print("Next: `clinrag metadata report`, check the CSV, then `clinrag metadata apply`.")


@metadata_app.command("report")
def metadata_report(
    raw: Path = typer.Option(None, help="Corpus file (default data/raw/guidelines.jsonl)."),
) -> None:
    """Summarise generated titles and write data/metadata/titles_review.csv."""
    from clinrag.ingestion.titles import load_cache, summarize, write_review_csv

    s = get_settings()
    cache_path, _, review = _metadata_paths(s)
    cache = load_cache(cache_path)
    if not cache:
        console.print("[yellow]No titles yet: run `clinrag metadata titles` first.[/yellow]")
        return
    _print_table(summarize(cache.values()), "Generated titles (all runs)")
    n = write_review_csv(cache.values(), _corpus_source(s, raw)[1], review)
    console.print(f"[green]{n} rows[/green] written to {review}")


@metadata_app.command("apply")
def metadata_apply(
    raw: Path = typer.Option(None, help="Corpus file (default data/raw/guidelines.jsonl)."),
) -> None:
    """Write accepted titles into the corpus file (original kept as a backup)."""
    from clinrag.ingestion.titles import apply_titles

    s = get_settings()
    cache_path, backup, _ = _metadata_paths(s)
    raw_path, _ = _corpus_source(s, raw)
    stats = apply_titles(raw_path, backup, cache_path)
    console.print(dict(stats))
    console.print(f"Original kept at {backup}. Next: `clinrag ingest` and `clinrag index`.")


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


@db_app.command("init")
def db_init() -> None:
    """Create tables and the read-only role (idempotent)."""
    from clinrag.sql.db import apply_sql_files, make_engine

    s = get_settings()
    applied = apply_sql_files(make_engine(s.database_admin_url), s.sql_dir)
    console.print(f"[green]Applied:[/green] {', '.join(applied)}")


@db_app.command("load")
def db_load(
    from_corpus: bool = typer.Option(
        True, help="Fill the `guidelines` table from processed corpus metadata first."
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Empty each table that has a CSV before loading it (use when reloading "
        "reviewed CSVs; without it rows are appended).",
    ),
) -> None:
    """Load data/structured/*.csv (and corpus metadata) into Postgres."""
    from clinrag.ingestion.pipeline import load_processed_documents
    from clinrag.sql.db import guideline_rows_from_documents, insert_rows, load_csv_dir, make_engine

    s = get_settings()
    engine = make_engine(s.database_admin_url)
    if from_corpus and s.documents_path.exists():
        docs = load_processed_documents(s.documents_path)
        n = insert_rows(engine, "guidelines", guideline_rows_from_documents(docs))
        console.print(f"guidelines (from corpus): {n} rows")
    for table, n in load_csv_dir(engine, s.structured_dir, replace=replace).items():
        console.print(f"{table}: {n} rows")


@db_app.command("extract")
def db_extract(
    max_chunks: int = typer.Option(2000, help="Upper bound on chunks sent to the LLM."),
    workers: int = 8,
) -> None:
    """LLM-extract doses / thresholds / graded recommendations into data/structured/*.csv."""
    from clinrag.ingestion.pipeline import load_chunks
    from clinrag.llm.client import make_llm
    from clinrag.sql.extract import extract_to_csv

    s = get_settings()
    counts = extract_to_csv(
        make_llm(s), load_chunks(s.chunks_path), s.structured_dir, max_chunks, workers
    )
    console.print(counts)
    console.print("[yellow]Review the CSVs before running `clinrag db load`.[/yellow]")


@db_app.command("query")
def db_query(question: str) -> None:
    """Run only the text-to-SQL tool (debugging)."""
    from clinrag.llm.client import make_llm
    from clinrag.sql.db import make_engine
    from clinrag.sql.text_to_sql import TextToSQL

    s = get_settings()
    tool = TextToSQL(make_llm(s), make_engine(s.database_url), row_limit=s.sql_row_limit)
    result = tool.run(question)
    console.print(f"[bold]SQL[/bold] (attempts={result.attempts}): {result.sql}")
    console.print(result.to_markdown())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

DEFAULT_EVAL = Path("data/eval/eval_set.jsonl")


@eval_app.command("summary")
def eval_summary(dataset: Path = typer.Option(DEFAULT_EVAL)) -> None:
    """Validate an eval file and print its composition."""
    from clinrag.evaluation.dataset import load_eval_set, summarize

    _print_table([summarize(load_eval_set(dataset))], str(dataset))


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: Path = typer.Option(DEFAULT_EVAL),
    modes: str = typer.Option("bm25,dense,hybrid,hybrid_rerank"),
    ks: str = typer.Option("1,3,5,10"),
    level: str = typer.Option("doc", help="doc | chunk"),
    mlflow_log: bool = typer.Option(True, "--mlflow/--no-mlflow"),
) -> None:
    """Retrieval ablation: Recall@k / MRR / nDCG per retrieval mode."""
    from clinrag.evaluation.runners import run_retrieval_ablation

    report = run_retrieval_ablation(
        get_settings(),
        dataset,
        modes=tuple(m.strip() for m in modes.split(",")),
        ks=tuple(int(k) for k in ks.split(",")),
        level=level,
        use_mlflow=mlflow_log,
    )
    keep = ("recall@", "mrr", "ndcg@5", "latency_p50")
    rows = [
        {"mode": r["mode"], **{k: v for k, v in r.items() if k.startswith(keep)}}
        for r in report.table()
    ]
    _print_table(rows, f"Retrieval ({level}-level)")


@eval_app.command("rag")
def eval_rag(
    dataset: Path = typer.Option(DEFAULT_EVAL),
    prompt: list[str] = typer.Option(None, "--prompt", help="Repeat for several versions."),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Run RAGAS (LLM judge)."),
    mlflow_log: bool = typer.Option(True, "--mlflow/--no-mlflow"),
    workers: int = 4,
    category: list[str] = typer.Option(None, "--category"),
    limit: int = typer.Option(None, help="Only the first N examples (smoke tests)."),
    metrics: str = typer.Option(
        None,
        "--metrics",
        help="Comma-separated RAGAS metrics (default: faithfulness, answer_relevancy, "
        "context_precision, context_recall). `faithfulness` alone needs ~1/4 of the calls.",
    ),
) -> None:
    """Run the assistant over the eval set and score it (RAGAS + refusal metrics)."""
    from clinrag.evaluation.ragas_eval import METRICS
    from clinrag.evaluation.runners import run_prompt_evals

    s = get_settings()
    versions = prompt or [s.prompt_version]
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
    unknown = sorted(set(metric_names or []) - set(METRICS))
    if unknown:
        raise typer.BadParameter(f"unknown metrics {unknown}; choose from {list(METRICS)}")
    results = run_prompt_evals(
        s,
        dataset,
        versions,
        judge=judge,
        use_mlflow=mlflow_log,
        workers=workers,
        categories=category,
        limit=limit,
        metric_names=metric_names,
    )
    keys = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "refusal_rate_out_of_corpus",
        "false_refusal_rate",
        "behavior_accuracy",
    )
    rows = [{"prompt": r.prompt_version, **{k: r.metrics.get(k) for k in keys}} for r in results]
    _print_table(rows, "RAG evaluation")
    for r in results:
        console.print(f"[dim]{r.prompt_version}: {r.output_dir}[/dim]")


@eval_app.command("build-testset")
def eval_build_testset(
    out: Path = typer.Option(Path("data/eval/eval_set.draft.jsonl")),
    model: str = typer.Option(None, help="Generator model (defaults to EVAL_JUDGE_MODEL)."),
    seed: int = 13,
) -> None:
    """Draft a 100-question eval set from the corpus for human review."""
    from clinrag.evaluation.testset_builder import TestsetSpec, build_testset
    from clinrag.ingestion.pipeline import load_chunks, load_processed_documents
    from clinrag.llm.client import make_llm
    from clinrag.schemas import write_jsonl

    s = get_settings()
    rows = build_testset(
        make_llm(s, model=model or s.eval_judge_model),
        load_chunks(s.chunks_path),
        load_processed_documents(s.documents_path),
        TestsetSpec(seed=seed),
    )
    write_jsonl(out, rows)
    console.print(f"[green]Wrote {len(rows)} draft questions to {out}[/green]")
    console.print("Review every row, then save it as data/eval/eval_set.jsonl.")


@eval_app.command("sync-langsmith")
def eval_sync_langsmith(
    dataset: Path = typer.Option(DEFAULT_EVAL),
    name: str = typer.Option("clinical-guideline-rag-eval"),
) -> None:
    """Upload the eval set as a LangSmith dataset."""
    from clinrag.evaluation.dataset import load_eval_set
    from clinrag.evaluation.tracking import sync_dataset

    examples = [e.model_dump() for e in load_eval_set(dataset)]
    dataset_id = sync_dataset(name, examples, description="Clinical guideline RAG eval set")
    console.print(
        f"[green]Synced {len(examples)} examples[/green] to LangSmith dataset {dataset_id}"
    )


@eval_app.command("compare")
def eval_compare(
    experiment: str = typer.Option("rag", help="rag | retrieval"),
) -> None:
    """Compare logged MLflow runs."""
    from clinrag.evaluation.runners import RAG_EXPERIMENT, RETRIEVAL_EXPERIMENT, compare_runs

    name = RAG_EXPERIMENT if experiment == "rag" else RETRIEVAL_EXPERIMENT
    rows = compare_runs(get_settings(), name)
    wanted = (
        (
            "run",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "refusal_rate_out_of_corpus",
            "false_refusal_rate",
        )
        if experiment == "rag"
        else ("run", "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5")
    )
    _print_table([{k: r.get(k) for k in wanted} for r in rows], name)


@eval_app.command("export")
def eval_export(run_dir: Path, out: Path = typer.Option(None)) -> None:
    """Export a run's per-question records to CSV for manual review."""
    import csv

    with (run_dir / "records.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    out = out or run_dir / "records.csv"
    cols = ["id", "category", "expected_behavior", "refused", "question", "answer", "citations"]
    score_cols = sorted({k for r in records for k in r.get("scores", {})})
    # utf-8-sig so Excel on Windows shows non-ASCII text correctly.
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols + score_cols)
        for r in records:
            w.writerow([r.get(c) for c in cols] + [r.get("scores", {}).get(s) for s in score_cols])
    console.print(f"[green]{out}[/green]")


if __name__ == "__main__":
    app()

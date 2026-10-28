"""raw files -> documents.jsonl -> chunks.jsonl"""

from __future__ import annotations

import logging
from pathlib import Path

from clinrag.config import Settings
from clinrag.ingestion.chunking import chunk_document
from clinrag.ingestion.loaders import load_documents
from clinrag.schemas import Chunk, Document, iter_jsonl, write_jsonl

log = logging.getLogger(__name__)


def ingest_corpus(settings: Settings, raw_dir: Path | None = None) -> tuple[int, int]:
    """Load, chunk and persist the corpus. Returns (n_documents, n_chunks)."""
    raw_dir = raw_dir or settings.raw_dir
    docs = load_documents(raw_dir)
    if not docs:
        raise FileNotFoundError(
            f"No supported documents found in {raw_dir}. See data/README.md for the expected layout."
        )
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                chunk_size=settings.chunk_size_words,
                overlap=settings.chunk_overlap_words,
                min_words=settings.min_chunk_words,
            )
        )
    # Page offsets are only needed for chunking; drop them from the persisted docs.
    write_jsonl(
        settings.documents_path,
        (d.model_copy(update={"page_starts": []}) for d in docs),
    )
    write_jsonl(settings.chunks_path, chunks)
    log.info("Ingested %d documents into %d chunks", len(docs), len(chunks))
    return len(docs), len(chunks)


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `clinrag ingest` first.")
    return [Chunk(**row) for row in iter_jsonl(path)]


def load_processed_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `clinrag ingest` first.")
    return [Document(**row) for row in iter_jsonl(path)]

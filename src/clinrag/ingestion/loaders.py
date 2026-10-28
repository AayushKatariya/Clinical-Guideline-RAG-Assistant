"""Load raw guideline files (PDF / HTML / Markdown / text / JSONL) into `Document`s."""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from clinrag.schemas import Document, iter_jsonl

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".jsonl"}
METADATA_FILENAME = "metadata.csv"


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_WS = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "treat-\nment" -> "treatment"
    text = "\n".join(_INLINE_WS.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def slugify(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_len] or "doc"


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------


def read_pdf(path: Path) -> tuple[str, list[int]]:
    """Return the text of a PDF and the character offset at which each page starts."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    for page in reader.pages:
        page_text = normalize_text(page.extract_text() or "")
        page_starts.append(offset)
        parts.append(page_text)
        offset += len(page_text) + 2  # account for the "\n\n" joiner
    return "\n\n".join(parts), page_starts


def read_html(path: Path) -> str:
    """Convert HTML to light markdown so section headings survive for chunking."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    for level in range(1, 5):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n\n{'#' * level} {h.get_text(' ', strip=True)}\n\n")
    for li in soup.find_all("li"):
        li.replace_with(f"\n- {li.get_text(' ', strip=True)}")
    for p in soup.find_all(["p", "div", "tr", "br"]):
        p.insert_after("\n")
    return normalize_text(soup.get_text())


def read_text(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def _first_heading(text: str) -> str | None:
    for line in text.split("\n")[:15]:
        line = line.strip().lstrip("#").strip()
        if 3 <= len(line) <= 200:
            return line
    return None


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


def load_metadata(raw_dir: Path) -> dict[str, dict[str, str]]:
    """Optional `metadata.csv` keyed by filename (relative to raw_dir)."""
    path = raw_dir / METADATA_FILENAME
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["filename"].strip(): row for row in rows if row.get("filename")}


_PLACEHOLDERS = {"", "none", "null", "nan"}
_JSONL_METADATA = {"overview", "country", "title_source"}


def _value(row: dict, *keys: str) -> Any:
    """First real value among `keys`. Public dumps often hold the string "None" for
    missing fields; that counts as missing."""
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip().lower() not in _PLACEHOLDERS:
            return value
    return None


def _jsonl_documents(path: Path) -> Iterator[Document]:
    """JSONL rows. Accepts common field-name variants so public corpora load as-is."""
    for i, row in enumerate(iter_jsonl(path)):
        text = _value(row, "text", "clean_text", "raw_text") or ""
        text = normalize_text(text)
        if not text:
            continue
        title = _value(row, "title") or _first_heading(text) or f"{path.stem}-{i}"
        doc_id = str(_value(row, "doc_id", "id") or slugify(title))
        yield Document(
            doc_id=doc_id,
            title=title,
            text=text,
            organization=_value(row, "organization", "source"),
            year=_to_int(_value(row, "year", "publication_year")),
            specialty=_value(row, "specialty"),
            url=_value(row, "url"),
            source_path=f"{path.name}#{i}",
            metadata={k: row[k] for k in _JSONL_METADATA if _value(row, k) is not None},
        )


def iter_documents(raw_dir: Path) -> Iterator[Document]:
    metadata = load_metadata(raw_dir)
    files = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                yield from _jsonl_documents(path)
                continue
            page_starts: list[int] = []
            if suffix == ".pdf":
                text, page_starts = read_pdf(path)
            elif suffix in {".html", ".htm"}:
                text = read_html(path)
            else:
                text = read_text(path)
        except Exception as exc:  # corrupt PDFs etc. should not kill a 1k-file ingest
            log.warning("Skipping %s: %s", path, exc)
            continue
        if not text.strip():
            log.warning("Skipping %s: no extractable text (scanned PDF?)", path)
            continue

        rel = str(path.relative_to(raw_dir))
        meta = metadata.get(rel) or metadata.get(path.name) or {}
        title = meta.get("title") or _first_heading(text) or path.stem
        yield Document(
            doc_id=meta.get("doc_id") or slugify(path.stem),
            title=title,
            text=text,
            organization=meta.get("organization") or None,
            year=_to_int(meta.get("year")),
            specialty=meta.get("specialty") or None,
            url=meta.get("url") or None,
            source_path=rel,
            page_starts=page_starts,
        )


def load_documents(raw_dir: Path) -> list[Document]:
    """Load every supported file, de-duplicating doc_ids."""
    docs: list[Document] = []
    seen: dict[str, int] = {}
    for doc in iter_documents(raw_dir):
        if doc.doc_id in seen:
            seen[doc.doc_id] += 1
            doc.doc_id = f"{doc.doc_id}-{seen[doc.doc_id]}"
        else:
            seen[doc.doc_id] = 0
        docs.append(doc)
    return docs

"""Descriptive titles for corpus documents that have none, written by the LLM.

946 of the 1,200 guidelines downloaded from `epfl-llm/guidelines` (every non-NICE source)
have the string "None" as title and URL, and their cleaned text has lost its title page.
Without a title, citations read "None > Recommendations" and chunks lose the contextual
header that helps retrieval.

`generate_titles` sends the section headings and opening text of several documents per
call and asks for a short descriptive title for each. These are *not* official titles:
they are stored with `title_source = "generated:<model>"`, and one is accepted only if
most of its topic words occur in the document itself. `apply_titles` writes them into the
corpus file, always starting from an untouched backup, and gives every remaining document
a plain fallback title. Missing URLs and years stay empty: they can't be recovered from
the text.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinrag.llm.client import LLMClient, loads_json_object
from clinrag.schemas import iter_jsonl, write_jsonl

log = logging.getLogger(__name__)

OK = "ok"  # accepted
REJECTED = "rejected"  # the model's title failed validation
MISSING = "missing"  # the model returned no title (e.g. fragmentary text)
STATUSES = (OK, REJECTED, MISSING)

_PLACEHOLDERS = {"", "none", "null", "nan"}


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip().lower() in _PLACEHOLDERS


def needs_title(row: dict) -> bool:
    return is_missing(row.get("title"))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_GENERIC_HEADING = re.compile(
    r"^(?:table of )?contents$|acknowledg|abbreviation|acronym|glossary|foreword|preface|"
    r"references|bibliography|annex|appendix|disclaimer|note to users|executive summary|"
    r"^summary$|^introduction$|^background$|^methods?$|^methodology$|^scope$|^purpose$|"
    r"^objectives?$|^conclusions?$|^discussion$|^results$|^recommendations?$|^tables?$|"
    r"^figures?$|^\W*$",
    re.I,
)


def headings(text: str, max_n: int = 12) -> list[str]:
    """Distinct, non-generic markdown headings, in document order."""
    out: list[str] = []
    for line in text.split("\n"):
        if not line.startswith("#"):
            continue
        h = re.sub(r"\s+", " ", line.lstrip("#")).strip(" .:")
        if 1 <= len(h.split()) <= 15 and not _GENERIC_HEADING.search(h) and h not in out:
            out.append(h)
            if len(out) >= max_n:
                break
    return out


def opening(text: str, n_words: int = 250) -> str:
    body = " ".join(line for line in text.split("\n") if not line.startswith("#"))
    return " ".join(body.split()[:n_words])


PROMPT = """\
You are cataloguing clinical practice guidelines whose titles were lost. For each document
below you get its publisher, some of its section headings and its opening text (which may
start mid-document or with front matter).

Write a short descriptive title (4-15 words) that names the clinical topic and the kind
of guidance, for example "Prevention and treatment of postpartum haemorrhage" or
"Investigating contacts of people with infectious tuberculosis".

Rules:
- Use only what the headings and text show. Do not invent an official title, edition,
  year, country or organisation name, and do not start with the publisher's name.
- Sentence case, no quotation marks, no trailing full stop.
- If the text is too fragmentary to tell what the document is about, use null.

Return only JSON: {{"titles": [{{"id": 1, "title": "..."}}, ...]}} with one entry per
document, using the document numbers below.

{documents}
"""


def build_prompt(batch: list[dict]) -> str:
    parts = []
    for i, row in enumerate(batch, 1):
        text = row.get("text") or ""
        heads = "; ".join(headings(text)) or "(none)"
        parts.append(
            f"Document {i} (publisher: {row.get('organization') or 'unknown'})\n"
            f"Headings: {heads}\n"
            f"Opening text: {opening(text)}"
        )
    return PROMPT.format(documents="\n\n".join(parts))


def parse_titles(content: str | None) -> dict[int, str | None]:
    """{document number: title or None}. Tolerates code fences and surrounding prose."""
    items = loads_json_object(content).get("titles")
    out: dict[int, str | None] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            key = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        title = item.get("title")
        out[key] = str(title) if title is not None else None
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_STOPWORDS = set(
    "about after also among and are based before being between both care clinical "
    "consensus document during evidence first following from further general guidance "
    "guide guideline guidelines including into management more other over patients people "
    "persons practice prevention recommendation recommendations report review settings "
    "statement such summary than that their them there these this those through treatment "
    "under update using what when where which while with within without".split()
)


def _norm(text: str) -> str:
    """Lower-case alphanumerics; hyphens and apostrophes split words ("pre-eclampsia")."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def topic_words(title: str) -> list[str]:
    return [w for w in _norm(title).split() if len(w) >= 4 and w.isalpha() and w not in _STOPWORDS]


def _expanded_acronyms(title: str, text: str) -> set[str]:
    """Title words that spell out an acronym the text uses, e.g. "positron emission
    tomography" when the text says "PET". Acronyms of 3+ capitals only, so ordinary
    two-letter words in all-caps headings don't count."""
    tokens = _norm(title).split()
    out: set[str] = set()
    for i in range(len(tokens)):
        for j in range(i + 3, min(i + 6, len(tokens)) + 1):
            acronym = "".join(t[0] for t in tokens[i:j]).upper()
            if re.search(rf"\b{acronym}s?\b", text):
                out.update(tokens[i:j])
    return out


def validate_title(
    title: str | None, text: str, min_grounded: float = 0.6
) -> tuple[str | None, str | None]:
    """(cleaned title, None) if acceptable, else (None, reason)."""
    if title is None or not str(title).strip():
        return None, "no title"
    title = re.sub(r"\s+", " ", str(title)).strip().strip("\"'“”‘’").rstrip(".")
    n_words = len(title.split())
    if not 3 <= n_words <= 20:
        return None, f"{n_words} words"
    words = topic_words(title)
    if not words:
        return None, "generic"
    doc = _norm(text)
    expanded = _expanded_acronyms(title, text)
    # A plural in the title may be singular in the text, so also try the word minus "s".
    grounded = [
        w for w in words if w in doc or (w.endswith("s") and w[:-1] in doc) or w in expanded
    ]
    if len(grounded) / len(words) < min_grounded:
        missing = sorted(set(words) - set(grounded))
        return None, "not in text: " + ", ".join(missing[:5])
    return title, None


# ---------------------------------------------------------------------------
# Generation, cache, report
# ---------------------------------------------------------------------------


@dataclass
class TitleRecord:
    doc_id: str
    organization: str | None
    status: str
    title: str | None = None
    reason: str | None = None
    raw_title: str | None = None  # exactly what the model returned, for auditing
    model: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def load_cache(path: Path) -> dict[str, dict]:
    """Latest record per doc_id (the cache is an append-only log)."""
    if not path.exists():
        return {}
    return {row["doc_id"]: row for row in iter_jsonl(path)}


def select_documents(
    source: Path, cache: dict[str, dict], limit: int | None = None, retry: bool = False
) -> list[dict]:
    todo = []
    for row in iter_jsonl(source):
        if not needs_title(row):
            continue
        prev = cache.get(row["doc_id"])
        if prev and (prev["status"] == OK or not retry):
            continue
        todo.append(row)
    return todo[:limit] if limit else todo


def batches(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[i : i + size] for i in range(0, len(rows), max(1, size))]


def _append(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_titles(
    docs: list[dict],
    llm: LLMClient,
    cache_path: Path,
    batch_size: int = 20,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[TitleRecord]:
    """One LLM call per batch; each batch's records are cached as soon as it returns.

    An LLM error (e.g. the daily quota running out) stops the run; finished batches stay
    cached, so re-running continues where it stopped.
    """
    records: list[TitleRecord] = []
    groups = batches(docs, batch_size)
    for n, batch in enumerate(groups, 1):
        reply = llm.chat([{"role": "user", "content": build_prompt(batch)}])
        titles = parse_titles(reply.content)
        out = []
        for i, row in enumerate(batch, 1):
            raw = titles.get(i)
            title, reason = validate_title(raw, row.get("text") or "")
            if title:
                status = OK
            elif raw is None or not str(raw).strip():
                status, reason = MISSING, None if i in titles else "not in reply"
            else:
                status = REJECTED
            out.append(
                TitleRecord(
                    doc_id=row["doc_id"],
                    organization=row.get("organization"),
                    status=status,
                    title=title,
                    reason=reason,
                    raw_title=raw,
                    model=llm.model,
                )
            )
        _append(cache_path, (asdict(r) for r in out))
        records.extend(out)
        if on_batch:
            on_batch(n, len(groups))
    return records


def summarize(records: Iterable[dict]) -> list[dict]:
    """One row per organisation with a count per status, plus a total row."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        counts[str(r.get("organization"))][r["status"]] += 1
    rows = [
        {"organization": org, **{s: c[s] for s in STATUSES}, "total": sum(c.values())}
        for org, c in sorted(counts.items())
    ]
    if rows:
        totals = {s: sum(r[s] for r in rows) for s in (*STATUSES, "total")}
        rows.append({"organization": "ALL", **totals})
    return rows


def write_review_csv(records: Iterable[dict], source: Path, out_path: Path) -> int:
    """Generated titles next to each document's opening words, for a quick human check."""
    starts = {row["doc_id"]: opening(row.get("text") or "", 30) for row in iter_jsonl(source)}
    rows = sorted(records, key=lambda r: (str(r.get("organization")), r["status"], r["doc_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows shows non-ASCII text correctly.
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["doc_id", "organization", "status", "title", "model_title", "reason", "opening_words"]
        )
        for r in rows:
            w.writerow(
                [
                    r["doc_id"],
                    r.get("organization"),
                    r["status"],
                    r.get("title"),
                    r.get("raw_title"),
                    r.get("reason"),
                    starts.get(r["doc_id"], ""),
                ]
            )
    return len(rows)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def fallback_title(row: dict) -> str:
    short = str(row["doc_id"]).split("-", 1)[-1][:8]
    org = row.get("organization")
    return f"{org} guideline {short}" if not is_missing(org) else f"Guideline {short}"


def apply_titles(raw_path: Path, backup_path: Path, cache_path: Path) -> Counter:
    """Rewrite the corpus file with accepted generated titles.

    Starts from the untouched original (copied to `backup_path` on first use), so it can
    be re-run after more titles are generated. Placeholder strings like "None" become
    null, and documents still without a title get a plain fallback title.
    """
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_path, backup_path)
    cache = load_cache(cache_path)
    stats: Counter = Counter()
    rows = []
    for row in iter_jsonl(backup_path):
        row = dict(row)
        for key in ("title", "url", "organization", "year", "specialty"):
            if key in row and is_missing(row[key]):
                row[key] = None
        if row.get("title"):
            row["title_source"] = "dataset"
            stats["dataset_titles"] += 1
        elif (rec := cache.get(row["doc_id"])) and rec["status"] == OK and rec.get("title"):
            row["title"], row["title_source"] = rec["title"], f"generated:{rec.get('model')}"
            stats["generated_titles"] += 1
        else:
            row["title"], row["title_source"] = fallback_title(row), "fallback"
            stats["fallback_titles"] += 1
        rows.append(row)
    stats["documents"] = write_jsonl(raw_path, rows)
    return stats

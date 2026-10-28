"""Section-aware chunking for clinical guidelines.

Guidelines are long, deeply sectioned documents where the *section* is often what makes a
passage answerable ("Recommendations > Pharmacological management > Adults"). The chunker:

1. walks the document line by line and tracks a heading stack (markdown `#`, numbered
   headings like `1.2.3 Title`, and short ALL-CAPS lines);
2. splits each section into sentence units, keeping paragraph boundaries;
3. packs units into windows of ~`chunk_size` words with a sentence-aligned overlap, never
   crossing a section boundary;
4. merges sections that are too short to stand alone into the next section.

Each chunk keeps its section path and page range so answers can cite them.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from clinrag.schemas import Chunk, Document

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUM_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+([A-Z][^\n]{1,118})$")
_DOT_LEADER = re.compile(r"\.{3,}|…")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[•\-–A-Z0-9])")
_ABBREV_END = re.compile(r"\b(?:e\.g|i\.e|vs|approx|Fig|Ref|Dr|No|et al)\.$", re.IGNORECASE)


@dataclass
class _Unit:
    text: str
    start: int
    end: int
    para: int

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class _Section:
    path: list[str]
    units: list[_Unit] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(u.words for u in self.units)


def detect_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) if `line` looks like a section heading."""
    s = line.strip()
    if not s or len(s) > 140 or _DOT_LEADER.search(s):
        return None
    m = _MD_HEADING.match(s)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _NUM_HEADING.match(s)
    if m and not s.endswith((".", ",", ";", ":")) and len(s.split()) <= 15:
        return m.group(1).count(".") + 1, f"{m.group(1)} {m.group(2).strip()}"
    letters = [c for c in s if c.isalpha()]
    if (
        len(letters) >= 4
        and sum(c.isupper() for c in letters) / len(letters) > 0.9
        and len(s.split()) <= 10
        and not s.endswith(".")
    ):
        return 2, s.title()
    return None


def _split_sentences(text: str, start: int, para: int) -> list[_Unit]:
    pieces = _SENTENCE_SPLIT.split(text)
    # Re-join splits that happened right after an abbreviation ("e.g. Metformin").
    merged: list[str] = []
    for piece in pieces:
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    units, cursor = [], 0
    for piece in merged:
        idx = text.find(piece[:30], cursor)
        idx = cursor if idx < 0 else idx
        units.append(_Unit(piece.strip(), start + idx, start + idx + len(piece), para))
        cursor = idx + len(piece)
    return [u for u in units if u.text]


def _split_long_unit(unit: _Unit, max_words: int) -> list[_Unit]:
    words = unit.text.split()
    if len(words) <= max_words:
        return [unit]
    out = []
    for i in range(0, len(words), max_words):
        out.append(_Unit(" ".join(words[i : i + max_words]), unit.start, unit.end, unit.para))
    return out


def split_into_sections(text: str) -> list[_Section]:
    sections: list[_Section] = [_Section(path=[])]
    stack: list[tuple[int, str]] = []
    para_lines: list[str] = []
    para_start = 0
    para_idx = 0
    offset = 0

    def flush_paragraph() -> None:
        nonlocal para_lines, para_idx
        if para_lines:
            para_text = " ".join(para_lines)
            sections[-1].units.extend(_split_sentences(para_text, para_start, para_idx))
            para_idx += 1
        para_lines = []

    for line in text.split("\n"):
        heading = detect_heading(line)
        if heading:
            flush_paragraph()
            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section(path=[t for _, t in stack]))
        elif not line.strip():
            flush_paragraph()
        else:
            if not para_lines:
                para_start = offset
            para_lines.append(line.strip())
        offset += len(line) + 1
    flush_paragraph()
    return [s for s in sections if s.units]


def _merge_small_sections(sections: list[_Section], min_words: int) -> list[_Section]:
    merged: list[_Section] = []
    pending: list[_Unit] = []
    for sec in sections:
        units = pending + sec.units
        if sum(u.words for u in units) < min_words:
            pending = units
            continue
        merged.append(_Section(path=sec.path, units=units))
        pending = []
    if pending:
        if merged:
            merged[-1].units.extend(pending)
        else:
            merged.append(_Section(path=sections[-1].path if sections else [], units=pending))
    return merged


def _join_units(units: list[_Unit]) -> str:
    out: list[str] = []
    prev_para = None
    for u in units:
        if prev_para is not None:
            out.append("\n\n" if u.para != prev_para else " ")
        out.append(u.text)
        prev_para = u.para
    return "".join(out)


def _page_of(offset: int, page_starts: list[int]) -> int | None:
    if not page_starts:
        return None
    return bisect.bisect_right(page_starts, offset)  # 1-indexed page number


def chunk_document(
    doc: Document,
    chunk_size: int = 350,
    overlap: int = 60,
    min_words: int = 40,
    max_section_depth: int = 3,
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    sections = _merge_small_sections(split_into_sections(doc.text), min_words)

    chunks: list[Chunk] = []

    def emit(units: list[_Unit], path: list[str]) -> None:
        text = _join_units(units)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{len(chunks):04d}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=" > ".join(path[-max_section_depth:]),
                text=text,
                position=len(chunks),
                page_start=_page_of(units[0].start, doc.page_starts),
                page_end=_page_of(max(units[-1].end - 1, units[0].start), doc.page_starts),
                organization=doc.organization,
                year=doc.year,
                specialty=doc.specialty,
            )
        )

    for sec in sections:
        units = [piece for u in sec.units for piece in _split_long_unit(u, chunk_size)]
        window: list[_Unit] = []
        window_words = 0
        fresh = 0  # units added since the last emit (excludes the carried-over overlap)
        for unit in units:
            if window and window_words + unit.words > chunk_size:
                emit(window, sec.path)
                tail: list[_Unit] = []
                tail_words = 0
                for prev in reversed(window):
                    if tail_words + prev.words > overlap:
                        break
                    tail.insert(0, prev)
                    tail_words += prev.words
                window, window_words, fresh = tail, tail_words, 0
            window.append(unit)
            window_words += unit.words
            fresh += 1
        if window and fresh:
            emit(window, sec.path)
    return chunks

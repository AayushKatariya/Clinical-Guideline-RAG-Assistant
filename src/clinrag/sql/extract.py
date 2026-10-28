"""LLM-assisted extraction of structured rows (doses, thresholds, graded recommendations)
from guideline chunks into CSVs that `clinrag db load` imports.

Output is meant to be *reviewed* before loading: open the CSVs, spot-check rows against
the cited chunk, delete bad rows. Each row keeps `source_chunk_id` in the notes so that
review is quick.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ValidationError

from clinrag.llm.client import LLMClient, loads_json_object
from clinrag.schemas import Chunk

log = logging.getLogger(__name__)

_DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|micrograms?|g|units?|IU|mL|mmol)(?:/kg)?\b", re.I
)
_THRESHOLD = re.compile(r"(?:≥|>=|≤|<=|<|>)\s?\d+(?:\.\d+)?\s?(?:%|mmol|mg|mmHg|kg)", re.I)
_RECO = re.compile(
    r"\b(?:strong|conditional|weak) recommendation\b|\bclass (?:I|IIa|IIb|III)\b"
    r"|\bgrade [A-D]\b|\b(?:high|moderate|low|very low)[- ](?:certainty|quality) evidence\b",
    re.I,
)


class DosingRow(BaseModel):
    drug_name: str
    indication: str | None = None
    population: str | None = None
    route: str | None = None
    dose: str | None = None
    frequency: str | None = None
    duration: str | None = None
    max_daily_dose: str | None = None
    renal_adjustment: str | None = None


class ThresholdRow(BaseModel):
    condition: str
    test_name: str
    comparator: str | None = None
    value: float | None = None
    unit: str | None = None
    population: str | None = None
    interpretation: str | None = None


class RecommendationRow(BaseModel):
    condition: str | None = None
    population: str | None = None
    intervention: str | None = None
    recommendation_text: str
    strength: str | None = None
    evidence_level: str | None = None


class Extraction(BaseModel):
    drug_dosing: list[DosingRow] = []
    diagnostic_thresholds: list[ThresholdRow] = []
    recommendations: list[RecommendationRow] = []


PROMPT = """\
Extract structured facts from this clinical guideline passage. Only extract what the
passage states explicitly; copy doses, units and wording exactly; never infer.
Return JSON with keys "drug_dosing", "diagnostic_thresholds", "recommendations"
(empty lists when absent).

drug_dosing items: drug_name (generic, lower case), indication, population, route, dose,
frequency, duration, max_daily_dose, renal_adjustment.
diagnostic_thresholds items: condition, test_name, comparator (<, <=, =, >=, >), value
(number), unit, population, interpretation.
recommendations items (only graded/explicit recommendations): condition, population,
intervention, recommendation_text (verbatim), strength, evidence_level.

Guideline: {title}
Section: {section}
Passage:
{text}
"""


def is_candidate(chunk: Chunk) -> bool:
    t = chunk.text
    return bool(_DOSE.search(t) or _THRESHOLD.search(t) or _RECO.search(t))


def extract_chunk(llm: LLMClient, chunk: Chunk) -> Extraction:
    prompt = PROMPT.format(title=chunk.title, section=chunk.section or "-", text=chunk.text)
    # Temperature comes from the client's configured default (Gemini 3+ wants 1.0).
    reply = llm.chat([{"role": "user", "content": prompt}], json_mode=True)
    try:
        return Extraction.model_validate(loads_json_object(reply.content))
    except ValidationError as exc:
        log.debug("Extraction failed for %s: %s", chunk.chunk_id, exc)
        return Extraction()


def extract_to_csv(
    llm: LLMClient,
    chunks: list[Chunk],
    out_dir: Path,
    max_chunks: int = 2000,
    workers: int = 8,
    seed: int = 13,
) -> dict[str, int]:
    """Extract rows from candidate chunks into CSVs.

    When `max_chunks` caps the run, a seeded random sample is taken across the whole
    corpus (in corpus order), so a small free-tier run still covers many guidelines
    instead of the first few.
    """
    candidates = [c for c in chunks if is_candidate(c)]
    if len(candidates) > max_chunks:
        keep = set(random.Random(seed).sample(range(len(candidates)), max_chunks))
        candidates = [c for i, c in enumerate(candidates) if i in keep]
    log.info("Extracting from %d candidate chunks", len(candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: extract_chunk(llm, c), candidates))

    tables: dict[str, list[dict]] = {
        "drug_dosing": [],
        "diagnostic_thresholds": [],
        "recommendations": [],
    }
    for chunk, ext in zip(candidates, results, strict=True):
        for row in ext.drug_dosing:
            tables["drug_dosing"].append(
                {"guideline_id": chunk.doc_id, **row.model_dump(), "notes": chunk.chunk_id}
            )
        for row in ext.diagnostic_thresholds:
            tables["diagnostic_thresholds"].append(
                {"guideline_id": chunk.doc_id, **row.model_dump()}
            )
        for i, row in enumerate(ext.recommendations):
            tables["recommendations"].append(
                {
                    "recommendation_id": f"{chunk.chunk_id}-r{i}",
                    "guideline_id": chunk.doc_id,
                    "section": chunk.section,
                    **row.model_dump(),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        if not rows:
            continue
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        counts[name] = len(rows)
    return counts

"""Build a guideline corpus from the open-access subset of `epfl-llm/guidelines`
(the clinical guideline corpus released with Meditron) on the Hugging Face Hub.

    pip install -e ".[data]"
    python scripts/download_guidelines.py --n 1200

Writes data/raw/guidelines.jsonl, which `clinrag ingest` reads directly. Check each source's
licence/terms before redistributing anything; the corpus is git-ignored by default.

You can skip this script entirely and drop your own PDFs / HTML / markdown files into
data/raw/ instead (optionally with data/raw/metadata.csv).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ORG_NAMES = {
    "cco": "CCO",
    "cdc": "CDC",
    "cma": "CMA",
    "icrc": "ICRC",
    "nice": "NICE",
    "spor": "SPOR",
    "who": "WHO",
    "wikidoc": "WikiDoc",
}
# WikiDoc is an encyclopaedia rather than a guideline publisher, so it is excluded by default.
DEFAULT_SOURCES = ["cco", "cdc", "cma", "icrc", "nice", "spor", "who"]


def first(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="number of guidelines to keep")
    ap.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/raw/guidelines.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("epfl-llm/guidelines", split="train")
    print(f"Loaded {len(ds)} rows; columns: {ds.column_names}")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        source = first(row, "source", "organization").lower()
        if args.sources and source not in args.sources:
            continue
        text = first(row, "clean_text", "text", "raw_text")
        n_words = len(text.split())
        if not (args.min_words <= n_words <= args.max_words):
            continue
        by_source[source].append(row)
    print("Eligible per source:", {k: len(v) for k, v in by_source.items()})

    # Stratified sample so no single publisher dominates the corpus.
    rng = random.Random(args.seed)
    for rows in by_source.values():
        rng.shuffle(rows)
    picked: list[dict] = []
    pools = {s: list(rows) for s, rows in by_source.items()}
    while len(picked) < args.n and any(pools.values()):
        for source in sorted(pools):
            if pools[source] and len(picked) < args.n:
                picked.append(pools[source].pop())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(picked):
            source = first(row, "source", "organization").lower()
            title = first(row, "title") or f"{source} guideline {i}"
            base = first(row, "id") or re.sub(r"[^a-z0-9]+", "-", title.lower())[:60]
            doc_id = f"{source}-{base}"[:90]
            while doc_id in seen:
                doc_id += "-x"
            seen.add(doc_id)
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": first(row, "clean_text", "text", "raw_text"),
                        "organization": ORG_NAMES.get(source, source.upper()),
                        "url": first(row, "url"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(picked)} guidelines to {args.out}")
    print("By organisation:", Counter(ORG_NAMES.get(first(r, "source").lower()) for r in picked))


if __name__ == "__main__":
    main()

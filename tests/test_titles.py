"""LLM-written titles for untitled corpus documents (offline, ScriptedLLM)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from clinrag.cli import app as cli_app
from clinrag.ingestion.loaders import load_documents
from clinrag.ingestion.titles import (
    MISSING,
    OK,
    REJECTED,
    apply_titles,
    build_prompt,
    generate_titles,
    headings,
    load_cache,
    parse_titles,
    select_documents,
    summarize,
    validate_title,
    write_review_csv,
)

from fakes import ScriptedLLM

MALARIA = (
    "# Contents\n\n# Acknowledgements\nThanks to the guideline group.\n\n"
    "# Severe malaria in adults\nSevere malaria should be treated with intravenous "
    "artesunate for at least 24 hours.\n\n# Malaria in pregnancy\nPregnant women with "
    "uncomplicated malaria should receive artemisinin-based combination therapy."
)
TB = "# Contact investigation\nHousehold contacts of people with infectious tuberculosis "
TB += "should be screened for active disease."
FLU = "# Influenza vaccination\nAnnual influenza vaccination is recommended for adults."


def doc(doc_id: str, text: str, title: str = "None", org: str = "WHO") -> dict:
    return {"doc_id": doc_id, "title": title, "url": "None", "organization": org, "text": text}


def reply(*titles) -> str:
    items = [{"id": i, "title": t} for i, t in enumerate(titles, 1)]
    return "```json\n" + json.dumps({"titles": items}) + "\n```"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------- prompt


def test_headings_skip_generic_sections():
    assert headings(MALARIA) == ["Severe malaria in adults", "Malaria in pregnancy"]


def test_prompt_numbers_documents_with_publisher_and_text():
    prompt = build_prompt([doc("who-1", MALARIA), doc("cdc-2", FLU, org="CDC")])
    assert "Document 1 (publisher: WHO)" in prompt and "Document 2 (publisher: CDC)" in prompt
    assert "Headings: Severe malaria in adults; Malaria in pregnancy" in prompt
    assert "intravenous artesunate" in prompt and '{"titles": [{"id": 1' in prompt
    assert "who-1" not in prompt  # documents are referred to by number, not hash ids


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (reply("A title", None), {1: "A title", 2: None}),
        ('Sure: {"titles": [{"id": "2", "title": "B"}]} done', {2: "B"}),
        ('{"titles": [{"title": "no id"}, "junk", {"id": 3, "title": "C"}]}', {3: "C"}),
        ("not json", {}),
        (None, {}),
    ],
)
def test_parse_titles(content, expected):
    assert parse_titles(content) == expected


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("title", "accepted", "reason"),
    [
        ('"Treatment of severe malaria in adults and pregnancy."', True, None),
        ("Management of diabetic ketoacidosis in children", False, "not in text"),
        ("Clinical practice guideline update", False, "generic"),
        ("Malaria", False, "1 words"),
        ("", False, "no title"),
    ],
)
def test_validate_title(title, accepted, reason):
    cleaned, why = validate_title(title, MALARIA)
    if accepted:
        assert cleaned == "Treatment of severe malaria in adults and pregnancy" and why is None
    else:
        assert cleaned is None and why.startswith(reason)


@pytest.mark.parametrize(
    ("title", "accepted"),
    [
        # hyphenated words are compared word by word, like the text
        ("Prevention of pre-eclampsia and COVID-19 in pregnancy", True),
        # an expansion of an acronym the text uses ("PET", "MRI") is grounded
        ("Positron emission tomography for staging oesophageal cancer", True),
        ("Breast cancer screening with magnetic resonance imaging", True),
        # ...but a two-letter all-caps word ("IN") does not ground "insulin needles"
        ("Insulin needles for oesophageal cancer", False),
    ],
)
def test_validate_title_hyphens_and_acronyms(title, accepted):
    text = (
        "# RECOMMENDATIONS IN PREGNANCY\nPre-eclampsia and COVID-19 risk in pregnancy. "
        "PET is used for staging oesophageal cancer. Breast cancer screening with MRI."
    )
    assert (validate_title(title, text)[0] is not None) is accepted


# ---------------------------------------------------------------- generation


def test_generate_titles_batches_validates_and_resumes(tmp_path):
    raw = tmp_path / "raw" / "guidelines.jsonl"
    rows = [
        doc("nice-0", "Text.", title="Asthma in adults", org="NICE"),
        doc("who-1", MALARIA),
        doc("who-2", TB),
        doc("cdc-3", FLU, org="CDC"),
    ]
    _write(raw, rows)
    cache_path = tmp_path / "metadata" / "generated_titles.jsonl"
    llm = ScriptedLLM(
        [
            reply("Treatment of severe malaria in adults", "Hip replacement surgery outcomes"),
            reply(None),
        ]
    )
    todo = select_documents(raw, load_cache(cache_path))
    assert [d["doc_id"] for d in todo] == ["who-1", "who-2", "cdc-3"]  # NICE has a title

    progress = []
    records = generate_titles(
        todo, llm, cache_path, batch_size=2, on_batch=lambda n, t: progress.append((n, t))
    )
    assert len(llm.calls) == 2 and progress == [(1, 2), (2, 2)]
    assert [(r.doc_id, r.status) for r in records] == [
        ("who-1", OK),
        ("who-2", REJECTED),
        ("cdc-3", MISSING),
    ]
    assert records[0].model == "scripted" and records[1].reason.startswith("not in text")
    assert records[1].raw_title == "Hip replacement surgery outcomes"

    cache = load_cache(cache_path)
    assert select_documents(raw, cache) == []  # resume: nothing left
    assert [d["doc_id"] for d in select_documents(raw, cache, retry=True)] == ["who-2", "cdc-3"]
    assert summarize(cache.values())[-1] == {
        "organization": "ALL",
        "ok": 1,
        "rejected": 1,
        "missing": 1,
        "total": 3,
    }
    review = tmp_path / "metadata" / "titles_review.csv"
    assert write_review_csv(cache.values(), raw, review) == 3
    assert "Household contacts" in review.read_text(encoding="utf-8")


def test_llm_error_keeps_finished_batches(tmp_path):
    cache_path = tmp_path / "titles.jsonl"
    calls = []

    def script(messages):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("429 quota exceeded")
        return reply("Treatment of severe malaria in adults")

    with pytest.raises(RuntimeError, match="quota"):
        generate_titles(
            [doc("who-1", MALARIA), doc("who-2", TB)], ScriptedLLM(script), cache_path, 1
        )
    assert list(load_cache(cache_path)) == ["who-1"]


# ---------------------------------------------------------------- apply


def test_apply_titles(tmp_path):
    raw = tmp_path / "raw" / "guidelines.jsonl"
    _write(
        raw,
        [
            doc("nice-1", "Text.", title="Asthma in adults", org="NICE"),
            doc("who-aaa", MALARIA),
            doc("who-bbbbbbbbbb", TB),
        ],
    )
    cache_path = tmp_path / "metadata" / "generated_titles.jsonl"
    _write(
        cache_path,
        [
            {"doc_id": "who-aaa", "status": OK, "title": "Severe malaria", "model": "gem"},
            {"doc_id": "who-bbbbbbbbbb", "status": REJECTED, "title": None, "model": "gem"},
        ],
    )
    backup = tmp_path / "metadata" / "guidelines.orig.jsonl"

    stats = apply_titles(raw, backup, cache_path)
    assert backup.exists()
    assert dict(stats) == {
        "dataset_titles": 1,
        "generated_titles": 1,
        "fallback_titles": 1,
        "documents": 3,
    }
    first = raw.read_text(encoding="utf-8")
    apply_titles(raw, backup, cache_path)  # re-running starts from the backup
    assert raw.read_text(encoding="utf-8") == first

    docs = {d.doc_id: d for d in load_documents(raw.parent)}  # backup is outside raw/
    assert set(docs) == {"nice-1", "who-aaa", "who-bbbbbbbbbb"}
    assert (docs["who-aaa"].title, docs["who-aaa"].url) == ("Severe malaria", None)
    assert docs["who-aaa"].metadata["title_source"] == "generated:gem"
    assert docs["who-bbbbbbbbbb"].title == "WHO guideline bbbbbbbb"
    assert docs["who-bbbbbbbbbb"].metadata["title_source"] == "fallback"
    assert docs["nice-1"].metadata["title_source"] == "dataset"


def test_cli_lists_metadata_commands():
    out = CliRunner().invoke(cli_app, ["metadata", "--help"])
    assert out.exit_code == 0
    assert all(cmd in out.output for cmd in ("titles", "report", "apply"))

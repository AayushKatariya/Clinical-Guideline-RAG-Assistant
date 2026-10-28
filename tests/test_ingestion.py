from __future__ import annotations

import json

import pytest

from clinrag.ingestion.chunking import chunk_document, detect_heading
from clinrag.ingestion.loaders import load_documents, normalize_text
from clinrag.schemas import Document


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("# Hypertension", (1, "Hypertension")),
        ("### 1.2 Treatment", (3, "1.2 Treatment")),
        ("1.2 Treatment of adults", (2, "1.2 Treatment of adults")),
        ("1.2.3 Pharmacological options", (3, "1.2.3 Pharmacological options")),
        ("RECOMMENDATIONS", (2, "Recommendations")),
        # numbered recommendation sentences are body text, not headings
        ("1.2.1 Offer a calcium-channel blocker to adults aged 55 and over.", None),
        ("Offer metformin as first-line treatment.", None),
        ("1.2 Treatment ........................ 14", None),  # table of contents
        ("", None),
    ],
)
def test_detect_heading(line, expected):
    assert detect_heading(line) == expected


def test_normalize_text_fixes_pdf_artifacts():
    raw = "Offer treat-\nment  to adults.\r\n\n\n\nNext   paragraph."
    assert normalize_text(raw) == "Offer treatment to adults.\n\nNext paragraph."


def test_chunks_respect_size_sections_and_ids(documents):
    doc = documents[0]
    chunks = chunk_document(doc, chunk_size=30, overlap=8, min_words=5)
    assert [c.chunk_id for c in chunks] == [f"{doc.doc_id}::c{i:04d}" for i in range(len(chunks))]
    assert all(len(c.text.split()) <= 30 for c in chunks)
    assert all(c.section.startswith("Hypertension in adults >") for c in chunks)
    sections = {c.section.split(" > ")[-1] for c in chunks}
    assert sections == {
        "1.1 Diagnosing hypertension",
        "1.4 Choosing antihypertensive drug treatment",
    }
    assert all(c.organization == "NICE" and c.year == 2019 for c in chunks)


def test_chunks_never_cross_sections(documents):
    chunks = chunk_document(documents[2], chunk_size=200, overlap=20, min_words=5)
    assert len(chunks) == 2  # one per section even though both would fit in one window
    assert "severe" not in chunks[0].text.lower()


def test_sentence_aligned_overlap():
    text = "## Section\n" + " ".join(f"Sentence number {i} is short." for i in range(20))
    doc = Document(doc_id="d", title="T", text=text)
    chunks = chunk_document(doc, chunk_size=25, overlap=10, min_words=1)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_sentence = a.text.split(". ")[-1].rstrip(".")
        assert last_sentence in b.text  # the tail of each chunk opens the next one


def test_small_sections_are_merged_forward():
    text = "# Title\n\n## Intro\nShort.\n\n## Body\n" + "Long sentence about dosing. " * 20
    chunks = chunk_document(Document(doc_id="d", title="T", text=text), 200, 20, min_words=10)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Short.")


def test_page_numbers_are_tracked():
    page1, page2 = "## Section A\n" + "alpha " * 30, "## Section B\n" + "beta " * 30
    text = f"{page1}\n\n{page2}"
    doc = Document(doc_id="d", title="T", text=text, page_starts=[0, len(page1) + 2])
    chunks = chunk_document(doc, chunk_size=100, overlap=10, min_words=5)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2)]
    assert "p. 2" in chunks[1].citation()


def test_load_documents_from_mixed_sources(tmp_path):
    (tmp_path / "asthma.md").write_text("# Asthma management\n\nUse a reliever inhaler.")
    (tmp_path / "copd.html").write_text(
        "<html><body><nav>menu</nav><h1>COPD</h1><p>Offer pulmonary rehabilitation.</p>"
        "<ul><li>Stop smoking</li></ul><script>x()</script></body></html>"
    )
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"id": "who-1", "title": "WHO TB", "clean_text": "Text.", "source": "WHO"})
        + "\n"
    )
    (tmp_path / "metadata.csv").write_text(
        "filename,doc_id,title,organization,year,specialty,url\n"
        "asthma.md,gina-asthma,GINA asthma strategy,GINA,2024,respiratory,\n"
    )
    docs = {d.doc_id: d for d in load_documents(tmp_path)}
    assert set(docs) == {"gina-asthma", "copd", "who-1"}
    assert docs["gina-asthma"].organization == "GINA" and docs["gina-asthma"].year == 2024
    assert docs["copd"].title == "COPD"
    assert "# COPD" in docs["copd"].text and "- Stop smoking" in docs["copd"].text
    assert "menu" not in docs["copd"].text and "x()" not in docs["copd"].text
    assert docs["who-1"].organization == "WHO"


def test_jsonl_placeholder_strings_count_as_missing(tmp_path):
    row = {"doc_id": "who-1", "title": "None", "url": "None", "year": "nan", "organization": "WHO"}
    row["text"] = "## Tuberculosis contacts\n\nScreen all household contacts."
    (tmp_path / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (doc,) = load_documents(tmp_path)
    assert doc.title == "Tuberculosis contacts"  # falls back to the first heading
    assert doc.url is None and doc.year is None and doc.organization == "WHO"


def test_duplicate_doc_ids_are_suffixed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "guide.md").write_text("# One\n\ntext")
    (tmp_path / "b" / "guide.md").write_text("# Two\n\ntext")
    assert sorted(d.doc_id for d in load_documents(tmp_path)) == ["guide", "guide-1"]

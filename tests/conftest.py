from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent))

from clinrag.config import PROJECT_ROOT  # noqa: E402
from clinrag.ingestion.chunking import chunk_document  # noqa: E402
from clinrag.retrieval import BM25Index, DenseIndex, HybridRetriever, RetrievalConfig  # noqa: E402
from clinrag.schemas import Document  # noqa: E402

from fakes import HashingEmbedder, OverlapReranker  # noqa: E402

PROMPTS_DIR = PROJECT_ROOT / "prompts"


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            doc_id="nice-hypertension",
            title="Hypertension in adults: diagnosis and management",
            organization="NICE",
            year=2019,
            specialty="cardiology",
            text=(
                "# Hypertension in adults\n\n"
                "## 1.1 Diagnosing hypertension\n"
                "If the clinic blood pressure is between 140/90 mmHg and 180/120 mmHg, offer "
                "ambulatory blood pressure monitoring (ABPM) to confirm the diagnosis of "
                "hypertension. If ABPM is unsuitable or the person is unable to tolerate it, "
                "offer home blood pressure monitoring (HBPM) to confirm the diagnosis.\n\n"
                "## 1.4 Choosing antihypertensive drug treatment\n"
                "Offer an ACE inhibitor or an angiotensin II receptor blocker (ARB) to adults "
                "with type 2 diabetes starting step 1 antihypertensive treatment. Offer a "
                "calcium-channel blocker (CCB) to adults aged 55 and over without type 2 "
                "diabetes. Do not combine an ACE inhibitor with an ARB to treat hypertension."
            ),
        ),
        Document(
            doc_id="who-tuberculosis",
            title="WHO consolidated guidelines on tuberculosis: treatment",
            organization="WHO",
            year=2022,
            specialty="infectious disease",
            text=(
                "## Drug-susceptible tuberculosis\n"
                "People with drug-susceptible pulmonary tuberculosis should receive the "
                "6-month regimen of isoniazid, rifampicin, pyrazinamide and ethambutol for two "
                "months followed by isoniazid and rifampicin for four months (2HRZE/4HR). "
                "Treatment should be supported by patient-centred care and adherence support."
            ),
        ),
        Document(
            doc_id="who-malaria",
            title="WHO guidelines for malaria",
            organization="WHO",
            year=2023,
            specialty="infectious disease",
            text=(
                "## Treating uncomplicated malaria\n"
                "Treat children and adults with uncomplicated Plasmodium falciparum malaria "
                "with an artemisinin-based combination therapy (ACT) for 3 days.\n\n"
                "## Treating severe malaria\n"
                "Treat adults and children with severe malaria with intravenous or "
                "intramuscular artesunate for at least 24 hours and until they can tolerate "
                "oral medication, then complete treatment with 3 days of an ACT."
            ),
        ),
        Document(
            doc_id="ada-diabetes",
            title="Standards of Care in Diabetes: classification and diagnosis",
            organization="ADA",
            year=2024,
            specialty="endocrinology",
            text=(
                "## Diagnostic criteria\n"
                "Diabetes may be diagnosed with an HbA1c of 6.5% or higher, a fasting plasma "
                "glucose of 126 mg/dL or higher, or a 2-hour plasma glucose of 200 mg/dL or "
                "higher during an oral glucose tolerance test. In the absence of unequivocal "
                "hyperglycaemia, diagnosis requires two abnormal test results."
            ),
        ),
    ]


@pytest.fixture
def chunks(documents):
    out = []
    for d in documents:
        out.extend(chunk_document(d, chunk_size=60, overlap=15, min_words=5))
    return out


@pytest.fixture
def retriever(chunks, tmp_path) -> HybridRetriever:
    embedder = HashingEmbedder()
    HybridRetriever.build_indexes(chunks, embedder, tmp_path)
    return HybridRetriever(
        chunks=chunks,
        bm25=BM25Index.load(tmp_path),
        dense=DenseIndex.load(tmp_path),
        embedder=embedder,
        reranker=OverlapReranker(),
        config=RetrievalConfig(bm25_top_k=20, dense_top_k=20, rerank_candidates=20, final_top_k=3),
    )


SQLITE_DDL = [
    """CREATE TABLE guidelines (guideline_id TEXT PRIMARY KEY, title TEXT NOT NULL,
       organization TEXT, specialty TEXT, country TEXT, publication_year INTEGER,
       last_updated DATE, url TEXT)""",
    """CREATE TABLE drug_dosing (dosing_id INTEGER PRIMARY KEY AUTOINCREMENT,
       guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id), drug_name TEXT NOT NULL,
       indication TEXT, population TEXT, route TEXT, dose TEXT, frequency TEXT, duration TEXT,
       max_daily_dose TEXT, renal_adjustment TEXT, notes TEXT)""",
    """CREATE TABLE diagnostic_thresholds (threshold_id INTEGER PRIMARY KEY AUTOINCREMENT,
       guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id), condition TEXT NOT NULL,
       test_name TEXT NOT NULL, comparator TEXT, value NUMERIC, unit TEXT, population TEXT,
       interpretation TEXT)""",
    """CREATE TABLE recommendations (recommendation_id TEXT PRIMARY KEY,
       guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id), section TEXT,
       condition TEXT, population TEXT, intervention TEXT, recommendation_text TEXT NOT NULL,
       strength TEXT, evidence_level TEXT)""",
]


@pytest.fixture
def sqlite_engine(documents):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with engine.begin() as conn:
        for ddl in SQLITE_DDL:
            conn.execute(text(ddl))
        for d in documents:
            conn.execute(
                text(
                    "INSERT INTO guidelines (guideline_id, title, organization, publication_year)"
                    " VALUES (:i, :t, :o, :y)"
                ),
                {"i": d.doc_id, "t": d.title, "o": d.organization, "y": d.year},
            )
        conn.execute(
            text(
                "INSERT INTO drug_dosing (guideline_id, drug_name, indication, population, route,"
                " dose, frequency) VALUES ('who-malaria', 'artesunate', 'severe malaria',"
                " 'adults and children >= 20 kg', 'IV or IM', '2.4 mg/kg',"
                " 'at 0, 12 and 24 hours, then once daily')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO diagnostic_thresholds (guideline_id, condition, test_name, comparator,"
                " value, unit) VALUES ('ada-diabetes', 'diabetes', 'HbA1c', '>=', 6.5, '%')"
            )
        )
    return engine

from __future__ import annotations

import pytest

from clinrag.schemas import SQLResult
from clinrag.sql.guard import SQLGuardError, validate_sql
from clinrag.sql.schema_context import describe_schema
from clinrag.sql.text_to_sql import TextToSQL, extract_sql

from fakes import ScriptedLLM

TABLES = {"guidelines", "recommendations", "drug_dosing", "diagnostic_thresholds"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM guidelines WHERE organization = 'WHO'",
        "SELECT g.title, d.dose FROM drug_dosing d JOIN guidelines g USING (guideline_id)",
        "WITH recent AS (SELECT * FROM guidelines WHERE publication_year > 2020) "
        "SELECT COUNT(*) FROM recent",
        "SELECT organization, COUNT(*) FROM guidelines GROUP BY organization",
        "SELECT title FROM guidelines UNION SELECT condition FROM recommendations",
        "SELECT * FROM guidelines;",
    ],
)
def test_guard_allows_reads(sql):
    out = validate_sql(sql, TABLES, row_limit=50)
    assert "LIMIT" in out.upper()


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE guidelines", "only SELECT"),
        ("DELETE FROM guidelines", "only SELECT"),
        ("UPDATE guidelines SET title = 'x'", "only SELECT"),
        ("INSERT INTO guidelines VALUES ('a', 'b')", "only SELECT"),
        ("SELECT 1; DROP TABLE guidelines", "exactly one"),
        ("SELECT * INTO backup FROM guidelines", "INTO"),
        ("SELECT * FROM guidelines FOR UPDATE", "LOCK"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT set_config('role', 'admin', false)", "set_config"),
        ("SELECT * FROM pg_catalog.pg_roles", "schema"),
        ("SELECT * FROM patients", "disallowed table"),
        ("SELECT * FROM generate_series(1, 10)", "table functions"),
        ("COPY guidelines TO '/tmp/out.csv'", "only SELECT"),
        ("", "empty"),
    ],
)
def test_guard_blocks_unsafe_sql(sql, reason):
    with pytest.raises(SQLGuardError, match=reason):
        validate_sql(sql, TABLES)


def test_guard_caps_limit():
    assert validate_sql("SELECT * FROM guidelines LIMIT 10", TABLES, 50).endswith("LIMIT 10")
    assert validate_sql("SELECT * FROM guidelines LIMIT 5000", TABLES, 50).endswith("LIMIT 50")
    assert validate_sql("SELECT * FROM guidelines", TABLES, 50).endswith("LIMIT 50")


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here you go:\n```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("SELECT 3") == "SELECT 3"
    assert extract_sql("NO_SQL: the schema has no patient data") is None
    assert extract_sql("") is None


def _tool(engine, script):
    llm = ScriptedLLM(script)
    return TextToSQL(llm, engine, row_limit=20, max_attempts=3), llm


def test_text_to_sql_happy_path(sqlite_engine):
    tool, llm = _tool(
        sqlite_engine,
        ["```sql\nSELECT COUNT(*) AS n FROM guidelines WHERE organization = 'WHO'\n```"],
    )
    result = tool.run("How many WHO guidelines are there?")
    assert result.ok and result.attempts == 1
    assert result.columns == ["n"] and result.rows == [[2]]
    system = llm.calls[0]["messages"][0]["content"]
    assert "TABLE guidelines" in system and "'WHO'" in system  # schema + sampled values


def test_text_to_sql_self_corrects_db_error(sqlite_engine):
    tool, llm = _tool(
        sqlite_engine,
        [
            "```sql\nSELECT drug, dose FROM drug_dosing\n```",  # bad column
            "```sql\nSELECT drug_name, dose FROM drug_dosing WHERE drug_name LIKE '%artesunate%'\n```",
        ],
    )
    result = tool.run("Artesunate dose for severe malaria?")
    assert result.ok and result.attempts == 2
    assert result.rows == [["artesunate", "2.4 mg/kg"]]
    feedback = llm.calls[1]["messages"][-1]["content"]
    assert "database returned an error" in feedback


def test_text_to_sql_rejects_then_recovers(sqlite_engine):
    tool, llm = _tool(
        sqlite_engine,
        ["DROP TABLE guidelines", "```sql\nSELECT title FROM guidelines LIMIT 1\n```"],
    )
    result = tool.run("Show one guideline title")
    assert result.ok and result.attempts == 2
    assert "rejected by the validator" in llm.calls[1]["messages"][-1]["content"]


def test_text_to_sql_retries_once_on_empty_result(sqlite_engine):
    tool, _ = _tool(
        sqlite_engine,
        [
            "SELECT * FROM drug_dosing WHERE drug_name = 'Artesunate'",
            "SELECT drug_name FROM drug_dosing WHERE lower(drug_name) LIKE '%artesunate%'",
        ],
    )
    result = tool.run("artesunate?")
    assert result.attempts == 2 and result.rows == [["artesunate"]]


def test_text_to_sql_no_sql(sqlite_engine):
    tool, _ = _tool(sqlite_engine, ["NO_SQL: no patient records in this database"])
    result = tool.run("List all patients with diabetes")
    assert not result.ok and "NO_SQL" in result.error


def test_text_to_sql_gives_up_after_max_attempts(sqlite_engine):
    tool, _ = _tool(sqlite_engine, ["SELECT nope FROM guidelines"] * 3)
    result = tool.run("?")
    assert not result.ok and result.attempts == 3


def test_describe_schema(sqlite_engine):
    text = describe_schema(sqlite_engine, ("guidelines", "drug_dosing", "missing_table"))
    assert "TABLE guidelines" in text and "TABLE drug_dosing" in text
    assert "missing_table" not in text
    assert "references guidelines.guideline_id" in text


def test_load_csv_dir_replace_avoids_duplicates(sqlite_engine, tmp_path):
    from sqlalchemy import text

    from clinrag.sql.db import load_csv_dir

    (tmp_path / "drug_dosing.csv").write_text(
        "guideline_id,drug_name,dose\nwho-malaria,artemether,1.6 mg/kg\n"
        "ada-diabetes,metformin,500 mg\n",
        encoding="utf-8",
    )

    def n_rows(table: str = "drug_dosing") -> int:
        with sqlite_engine.connect() as conn:
            return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

    before, guidelines = n_rows(), n_rows("guidelines")  # fixture rows
    assert load_csv_dir(sqlite_engine, tmp_path) == {"drug_dosing": 2}
    load_csv_dir(sqlite_engine, tmp_path)
    assert n_rows() == before + 4  # appending duplicates the serial-id rows
    assert load_csv_dir(sqlite_engine, tmp_path, replace=True) == {"drug_dosing": 2}
    assert n_rows() == 2  # emptied first, then loaded once
    assert n_rows("guidelines") == guidelines  # tables without a CSV are untouched


def test_extract_to_csv_samples_across_corpus(tmp_path):
    from clinrag.schemas import Chunk
    from clinrag.sql.extract import extract_to_csv

    chunks = [
        Chunk(
            chunk_id=f"doc{d:02d}::c{c:04d}",
            doc_id=f"doc{d:02d}",
            title=f"Guideline {d}",
            text=f"Give drug{d} 500 mg twice daily for {c + 1} days.",
        )
        for d in range(20)
        for c in range(10)
    ]  # 200 candidates, 10 per guideline

    def script(messages):
        chunk_text = messages[0]["content"].split("Passage:\n", 1)[1]
        drug = chunk_text.split()[1]
        return f'```json\n{{"drug_dosing": [{{"drug_name": "{drug}", "dose": "500 mg"}}]}}\n```'

    llm = ScriptedLLM(script)
    counts = extract_to_csv(llm, chunks, tmp_path, max_chunks=10, workers=1, seed=1)
    assert counts == {"drug_dosing": 10} and len(llm.calls) == 10
    rows = (tmp_path / "drug_dosing.csv").read_text(encoding="utf-8").splitlines()[1:]
    sampled_docs = {r.split(",")[0] for r in rows}
    assert len(sampled_docs) > 3  # the first 10 candidates would all be doc00
    # Same seed, same sample.
    extract_to_csv(ScriptedLLM(script), chunks, tmp_path / "again", 10, 1, seed=1)
    assert (tmp_path / "again" / "drug_dosing.csv").read_text(encoding="utf-8").splitlines() == [
        "guideline_id,drug_name,indication,population,route,dose,frequency,duration,"
        "max_daily_dose,renal_adjustment,notes",
        *rows,
    ]


def test_sql_result_markdown():
    r = SQLResult(question="q", sql="SELECT 1", columns=["a", "b"], rows=[[1, None], [2, "x"]])
    md = r.to_markdown()
    assert md.splitlines()[0] == "| a | b |" and "| 1 |  |" in md
    assert "no rows" in SQLResult(question="q", sql="SELECT 1").to_markdown()
    assert "failed" in SQLResult(question="q", error="boom").to_markdown()

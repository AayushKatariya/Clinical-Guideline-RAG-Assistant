"""Postgres access: read-only query execution, schema setup and CSV loading."""

from __future__ import annotations

import csv
import datetime as dt
import decimal
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, MetaData, Table, create_engine, text

from clinrag.schemas import Document

log = logging.getLogger(__name__)

TABLES: tuple[str, ...] = ("guidelines", "recommendations", "drug_dosing", "diagnostic_thresholds")
# Load order respects foreign keys.
LOAD_ORDER = TABLES


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def sqlglot_dialect(engine: Engine) -> str:
    name = engine.dialect.name
    return {"postgresql": "postgres"}.get(name, name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def execute_readonly(
    engine: Engine, sql: str, timeout_ms: int = 5000, max_rows: int = 50
) -> tuple[list[str], list[list[Any]]]:
    """Run a (pre-validated) SELECT in a read-only transaction with a timeout."""
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [[_jsonable(v) for v in row] for row in result.fetchmany(max_rows)]
        conn.rollback()
    return columns, rows


# ---------------------------------------------------------------------------
# Admin helpers (used by `clinrag db ...`, never by the assistant)
# ---------------------------------------------------------------------------


def apply_sql_files(admin_engine: Engine, sql_dir: Path) -> list[str]:
    """Execute every `*.sql` file in order (schema, then the read-only role)."""
    applied = []
    raw = admin_engine.raw_connection()
    try:
        cur = raw.cursor()
        for path in sorted(sql_dir.glob("*.sql")):
            cur.execute(path.read_text(encoding="utf-8"))
            applied.append(path.name)
        raw.commit()
    finally:
        raw.close()
    return applied


def _coerce(row: dict[str, str], table: Table) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in table.columns:
        if col.name not in row:
            continue
        value = row[col.name]
        if value is None or str(value).strip() == "":
            out[col.name] = None
            continue
        py = getattr(col.type, "python_type", str)
        try:
            if py is int:
                out[col.name] = int(float(value))
            elif py in (float, decimal.Decimal):
                out[col.name] = float(value)
            elif py is dt.date:
                out[col.name] = dt.date.fromisoformat(str(value)[:10])
            else:
                out[col.name] = str(value).strip()
        except (ValueError, NotImplementedError):
            out[col.name] = None
    return out


def _insert_ignore(table: Table, dialect: str):
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover
        raise NotImplementedError(dialect)
    return insert(table).on_conflict_do_nothing()


def insert_rows(admin_engine: Engine, table_name: str, rows: Iterable[dict[str, Any]]) -> int:
    meta = MetaData()
    table = Table(table_name, meta, autoload_with=admin_engine)
    batch = [_coerce(r, table) for r in rows]
    batch = [r for r in batch if r]
    if not batch:
        return 0
    with admin_engine.begin() as conn:
        conn.execute(_insert_ignore(table, admin_engine.dialect.name), batch)
    return len(batch)


def load_csv_dir(
    admin_engine: Engine, structured_dir: Path, replace: bool = False
) -> dict[str, int]:
    """Load `<table>.csv` files from `structured_dir` (missing files are skipped).

    Without `replace`, rows are appended: tables with serial ids (`drug_dosing`,
    `diagnostic_thresholds`) then get duplicates when a reviewed CSV is reloaded. With
    `replace`, every table that has a CSV is emptied first (children before parents).
    """
    present = [t for t in LOAD_ORDER if (structured_dir / f"{t}.csv").exists()]
    if replace and present:
        with admin_engine.begin() as conn:
            for table in reversed(present):  # names come from TABLES, never from input
                conn.execute(text(f"DELETE FROM {table}"))
                log.info("Cleared %s", table)
    counts: dict[str, int] = {}
    for table in present:
        path = structured_dir / f"{table}.csv"
        with path.open(newline="", encoding="utf-8") as f:
            counts[table] = insert_rows(admin_engine, table, csv.DictReader(f))
        log.info("Loaded %d rows into %s", counts[table], table)
    return counts


def guideline_rows_from_documents(documents: Iterable[Document]) -> list[dict[str, Any]]:
    """The `guidelines` table can be filled straight from corpus metadata."""
    return [
        {
            "guideline_id": d.doc_id,
            "title": d.title,
            "organization": d.organization,
            "specialty": d.specialty,
            "country": d.metadata.get("country"),
            "publication_year": d.year,
            "url": d.url,
        }
        for d in documents
    ]

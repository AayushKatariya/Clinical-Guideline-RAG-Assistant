"""Builds the schema description the text-to-SQL prompt is grounded on.

Column comments come straight from the database (see sql/001_schema.sql) and a few distinct
values are sampled for categorical columns, so the model writes `organization = 'WHO'`
instead of guessing `'World Health Organization'`.
"""

from __future__ import annotations

from sqlalchemy import Engine, inspect, text

CATEGORICAL_COLUMNS = {
    "organization",
    "specialty",
    "country",
    "strength",
    "evidence_level",
    "population",
    "route",
    "comparator",
    "unit",
}
MAX_SAMPLE_VALUES = 12


def describe_schema(engine: Engine, tables: tuple[str, ...], sample_values: bool = True) -> str:
    insp = inspect(engine)
    existing = set(insp.get_table_names())
    blocks: list[str] = []
    for table in tables:
        if table not in existing:
            continue
        try:
            comment = (insp.get_table_comment(table) or {}).get("text")
        except NotImplementedError:
            comment = None
        lines = [f"TABLE {table}" + (f"  -- {comment}" if comment else "")]
        fks = {
            col: f"{fk['referred_table']}.{fk['referred_columns'][i]}"
            for fk in insp.get_foreign_keys(table)
            for i, col in enumerate(fk["constrained_columns"])
        }
        for col in insp.get_columns(table):
            name = col["name"]
            parts = [f"  {name} {col['type']}"]
            notes = []
            if col.get("comment"):
                notes.append(col["comment"].rstrip("."))
            if name in fks:
                notes.append(f"references {fks[name]}")
            if sample_values and name in CATEGORICAL_COLUMNS:
                values = _sample_values(engine, table, name)
                if values:
                    notes.append("values include: " + ", ".join(repr(v) for v in values))
            if notes:
                parts.append("  -- " + "; ".join(notes))
            lines.append("".join(parts))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _sample_values(engine: Engine, table: str, column: str) -> list[str]:
    # Identifiers come from the inspector (not user input), so formatting them is safe.
    sql = text(
        f'SELECT "{column}" AS v, COUNT(*) AS n FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL GROUP BY "{column}" ORDER BY n DESC LIMIT {MAX_SAMPLE_VALUES}'
    )
    try:
        with engine.connect() as conn:
            return [str(r.v) for r in conn.execute(sql)]
    except Exception:
        return []

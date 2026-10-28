"""Text-to-SQL tool with validation and execution-guided self-correction."""

from __future__ import annotations

import logging
import re

from langsmith import traceable
from sqlalchemy import Engine

from clinrag.llm.client import LLMClient
from clinrag.schemas import SQLResult
from clinrag.sql.db import TABLES, execute_readonly, sqlglot_dialect
from clinrag.sql.guard import SQLGuardError, validate_sql
from clinrag.sql.schema_context import describe_schema

log = logging.getLogger(__name__)

NO_SQL = "NO_SQL"

SYSTEM_PROMPT = """\
You write a single read-only {dialect} SQL query that answers a question about a database \
of clinical practice guidelines.

Schema:
{schema}

Rules:
- Use only the tables and columns listed above. Never invent columns.
- Write exactly one SELECT statement. No INSERT/UPDATE/DELETE/DDL.
- Match free-text columns case-insensitively and partially, e.g. \
lower(drug_name) LIKE '%metformin%' or condition ILIKE '%hypertension%'.
- When returning guideline-level facts, also select guidelines.guideline_id and \
guidelines.title so the answer can be cited.
- Prefer explicit JOINs on guideline_id. Use COUNT/GROUP BY for "how many" questions.
- Add LIMIT {row_limit} unless the query is an aggregate that returns few rows.
- If the question cannot be answered from this schema, reply with exactly: {no_sql}: <reason>

Return only the SQL inside a ```sql code block.

Examples:
Q: How many WHO guidelines were published since 2020?
```sql
SELECT COUNT(*) AS n_guidelines FROM guidelines
WHERE organization ILIKE 'WHO' AND publication_year >= 2020;
```
Q: What is the adult dose of amoxicillin for community-acquired pneumonia?
```sql
SELECT g.guideline_id, g.title, d.drug_name, d.population, d.route, d.dose, d.frequency, d.duration
FROM drug_dosing d JOIN guidelines g ON g.guideline_id = d.guideline_id
WHERE lower(d.drug_name) LIKE '%amoxicillin%' AND d.indication ILIKE '%pneumonia%'
  AND (d.population ILIKE '%adult%' OR d.population IS NULL)
LIMIT {row_limit};
```
"""

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str | None) -> str | None:
    if not text:
        return None
    if text.strip().upper().startswith(NO_SQL):
        return None
    m = _SQL_BLOCK.search(text)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip() or None


class TextToSQL:
    def __init__(
        self,
        llm: LLMClient,
        engine: Engine,
        allowed_tables: tuple[str, ...] = TABLES,
        row_limit: int = 50,
        timeout_ms: int = 5000,
        max_attempts: int = 3,
        schema_text: str | None = None,
    ):
        self.llm = llm
        self.engine = engine
        self.allowed_tables = allowed_tables
        self.row_limit = row_limit
        self.timeout_ms = timeout_ms
        self.max_attempts = max_attempts
        self.dialect = sqlglot_dialect(engine)
        self._schema_text = schema_text

    @property
    def schema_text(self) -> str:
        if self._schema_text is None:
            self._schema_text = describe_schema(self.engine, self.allowed_tables)
        return self._schema_text

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            dialect="PostgreSQL" if self.dialect == "postgres" else self.dialect,
            schema=self.schema_text,
            row_limit=self.row_limit,
            no_sql=NO_SQL,
        )

    @traceable(run_type="tool", name="text_to_sql")
    def run(self, question: str) -> SQLResult:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        result = SQLResult(question=question)
        retried_empty = False
        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            reply = self.llm.chat(messages).content or ""  # configured default temperature
            raw_sql = extract_sql(reply)
            if raw_sql is None:
                result.error = reply.strip() or "model returned no SQL"
                return result
            messages.append({"role": "assistant", "content": f"```sql\n{raw_sql}\n```"})
            try:
                safe_sql = validate_sql(
                    raw_sql, set(self.allowed_tables), self.row_limit, dialect=self.dialect
                )
                columns, rows = execute_readonly(
                    self.engine, safe_sql, timeout_ms=self.timeout_ms, max_rows=self.row_limit
                )
            except SQLGuardError as exc:
                feedback = f"That query was rejected by the validator: {exc}. Rewrite it."
            except Exception as exc:  # DB errors: bad column, type mismatch, timeout...
                feedback = (
                    f"The database returned an error: {str(exc).splitlines()[0][:300]}. "
                    "Fix the query."
                )
            else:
                result.sql, result.columns, result.rows, result.error = (
                    safe_sql,
                    columns,
                    rows,
                    None,
                )
                if rows or retried_empty or attempt == self.max_attempts:
                    return result
                retried_empty = True
                feedback = (
                    "The query ran but returned no rows. If you used exact matching, retry with "
                    "case-insensitive partial matching or fewer filters. If the data is truly "
                    "absent, return the same query."
                )
            log.debug("text-to-sql attempt %d failed: %s", attempt, feedback)
            result.sql, result.error = raw_sql, feedback
            messages.append({"role": "user", "content": feedback})
        return result

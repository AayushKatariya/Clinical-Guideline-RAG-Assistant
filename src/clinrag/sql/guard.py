"""Static safety checks for LLM-generated SQL.

Defence in depth: the database role is read-only with a statement timeout, *and* every
query is parsed and validated here before it is sent:

* exactly one statement, and it must be a SELECT (or UNION/INTERSECT/EXCEPT of SELECTs);
* no DML/DDL anywhere in the tree (incl. SELECT ... INTO and FOR UPDATE);
* only allow-listed tables (CTE names are fine), no other schemas;
* no system / file / sleep functions (pg_*, lo_*, dblink, set_config, ...);
* a LIMIT is always present and never above the configured row cap.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SQLGuardError(ValueError):
    pass


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "Command",
        "Copy",
        "Merge",
        "TruncateTable",
        "Into",
        "Lock",
        "Grant",
        "Set",
        "Pragma",
        "Transaction",
        "Commit",
        "Rollback",
    )
    if hasattr(exp, name)
)
_SET_OPS: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, n) for n in ("SetOperation", "Union", "Intersect", "Except") if hasattr(exp, n)
)
_FORBIDDEN_FUNC_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "txid_")
_FORBIDDEN_FUNCS = {"version", "inet_server_addr", "query_to_xml", "copy"}
_ALLOWED_SCHEMAS = {"", "public", "main"}


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def validate_sql(
    sql: str,
    allowed_tables: set[str] | frozenset[str],
    row_limit: int = 50,
    dialect: str = "postgres",
) -> str:
    """Validate `sql` and return a normalised, LIMIT-capped version of it."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SQLGuardError("empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=dialect) if s is not None]
    except ParseError as exc:
        raise SQLGuardError(f"could not parse SQL: {exc}") from exc
    if len(statements) != 1:
        raise SQLGuardError("exactly one SQL statement is allowed")
    root = statements[0]
    if not isinstance(root, (exp.Select, *_SET_OPS)):
        raise SQLGuardError(f"only SELECT queries are allowed (got {root.key.upper()})")

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError(f"forbidden SQL construct: {node.key.upper()}")
        if isinstance(node, exp.Func):
            name = _func_name(node)
            if name in _FORBIDDEN_FUNCS or name.startswith(_FORBIDDEN_FUNC_PREFIXES):
                raise SQLGuardError(f"function not allowed: {name}")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        if not name:  # e.g. table-valued function calls
            raise SQLGuardError("table functions are not allowed")
        if name in cte_names and not table.db:
            continue
        if (table.db or "").lower() not in _ALLOWED_SCHEMAS:
            raise SQLGuardError(f"schema not allowed: {table.db}")
        if name not in allowed:
            raise SQLGuardError(f"unknown or disallowed table: {name}")

    _cap_limit(root, row_limit)
    return root.sql(dialect=dialect)


def _cap_limit(root: exp.Expression, row_limit: int) -> None:
    limit = root.args.get("limit")
    if limit is not None:
        value = limit.args.get("expression")
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= row_limit:
            return
    root.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))

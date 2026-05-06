"""NL → DuckDB SQL via Claude Sonnet 4.6.

The dashboard's "Talk to data" tab feeds user questions into this
module, which:

1. Introspects the DuckDB schema once on first use.
2. Asks Claude Sonnet 4.6 to translate the question into a single
   read-only SQL statement.
3. Returns the SQL string so the caller can run it on the read-only
   DuckDB connection.

The model is intentionally instructed to emit only SQL (no prose, no
markdown fences). Errors raised here propagate up to the state layer,
which renders them in the UI.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import anthropic

from esp_dashboard.db import get_connection

logger = logging.getLogger("dashboard.llm")

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 512


@lru_cache(maxsize=1)
def _schema_text() -> str:
    """Render the DuckDB schema as a compact string for prompt injection.

    Walks every user table in the dashboard DuckDB and concatenates a
    one-line summary per column. Cached because the schema is fixed
    for the lifetime of the process.

    Returns
    -------
    str
        A multi-line string with one ``CREATE TABLE`` block per table,
        suitable for embedding into the system prompt.
    """
    con = get_connection()
    table_names = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    ]
    parts: list[str] = []
    for table in table_names:
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        col_lines = ",\n  ".join(f'"{name}" {dtype}' for name, dtype in cols)
        parts.append(f'CREATE TABLE "{table}" (\n  {col_lines}\n);')
    return "\n\n".join(parts)


SYSTEM_PROMPT_TEMPLATE = (
    "You translate natural-language questions into a single read-only DuckDB SQL "
    "query against the schema below.\n\n"
    "Rules:\n"
    "- Output ONLY one SQL statement. No prose, no markdown fences, no trailing "
    "semicolons.\n"
    "- Use only the tables and columns defined below. Quote identifiers that "
    'contain special characters (e.g. "class", "order").\n'
    "- Keep result sets small (LIMIT 100 by default unless the user explicitly "
    "asks for more).\n"
    "- For 'top N' questions order by the relevant column and apply LIMIT.\n"
    "- Round float aggregates to 2 decimals when displayed.\n"
    "- For GROUP BY queries on dimension columns (taxonomy, license, country, "
    "dataset name, etc.) exclude NULL and empty-string values from the "
    "dimension(s) being grouped, unless the user explicitly asks to include "
    "them. Use `WHERE col IS NOT NULL AND col <> ''`.\n"
    "- Do not use DDL, DML, PRAGMA, ATTACH, or COPY.\n\n"
    "Schema:\n{schema}\n"
)


def _client() -> anthropic.Anthropic:
    """Construct the Anthropic SDK client.

    Returns
    -------
    anthropic.Anthropic
        Reads the API key from `ANTHROPIC_API_KEY` (the SDK default).

    Raises
    ------
    RuntimeError
        If `ANTHROPIC_API_KEY` is not set.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; the Talk-to-Data tab cannot call Claude.")
    return anthropic.Anthropic()


def generate_sql(question: str) -> str:
    """Translate a natural-language question into a DuckDB SQL string.

    Parameters
    ----------
    question : str
        The user's question, e.g. ``"Top 10 datasets by hours of audio"``.

    Returns
    -------
    str
        A single SQL statement, stripped of leading/trailing whitespace
        and any wrapping ``\\`\\`\\`sql ... \\`\\`\\``` fences.

    Raises
    ------
    RuntimeError
        If the API key is missing or Claude returns no text.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=_schema_text())
    client = _client()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    text_blocks = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
    if not text_blocks:
        raise RuntimeError("Claude returned no text content.")
    sql = text_blocks[0].strip()
    if sql.startswith("```"):
        # Strip a single leading/trailing fence; tolerate `sql` language hint.
        sql = sql.split("\n", 1)[-1] if "\n" in sql else sql.strip("`")
        if sql.endswith("```"):
            sql = sql.rsplit("```", 1)[0]
    return sql.strip().rstrip(";")

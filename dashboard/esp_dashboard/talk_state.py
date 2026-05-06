"""Reflex state for the ``/talk`` (Talk-to-Data) tab.

Holds the user's question, the SQL Claude generated, and the result
set. Submission goes through `esp_dashboard.llm.generate_sql` and then
runs the SQL on the same read-only DuckDB connection the rest of the
dashboard uses.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from esp_dashboard.db import get_connection
from esp_dashboard.llm import generate_sql

EXAMPLE_QUERIES: tuple[str, ...] = (
    "Top 10 datasets by total hours of audio",
    "Recordings by license, sorted by count",
    "Total recordings per phylum",
    "Top 20 families across all datasets",
    "Which datasets have geographic metadata?",
    "Average clip duration per dataset",
)


class TalkState(rx.State):
    """State for the Talk-to-Data tab.

    The submit handler is intentionally synchronous — Claude calls and
    DuckDB queries are both fast (<2s typical). If we ever need
    streaming or progress, this is the place to switch to async.
    """

    question: str = ""
    sql: str = ""
    columns: list[str] = []
    rows: list[list[str]] = []
    chart: list[dict[str, Any]] = []
    chart_kind: str = ""  # "bar" | "scatter" | "" (none)
    chart_x_key: str = "x"
    chart_y_key: str = "y"
    error: str = ""
    is_loading: bool = False

    def set_question(self, value: str) -> None:
        """Setter bound to the input field.

        Parameters
        ----------
        value : str
            New question text.
        """
        self.question = value

    def use_example(self, example: str) -> None:
        """Populate the question with an example chip and submit it.

        Parameters
        ----------
        example : str
            One of `EXAMPLE_QUERIES`.
        """
        self.question = example
        return TalkState.submit  # type: ignore[return-value]

    def submit(self) -> None:
        """Translate the current question to SQL, execute it, render results.

        On any failure (missing API key, malformed SQL, runtime error)
        `error` is populated and the table/chart fields are cleared.
        """
        question = (self.question or "").strip()
        if not question:
            self.error = "Type a question first."
            return

        self.is_loading = True
        self.error = ""
        self.sql = ""
        self.columns = []
        self.rows = []
        self.chart = []
        self.chart_kind = ""

        try:
            sql = generate_sql(question)
        except Exception as exc:
            self.error = f"LLM call failed: {exc}"
            self.is_loading = False
            return

        self.sql = sql
        try:
            con = get_connection()
            cur = con.execute(sql)
            description = cur.description or []
            self.columns = [d[0] for d in description]
            raw_rows = cur.fetchmany(200)
        except Exception as exc:
            self.error = f"SQL failed: {exc}"
            self.is_loading = False
            return

        # Stringify cell values for a stable table render.
        def _fmt(v: object) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:,.2f}"
            if isinstance(v, int):
                return f"{v:,}"
            return str(v)

        self.rows = [[_fmt(c) for c in r] for r in raw_rows]
        self._build_chart(self.columns, raw_rows)
        self.is_loading = False

    def _build_chart(self, columns: list[str], rows: list[tuple]) -> None:
        """Pick a chart shape based on the result columns.

        Heuristic:

        - 1 categorical (string) + 1 numeric column → bar chart
        - 2 numeric columns → scatter
        - Anything else → table only

        Parameters
        ----------
        columns : list[str]
            Column names from the DuckDB cursor description.
        rows : list[tuple]
            Raw row tuples from DuckDB (numeric values still numeric).
        """
        if len(columns) != 2 or not rows:
            return
        sample = rows[0]
        is_num = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample]

        if not is_num[0] and is_num[1]:
            self.chart = [{"x": str(r[0]), "y": float(r[1] or 0)} for r in rows[:50]]
            self.chart_kind = "bar"
            self.chart_x_key = "x"
            self.chart_y_key = "y"
        elif is_num[0] and is_num[1]:
            self.chart = [{"x": float(r[0] or 0), "y": float(r[1] or 0)} for r in rows[:200]]
            self.chart_kind = "scatter"
            self.chart_x_key = "x"
            self.chart_y_key = "y"

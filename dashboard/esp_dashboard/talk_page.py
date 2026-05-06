"""``/talk`` page — natural-language exploration of the dashboard DuckDB.

Users type a question, Claude Sonnet 4.6 turns it into a single
read-only SQL query, the dashboard runs it, and the result lands as a
table plus an auto-picked chart (when the shape fits).
"""

from __future__ import annotations

import reflex as rx

from esp_dashboard.header import app_header
from esp_dashboard.talk_state import EXAMPLE_QUERIES, TalkState


def _example_chip(text: str) -> rx.Component:
    """Render one clickable example-question pill.

    Parameters
    ----------
    text : str
        The example question.

    Returns
    -------
    rx.Component
        A small button that fills the input box with the example and
        submits it.
    """
    return rx.button(
        text,
        on_click=lambda: TalkState.use_example(text),
        size="1",
        variant="surface",
        color_scheme="gray",
    )


def _table() -> rx.Component:
    """Render the result table.

    Returns
    -------
    rx.Component
        A Radix-styled table with ``columns``/``rows`` from the state.
    """
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    TalkState.columns,
                    lambda c: rx.table.column_header_cell(c),
                ),
            ),
        ),
        rx.table.body(
            rx.foreach(
                TalkState.rows,
                lambda r: rx.table.row(
                    rx.foreach(r, lambda v: rx.table.cell(v)),
                ),
            ),
        ),
        variant="surface",
        size="1",
    )


def _chart() -> rx.Component:
    """Render the auto-picked chart for the current result.

    Returns
    -------
    rx.Component
        A Recharts bar/scatter chart, or an empty fragment when the
        result shape isn't auto-chartable.
    """
    return rx.cond(
        TalkState.chart_kind == "bar",
        rx.recharts.bar_chart(
            rx.recharts.bar(data_key=TalkState.chart_y_key, fill="#04D78A"),
            rx.recharts.x_axis(
                data_key=TalkState.chart_x_key,
                angle=-30,
                text_anchor="end",
                interval=0,
                height=120,
            ),
            rx.recharts.y_axis(),
            rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
            rx.recharts.graphing_tooltip(),
            data=TalkState.chart,
            width="100%",
            height=380,
        ),
        rx.cond(
            TalkState.chart_kind == "scatter",
            rx.recharts.scatter_chart(
                rx.recharts.scatter(
                    data=TalkState.chart,
                    fill="#04D78A",
                ),
                rx.recharts.x_axis(data_key=TalkState.chart_x_key, type_="number"),
                rx.recharts.y_axis(data_key=TalkState.chart_y_key, type_="number"),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                rx.recharts.graphing_tooltip(),
                width="100%",
                height=380,
            ),
            rx.fragment(),
        ),
    )


def talk_page() -> rx.Component:
    """Top-level component for ``/talk``.

    Returns
    -------
    rx.Component
        The full page tree: header, input, example chips, generated SQL
        block, and result table + chart.
    """
    return rx.container(
        rx.vstack(
            app_header(),
            rx.link(
                rx.text("← back to landing", size="2"),
                href="/",
                color_scheme="gray",
                margin_top="2",
            ),
            rx.heading("Talk to data", size="8"),
            rx.text(
                "Ask a question; Claude turns it into a DuckDB query and runs it "
                "against the dashboard's tables. Read-only, experimental.",
                size="3",
                color_scheme="gray",
            ),
            rx.form(
                rx.hstack(
                    rx.input(
                        placeholder="e.g. Top 10 families by recording count",
                        value=TalkState.question,
                        on_change=TalkState.set_question,
                        size="3",
                        flex_grow="1",
                    ),
                    rx.button(
                        "Ask",
                        type="submit",
                        loading=TalkState.is_loading,
                        size="3",
                    ),
                    width="100%",
                    spacing="2",
                ),
                on_submit=lambda _: TalkState.submit,
                reset_on_submit=False,
                width="100%",
                margin_top="3",
            ),
            rx.flex(
                *[_example_chip(q) for q in EXAMPLE_QUERIES],
                wrap="wrap",
                gap="2",
                margin_top="2",
            ),
            rx.cond(
                TalkState.error != "",
                rx.callout(
                    TalkState.error,
                    icon="triangle_alert",
                    color_scheme="red",
                    margin_top="3",
                ),
                rx.fragment(),
            ),
            rx.cond(
                TalkState.sql != "",
                rx.vstack(
                    rx.heading("Generated SQL", size="4", margin_top="5"),
                    rx.code_block(
                        TalkState.sql,
                        language="sql",
                        theme=rx.code_block.themes.one_dark,
                        wrap_long_lines=True,
                        width="100%",
                    ),
                    spacing="2",
                    align="start",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.cond(
                TalkState.rows.length() > 0,
                rx.vstack(
                    rx.heading("Result", size="4", margin_top="5"),
                    rx.box(_table(), width="100%", overflow_x="auto"),
                    rx.cond(
                        TalkState.chart_kind != "",
                        rx.box(_chart(), width="100%", margin_top="3"),
                        rx.fragment(),
                    ),
                    spacing="2",
                    align="start",
                    width="100%",
                ),
                rx.fragment(),
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="4",
        padding_y="6",
    )

"""Reflex state for the dashboard pages.

Reads aggregate stats from the same DuckDB connection the FastAPI
routes use; we go directly through `esp_dashboard.db` rather than
making an HTTP call to ourselves.
"""

from __future__ import annotations

import plotly.graph_objects as go
import reflex as rx

from esp_dashboard.db import get_connection

# ESP extended chart palette. Cycled across each set of sibling wedges
# so adjacent slices in the sunburst stay visually distinct, and reused
# for any other categorical color encoding the dashboard needs.
_SUNBURST_PALETTE: tuple[str, ...] = (
    "#04D78A",  # Sea Green
    "#FFBF00",  # amber
    "#C3E409",  # Pear Green
    "#129C7B",  # Jungle Green
    "#5F4BB6",  # purple
    "#86A5D9",  # blue-gray
    "#DA3E52",  # red
    "#FD96A9",  # pink
    "#2D82B7",  # blue
    "#A3D9FF",  # sky
    "#A7ED99",  # Amphibian Green
    "#D5DCF9",  # lavender
)


def _format_compact(n: int | float) -> str:
    """Format a number as a compact string (e.g. ``1.2M``).

    Parameters
    ----------
    n : int or float
        Value to format. Treated as 0 if `None`.

    Returns
    -------
    str
        ``""`` if `n` is zero, otherwise the value rounded to one decimal
        and suffixed with ``K``, ``M``, or ``B`` as appropriate. Values
        below 1000 are rendered with thousands separators.
    """
    n = float(n or 0)
    if n == 0:
        return "0"
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs_n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"


class LandingState(rx.State):
    """State for the landing page.

    Populated by `load_stats` on page mount. Fields are intentionally
    pre-formatted strings so the view layer doesn't need formatting
    logic.
    """

    n_datasets: str = "—"
    n_recordings: str = "—"
    total_hours: str = "—"
    n_phyla: str = "—"
    n_classes: str = "—"
    n_orders: str = "—"
    n_families: str = "—"
    error: str = ""
    sunburst: go.Figure = go.Figure()

    def _build_sunburst(self) -> go.Figure:
        """Build the taxonomy sunburst figure from DuckDB.

        Aggregates `dataset_taxonomy` from kingdom down to family and
        wraps the result in a Plotly `Sunburst` trace with click-drill
        behavior. The layout uses a transparent background so the chart
        blends with the dashboard's dark theme.

        Returns
        -------
        plotly.graph_objects.Figure
            The configured figure ready for `rx.plotly`.
        """
        from esp_dashboard.api import _TAXONOMY_TREE_SQL

        con = get_connection()
        rows = con.execute(_TAXONOMY_TREE_SQL).fetchall()
        ids: list[str] = [r[0] for r in rows]
        parents: list[str] = [r[1] for r in rows]
        labels: list[str] = [r[2] for r in rows]
        values: list[int] = [int(r[3] or 0) for r in rows]

        # Cycle the palette within each group of siblings so that
        # neighbouring wedges always read as distinct colors. Top-level
        # nodes (parent == "") share the same group.
        sibling_index: dict[str, int] = {}
        colors: list[str] = []
        for _node_id, parent in zip(ids, parents, strict=True):
            k = sibling_index.get(parent, 0)
            colors.append(_SUNBURST_PALETTE[k % len(_SUNBURST_PALETTE)])
            sibling_index[parent] = k + 1

        fig = go.Figure(
            go.Sunburst(
                ids=ids,
                parents=parents,
                labels=labels,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>%{value:,} recordings<extra></extra>",
                insidetextorientation="radial",
                maxdepth=3,
                marker=dict(
                    colors=colors,
                    line=dict(color="rgba(15,15,20,0.85)", width=1),
                ),
            )
        )
        fig.update_layout(
            margin=dict(t=10, l=10, r=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6e6e6", size=14),
            height=900,
        )
        return fig

    def load_stats(self) -> None:
        """Load aggregate stats from DuckDB into the state.

        Called by Reflex when the landing page mounts. On failure (e.g.
        DuckDB file missing), the field values stay at their dash
        placeholders and `error` is set.
        """
        try:
            con = get_connection()
            n_datasets, n_recordings, total_hours = con.execute(
                """
                SELECT COUNT(*),
                       SUM(num_files),
                       SUM(total_duration_seconds) / 3600.0
                FROM dataset_manifest_stats
                """
            ).fetchone()
            n_phyla, n_classes, n_orders, n_families = con.execute(
                """
                SELECT COUNT(DISTINCT phylum),
                       COUNT(DISTINCT "class"),
                       COUNT(DISTINCT "order"),
                       COUNT(DISTINCT family)
                FROM dataset_taxonomy
                """
            ).fetchone()
        except Exception as exc:
            self.error = str(exc)
            return

        self.n_datasets = _format_compact(n_datasets)
        self.n_recordings = _format_compact(n_recordings)
        self.total_hours = _format_compact(total_hours)
        self.n_phyla = _format_compact(n_phyla)
        self.n_classes = _format_compact(n_classes)
        self.n_orders = _format_compact(n_orders)
        self.n_families = _format_compact(n_families)

        try:
            self.sunburst = self._build_sunburst()
        except Exception as exc:
            self.error = f"Sunburst load failed: {exc}"
            return
        self.error = ""

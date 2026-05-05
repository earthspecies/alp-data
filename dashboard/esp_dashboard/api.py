"""FastAPI routes mounted onto the Reflex app.

The Reflex `App` exposes a FastAPI instance via `api_transformer`. We
attach backend endpoints here so the frontend (Reflex/React) can call
them via HTTP. Routes read from the precomputed DuckDB file via
`esp_dashboard.db.get_connection`.
"""

from typing import Any

from fastapi import APIRouter

from esp_dashboard.db import get_connection

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Cloud Run and local smoke tests.

    Returns
    -------
    dict[str, str]
        A small JSON payload with a `status` key.
    """
    return {"status": "ok"}


@router.get("/aggregate_stats")
def aggregate_stats() -> dict[str, Any]:
    """Headline aggregate stats for the landing page.

    Computes totals across all public datasets bundled in
    `dataset_manifest_stats`, plus distinct-count rollups from
    `dataset_taxonomy` for the taxonomy headline numbers. The taxonomy
    distinct counts are exact across datasets (a family present in
    multiple datasets is counted once).

    Returns
    -------
    dict[str, Any]
        Keys: ``n_datasets``, ``n_recordings``, ``total_hours``,
        ``n_kingdoms``, ``n_phyla``, ``n_classes``, ``n_orders``,
        ``n_families``. Values are integers / floats.
    """
    con = get_connection()
    n_datasets, n_recordings, total_hours = con.execute(
        """
        SELECT COUNT(*),
               SUM(num_files),
               SUM(total_duration_seconds) / 3600.0
        FROM dataset_manifest_stats
        """
    ).fetchone()
    (n_kingdoms, n_phyla, n_classes, n_orders, n_families) = con.execute(
        """
        SELECT COUNT(DISTINCT kingdom),
               COUNT(DISTINCT phylum),
               COUNT(DISTINCT "class"),
               COUNT(DISTINCT "order"),
               COUNT(DISTINCT family)
        FROM dataset_taxonomy
        """
    ).fetchone()
    return {
        "n_datasets": int(n_datasets or 0),
        "n_recordings": int(n_recordings or 0),
        "total_hours": float(total_hours or 0.0),
        "n_kingdoms": int(n_kingdoms or 0),
        "n_phyla": int(n_phyla or 0),
        "n_classes": int(n_classes or 0),
        "n_orders": int(n_orders or 0),
        "n_families": int(n_families or 0),
    }


# Each tier of the sunburst contributes a SELECT that emits one row per
# distinct path. `id` is the slash-joined path (unique), `parent` is the
# id of the level above, `label` is the bare name shown on the wedge.
_TAXONOMY_TREE_SQL = """
WITH base AS (
    SELECT kingdom, phylum, "class" AS clazz, "order" AS ord,
           family, SUM(n_recordings) AS n
    FROM dataset_taxonomy
    WHERE kingdom IS NOT NULL
    GROUP BY kingdom, phylum, clazz, ord, family
)
SELECT kingdom AS id, '' AS parent, kingdom AS label, SUM(n) AS value
FROM base GROUP BY kingdom
UNION ALL
SELECT kingdom || '/' || phylum, kingdom, phylum, SUM(n)
FROM base WHERE phylum IS NOT NULL GROUP BY kingdom, phylum
UNION ALL
SELECT kingdom || '/' || phylum || '/' || clazz,
       kingdom || '/' || phylum, clazz, SUM(n)
FROM base WHERE clazz IS NOT NULL GROUP BY kingdom, phylum, clazz
UNION ALL
SELECT kingdom || '/' || phylum || '/' || clazz || '/' || ord,
       kingdom || '/' || phylum || '/' || clazz, ord, SUM(n)
FROM base WHERE ord IS NOT NULL GROUP BY kingdom, phylum, clazz, ord
UNION ALL
SELECT kingdom || '/' || phylum || '/' || clazz || '/' || ord || '/' || family,
       kingdom || '/' || phylum || '/' || clazz || '/' || ord, family, SUM(n)
FROM base WHERE family IS NOT NULL GROUP BY kingdom, phylum, clazz, ord, family
"""


@router.get("/taxonomy_tree")
def taxonomy_tree() -> dict[str, list[Any]]:
    """Hierarchical taxonomy rollup formatted for a Plotly sunburst.

    Returns parallel arrays (`ids`, `parents`, `labels`, `values`)
    describing every taxonomic node from kingdom down to family,
    aggregated across all datasets in `dataset_taxonomy`. Path-style
    ids ensure uniqueness when the same name (e.g. a family) appears
    under different parents.

    Returns
    -------
    dict[str, list[Any]]
        Keys ``ids``, ``parents``, ``labels``, ``values``. The arrays
        are aligned and can be passed straight to
        ``plotly.graph_objects.Sunburst``.
    """
    con = get_connection()
    rows = con.execute(_TAXONOMY_TREE_SQL).fetchall()
    ids: list[str] = []
    parents: list[str] = []
    labels: list[str] = []
    values: list[int] = []
    for r in rows:
        ids.append(r[0])
        parents.append(r[1])
        labels.append(r[2])
        values.append(int(r[3] or 0))
    return {"ids": ids, "parents": parents, "labels": labels, "values": values}

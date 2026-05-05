"""DuckDB connection helper for the dashboard backend.

The dashboard ships with a precomputed DuckDB file
(`dashboard/assets/dashboard.duckdb`) produced by
`scripts/dashboard/build_assets.py`. The FastAPI routes open a single
read-only connection lazily on first use and reuse it for every request
(DuckDB read-only connections are thread-safe).

The path can be overridden via the `ESP_DASHBOARD_DB` env var, useful in
the container where the file lives at a different absolute path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "assets" / "dashboard.duckdb"


def _resolve_db_path() -> Path:
    """Resolve the DuckDB file path from env or default.

    Returns
    -------
    Path
        Absolute path to the DuckDB file the backend should open.
    """
    override = os.environ.get("ESP_DASHBOARD_DB")
    if override:
        return Path(override)
    return DEFAULT_DB_PATH


@lru_cache(maxsize=1)
def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the process-wide read-only DuckDB connection.

    The connection is opened on first call and cached for the lifetime
    of the process. DuckDB's read-only connections are safe to share
    across threads.

    Returns
    -------
    duckdb.DuckDBPyConnection
        A read-only connection to the dashboard DuckDB file.

    Raises
    ------
    FileNotFoundError
        If the DuckDB file resolved from `_resolve_db_path` does not
        exist on disk.
    """
    path = _resolve_db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dashboard DuckDB not found at {path}. "
            "Run `scripts/dashboard/build_assets.py build-stats` "
            "and `build-cards` to create it, or set ESP_DASHBOARD_DB."
        )
    return duckdb.connect(str(path), read_only=True)

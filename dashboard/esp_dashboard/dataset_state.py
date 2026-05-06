"""Reflex state for the per-dataset detail page.

Fetches the dataset's manifest stats, card stats, and sample manifest
from the precomputed DuckDB and exposes them to the view layer in
ready-to-render shapes (formatted strings, list[dict]).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import reflex as rx

from esp_dashboard.db import get_connection
from esp_dashboard.state import _format_compact

_FAMILY_COMMON_PATH = Path(__file__).resolve().parent.parent / "assets" / "family_common_names.json"
_BLURBS_PATH = Path(__file__).resolve().parent.parent / "assets" / "dataset_blurbs.json"


@lru_cache(maxsize=1)
def _dataset_blurbs() -> dict[str, dict[str, Any]]:
    """Load curated TLDR + label-vocabulary notes for each dataset.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from `info_name` to a dict with keys ``tldr`` (str) and
        ``label_vocab`` (str | None). Empty if the JSON file is missing.
    """
    if not _BLURBS_PATH.exists():
        return {}
    with _BLURBS_PATH.open() as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _family_common_names() -> dict[str, str]:
    """Load the curated family → English-name mapping.

    Returns
    -------
    dict[str, str]
        Mapping from Latin family (e.g. ``"Fringillidae"``) to a short
        English vernacular (e.g. ``"true finches"``). Empty if the JSON
        file is missing.
    """
    if not _FAMILY_COMMON_PATH.exists():
        return {}
    with _FAMILY_COMMON_PATH.open() as f:
        return json.load(f)


class DatasetState(rx.State):
    """State for ``/datasets/[name]`` pages.

    Populated by :meth:`load` on page mount, which inspects the route
    parameter ``name`` and queries DuckDB. If the dataset has no samples
    rendered, `samples` is empty and the player section is hidden by
    the view layer.
    """

    info_name: str = ""
    title: str = ""
    license_text: str = ""
    version: str = ""
    n_recordings: str = "—"
    total_hours: str = "—"
    n_species: str = "—"
    n_families: str = "—"
    samples: list[dict[str, Any]] = []
    family_chart: list[dict[str, Any]] = []
    family_other_count: int = 0
    family_other_total: int = 0
    tldr: str = ""
    label_vocab: str = ""
    error: str = ""

    def load(self) -> None:
        """Read dataset metadata + samples from DuckDB.

        Resolves the dataset name from `self.router.page.params` and
        joins the manifest stats, card stats, and sample manifest
        tables. Sets `error` if the dataset is unknown.
        """
        name = self.router.page.params.get("name", "")
        if not name:
            self.error = "No dataset specified."
            return
        self.info_name = name
        self.title = name
        self.error = ""

        blurb = _dataset_blurbs().get(name, {})
        self.tldr = blurb.get("tldr") or ""
        self.label_vocab = blurb.get("label_vocab") or ""

        con = get_connection()
        manifest_row = con.execute(
            """
            SELECT version, license, num_files, total_duration_seconds
            FROM dataset_manifest_stats
            WHERE info_name = ?
            """,
            [name],
        ).fetchone()
        if manifest_row is None:
            self.error = f"Unknown dataset: {name}"
            return
        version, license_text, num_files, total_seconds = manifest_row
        self.version = version or ""
        self.license_text = license_text or ""
        self.n_recordings = _format_compact(num_files)
        self.total_hours = _format_compact((total_seconds or 0.0) / 3600.0)

        card_row = con.execute(
            """
            SELECT n_species, n_families
            FROM dataset_card_stats
            WHERE info_name = ?
            """,
            [name],
        ).fetchone()
        if card_row is not None:
            n_species, n_families = card_row
            self.n_species = _format_compact(n_species)
            self.n_families = _format_compact(n_families)
        else:
            self.n_species = "—"
            self.n_families = "—"

        sample_rows = con.execute(
            """
            SELECT sample_idx, label, license, duration_s,
                   audio_rel, spec_rel, source_url
            FROM dataset_samples
            WHERE info_name = ?
            ORDER BY sample_idx
            """,
            [name],
        ).fetchall()
        self.samples = [
            {
                "sample_idx": r[0],
                "label": r[1] or "",
                "license": r[2] or "",
                "duration": f"{r[3]:.1f}s",
                "audio_url": "/" + r[4],
                "spec_url": "/" + r[5],
                "source_url": r[6] or "",
            }
            for r in sample_rows
        ]

        # Top-N families by recording count, with an "other" bucket for
        # the long tail. Only meaningful for datasets that contributed
        # to `dataset_taxonomy` (i.e. have full kingdom→family columns);
        # other datasets return no rows and the chart is hidden.
        top_n = 30
        family_rows = con.execute(
            """
            SELECT family, SUM(n_recordings) AS n
            FROM dataset_taxonomy
            WHERE info_name = ? AND family IS NOT NULL
            GROUP BY family
            ORDER BY n DESC
            """,
            [name],
        ).fetchall()
        if family_rows:
            common = _family_common_names()
            top = family_rows[:top_n]
            tail = family_rows[top_n:]
            chart: list[dict[str, Any]] = []
            for fam, n in top:
                vernacular = common.get(fam)
                label = f"{fam} · {vernacular}" if vernacular else fam
                chart.append({"name": label, "value": int(n or 0)})
            self.family_chart = chart
            self.family_other_count = len(tail)
            self.family_other_total = sum(int(r[1] or 0) for r in tail)
        else:
            self.family_chart = []
            self.family_other_count = 0
            self.family_other_total = 0

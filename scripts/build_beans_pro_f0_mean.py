#!/usr/bin/env python3
"""Build BEANS-Pro f0_mean evaluation splits from f0_bioacoustic data.

Loads the f0_bioacoustic dataset CSVs, computes the mean F0 for each
vocalization, and writes two pre-computed JSONL files matching the
beans_pro schema:

- ``f0-mean-seen-taxa``: val split, 9 seen taxa
- ``f0-mean-heldout-taxa``: all split, spotted hyenas (held-out taxon)

Usage::

    uv run python scripts/build_beans_pro_f0_mean.py
    uv run python scripts/build_beans_pro_f0_mean.py --output-dir /tmp/f0_mean_splits
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.io import filesystem_from_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

INSTRUCTION_TEXT = (
    "What is the mean fundamental frequency of this vocalization?"
)
INSTRUCTION = (
    f"<Audio><AudioHere></Audio> {INSTRUCTION_TEXT}"
)
SOURCE_DOI = "doi:10.1080/09524622.2025.2500380"
LICENSE = "CC0-1.0"

SEEN_TAXA = [
    "canids",
    "hummingbirds",
    "La_Palma_chaffinches",
    "lions",
    "little_owls",
    "long-billed_hermits",
    "monk_parakeets",
    "orangutans",
    "Reunion_grey_white_eyes",
]

HELDOUT_TAXA = ["spotted_hyenas"]

SPLIT_CONFIGS = {
    "f0-mean-seen-taxa": {
        "csv_path": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_val.csv",
        "taxa": SEEN_TAXA,
    },
    "f0-mean-heldout-taxa": {
        "csv_path": "gs://esp-data-ingestion/f0-prediction/f0_bioacoustic_normalized.csv",
        "taxa": HELDOUT_TAXA,
    },
}


# ── F0 helpers (matching esp-research f0_features.py) ────────────────────


def _parse_f0(tsv: str) -> pd.DataFrame | None:
    """Parse TSV F0 contour string into a DataFrame.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with ``time_s`` and ``freq_hz`` columns, or ``None``
        if parsing fails or the result has fewer than 2 rows.
    """
    if not tsv or not isinstance(tsv, str):
        return None
    try:
        df = pd.read_csv(StringIO(tsv), sep="\t")
        df = df.rename(columns={"Time": "time_s", "Freq": "freq_hz"})
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["freq_hz"] = pd.to_numeric(df["freq_hz"], errors="coerce")
        df = df.dropna()
        return df if len(df) >= 2 else None
    except Exception:
        return None


def _round_freq(f: float, step: int = 10) -> int:
    """Round frequency to the nearest ``step`` Hz.

    Returns
    -------
    int
        Frequency rounded to the nearest multiple of ``step``.
    """
    return int(round(f / step) * step)


def compute_f0_mean(tsv: str) -> str | None:
    """Compute formatted mean F0 from a TSV contour string.

    Returns
    -------
    str | None
        Formatted string like ``"1240 Hz"``, or ``None`` if the contour
        is invalid.
    """
    df = _parse_f0(tsv)
    if df is None:
        return None
    mean_freq = float(df["freq_hz"].mean())
    return f"{_round_freq(mean_freq)} Hz"


# ── JSONL builder ────────────────────────────────────────────────────────


def build_jsonl_rows(csv_path: str, taxa: list[str], split_name: str) -> list[dict]:
    """Load a f0_bioacoustic CSV, filter by taxa, compute f0_mean, build rows.

    Parameters
    ----------
    csv_path
        GCS path to the f0_bioacoustic CSV.
    taxa
        List of taxon names to include.
    split_name
        Name for the ``dataset_name`` field in the output.

    Returns
    -------
    list[dict]
        List of JSONL-ready dictionaries in beans_pro schema.
    """
    logger.info("Loading %s", csv_path)
    fs = filesystem_from_path(csv_path)
    with fs.open(csv_path, "r") as f:
        df = pd.read_csv(f)
    logger.info("  Loaded %d rows, columns: %s", len(df), list(df.columns))

    taxa_set = set(taxa)
    df = df[df["taxon"].isin(taxa_set)]
    logger.info("  After taxa filter (%s): %d rows", taxa, len(df))

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        f0_mean = compute_f0_mean(row.get("f0_contour", ""))
        if f0_mean is None:
            skipped += 1
            continue

        metadata = {
            "taxon": row.get("taxon", ""),
            "species": row.get("species", ""),
            "canonical_name": row.get("canonical_name", ""),
            "species_common": row.get("species_common", ""),
        }

        audio_path = row["audio_path"]
        rows.append({
            "source_dataset": SOURCE_DOI,
            "dataset_name": split_name,
            "output": f0_mean,
            "instruction_text": INSTRUCTION_TEXT,
            "instruction": INSTRUCTION,
            "task": "f0_mean",
            "file_name": audio_path,
            "license": LICENSE,
            "id": str(uuid.uuid4()),
            "metadata": json.dumps(metadata),
            "audio_path_original_sample_rate": audio_path,
        })

    logger.info("  Built %d rows, skipped %d (invalid/empty f0_contour)", len(rows), skipped)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def print_summary(rows: list[dict], split_name: str) -> None:
    """Print summary statistics for a set of rows."""
    if not rows:
        logger.warning("  %s: no rows!", split_name)
        return

    f0_values = []
    taxon_counts: dict[str, int] = {}
    for row in rows:
        val = int(row["output"].replace(" Hz", ""))
        f0_values.append(val)
        meta = json.loads(row["metadata"])
        taxon = meta.get("taxon", "unknown")
        taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1

    arr = np.array(f0_values)
    logger.info("  %s: %d rows", split_name, len(rows))
    logger.info("    f0_mean: min=%d, max=%d, median=%d, mean=%.0f Hz",
                arr.min(), arr.max(), np.median(arr), arr.mean())
    for taxon in sorted(taxon_counts):
        logger.info("    %s: %d rows", taxon, taxon_counts[taxon])


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_f0_mean",
        help="Directory for output JSONL files (default: data/beans_pro_f0_mean/)",
    )
    args = parser.parse_args()

    for split_name, cfg in SPLIT_CONFIGS.items():
        rows = build_jsonl_rows(cfg["csv_path"], cfg["taxa"], split_name)
        print_summary(rows, split_name)
        out_path = args.output_dir / f"{split_name.replace('-', '_')}.jsonl"
        write_jsonl(rows, out_path)


if __name__ == "__main__":
    main()

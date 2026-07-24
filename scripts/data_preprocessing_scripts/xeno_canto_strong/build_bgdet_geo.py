"""Geo-enrich the XC strong background-detection multilabel split.

``train_strong_unseen_top100_bgdet.csv`` carries per-file selection tables
(background-species detections) but NO location/time metadata — only
``relative_path`` + labels. The original Xeno-Canto metadata
(``all_20260622_v1.csv``) has full coordinates/locality/date keyed by the same
XC id (embedded in ``relative_path`` as ``XC<digits>``). This joins the two by
``xc_id`` and writes an enriched split so ``context_builder`` can produce real
location/time context for the multilabel tasks. Left join (unmatched files keep
empty context and fall back gracefully).

Run on a host with GCS access (``ssh slurm-login``).
"""

from __future__ import annotations

import argparse
import re

import polars as pl

from esp_data.backends.polars_backend import PolarsBackend

RAW = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"
STRONG = f"{RAW}/train_strong_unseen_top100_bgdet.csv"
META = f"{RAW}/all_20260622_v1.csv"
OUT = f"{RAW}/train_strong_unseen_top100_bgdet_geo.csv"
# ContextBuilder-relevant fields: location + temporal + notes.
GEO_COLS = [
    "latitudeDecimal", "longitudeDecimal", "locality", "country_code",
    "eventDate", "eventTime", "fieldNotes",
]
_XCID = re.compile(r"XC(\d+)")


def _load(path: str, cols: list[str] | None = None) -> pl.DataFrame:
    b = PolarsBackend.from_csv(
        path, keep_default_na=False, na_values=[""], **({"columns": cols} if cols else {})
    )
    return b._df if hasattr(b, "_df") else b.unwrap


def _xcid(s: str | None) -> str | None:
    m = _XCID.search(s or "")
    return m.group(1) if m else None


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=OUT)
    p.add_argument("--local-out", default="/tmp/train_strong_unseen_top100_bgdet_geo.csv")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()

    strong = _load(STRONG)
    meta = _load(META, ["relative_path", *GEO_COLS])
    print(f"strong={strong.height:,}  meta={meta.height:,}", flush=True)

    strong = strong.with_columns(
        pl.col("relative_path").map_elements(_xcid, return_dtype=pl.Utf8).alias("_xcid")
    )
    meta = meta.with_columns(
        pl.col("relative_path").map_elements(_xcid, return_dtype=pl.Utf8).alias("_xcid")
    ).drop("relative_path").unique("_xcid")

    out = strong.join(meta, on="_xcid", how="left").drop("_xcid")
    matched = out.filter(
        pl.col("latitudeDecimal").cast(pl.Utf8).fill_null("") != ""
    ).height
    print(f"enriched={out.height:,}  with-latlong={matched:,} ({100 * matched / out.height:.1f}%)",
          flush=True)
    print(f"distinct countries={out['country_code'].n_unique()}  "
          f"localities={out['locality'].n_unique()}", flush=True)

    out.write_csv(args.local_out)
    print(f"wrote {args.local_out}", flush=True)
    if args.upload:
        import subprocess
        subprocess.run(["gsutil", "-q", "cp", args.local_out, args.out], check=True)
        print(f"uploaded -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

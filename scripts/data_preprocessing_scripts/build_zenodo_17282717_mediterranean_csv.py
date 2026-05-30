"""
Build the SuperWhales-compatible manifest CSV for Zenodo 17282717
(Multi-platform deployments of low-cost devices for cetacean PAM,
Jankauskaite et al. 2025, Western Mediterranean, HydroMoth).

Inputs (NFS, populated by zenodo_17282717_stage.sh):
    /mnt/home/zenodo_17282717_staging/extract/<species>/
        1-annotation-tables/<deployment>_<sp>_annotations.txt   (Raven TSV)
        2-annotation-clips/<selectionID>.wav                    (per-event clip)
        2-annotation-clips/clip_lists_txt/<deployment>_<sp>_clips.txt

Raven .txt schema (tab-separated):
    Selection View Channel Begin File End File
    Begin Time (s) End Time (s) Delta Time (s)
    Begin Date Time End Date Time
    Low Freq (Hz) High Freq (Hz) Peak Freq (Hz)
    signalType species quality deploymentType deploymentID selectionID

Each annotation row → one CSV row pointing to its extracted clip
``<selectionID>.wav``. Within the clip, the event spans 0..duration, so the
embedded selection_table TSV records a single event row at Begin Time=0,
End Time=duration_s (frequencies preserved from the Raven row).

Output CSV schema (matches superwhale_detection.csv layout):
    audio_path, sample_rate_hz, duration_s, species,
    selection_table, source_dataset, source_url, license,
    source_paper_doi, all_cetaceans_labeled,
    scientific_name_unified_original, canonical_name, gbif_link_ok,
    gbifID, kingdom, phylum, class, order, family, genus,
    species_common, 16khz_path, 32khz_path

(16khz_path / 32khz_path left blank here; Phase 3 populates them.)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import soundfile as sf

from esp_data.discover.gbif_taxonomy import GBIFConverter

GCS_DST_PREFIX = "zenodo_17282717"  # audio_path / 16khz_path / 32khz_path prefix
GCS_ROOT = "gs://esp-data-ingestion/superwhale/v0.1.0/raw"
SOURCE_DATASET = "zenodo_17282717_mediterranean_cetacean_clips"
SOURCE_URL = "https://zenodo.org/records/17282717"
SOURCE_PAPER_DOI = "10.5281/zenodo.17282717"
LICENSE = "CC-BY-4.0"

# Selection-table column order — must match what existing SuperWhales rows use.
ST_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "species",
    "taxon",
    "taxon_rank",
    "call_type",
    "coarse_call_type",
    "confidence",
    "canonical_name",
    "genus",
    "family",
    "species_common",
    "gbifID",
]

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Species → metadata for the GBIF link. Each entry carries the SuperWhales
# selection-table fields (taxon, taxon_rank) and a `gbif_query` used to
# resolve the taxonomy. Delphinidae is a family-level aggregate: GBIFConverter
# does not resolve family names, so we provide a manual `taxonomy_override`
# block that gets used when the GBIF query fails (or is empty).
SPECIES_META: dict[str, dict[str, str | dict[str, str]]] = {
    "Delphinidae": {
        "taxon": "Delphinidae",
        "taxon_rank": "family",
        "gbif_query": "Delphinidae",  # will fail; falls back to override below
        "taxonomy_override": {
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Cetacea",
            "family": "Delphinidae",
            "genus": "",
            "canonical_name": "",  # no species-level identification
            "species_common": "oceanic dolphins",
            "gbifID": "",
        },
    },
    "Globicephala_melas": {
        "taxon": "Globicephala melas",
        "taxon_rank": "species",
        "gbif_query": "Globicephala melas",
    },
    "Grampus_griseus": {
        "taxon": "Grampus griseus",
        "taxon_rank": "species",
        "gbif_query": "Grampus griseus",
    },
    "Physeter_macrocephalus": {
        "taxon": "Physeter macrocephalus",
        "taxon_rank": "species",
        "gbif_query": "Physeter macrocephalus",
    },
    "Stenella_coeruleoalba": {
        "taxon": "Stenella coeruleoalba",
        "taxon_rank": "species",
        "gbif_query": "Stenella coeruleoalba",
    },
    "Tursiops_truncatus": {
        "taxon": "Tursiops truncatus",
        "taxon_rank": "species",
        "gbif_query": "Tursiops truncatus",
    },
}

# Raven `signalType` → SuperWhales `coarse_call_type`
COARSE_CALL_MAP = {
    "clicks": "click",
    "click": "click",
    "whistles": "whistle",
    "whistle": "whistle",
    "clicks_whistles": "mixed",
    "whistles_clicks": "mixed",
    "buzz": "click",
    "mixed": "mixed",
    "": "unknown",
}


def _coarse_call(signal_type: str) -> str:
    s = (signal_type or "").strip().lower()
    if s in COARSE_CALL_MAP:
        return COARSE_CALL_MAP[s]
    # Fallback heuristics for compound signalType strings
    if "click" in s and "whistle" in s:
        return "mixed"
    if "click" in s:
        return "click"
    if "whistle" in s:
        return "whistle"
    return "unknown"


def _gbif_for(converter: GBIFConverter, species_key: str, cache: dict) -> dict:
    """Return the GBIF-resolved taxonomy dict for a SPECIES_META key.

    Returns
    -------
    dict
        Keys: canonical_name, genus, family, species_common, gbifID,
        gbif_link_ok (bool), plus kingdom/phylum/class/order/family/genus.
        Empty strings (and gbif_link_ok=False) when unresolved. Cached in-place
        via the ``cache`` dict.
    """
    if species_key in cache:
        return cache[species_key]
    meta = SPECIES_META[species_key]
    out = {
        "canonical_name": "",
        "genus": "",
        "family": "",
        "species_common": "",
        "gbifID": "",
        "gbif_link_ok": False,
    } | {r: "" for r in TAXONOMY_RANKS}
    info, ok = converter(meta["gbif_query"])
    if ok:
        out["canonical_name"] = info.get("canonicalName", "")
        out["genus"] = info.get("genus", "")
        out["family"] = info.get("family", "")
        out["species_common"] = info.get("vernacularName", "") or ""
        out["gbifID"] = str(info.get("taxonID", ""))
        out["gbif_link_ok"] = True
        for r in TAXONOMY_RANKS:
            out[r] = info.get(r, "")
    elif "taxonomy_override" in meta:
        # Family- or higher-rank aggregates: GBIFConverter only resolves
        # species binomials. Use a manually-curated taxonomy block.
        ovr = meta["taxonomy_override"]
        for k in ("canonical_name", "genus", "family", "species_common", "gbifID"):
            out[k] = ovr.get(k, "")
        for r in TAXONOMY_RANKS:
            out[r] = ovr.get(r, "")
        out["gbif_link_ok"] = False  # not a GBIF link, manual override
    cache[species_key] = out
    return out


def _read_raven_table(path: Path) -> pd.DataFrame:
    """Read a Raven selection table .txt (TSV).

    Returns
    -------
    pd.DataFrame
        Schema-tolerant string-typed DataFrame; missing cells become empty
        strings so downstream accesses are safe.
    """
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], dtype=str)
    df = df.fillna("")
    return df


def _build_st_tsv(row: dict, duration_s: float, gbif: dict) -> str:
    """Build a single-event selection_table TSV blob for one annotation.

    Returns
    -------
    str
        TSV (tab-separated values) string with a header row matching
        ``ST_COLUMNS`` and a single event row spanning the clip
        (``Begin Time = 0``, ``End Time = duration_s``).
    """
    species_key = row["_species_key"]
    meta = SPECIES_META[species_key]
    event = {
        "Begin Time (s)": f"{0.0:.6f}",
        "End Time (s)": f"{float(duration_s):.6f}",
        "Low Freq (Hz)": row.get("Low Freq (Hz)", ""),
        "High Freq (Hz)": row.get("High Freq (Hz)", ""),
        "species": meta["taxon"] if meta["taxon_rank"] == "species" else "",
        "taxon": meta["taxon"],
        "taxon_rank": meta["taxon_rank"],
        "call_type": row.get("signalType", ""),
        "coarse_call_type": _coarse_call(row.get("signalType", "")),
        "confidence": row.get("quality", ""),
        "canonical_name": gbif["canonical_name"],
        "genus": gbif["genus"],
        "family": gbif["family"],
        "species_common": gbif["species_common"],
        "gbifID": gbif["gbifID"],
    }
    df = pd.DataFrame([event], columns=ST_COLUMNS)
    return df.to_csv(sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extract-root", default="/mnt/home/zenodo_17282717_staging/extract")
    p.add_argument(
        "--out-local",
        default="/mnt/home/zenodo_17282717_staging/zenodo_17282717_mediterranean_cetacean_clips.csv",
    )
    p.add_argument(
        "--out-gcs",
        default=f"{GCS_ROOT}/zenodo_17282717_mediterranean_cetacean_clips.csv",
    )
    p.add_argument(
        "--gbif-cache",
        default="/mnt/home/superwhale_merge/gbif_animals.tsv",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_local), exist_ok=True)
    os.makedirs(os.path.dirname(args.gbif_cache), exist_ok=True)

    extract_root = Path(args.extract_root)
    species_dirs = sorted(d for d in extract_root.iterdir() if d.is_dir())
    print(f"[scan] {len(species_dirs)} species dirs: {[d.name for d in species_dirs]}")

    converter = GBIFConverter(cache_path=args.gbif_cache)
    gbif_cache: dict = {}

    rows_out: list[dict] = []
    missing_clips = 0

    for sp_dir in species_dirs:
        sp_key = sp_dir.name
        if sp_key not in SPECIES_META:
            print(f"[warn] unknown species dir {sp_key!r} (skipping)")
            continue
        tables_dir = sp_dir / "1-annotation-tables"
        clips_dir = sp_dir / "2-annotation-clips"
        if not tables_dir.is_dir() or not clips_dir.is_dir():
            print(f"[warn] {sp_key}: missing annotation-tables or clips dir; skipping")
            continue

        gbif = _gbif_for(converter, sp_key, gbif_cache)
        n_table_rows = 0
        n_kept = 0
        for txt in sorted(tables_dir.glob("*.txt")):
            df = _read_raven_table(txt)
            n_table_rows += len(df)
            for _, r in df.iterrows():
                sel_id = str(r.get("selectionID", "")).strip()
                if not sel_id:
                    continue
                clip = clips_dir / f"{sel_id}.wav"
                if not clip.is_file():
                    missing_clips += 1
                    continue
                # Probe SR + duration from clip header (no decode)
                info = sf.info(str(clip))
                duration_s = info.frames / float(info.samplerate)
                sample_rate_hz = int(info.samplerate)

                row_for_st = dict(r) | {"_species_key": sp_key}
                st_tsv = _build_st_tsv(row_for_st, duration_s, gbif)

                rel_audio = f"{GCS_DST_PREFIX}/{sp_key}/2-annotation-clips/{sel_id}.wav"
                rows_out.append(
                    {
                        "audio_path": rel_audio,
                        "sample_rate_hz": str(sample_rate_hz),
                        "duration_s": f"{duration_s:.6f}",
                        "species": gbif["canonical_name"] or SPECIES_META[sp_key]["taxon"],
                        "selection_table": st_tsv,
                        "source_dataset": SOURCE_DATASET,
                        "source_url": SOURCE_URL,
                        "license": LICENSE,
                        "source_paper_doi": SOURCE_PAPER_DOI,
                        "all_cetaceans_labeled": "True",
                        "scientific_name_unified_original": SPECIES_META[sp_key]["taxon"],
                        "canonical_name": gbif["canonical_name"],
                        "gbif_link_ok": "True" if gbif["gbif_link_ok"] else "False",
                        "gbifID": gbif["gbifID"],
                        **{r_: gbif[r_] for r_ in TAXONOMY_RANKS},
                        "species_common": gbif["species_common"],
                        "16khz_path": "",  # populated in Phase 3
                        "32khz_path": "",
                    }
                )
                n_kept += 1
        print(f"  {sp_key}: {n_table_rows} table rows -> {n_kept} CSV rows")

    out_df = pd.DataFrame(rows_out)
    print(f"\n[summary] {len(out_df)} rows; missing-clip skips: {missing_clips}")
    if len(out_df) == 0:
        raise RuntimeError("no rows built; check --extract-root contents")

    print("[summary] source_dataset:", out_df["source_dataset"].iloc[0])
    print("[summary] per-species row counts:")
    print(out_df["scientific_name_unified_original"].value_counts().to_string())
    print("[summary] sample_rates seen:")
    print(out_df["sample_rate_hz"].value_counts().to_string())

    # Inspect first selection_table to sanity-check
    print("\n[first selection_table]")
    print(out_df.iloc[0]["selection_table"])

    print(f"\n[write] local: {args.out_local}")
    out_df.to_csv(args.out_local, index=False)
    print(f"  size: {os.path.getsize(args.out_local) / 1e6:.2f} MB")

    if args.dry_run:
        print("[dry-run] not uploading.")
        return

    print(f"[upload] {args.out_gcs}")
    rc = os.system(f"gsutil -q cp {args.out_local} {args.out_gcs}")
    if rc != 0:
        raise RuntimeError(f"gsutil cp failed rc={rc}")
    print("Done.")


if __name__ == "__main__":
    main()

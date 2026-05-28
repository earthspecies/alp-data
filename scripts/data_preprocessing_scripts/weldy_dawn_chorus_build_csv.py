"""
Build Weldy NW dawn-chorus split CSVs (WABAD-style multi-label) from the
staged TSV annotation files.

Each row = one 5-min recording with an embedded tab-separated ``selection_table``
listing every annotated 2-s window label (multi-label per window). Writes
``all.csv``, ``complete.csv``, ``partial.csv``, ``labeled.csv``,
``unlabeled.csv`` and ``species_labels.csv``, uploaded to the GCS dataset root.

Selection-table columns (per labelled 2-s window):
    Begin Time (s), End Time (s), Species (GBIF canonical),
    Species Code (eBird_2021), Common Name, Sonotype (song/call/...),
    Category (species/non-biotic/biotic-aggregate/method), Label (raw Weldy),
    clip_complete (TRUE/FALSE)

Inputs (local, from weldy_dawn_chorus_stage.sh):
    meta/acoustic_annotations.tsv      (53,509 rows, 156 fully-annotated files)
    meta/partial_annotations.tsv       (5,500 rows, 215 partially-annotated files)
    meta/annotation_metadata.tsv       (label vocabulary)
    meta/acoustic_files.tsv            (per-recording status: complete/partial/no)
    weldy_species_taxonomy.csv         (eBird code -> GBIF canonical)
    extract/**/Site_NNN_Rep_X.wav      (used only for audio_duration headers)
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd
import soundfile as sf

GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
UNKNOWN = "Unknown"
ST_COLUMNS = [
    "Begin Time (s)",
    "End Time (s)",
    "Species",
    "Species Code",
    "Common Name",
    "Sonotype",
    "Category",
    "Label",
    "clip_complete",
]
METHOD_LABELS = {"complete", "empty", "unknown", "UNK_chip", "unk", "impossible"}
NON_BIOTIC_HINTS = re.compile(
    r"\b(rain|wind|airplane|engine|noise|truck|helicop|saw|vehicle|motor|traffic|car|beep|gun|chainsaw)\b",
    re.I,
)


def categorize(label: str, sci: str) -> str:
    if label in METHOD_LABELS:
        return "method"
    sci = sci.strip()
    if not sci or sci.lower() == "nan":
        if NON_BIOTIC_HINTS.search(label):
            return "non-biotic"
        return "method"
    if "spp" in sci.lower() or sci in ("Aves", "Insecta"):
        return "biotic-aggregate"
    return "species"


def sonotype_of(sound: str) -> str:
    s = (sound or "").strip()
    return s.split("_")[0] if "_" in s else s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/mnt/home/weldy_staging")
    p.add_argument("--taxonomy", default="weldy_species_taxonomy.csv")
    p.add_argument("--gcs-root", default=GCS_ROOT)
    p.add_argument("--out-dir", default="/mnt/home/weldy_staging/csv")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta_dir = os.path.join(args.work, "meta")
    extract_dir = os.path.join(args.work, "extract")

    F = pd.read_csv(os.path.join(meta_dir, "acoustic_files.tsv"), sep="\t").fillna("")
    M = pd.read_csv(os.path.join(meta_dir, "annotation_metadata.tsv"), sep="\t").fillna("")
    A = pd.read_csv(os.path.join(meta_dir, "acoustic_annotations.tsv"), sep="\t").fillna("")
    P = pd.read_csv(os.path.join(meta_dir, "partial_annotations.tsv"), sep="\t").fillna("")
    T = pd.read_csv(args.taxonomy, keep_default_na=False).fillna("")
    print(
        f"files={len(F)} | full-ann rows={len(A)} | partial-ann rows={len(P)} "
        f"| labels={len(M)} | taxonomy={len(T)}"
    )

    # label -> (common, sound/sonotype, sci, category)
    lbl_info: dict[str, dict] = {}
    for _, r in M.iterrows():
        lbl = str(r["label"])
        lbl_info[lbl] = {
            "common": str(r["common_name"]),
            "sonotype": sonotype_of(str(r["sound"])),
            "sci": str(r["scientific_name"]),
            "category": categorize(lbl, str(r["scientific_name"])),
            "code": str(r["eBird_2021"]),
        }
    # eBird code -> canonical (Species)
    code_to_canonical: dict[str, str] = dict(
        zip(T["eBird_2021"].astype(str), T["canonical_name"].astype(str), strict=False)
    )

    # index local WAV paths for durations (header read; no decode)
    wav_paths = glob.glob(os.path.join(extract_dir, "**", "*.wav"), recursive=True)
    wav_index = {os.path.basename(p): p for p in wav_paths}
    print(f"local wav indexed: {len(wav_index)}")

    A_by = dict(tuple(A.groupby("file")))
    P_by = dict(tuple(P.groupby("file")))

    def build_st(rows_df: pd.DataFrame, partial: bool) -> tuple[str, int]:
        if rows_df is None or rows_df.empty:
            return pd.DataFrame(columns=ST_COLUMNS).to_csv(sep="\t", index=False), 0
        out = []
        for _, ev in rows_df.iterrows():
            lbl = str(ev["label"])
            info = lbl_info.get(lbl) or {
                "common": "",
                "sonotype": "",
                "sci": "",
                "category": "method",
                "code": str(ev.get("eBird_2021", "")),
            }
            cc = (str(ev["clip_complete"]).upper() == "TRUE") if partial else True
            out.append(
                {
                    "Begin Time (s)": float(ev["start"]),
                    "End Time (s)": float(ev["end"]),
                    "Species": code_to_canonical.get(info["code"], ""),
                    "Species Code": info["code"],
                    "Common Name": info["common"],
                    "Sonotype": info["sonotype"],
                    "Category": info["category"],
                    "Label": lbl,
                    "clip_complete": cc,
                }
            )
        df = pd.DataFrame(out, columns=ST_COLUMNS)
        return df.to_csv(sep="\t", index=False), len(df)

    rows = []
    for _, r in F.iterrows():
        fn = str(r["file"])
        stem = os.path.splitext(fn)[0]
        status = str(r["annotated"]).strip()  # complete / partial / no
        # selection_table per status
        if status == "complete":
            st_str, n_ev = build_st(A_by.get(fn), partial=False)
        elif status == "partial":
            st_str, n_ev = build_st(P_by.get(fn), partial=True)
        else:
            st_str, n_ev = pd.DataFrame(columns=ST_COLUMNS).to_csv(sep="\t", index=False), 0
        # duration from local header (fallback 300 s)
        dur, sr = 300.0, ""
        lp = wav_index.get(fn)
        if lp and os.path.exists(lp):
            try:
                info = sf.info(lp)
                dur = round(float(info.frames) / float(info.samplerate), 6)
                sr = int(info.samplerate)
            except Exception:
                pass
        rows.append(
            {
                "fn": stem,
                "file": fn,
                "site": r["site"],
                "replicate": r["replicate"],
                "date": r["recording_date"],
                "annotation_status": status,
                "audio_fp": f"recordings/{fn}",
                "16khz_path": f"audio_16k/{fn}",
                "32khz_path": f"audio_32k/{fn}",
                "audio_duration": dur,
                "sample_rate": sr,
                "n_events": n_ev,
                "selection_table": st_str,
            }
        )

    all_df = pd.DataFrame(rows)
    status = all_df["annotation_status"]

    def _save(df: pd.DataFrame, name: str) -> None:
        local = os.path.join(args.out_dir, name)
        df.to_csv(local, index=False)
        os.system(f"gsutil -q cp {local} {args.gcs_root}/{name}")
        print(f"  {name}: {len(df)} rows -> {args.gcs_root}/{name}")

    _save(all_df, "all.csv")
    _save(all_df[status == "complete"].reset_index(drop=True), "complete.csv")
    _save(all_df[status == "partial"].reset_index(drop=True), "partial.csv")
    labeled_mask = status.isin(["complete", "partial"])
    _save(all_df[labeled_mask].reset_index(drop=True), "labeled.csv")
    _save(all_df[~labeled_mask].reset_index(drop=True), "unlabeled.csv")

    # species vocabulary = GBIF canonical names actually used as labels
    canon_used = {code_to_canonical.get(info["code"], "") for info in lbl_info.values()}
    used_canon = sorted(canon_used - {""})
    _save(pd.DataFrame({"Species": used_canon}), "species_labels.csv")

    print(f"\nTOTAL recordings: {len(all_df)} | total events: {int(all_df['n_events'].sum())}")
    print("by annotation_status:\n", status.value_counts().to_string())


if __name__ == "__main__":
    main()

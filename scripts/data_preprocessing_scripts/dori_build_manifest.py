"""
Build the DORI Phase-1 ingestion manifest from DORI.csv.

Phase 1 = HF-hosted sources only (ONC, Orcasound, OOI). For each labeled
segment we record the HF repo + path + the labeled window + labels (species
coerced to GBIF canonical, plus ecotype / call-type / source / license / split).
We also sample ONC "negative" (unlabeled) clips ~1/5 of the ONC positive count.

Audio is later cropped-on-download to the labeled ~15 s window (the ONC source
recordings are ~5-7 min / ~32 MB each), so we only keep the windows.

Output: dori_phase1_manifest.csv (local + gs://esp-data-ingestion/dori/v0.1.0/metadata/).
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from huggingface_hub import HfApi

DORI_CSV = "https://huggingface.co/datasets/DORI-SRKW/DORI-ONC/resolve/main/DORI.csv"
REPOS = {
    "onc": "DORI-SRKW/DORI-ONC",
    "orcasound": "DORI-SRKW/DORI-Orcasound",
    "ooi": "DORI-SRKW/DORI-OOI",
}
# species_label_clean -> GBIF canonical (verified marine-mammal binomials).
# Ambiguous / non-species map to "" (kept as common name only).
GBIF = {
    "orca": "Orcinus orca",
    "humpback": "Megaptera novaeangliae",
    "pacific white sided dolphin": "Lagenorhynchus obliquidens",
    "fin whale": "Balaenoptera physalus",
    "false killer whale": "Pseudorca crassidens",
    "sperm whale": "Physeter macrocephalus",
    "northern right whale dolphin": "Lissodelphis borealis",
    "gray": "Eschrichtius robustus",
    "minke": "Balaenoptera acutorostrata",
    "sea lion": "",  # genus ambiguous (Zalophus/Eumetopias) -> common only
    "multiple classes": "",
    "noise": "",
    "uncertain": "",
}
ONC_PREFIX = re.compile(r"^(JASCO|ICLISTEN|ICListen|NAXYS|IOS3|ICHYDR|[0-9]{8,})")
ORCA_PREFIX = re.compile(r"^(OS_|rpi-)")
OOI_PREFIX = re.compile(r"^OO-")
DEFAULT_NEG_WINDOW = (60.0, 75.0)  # 15 s window for unlabeled ONC negatives


def source_of(fn: str) -> str | None:
    if ONC_PREFIX.match(fn):
        return "onc"
    if ORCA_PREFIX.match(fn):
        return "orcasound"
    if OOI_PREFIX.match(fn):
        return "ooi"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/dori_staging/dori_phase1_manifest.csv"))
    ap.add_argument(
        "--gcs", default="gs://esp-data-ingestion/dori/v0.1.0/metadata/dori_phase1_manifest.csv"
    )
    ap.add_argument("--neg-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(DORI_CSV)
    df["base"] = df["filename"].astype(str)
    df["src"] = df["base"].map(source_of)
    df = df[df["src"].notna()].copy()
    print("Phase-1 labeled rows by source:\n", df["src"].value_counts().to_string())

    # list ONC repo once -> basename -> (path, split)
    api = HfApi()
    onc_map: dict[str, tuple[str, str]] = {}
    print("listing DORI-ONC repo ...")
    for f in api.list_repo_tree("DORI-SRKW/DORI-ONC", repo_type="dataset", recursive=True):
        p = getattr(f, "path", None)
        if p and p.endswith(".flac"):
            top = p.split("/")[0]
            split = "test" if top == "test" else "train"
            onc_map[p.split("/")[-1]] = (p, split)
    print(f"ONC repo flac: {len(onc_map)}")

    rows = []
    missing = 0
    for _, r in df.iterrows():
        src = r["src"]
        base = r["base"]
        if src == "onc":
            hit = onc_map.get(base)
            if hit is None:
                missing += 1
                continue
            repo_path, split = hit
        else:
            repo_path, split = f"data/{base}", "train"
        sp_clean = str(r.get("species_label_clean", "")).strip().lower()
        rows.append(
            {
                "clip_id": os.path.splitext(base)[0],
                "source": src,
                "repo_id": REPOS[src],
                "repo_path": repo_path,
                "filename": base,
                "segment_start": r.get("segment_start", 0.0),
                "segment_end": r.get("segment_end", 0.0),
                "species_common": sp_clean,
                "species": GBIF.get(sp_clean, ""),
                "ecotype": str(r.get("ecotype_label_clean", "") or "").strip(),
                "call_type": str(r.get("call_annotation_clean", "") or "").strip(),
                "label_source": r.get("species_label_source", ""),
                "license": r.get("license", ""),
                "split": split,
                "is_negative": False,
            }
        )
    print(f"positives: {len(rows)} (ONC unresolved dropped: {missing})")

    # ONC negatives: unlabeled ONC files not referenced as positives
    pos_onc = {r["filename"] for r in rows if r["source"] == "onc"}
    n_onc_pos = len(pos_onc)
    neg_pool = sorted(set(onc_map) - set(df.loc[df["src"] == "onc", "base"]))
    n_neg = min(int(round(args.neg_frac * n_onc_pos)), len(neg_pool))
    import random

    rng = random.Random(args.seed)
    for base in rng.sample(neg_pool, n_neg):
        repo_path, split = onc_map[base]
        rows.append(
            {
                "clip_id": os.path.splitext(base)[0],
                "source": "onc",
                "repo_id": REPOS["onc"],
                "repo_path": repo_path,
                "filename": base,
                "segment_start": DEFAULT_NEG_WINDOW[0],
                "segment_end": DEFAULT_NEG_WINDOW[1],
                "species_common": "",
                "species": "",
                "ecotype": "",
                "call_type": "",
                "label_source": "unlabeled",
                "license": "",
                "split": split,
                "is_negative": True,
            }
        )
    print(f"ONC negatives added: {n_neg} (1/{1 / args.neg_frac:.0f} of {n_onc_pos} ONC positives)")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nmanifest rows: {len(out)} -> {args.out}")
    print("by source:\n", out["source"].value_counts().to_string())
    print("by split:\n", out["split"].value_counts().to_string())
    sp = out["species"].replace("", "(none)").value_counts().head(12)
    print("species (canonical, top):\n", sp.to_string())
    os.system(f"gsutil -q cp {args.out} {args.gcs}")
    print(f"uploaded -> {args.gcs}")


if __name__ == "__main__":
    main()

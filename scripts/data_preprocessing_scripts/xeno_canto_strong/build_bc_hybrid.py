"""Build the XC `train_strong_unseen_bchybrid` strong manifest.

Per unseen XC recording, the per-window pseudo-annotations are:
  - focal (human ``canonical_name``) as a full-span event [0, span], and
  - background events from BirdCODE (>=0.40) filtered by a CEB-validated hybrid:
      * recordings WITH Associated Taxa: keep BC species that also appear in AT
        (human-corroborated; ~80% precise, localized by BC times);
      * recordings WITHOUT AT: keep BC species scoring >= 0.70 (~82% precise).

``span`` = max BirdCODE detection end (+pad); this is the annotated region (no
labels exist past the last detection, so true file duration is unnecessary for
windowing). Recordings with no BC detections are dropped.

Output manifest mirrors ``train_strong_unseen_top100_bgdet`` columns, with a
``selection_table`` TSV of ``Begin Time (s)/End Time (s)/Species/Species_Common/
Species_Taxonomic``. Uploaded to
``gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/train_strong_unseen_bchybrid.csv``.
"""

import argparse
import glob
import os
import re
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

SHARDS = os.environ["SHARDS40"]
META = os.environ["META_UNSEEN"]
OUT_GCS = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/train_strong_unseen_bchybrid.csv"
PAD = 1.0
SPAN_CAP = 120.0  # cap the windowed span; long recordings would blow up window counts
DEFAULT_SPAN = 10.0  # window span for recordings with no BirdCODE detection (focal-only)
BC_LO = 0.40
NONAT_HI = float(os.environ.get("NONAT_HI", "0.70"))  # non-AT branch threshold (~79% added-bg precision vs ~70% at 0.60)
_X = re.compile(r"XC(\d+)")
_PREFIX = re.compile(r"(?i)^\s*(has background sounds?|also|background|other)\s*[:\-]\s*")
conv = GBIFConverter(gbif_animals_tsv_fp=os.environ["GBIF_TSV"], cache_path=None)
_SPELLFIX = {"Phylloscopus sibillatrix": "Phylloscopus sibilatrix"}
_cc = {}


def canon(name: str) -> str | None:
    name = _SPELLFIX.get(str(name).strip(), str(name).strip())
    if not name:
        return None
    if name not in _cc:
        info, ok = conv(name)
        cn = info.get("canonicalName") if ok else None
        _cc[name] = " ".join(cn.split()[:2]) if cn and len(cn.split()) >= 2 else (
            name if len(name.split()) >= 2 else None)
    return _cc[name]


def parse_assoc(s: str) -> set[str]:
    if not s or str(s).strip() in ("", "nan"):
        return set()
    out = set()
    for p in re.split(r"[|;]", _PREFIX.sub("", str(s))):
        p = _PREFIX.sub("", p).strip()
        low = p.lower()
        if not p or "sp." in low or low.startswith(("unident", "unknown")):
            continue
        if (c := canon(" ".join(p.split()[:2]))):
            out.add(c)
    return out


def main(limit: int | None = None) -> None:
    # gcs_path is REQUIRED: XenoCanto._resolve_remote_audio_paths uses it to
    # rewrite 32khz_path/16khz_path to the correct per-row bucket (audio lives in
    # both esp-ml-datasets and esp-data-ingestion). Without it the loader falls
    # back to a fixed bucket and can't find the ~38% esp-data-ingestion-origin
    # recordings' presampled audio.
    md = pd.read_csv(META, dtype=str, keep_default_na=False,
                     usecols=["xc_id", "relative_path", "gcs_path", "canonical_name",
                              "species_common", "phylum", "class", "order", "family", "genus",
                              "Associated Taxa", "32khz_path", "16khz_path",
                              "latitudeDecimal", "longitudeDecimal", "locality"])
    md = md.rename(columns={"Associated Taxa": "assoc", "class": "cls",
                            "32khz_path": "p32", "16khz_path": "p16"})
    md["xid"] = md["relative_path"].map(lambda s: (m := _X.search(s or "")) and m.group(1))
    md = md[md["xid"].notna()].drop_duplicates("xid", keep="first")
    unseen = set(md["xid"])
    print(f"unseen recordings: {len(unseen):,}", flush=True)

    # canonical -> taxonomy string + canonical -> common name, from metadata
    # (used to fill background events, which only carry a scientific name)
    tax, common = {}, {}
    for r in md.itertuples(index=False):
        c = canon(r.canonical_name)
        if not c:
            continue
        if c not in tax and r.phylum:
            tax[c] = f"{r.phylum} {r.cls} {r.order} {r.family} {c}".strip()
        if c not in common and r.species_common:
            common[c] = r.species_common
    print(f"taxonomy map entries: {len(tax):,} | common-name map: {len(common):,}", flush=True)

    # BirdCODE >=0.40 detections for unseen recordings
    det = defaultdict(list)  # xid -> [(species_canon, begin, end, score)]
    for sp in sorted(glob.glob(f"{SHARDS}/shard_*.npz")):
        d = np.load(sp, allow_pickle=True)
        fids = d["file_ids"]
        for i in range(len(fids)):
            m = _X.search(str(fids[i]))
            if not m or m.group(1) not in unseen:
                continue
            xid = m.group(1)
            for ln in str(d[f"table_{i}"][0]).splitlines()[1:]:
                p = ln.split("\t")
                if len(p) < 4:
                    continue
                try:
                    b, e, s = float(p[0]), float(p[1]), float(p[3])
                except ValueError:
                    continue
                if (c := canon(p[2])):
                    det[xid].append((c, b, e, s))
    print(f"recordings with BC detections: {len(det):,}", flush=True)

    rows = []
    n_at = n_multi = n_nobc = 0
    # iterate ALL unseen recordings so the split covers the full train_unseen
    # set; recordings without any BirdCODE detection get a focal-only table
    # (no extra species) at a default span.
    rows_iter = list(md.itertuples(index=False))
    if limit:
        rows_iter = rows_iter[:limit]
    for meta in rows_iter:
        xid = meta.xid
        dets = det.get(xid, [])
        focal = canon(meta.canonical_name)
        at = parse_assoc(meta.assoc)
        at_covered = len(at) > 0
        if dets:
            span = min(max(e for _, _, e, _ in dets) + PAD, SPAN_CAP)
        else:
            span = DEFAULT_SPAN
            n_nobc += 1
        # focal is a full-span event; background events kept per the hybrid rule
        seln = []
        if focal:
            seln.append((0.0, round(span, 4), focal, meta.species_common,
                         tax.get(focal, focal)))
        for (c, b, e, s) in sorted(dets, key=lambda x: x[1]):
            if c == focal or b >= span:
                continue
            keep = (c in at) if at_covered else (s >= NONAT_HI)
            if not keep:
                continue
            seln.append((round(b, 4), round(min(e, span), 4), c, common.get(c, ""), tax.get(c, c)))
        n_sp = len({r[2] for r in seln})
        if n_sp == 0:
            continue
        if at_covered and dets:
            n_at += 1
        if n_sp >= 2:
            n_multi += 1
        st = "Begin Time (s)\tEnd Time (s)\tSpecies\tSpecies_Common\tSpecies_Taxonomic\n"
        st += "\n".join(f"{a}\t{b}\t{c}\t{cm}\t{tx}" for (a, b, c, cm, tx) in seln)
        rows.append({
            "relative_path": meta.relative_path, "gcs_path": meta.gcs_path,
            "32khz_path": meta.p32, "16khz_path": meta.p16,
            "canonical_name": meta.canonical_name, "species_common": meta.species_common,
            "phylum": meta.phylum, "class": meta.cls, "order": meta.order,
            "family": meta.family, "genus": meta.genus, "audio_duration": round(span, 4),
            "selection_table": st, "xc_id": xid, "n_species": n_sp, "has_at": at_covered,
            "latitudeDecimal": meta.latitudeDecimal, "longitudeDecimal": meta.longitudeDecimal,
            "locality": meta.locality,
        })
    out = pd.DataFrame(rows)
    print(f"\nmanifest rows: {len(out):,} | with AT: {n_at:,} | "
          f"multi-species (>=2): {n_multi:,} | focal-only (no BC): {n_nobc:,}", flush=True)
    print("n_species distribution:", out["n_species"].value_counts().sort_index().head(8).to_dict())
    # write to a persistent (NFS) location so the manifest survives even if the
    # GCS upload fails on expired creds; upload is best-effort.
    local = os.environ.get("LOCAL_OUT", "/tmp/train_strong_unseen_bchybrid.csv")
    out.to_csv(local, index=False)
    print(f"wrote {local}", flush=True)
    if not limit:
        r = subprocess.run(["gsutil", "-q", "cp", local, OUT_GCS])
        print(f"uploaded -> {OUT_GCS}" if r.returncode == 0
              else f"UPLOAD FAILED (rc={r.returncode}); manifest at {local}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)

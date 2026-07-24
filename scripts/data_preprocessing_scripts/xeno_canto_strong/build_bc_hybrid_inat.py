"""Build the iNaturalist `train_strong_unseen_bchybrid` strong manifest.

Mirror of ``build_bc_hybrid.py`` for iNat. Join key = stem of ``originals_path``
(== BirdCODE file_id minus ``audio/`` prefix and extension; 96.8% match). iNat
observations are single-species with (near-)empty Associated Taxa, so nearly all
recordings take the non-AT branch (BirdCODE >= NONAT_HI). ``32khz_path`` /
``16khz_path`` are already full gs:// URIs and are copied through unchanged.

Output columns mirror the XC bchybrid manifest; uploaded (best-effort, else left
on NFS via LOCAL_OUT) to
``gs://esp-data-ingestion/inaturalist/v0.1.0/raw/train_strong_unseen_bchybrid.csv``.
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

SHARDS = os.environ["SHARDS_INAT"]
META = os.environ["META_INAT"]
OUT_GCS = "gs://esp-data-ingestion/inaturalist/v0.1.0/raw/train_strong_unseen_bchybrid.csv"
PAD = 1.0
SPAN_CAP = 120.0
DEFAULT_SPAN = 10.0
NONAT_HI = float(os.environ.get("NONAT_HI", "0.70"))  # non-AT branch threshold (~79% added-bg precision vs ~70% at 0.60)
_PREFIX = re.compile(r"(?i)^\s*(has background sounds?|also|background|other)\s*[:\-]\s*")
conv = GBIFConverter(gbif_animals_tsv_fp=os.environ["GBIF_TSV"], cache_path=None)
_SPELLFIX = {"Phylloscopus sibillatrix": "Phylloscopus sibilatrix"}
_cc = {}


def stem(p: str) -> str:
    # originals_path / file_id -> join key (drop `audio/` prefix + extension)
    p = re.sub(r"^audio/", "", str(p))
    return re.sub(r"\.[^./]+$", "", p)


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
    md = pd.read_csv(META, dtype=str, keep_default_na=False,
                     usecols=["originals_path", "canonical_name", "species_common",
                              "phylum", "class", "order", "family", "genus",
                              "Associated Taxa", "32khz_path", "16khz_path"])
    md = md.rename(columns={"Associated Taxa": "assoc", "class": "cls",
                            "originals_path": "relative_path",
                            "32khz_path": "p32", "16khz_path": "p16"})
    md["key"] = md["relative_path"].map(stem)
    md = md[(md["key"] != "") & md["p32"].astype(bool)].drop_duplicates("key", keep="first")
    keys = set(md["key"])
    print(f"iNat unseen recordings: {len(keys):,}", flush=True)

    tax, common = {}, {}
    for r in md.itertuples(index=False):
        c = canon(r.canonical_name)
        if not c:
            continue
        if c not in tax and r.phylum:
            tax[c] = f"{r.phylum} {r.cls} {r.order} {r.family} {c}".strip()
        if c not in common and r.species_common:
            common[c] = r.species_common
    print(f"taxonomy map: {len(tax):,} | common-name map: {len(common):,}", flush=True)

    det = defaultdict(list)  # key -> [(species_canon, begin, end, score)]
    for sp in sorted(glob.glob(f"{SHARDS}/shard_*.npz")):
        d = np.load(sp, allow_pickle=True)
        fids = d["file_ids"]
        for i in range(len(fids)):
            k = stem(str(fids[i]))
            if k not in keys:
                continue
            for ln in str(d[f"table_{i}"][0]).splitlines()[1:]:
                p = ln.split("\t")
                if len(p) < 4:
                    continue
                try:
                    b, e, s = float(p[0]), float(p[1]), float(p[3])
                except ValueError:
                    continue
                if (c := canon(p[2])):
                    det[k].append((c, b, e, s))
    print(f"recordings with BC detections: {len(det):,}", flush=True)

    rows = []
    n_multi = n_nobc = 0
    rows_iter = list(md.itertuples(index=False))
    if limit:
        rows_iter = rows_iter[:limit]
    for meta in rows_iter:
        k = meta.key
        dets = det.get(k, [])
        focal = canon(meta.canonical_name)
        at = parse_assoc(meta.assoc)
        at_covered = len(at) > 0
        if dets:
            span = min(max(e for _, _, e, _ in dets) + PAD, SPAN_CAP)
        else:
            span = DEFAULT_SPAN
            n_nobc += 1
        seln = []
        if focal:
            seln.append((0.0, round(span, 4), focal, meta.species_common,
                         tax.get(focal, focal)))
        # BirdCODE only predicts birds: only add background to Aves-focal
        # recordings; non-bird recordings (frogs/insects/mammals) stay
        # focal-only to avoid hallucinated bird labels on non-bird audio.
        bg_dets = dets if meta.cls == "Aves" else []
        for (c, b, e, s) in sorted(bg_dets, key=lambda x: x[1]):
            if c == focal or b >= span:
                continue
            keep = (c in at) if at_covered else (s >= NONAT_HI)
            if not keep:
                continue
            seln.append((round(b, 4), round(min(e, span), 4), c, common.get(c, ""), tax.get(c, c)))
        n_sp = len({r[2] for r in seln})
        if n_sp == 0:
            continue
        if n_sp >= 2:
            n_multi += 1
        st = "Begin Time (s)\tEnd Time (s)\tSpecies\tSpecies_Common\tSpecies_Taxonomic\n"
        st += "\n".join(f"{a}\t{b}\t{c}\t{cm}\t{tx}" for (a, b, c, cm, tx) in seln)
        rows.append({
            "relative_path": meta.relative_path, "32khz_path": meta.p32, "16khz_path": meta.p16,
            "canonical_name": meta.canonical_name, "species_common": meta.species_common,
            "phylum": meta.phylum, "class": meta.cls, "order": meta.order,
            "family": meta.family, "genus": meta.genus, "audio_duration": round(span, 4),
            "selection_table": st, "sound_id": k, "n_species": n_sp, "has_at": at_covered,
        })
    out = pd.DataFrame(rows)
    print(f"\nmanifest rows: {len(out):,} | multi-species (>=2): {n_multi:,} | "
          f"focal-only (no BC): {n_nobc:,}", flush=True)
    print("n_species distribution:", out["n_species"].value_counts().sort_index().head(8).to_dict())
    local = os.environ.get("LOCAL_OUT", "/tmp/train_strong_unseen_bchybrid_inat.csv")
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

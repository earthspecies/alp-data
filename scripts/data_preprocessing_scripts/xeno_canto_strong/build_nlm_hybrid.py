"""Build the NatureLM-corroborated XC pseudolabel split `train_strong_unseen_nlmhybrid`.

Extends build_bc_hybrid.py. Decision rule = multi-source corroboration (each tier
independently ~80% precise on CEB); location comes from BirdCODE (tight), with a
low-threshold frame-based backfill for NLM-added species that BirdCODE missed at
≥0.40, and an NLM-window fallback only as last resort.

Per unseen XC recording, PRESENCE (which species):
  focal (human)                                     always
  BC∩AT   : BC≥0.40 ∈ AssociatedTaxa                if has_at
  BC≥0.70                                           if not has_at
  NLM∩BC  : BC≥NLM_BC_LO(0.50) ∩ NLM                always
  NLM∩AT  : NLM ∈ AssociatedTaxa                    if has_at
`label_corroborated` = any non-focal species came from AT- or NLM-corroboration.

LOCATION (per admitted species, in priority order):
  1. BirdCODE ≥0.40 boxes for that species (tight), else
  2. frame-based BirdCODE boxes at LOC_THR (0.30) re-thresholded from raw
     per-frame scores (for NLM∩AT species BC missed at 0.40), else
  3. NLM-firing windows (coarse 10 s), else
  4. focal full-span.

Env: META_UNSEEN, SHARDS40 (BC≥0.40 npz), FB_SHARDS (frame_based npz),
     NLM_PREDS (jsonl: {xc_id, nlm_species:[canon...], nlm_windows:[[s,e,[sp..]]]}),
     GBIF_TSV, NONAT_HI(0.70), NLM_BC_LO(0.50), LOC_THR(0.30), MERGE_GAP(0.3),
     MIN_DUR(0.1), SPAN_CAP(120), OUT
Selection-table cols (bgdet schema): Begin Time (s), End Time (s), Species,
Species_Common, Species_Taxonomic  (+ provenance column `Source`).
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from esp_data.discover.gbif_taxonomy import GBIFConverter

META = os.environ["META_UNSEEN"]
SHARDS = os.environ["SHARDS40"]
FB = os.environ["FB_SHARDS"]
NLM_PREDS = os.environ["NLM_PREDS"]
NONAT_HI = float(os.environ.get("NONAT_HI", "0.70"))
NLM_BC_LO = float(os.environ.get("NLM_BC_LO", "0.50"))
LOC_THR = float(os.environ.get("LOC_THR", "0.30"))
MERGE_GAP = float(os.environ.get("MERGE_GAP", "0.3"))
MIN_DUR = float(os.environ.get("MIN_DUR", "0.1"))
SPAN_CAP = float(os.environ.get("SPAN_CAP", "120"))
OUT = os.environ.get("OUT", "/tmp/train_strong_unseen_nlmhybrid.csv")
PAD = 1.0
_X = re.compile(r"XC(\d+)")

conv = GBIFConverter(gbif_animals_tsv_fp=os.environ["GBIF_TSV"], cache_path=None)
_SPELLFIX = {"Phylloscopus sibillatrix": "Phylloscopus sibilatrix"}
_cache: dict[str, str | None] = {}


def canon(name):
    name = _SPELLFIX.get(str(name).strip(), str(name).strip())
    if not name:
        return None
    if name not in _cache:
        try:
            info, ok = conv(name)
            cn = info.get("canonicalName") if ok else None
        except Exception:  # noqa: BLE001
            cn = None
        _cache[name] = (" ".join(cn.split()[:2]) if cn and len(cn.split()) >= 2
                        else (name if len(name.split()) >= 2 else None))
    return _cache[name]


_ATP = re.compile(r"^(has background sounds|also|background|other)\s*:?", re.I)


def parse_at(s):
    if not s:
        return set()
    return {c for tok in re.split(r"[|;,]+", _ATP.sub("", str(s)).strip()) if (c := canon(tok.strip()))}


def frames_to_boxes(prob, fr, thr):
    on = prob >= thr
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        out.append([i / fr, (j + 1) / fr])
        i = j + 1
    merged = []
    for b in out:
        if merged and b[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [tuple(m) for m in merged if m[1] - m[0] >= MIN_DUR]


def main():
    """See module docstring."""
    md = pd.read_csv(META, dtype=str, keep_default_na=False)
    md["xid"] = md["xc_id"].map(lambda s: (_X.search("XC" + str(s)) or _X.search(str(s))).group(1)
                                if re.search(r"\d", str(s)) else None)
    tax, common = {}, {}
    for r in md.itertuples(index=False):
        c = canon(r.canonical_name)
        if c:
            tax.setdefault(c, f"{r.phylum} {getattr(r, 'class')} {r.order} {r.family} {c}".strip())
            common.setdefault(c, r.species_common)

    # NLM predictions per recording (species set + per-window firings for fallback).
    nlm_sp, nlm_win = defaultdict(set), defaultdict(list)
    for x in open(NLM_PREDS):
        d = json.loads(x)
        xid = d["xc_id"]
        nlm_sp[xid] |= {canon(s) for s in d.get("nlm_species", []) if canon(s)}
        for w in d.get("nlm_windows", []):
            nlm_win[xid].append((w[0], w[1], {canon(s) for s in w[2] if canon(s)}))

    # BirdCODE >=0.40 detections (species -> boxes + max score).
    bc = defaultdict(lambda: defaultdict(list))   # xid -> sp -> [(b,e,score)]
    want = set(md["xid"])
    for sp_f in sorted(glob.glob(f"{SHARDS}/shard_*.npz")):
        d = np.load(sp_f, allow_pickle=True)
        for i in range(len(d["file_ids"])):
            m = _X.search(str(d["file_ids"][i]))
            if not m or m.group(1) not in want:
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
                    bc[m.group(1)][c].append((b, e, s))

    # frame-based lookup is opened lazily per shard only for xids needing backfill.
    # (Two-pass: first decide presence + which (xid,sp) need FB boxes, then scan FB.)
    rows, need_fb = [], defaultdict(set)
    plan = {}
    for meta in md.itertuples(index=False):
        xid = meta.xid
        if not xid:
            continue
        focal = canon(meta.canonical_name)
        at = parse_at(meta.assoc if hasattr(meta, "assoc") else getattr(meta, "Associated Taxa", ""))
        bcd = bc.get(xid, {})
        bc_hi = {c for c, evs in bcd.items() if max(s for _, _, s in evs) >= NONAT_HI}
        bc_mid = {c for c, evs in bcd.items() if max(s for _, _, s in evs) >= NLM_BC_LO}
        nlm = nlm_sp.get(xid, set())
        present = set()
        if focal:
            present.add(focal)
        present |= (set(bcd) & at) if at else bc_hi        # BC∩AT or BC≥0.7
        present |= (bc_mid & nlm)                           # NLM∩BC
        if at:
            present |= (nlm & at)                           # NLM∩AT
        present.discard(None)
        if not present:
            continue
        corrob = bool(at) or bool((bc_mid & nlm) | (nlm & at))
        span = min(max((e for evs in bcd.values() for _, e, _ in evs), default=10.0) + PAD, SPAN_CAP)
        plan[xid] = {"meta": meta, "focal": focal, "present": present, "corrob": corrob, "span": span}
        for sp in present:
            if sp != focal and sp not in bcd:               # needs FB (or window) localization
                need_fb[xid].add(sp)

    # scan FB shards once for the needed (xid, sp) boxes
    fb_box = defaultdict(dict)
    for shard in sorted(glob.glob(f"{FB}/shard_*.npz")):
        d = np.load(shard, allow_pickle=True)
        fids, frs = d["file_ids"], d["framerates"]
        for i in range(len(fids)):
            m = _X.search(str(fids[i]))
            if not m or m.group(1) not in need_fb:
                continue
            xid = m.group(1); fr = float(frs[i])
            preds = d[f"preds_{i}"]; classes = list(d[f"classes_{i}"])
            if preds.ndim == 2 and preds.shape[0] == len(classes) and preds.shape[1] != len(classes):
                preds = preds.T
            col = {}
            for k, cn in enumerate(classes):
                if (c := canon(str(cn))):
                    col.setdefault(c, k)
            for sp in need_fb[xid]:
                if sp in col and sp not in fb_box[xid]:
                    bx = frames_to_boxes(np.asarray(preds[:, col[sp]], dtype=np.float32), fr, LOC_THR)
                    if bx:
                        fb_box[xid][sp] = bx

    # assemble selection tables
    for xid, pl in plan.items():
        meta, focal, present, span = pl["meta"], pl["focal"], pl["present"], pl["span"]
        bcd = bc.get(xid, {})
        seln = []
        for sp in sorted(present):
            if sp == focal:
                seln.append((0.0, round(span, 4), sp, "focal"))
                continue
            if sp in bcd:                                   # BC ≥0.40 boxes
                for (b, e, _) in bcd[sp]:
                    if b < span:
                        seln.append((round(b, 4), round(min(e, span), 4), sp, "bc"))
            elif sp in fb_box.get(xid, {}):                 # frame-based backfill
                for (b, e) in fb_box[xid][sp]:
                    if b < span:
                        seln.append((round(b, 4), round(min(e, span), 4), sp, "fb"))
            elif nlm_win.get(xid):                          # NLM-window fallback
                for (ws, we, sps) in nlm_win[xid]:
                    if sp in sps and ws < span:
                        seln.append((round(ws, 4), round(min(we, span), 4), sp, "nlmwin"))
            else:
                seln.append((0.0, round(span, 4), sp, "span"))
        if not seln:
            continue
        st = "Begin Time (s)\tEnd Time (s)\tSpecies\tSpecies_Common\tSpecies_Taxonomic\tSource\n"
        st += "\n".join(f"{a}\t{b}\t{c}\t{common.get(c, '')}\t{tax.get(c, c)}\t{src}"
                        for (a, b, c, src) in seln)
        rows.append({
            "relative_path": meta.relative_path, "gcs_path": meta.gcs_path,
            "32khz_path": meta.p32 if hasattr(meta, "p32") else getattr(meta, "32khz_path", ""),
            "16khz_path": meta.p16 if hasattr(meta, "p16") else getattr(meta, "16khz_path", ""),
            "canonical_name": meta.canonical_name, "species_common": meta.species_common,
            "phylum": meta.phylum, "class": getattr(meta, "class"), "order": meta.order,
            "family": meta.family, "genus": meta.genus,
            "audio_duration": round(span, 4), "selection_table": st, "xc_id": xid,
            "n_species": len(present), "has_at": bool(parse_at(getattr(meta, "assoc", ""))),
            "label_corroborated": pl["corrob"],
        })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(rows)} recordings", flush=True)


if __name__ == "__main__":
    main()

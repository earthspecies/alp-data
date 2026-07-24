"""Ingest the Bengalese Finch Song Repository (Nicholson, Queen & Sober 2017).

Source: figshare 10.6084/m9.figshare.4805749 (CC BY 4.0). Four birds
(bl26lb16, gr41rd51, gy6or6, or60yw70), song bouts recorded at 32 kHz as
``.cbin`` with per-syllable ``.cbin.not.mat`` annotations (evsonganaly):
onsets/offsets in ms + a single-character label per syllable.

This script:
  1. decodes each ``.cbin`` to a 32 kHz mono wav (evfuncs; no resample — the
     recordings are already 32 kHz, matching the NatureLM stack),
  2. parses the ``.not.mat`` into a WABAD/CEB-shaped inline ``selection_table``
     TSV (``Begin Time (s)``, ``End Time (s)``, ``Annotation``), and
  3. writes one manifest row per song bout.

Used as a held-out subsegmentation evaluation (syllable boundary detection).
Run with the ingest venv (evfuncs, soundfile). Audio + manifest are then
staged to gs://esp-data-ingestion/bengalese_finch/.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
from pathlib import Path

import evfuncs
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_BIRDS = ("bl26lb16", "gr41rd51", "gy6or6", "or60yw70")


def _selection_table(onsets_ms: np.ndarray, offsets_ms: np.ndarray, labels: str) -> str:
    """Build the inline selection-table TSV from a .not.mat annotation."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    w.writerow(["Begin Time (s)", "End Time (s)", "Annotation"])
    for on, off, lab in zip(np.atleast_1d(onsets_ms), np.atleast_1d(offsets_ms), labels, strict=False):
        if off <= on:
            continue
        w.writerow([f"{on / 1000.0:.4f}", f"{off / 1000.0:.4f}", lab])
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="Dir with extracted <bird>/<day>/*.cbin(+.not.mat)")
    ap.add_argument("--wav-out", type=Path, required=True, help="Output dir for decoded wavs")
    ap.add_argument("--manifest", type=Path, required=True, help="Output manifest CSV")
    args = ap.parse_args()

    args.wav_out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_bad = 0
    cbins = sorted(args.raw_dir.rglob("*.cbin"))
    logger.info(f"found {len(cbins)} cbin files under {args.raw_dir}")
    for cbin in cbins:
        notmat = cbin.with_suffix(".cbin.not.mat")
        if not notmat.exists():
            n_bad += 1
            continue
        # bird / day from path (<bird>/<day>/<name>.cbin)
        parts = cbin.relative_to(args.raw_dir).parts
        bird = next((p for p in parts if p in _BIRDS), parts[0])
        day = cbin.parent.name
        try:
            data, fs = evfuncs.load_cbin(str(cbin))
            ann = evfuncs.load_notmat(str(cbin))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"skip {cbin.name}: {e}")
            n_bad += 1
            continue
        rel = f"{bird}/{day}/{cbin.stem}.wav"
        out_wav = args.wav_out / rel
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        # cbin is int16; write a 32 kHz PCM_16 wav
        sf.write(out_wav, np.asarray(data).astype(np.int16), int(fs), subtype="PCM_16")
        onsets = np.atleast_1d(ann["onsets"]); offsets = np.atleast_1d(ann["offsets"])
        st = _selection_table(onsets, offsets, str(ann["labels"]))
        rows.append({
            "filepath": rel,
            "audio_fp": rel,
            "bird_id": bird,
            "day": day,
            "source_cbin": cbin.name,
            "sample_rate": int(fs),
            "duration_s": round(len(data) / float(fs), 3),
            "n_syllables": int(len(onsets)),
            "selection_table": st,
        })
    logger.info(f"decoded {len(rows)} recordings ({n_bad} skipped)")

    cols = ["filepath", "audio_fp", "bird_id", "day", "source_cbin",
            "sample_rate", "duration_s", "n_syllables", "selection_table"]
    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # summary
    from collections import Counter
    per_bird = Counter(r["bird_id"] for r in rows)
    tot_syl = sum(r["n_syllables"] for r in rows)
    logger.info(f"per-bird recordings: {dict(per_bird)}")
    logger.info(f"total syllables: {tot_syl} | manifest -> {args.manifest}")


if __name__ == "__main__":
    main()

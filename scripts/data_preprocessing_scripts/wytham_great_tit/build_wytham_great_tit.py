"""Ingest the Wytham Great Tit Song Dataset (Merino Recalde et al. 2024).

Source: OSF 10.17605/OSF.IO/N8AC9 (CC BY 4.0). 109,963 songs (one 22.05 kHz
mono wav each) from 455 birds, with 1,161,033 note segments in
``great-tit-hits-crowsetta.csv`` (per-note ``onset_s`` / ``offset_s`` +
``notated_path`` = song wav; ``label`` = per-song song-type). Median 10
notes/song.

This script builds a WABAD/CEB-shaped manifest (one row per song, inline
``selection_table`` TSV of note onsets/offsets) for every song whose wav is
present under ``--wav-dir`` (the eval subset, pre-extracted from the OSF
split-zip with ``7z x sf.zip -i@include.txt`` — the ``zip -s`` volumes can't be
read via Python ``zipfile`` after a plain ``cat``). Audio is left at 22.05 kHz
(the dataset class resamples to 32 kHz on load). Used as a held-out
subsegmentation evaluation (note boundary detection).
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
csv.field_size_limit(1 << 30)


def _selection_table(notes: list[tuple[float, float]], label: str) -> str:
    """Inline selection-table TSV from a song's note (onset_s, offset_s) list."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    w.writerow(["Begin Time (s)", "End Time (s)", "Annotation"])
    for on, off in sorted(notes):
        if off > on:
            w.writerow([f"{on:.4f}", f"{off:.4f}", label])
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crowsetta", type=Path, required=True, help="great-tit-hits-crowsetta.csv")
    ap.add_argument("--wav-dir", type=Path, required=True,
                    help="dir of pre-extracted eval-subset wavs (recursively globbed for *.wav)")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    present = {p.name: p for p in args.wav_dir.rglob("*.wav")}
    logger.info(f"{len(present)} wavs present under {args.wav_dir}")

    # Group notes by song (only for songs we have audio for), keep bird + song-type.
    per_song: dict[str, list[tuple[float, float]]] = defaultdict(list)
    song_meta: dict[str, tuple[str, str]] = {}
    with args.crowsetta.open(newline="") as f:
        for r in csv.DictReader(f):
            song = r["notated_path"]
            if song not in present:
                continue
            per_song[song].append((float(r["onset_s"]), float(r["offset_s"])))
            if song not in song_meta:
                song_meta[song] = (song.split("_")[0], r["label"])
    logger.info(f"crowsetta notes for present songs: {sum(len(v) for v in per_song.values())} "
                f"across {len(per_song)} songs")

    rows = []
    for song in sorted(per_song):
        bird, label = song_meta[song]
        notes = per_song[song]
        rows.append({
            "filepath": song,
            "audio_fp": song,
            "bird_id": bird,
            "song_type": label,
            "n_notes": len(notes),
            "selection_table": _selection_table(notes, label),
        })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "audio_fp", "bird_id", "song_type", "n_notes", "selection_table"]
    with args.manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"wrote {len(rows)} songs / {sum(r['n_notes'] for r in rows)} notes -> {args.manifest}")


if __name__ == "__main__":
    main()

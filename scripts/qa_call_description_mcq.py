#!/usr/bin/env python3
"""QA the call-description MCQ dataset by rendering cropped spectrograms.

Loads rows through the ROOTS adapter (so this doubles as a load test of the
``__crop_`` + 32 kHz redirect path), then renders a labelled mel-spectrogram
grid for a stratified sample so the crops can be visually checked: each panel
shows the focal common name, the gold description, the recordist behavior tag,
and the crop bounds. Saves a PNG montage.

Usage::

    uv run python scripts/qa_call_description_mcq.py \
        --jsonl data/roots_call_description_mcq/call_description_mcq_iconic_v1.jsonl \
        --out data/roots_call_description_mcq/qa_grid.png --n 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.datasets.roots import ROOTS  # noqa: E402


def pick_indices(jsonl: Path, n: int, seed: int = 0) -> list[int]:
    """Pick ``n`` row indices spread across distinct focal species + sources."""
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[int]] = {}
    with open(jsonl) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            meta = json.loads(row["metadata"])
            key = (meta["correct_species"], row["source_dataset"])
            by_key.setdefault(key, []).append(i)
    keys = list(by_key)
    rng.shuffle(keys)
    picks = [rng.choice(by_key[k]) for k in keys[:n]]
    return picks


def main() -> None:
    """Render a QA spectrogram grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "roots_call_description_mcq"
        / "call_description_mcq_iconic_v1.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "roots_call_description_mcq" / "qa_grid.png",
    )
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = ROOTS(split="_qa", jsonl_path=str(args.jsonl), sample_rate=32000)
    indices = pick_indices(args.jsonl, args.n, args.seed)

    ncol = 3
    nrow = (len(indices) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices, strict=False):
        item = ds[idx]
        meta = json.loads(item["metadata"])
        audio = item["audio"]
        sr = item["sample_rate"]
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr // 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, fmax=sr // 2)
        dur = len(audio) / sr
        desc = meta["correct_common_name"]
        title = (
            f"{desc} [{item['source_dataset'][:4]}] "
            f"voc={meta['vocalization_type']} beh={meta['behavior'] or '-'}\n"
            f"crop {meta['crop_start_sec']:.1f}-{meta['crop_end_sec']:.1f}s "
            f"({dur:.1f}s) det={meta['detection_score']:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved QA grid to {args.out}")

    # Print the textual description matched to each panel for cross-checking.
    for idx in indices:
        with open(args.jsonl) as f:
            for i, line in enumerate(f):
                if i == idx:
                    row = json.loads(line)
                    meta = json.loads(row["metadata"])
                    print(
                        f"[{idx}] {meta['correct_common_name']} "
                        f"({meta['correct_species']}) <- "
                        f"\"{[e for e in [meta['correct']]][0]}\" "
                        f"| {row['source_dataset']}"
                    )
                    break


if __name__ == "__main__":
    main()

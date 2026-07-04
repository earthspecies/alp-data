# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "soundfile",
#     "librosa",
#     "matplotlib",
#     "google-cloud-storage",
# ]
# ///
"""Render a spectrogram montage of a random sample of extracted noise clips for
visual verification that no (annotated or unannotated) vocalisations leaked in.

Reads the per-dataset noise manifest from GCS, samples ``--n`` clips, downloads
them, and renders a grid of dB spectrograms to a PNG. Also prints manifest
summary stats (clip count, total duration, RMS distribution).
"""

from __future__ import annotations

import argparse
import io
import os

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from google.cloud import storage

DEST_BASE = "gs://foundation-model-data/audio_32k/noise"


def _split_gs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dest = f"{DEST_BASE}/{args.dataset}"
    out = args.out or f"noise_review_{args.dataset}.png"
    client = storage.Client()
    bucket_name, prefix = _split_gs(dest)
    bucket = client.bucket(bucket_name)

    mraw = bucket.blob(f"{prefix}/noise_manifest_{args.dataset}.csv").download_as_bytes()
    m = pd.read_csv(io.BytesIO(mraw))
    print(f"{args.dataset}: {len(m)} clips from {m['source_fn'].nunique()} recordings, "
          f"{m['dur_s'].sum() / 3600:.2f} h total")
    print("dur_s:", m["dur_s"].describe()[["min", "50%", "max"]].to_dict())
    print("rms percentiles:", {p: round(float(np.percentile(m["rms"], p)), 5) for p in (5, 50, 95)})

    sample = m.sample(n=min(args.n, len(m)), random_state=args.seed).reset_index(drop=True)
    ncol, nrow = 5, int(np.ceil(len(sample) / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, row in sample.iterrows():
        ax = axes[i]
        raw = bucket.blob(f"{prefix}/{row['clip']}").download_as_bytes()
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
        librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax,
                                 cmap="magma")
        ax.set_title(f"{row['clip'][:28]}\nrms={row['rms']:.4f} {row['dur_s']}s", fontsize=6)
        ax.label_outer()
        ax.tick_params(labelsize=5)
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"noise sample — {args.dataset} (n={len(sample)}, seed={args.seed})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=90)
    print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()

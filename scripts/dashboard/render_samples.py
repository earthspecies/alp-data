"""Audio + spectrogram rendering helpers for the dashboard precompute.

Produces per-sample artifacts in `dashboard/assets/samples/<dataset>/`:
- `<idx>.mp3` — audio for browser playback (transcoded via ffmpeg)
- `<idx>.png` — log-mel spectrogram

Used by `scripts.dashboard.build_assets.cmd_build_samples`. Kept
separate so the heavy imports (`librosa`, `matplotlib`, `soundfile`) are
only loaded when this stage runs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

logger = logging.getLogger("dashboard.render_samples")


def center_crop(audio: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    """Center-crop an audio array if it exceeds `max_seconds`.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio samples.
    sr : int
        Sample rate of `audio`.
    max_seconds : float
        Maximum clip length in seconds. Shorter clips are returned as-is.

    Returns
    -------
    np.ndarray
        The (possibly cropped) audio array.
    """
    duration = len(audio) / sr
    if duration <= max_seconds:
        return audio
    half = int(max_seconds * sr) // 2
    center = len(audio) // 2
    return audio[max(0, center - half) : center + half]


def render_log_mel_png(
    audio: np.ndarray,
    sr: int,
    out_path: Path,
    *,
    n_fft: int,
    hop: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> None:
    """Render a log-mel spectrogram PNG to `out_path`.

    Uses a fixed 8x4 figure with no axes for a clean visual that fits
    inside a card. The colormap (`magma`) reads well on dark UI.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio samples in floating-point ``[-1, 1]`` range.
    sr : int
        Sample rate of `audio`.
    out_path : Path
        Destination path for the PNG. Parent directory must exist.
    n_fft, hop, n_mels, fmin, fmax : int
        Mel-spectrogram parameters forwarded to
        `librosa.feature.melspectrogram`.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    librosa.display.specshow(
        log_mel,
        sr=sr,
        hop_length=hop,
        fmin=fmin,
        fmax=fmax,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("#0e0e12")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


def transcode_mp3(audio: np.ndarray, sr: int, out_path: Path) -> None:
    """Write `audio` to a 128kbps VBR MP3 at `out_path`.

    Writes a temporary 16-bit WAV and shells out to ``ffmpeg`` for the
    MP3 encode. Mono (16-bit PCM) keeps the WAV small.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio samples (float in ``[-1, 1]``).
    sr : int
        Sample rate of `audio`.
    out_path : Path
        Destination path (``.mp3``).

    Raises
    ------
    RuntimeError
        If `ffmpeg` is not on PATH or the encode subprocess fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it on the build host.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        sf.write(tmp_wav, audio.astype(np.float32), sr, subtype="PCM_16")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp_wav),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "4",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()}")
    finally:
        tmp_wav.unlink(missing_ok=True)

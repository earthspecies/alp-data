"""Build the synthetic ``whales`` dataset.

Drops Watkins marine-mammal clips onto annotated-as-empty windows of DCLDE
2026 hydrophone recordings, plus a matched number of pure-background
negatives. Writes 32 kHz mono WAV clips and a manifest CSV to
``gs://foundation-model-data/synthetic/whales/v0.1.0/``.

See ``esp_data/datasets/whales.py`` for the dataset class that consumes
the manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from esp_data.datasets.dclde2026 import DCLDE2026
from esp_data.datasets.watkins import Watkins
from esp_data.io import audio_stereo_to_mono, filesystem_from_path, read_audio

logger = logging.getLogger("build_whales_synthetic")

TARGET_SR = 32000
ANNOTATION_PAD_S = 1.0  # buffer added around each DCLDE annotation when finding gaps
RMS_FLOOR = 1e-6  # avoid div-by-zero on perfectly silent segments
PEAK_TARGET = 0.95  # peak-normalize mixed clip to this absolute amplitude

# Watkins columns copied onto positive rows. Negatives leave these empty.
WATKINS_LABEL_COLUMNS: tuple[str, ...] = (
    "species",
    "canonical_name",
    "species_common",
    "call_type",
    "coarse_call_type",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
    "gbifID",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "32khz_path",
    "duration_s",
    "is_positive",
    *WATKINS_LABEL_COLUMNS,
    "event_start_s",
    "event_end_s",
    "snr_db",
    "watkins_source_path",
    "dclde_source_path",
    "dclde_window_start_s",
    "dclde_window_end_s",
    "dclde_provider",
    "dclde_audio_id",
    "mixup_group",
)


# ── Empty-window discovery ────────────────────────────────────────────────


@dataclass(frozen=True)
class EmptyWindow:
    """One contiguous gap in a DCLDE audio file with no annotated event."""

    audio_path: str  # GCS path to the DCLDE 32 kHz WAV
    provider: str
    audio_id: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _parse_event_times(blob: str) -> list[tuple[float, float]]:
    """Extract ``(begin, end)`` tuples from an inline TSV selection table.

    Returns
    -------
    list[tuple[float, float]]
        Annotated event intervals in seconds. Empty if ``blob`` is empty or
        unparsable.
    """
    if not blob or not isinstance(blob, str) or not blob.strip():
        return []
    try:
        df = pd.read_csv(StringIO(blob), sep="\t", keep_default_na=False, na_values=[""])
    except Exception as exc:
        logger.warning("Failed to parse selection table: %s", exc)
        return []
    if "Begin Time (s)" not in df.columns or "End Time (s)" not in df.columns:
        return []
    begin = pd.to_numeric(df["Begin Time (s)"], errors="coerce")
    end = pd.to_numeric(df["End Time (s)"], errors="coerce")
    pairs = list(zip(begin.tolist(), end.tolist(), strict=True))
    return [(float(b), float(e)) for b, e in pairs if pd.notna(b) and pd.notna(e) and e > b]


def find_empty_windows(
    events: list[tuple[float, float]],
    audio_duration_s: float,
    min_gap_s: float,
    pad_s: float = ANNOTATION_PAD_S,
) -> list[tuple[float, float]]:
    """Find unannotated time intervals inside ``[0, audio_duration_s]``.

    Each annotated event is widened by ``pad_s`` on both sides before
    subtraction so a gap never abuts a real call. Only gaps of length
    ``>= min_gap_s`` are returned.

    Parameters
    ----------
    events : list[tuple[float, float]]
        ``(begin, end)`` pairs in seconds.
    audio_duration_s : float
        Total duration of the source audio.
    min_gap_s : float
        Minimum acceptable gap length.
    pad_s : float
        Per-side padding added to every event.

    Returns
    -------
    list[tuple[float, float]]
        ``(start, end)`` pairs of empty windows.
    """
    if audio_duration_s <= 0:
        return []

    padded: list[tuple[float, float]] = []
    for begin, end in events:
        padded.append((max(0.0, begin - pad_s), min(audio_duration_s, end + pad_s)))

    if not padded:
        if audio_duration_s >= min_gap_s:
            return [(0.0, audio_duration_s)]
        return []

    padded.sort()
    merged: list[tuple[float, float]] = []
    cur_begin, cur_end = padded[0]
    for begin, end in padded[1:]:
        if begin <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_begin, cur_end))
            cur_begin, cur_end = begin, end
    merged.append((cur_begin, cur_end))

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for begin, end in merged:
        if begin - cursor >= min_gap_s:
            gaps.append((cursor, begin))
        cursor = max(cursor, end)
    if audio_duration_s - cursor >= min_gap_s:
        gaps.append((cursor, audio_duration_s))
    return gaps


def collect_empty_windows(
    dclde: DCLDE2026,
    min_gap_s: float,
) -> list[EmptyWindow]:
    """Walk DCLDE rows and accumulate empty windows from each file.

    Skips rows where the 32 kHz pre-resampled path is missing.

    Parameters
    ----------
    dclde : DCLDE2026
        DCLDE dataset (loaded; metadata-only access via ``_data``).
    min_gap_s : float
        Minimum gap duration to keep.

    Returns
    -------
    list[EmptyWindow]
    """
    out: list[EmptyWindow] = []
    n_rows = len(dclde._data)  # type: ignore[arg-type]
    for idx in range(n_rows):
        row = dclde._data[idx]  # type: ignore[index]
        path_32k = row.get("32khz_path")
        if not path_32k or not str(path_32k).strip():
            continue
        audio_duration = row.get("audio_duration")
        if audio_duration is None or float(audio_duration) <= 0:
            continue
        events = _parse_event_times(row.get("selection_table", ""))
        gaps = find_empty_windows(events, float(audio_duration), min_gap_s=min_gap_s)
        if not gaps:
            continue
        full_path = str(dclde.data_root / "audio_32k" / str(path_32k))
        provider = str(row.get("provider", "") or "")
        audio_id = str(row.get("audio_id", "") or row.get("audio_path", "") or "")
        for begin, end in gaps:
            out.append(
                EmptyWindow(
                    audio_path=full_path,
                    provider=provider,
                    audio_id=audio_id,
                    start_s=begin,
                    end_s=end,
                )
            )
    return out


# ── Mixing primitives ─────────────────────────────────────────────────────


def _rms(x: np.ndarray) -> float:
    """RMS of ``x`` with a small floor to avoid divide-by-zero on silence.

    Returns
    -------
    float
    """
    if x.size == 0:
        return RMS_FLOOR
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + RMS_FLOOR)


def _best_energy_crop(
    audio: np.ndarray, target_samples: int, sr: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Return the highest-energy contiguous crop of ``target_samples`` samples.

    Uses a coarse stride (~10 ms hops) and picks the window with the largest
    summed squared amplitude, breaking ties randomly. Falls back to a random
    crop if the audio is short enough that no choice is needed.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio.
    target_samples : int
        Crop length.
    sr : int
        Sample rate (used to derive the hop length).
    rng : np.random.Generator
        Source of randomness for tie-breaking.

    Returns
    -------
    tuple[np.ndarray, int]
        The cropped audio and its start-sample offset within ``audio``.
    """
    n = len(audio)
    if n <= target_samples:
        return audio, 0
    hop = max(1, sr // 100)  # ~10 ms
    energy = audio.astype(np.float64) ** 2
    cum = np.concatenate(([0.0], np.cumsum(energy)))
    best_score = -np.inf
    candidates: list[int] = []
    for start in range(0, n - target_samples + 1, hop):
        score = cum[start + target_samples] - cum[start]
        if score > best_score:
            best_score = score
            candidates = [start]
        elif score == best_score:
            candidates.append(start)
    start = int(rng.choice(candidates)) if candidates else 0
    return audio[start : start + target_samples], start


def mix_event_onto_background(
    event: np.ndarray,
    background: np.ndarray,
    event_offset_samples: int,
    snr_db: float,
) -> tuple[np.ndarray, float]:
    """Mix a (possibly shorter) event into a fixed-length background at a target SNR.

    The event is rescaled so ``rms(event) / rms(background) == 10 ** (snr_db / 20)``,
    then summed into ``background`` starting at ``event_offset_samples``. The
    resulting clip is peak-normalized to :data:`PEAK_TARGET`.

    Parameters
    ----------
    event : np.ndarray
        Mono float32 audio of the Watkins event (length <= background).
    background : np.ndarray
        Mono float32 audio of the DCLDE background.
    event_offset_samples : int
        Sample index inside ``background`` where the event starts.
    snr_db : float
        Target SNR of event over background, in dB.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(mixed_audio, peak_gain)`` — the mixed clip and the gain that was
        applied to reach the peak target. Useful for diagnostics.
    """
    bg_rms = _rms(background)
    ev_rms = _rms(event)
    target_ev_rms = bg_rms * (10.0 ** (snr_db / 20.0))
    event_scaled = event * (target_ev_rms / ev_rms)

    mixed = background.copy()
    end = min(event_offset_samples + len(event_scaled), len(mixed))
    mixed[event_offset_samples:end] = (
        mixed[event_offset_samples:end] + event_scaled[: end - event_offset_samples]
    )

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > PEAK_TARGET:
        gain = PEAK_TARGET / peak
        mixed = mixed * gain
        return mixed.astype(np.float32), gain
    return mixed.astype(np.float32), 1.0


def peak_normalize(audio: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """Scale ``audio`` down so its peak amplitude does not exceed ``target``.

    Returns
    -------
    np.ndarray
        Float32 audio. Returned unchanged when the peak is already at or
        below the target.
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32)
    if peak <= target:
        return audio.astype(np.float32)
    return (audio * (target / peak)).astype(np.float32)


# ── I/O ───────────────────────────────────────────────────────────────────


def _read_segment(audio_path: str, start_s: float, end_s: float, target_sr: int) -> np.ndarray:
    """Read a mono float32 segment from ``audio_path`` resampled to ``target_sr``.

    Returns
    -------
    np.ndarray
        Mono float32 audio at ``target_sr``.
    """
    audio, sr = read_audio(audio_path, start_time=start_s, end_time=end_s)
    audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=target_sr,
            scale=True,
            res_type="kaiser_best",
        ).astype(np.float32)
    return audio


def _read_full(audio_path: str, target_sr: int) -> np.ndarray:
    """Read full mono float32 audio from ``audio_path`` resampled to ``target_sr``.

    Returns
    -------
    np.ndarray
        Mono float32 audio at ``target_sr``.
    """
    audio, sr = read_audio(audio_path)
    audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=target_sr,
            scale=True,
            res_type="kaiser_best",
        ).astype(np.float32)
    return audio


def _write_wav(gcs_path: str, audio: np.ndarray, sr: int) -> None:
    """Encode ``audio`` as 16-bit WAV and upload to ``gcs_path``."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    fs = filesystem_from_path(gcs_path)
    with fs.open(gcs_path, "wb") as handle:
        handle.write(buf.read())


# ── Clip-duration sampling ────────────────────────────────────────────────


def sample_clip_duration(
    watkins_duration_s: float | None,
    rng: np.random.Generator,
    duration_min: float,
    duration_max: float,
    short_clip_threshold_s: float = 5.0,
) -> float:
    """Sample the synthesised clip duration given the source Watkins length.

    Rules
    -----
    - Negatives (``watkins_duration_s is None``) draw uniformly from
      ``[duration_min, duration_max]``.
    - Short Watkins clips (``<= short_clip_threshold_s``) expand the clip
      duration so the full Watkins audio fits, never randomly cropping out
      the call.
    - Long Watkins clips truncate to a ``[duration_min, duration_max]``
      random crop; the source is energetic-cropped downstream.

    Returns
    -------
    float
        Clip duration in seconds.
    """
    target = float(rng.uniform(duration_min, duration_max))
    if watkins_duration_s is None:
        return target
    if watkins_duration_s <= short_clip_threshold_s:
        return max(target, float(watkins_duration_s))
    return target


# ── Row builders ──────────────────────────────────────────────────────────


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _empty_row_template() -> dict[str, str]:
    return {col: "" for col in MANIFEST_COLUMNS}


def build_positive_row(
    *,
    idx: int,
    watkins_row: dict[str, object],
    window: EmptyWindow,
    out_audio_root: str,
    rng: np.random.Generator,
    snr_min: float,
    snr_max: float,
    duration_min: float,
    duration_max: float,
) -> dict[str, str]:
    """Synthesise one positive (Watkins-on-DCLDE) clip and return its manifest row.

    Raises
    ------
    ValueError
        If the Watkins row is missing a usable ``32khz_path`` column.

    Returns
    -------
    dict[str, str]
        Manifest row keyed by :data:`MANIFEST_COLUMNS`.
    """
    watkins_duration = watkins_row.get("duration_s")
    watkins_duration_f: float | None
    try:
        watkins_duration_f = float(watkins_duration) if watkins_duration is not None else None
    except (TypeError, ValueError):
        watkins_duration_f = None

    clip_duration = sample_clip_duration(
        watkins_duration_f,
        rng,
        duration_min=duration_min,
        duration_max=duration_max,
    )
    clip_samples = int(round(clip_duration * TARGET_SR))

    # Background window: a random clip-length sub-window inside the chosen gap.
    max_bg_start = max(window.start_s, window.end_s - clip_duration)
    if max_bg_start > window.start_s:
        bg_start = float(rng.uniform(window.start_s, max_bg_start))
    else:
        bg_start = window.start_s
    bg_end = bg_start + clip_duration
    background = _read_segment(window.audio_path, bg_start, bg_end, TARGET_SR)
    if len(background) < clip_samples:
        background = np.pad(background, (0, clip_samples - len(background)))
    else:
        background = background[:clip_samples]

    # Watkins event audio.
    watkins_path = str(watkins_row.get("32khz_path") or "")
    if not watkins_path:
        raise ValueError(f"Watkins row missing 32khz_path: {watkins_row.get('audio_path')}")
    watkins_full = "gs://esp-data-ingestion/watkins/v0.1.0/" + watkins_path
    event_audio = _read_full(watkins_full, TARGET_SR)

    if len(event_audio) > clip_samples:
        event_audio, _crop_start = _best_energy_crop(event_audio, clip_samples, TARGET_SR, rng)

    event_samples = len(event_audio)
    max_offset = max(0, clip_samples - event_samples)
    event_offset_samples = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0

    snr_db = float(rng.uniform(snr_min, snr_max))
    mixed, _peak_gain = mix_event_onto_background(
        event=event_audio,
        background=background,
        event_offset_samples=event_offset_samples,
        snr_db=snr_db,
    )

    clip_id = f"pos_{idx:07d}"
    rel_path = f"audio_32k/{clip_id}.wav"
    gcs_path = f"{out_audio_root}/{rel_path}"
    _write_wav(gcs_path, mixed, TARGET_SR)

    row = _empty_row_template()
    row["clip_id"] = clip_id
    row["32khz_path"] = rel_path
    row["duration_s"] = f"{clip_duration:.6f}"
    row["is_positive"] = "True"
    for col in WATKINS_LABEL_COLUMNS:
        row[col] = _format_value(watkins_row.get(col))
    row["event_start_s"] = f"{event_offset_samples / TARGET_SR:.6f}"
    row["event_end_s"] = f"{(event_offset_samples + event_samples) / TARGET_SR:.6f}"
    row["snr_db"] = f"{snr_db:.3f}"
    row["watkins_source_path"] = watkins_full
    row["dclde_source_path"] = window.audio_path
    row["dclde_window_start_s"] = f"{bg_start:.6f}"
    row["dclde_window_end_s"] = f"{bg_end:.6f}"
    row["dclde_provider"] = window.provider
    row["dclde_audio_id"] = window.audio_id
    row["mixup_group"] = "marine_mammal"
    return row


def build_negative_row(
    *,
    idx: int,
    window: EmptyWindow,
    out_audio_root: str,
    rng: np.random.Generator,
    duration_min: float,
    duration_max: float,
) -> dict[str, str]:
    """Synthesise one pure-background negative clip and return its manifest row.

    Returns
    -------
    dict[str, str]
        Manifest row keyed by :data:`MANIFEST_COLUMNS`.
    """
    clip_duration = sample_clip_duration(
        None,
        rng,
        duration_min=duration_min,
        duration_max=duration_max,
    )
    clip_samples = int(round(clip_duration * TARGET_SR))

    max_bg_start = max(window.start_s, window.end_s - clip_duration)
    if max_bg_start > window.start_s:
        bg_start = float(rng.uniform(window.start_s, max_bg_start))
    else:
        bg_start = window.start_s
    bg_end = bg_start + clip_duration
    background = _read_segment(window.audio_path, bg_start, bg_end, TARGET_SR)
    if len(background) < clip_samples:
        background = np.pad(background, (0, clip_samples - len(background)))
    else:
        background = background[:clip_samples]

    background = peak_normalize(background)

    clip_id = f"neg_{idx:07d}"
    rel_path = f"audio_32k/{clip_id}.wav"
    gcs_path = f"{out_audio_root}/{rel_path}"
    _write_wav(gcs_path, background, TARGET_SR)

    row = _empty_row_template()
    row["clip_id"] = clip_id
    row["32khz_path"] = rel_path
    row["duration_s"] = f"{clip_duration:.6f}"
    row["is_positive"] = "False"
    row["dclde_source_path"] = window.audio_path
    row["dclde_window_start_s"] = f"{bg_start:.6f}"
    row["dclde_window_end_s"] = f"{bg_end:.6f}"
    row["dclde_provider"] = window.provider
    row["dclde_audio_id"] = window.audio_id
    row["mixup_group"] = "marine_mammal"
    return row


# ── Provider-balanced sampling ────────────────────────────────────────────


def _provider_balanced_sampler(
    windows: list[EmptyWindow], rng: np.random.Generator
) -> "callable[[], EmptyWindow]":
    """Return a sampler that draws windows weighted to equalize providers.

    Each provider is sampled uniformly; within a provider, windows are
    sampled uniformly. This prevents the largest-provider files from
    dominating the negatives.

    Raises
    ------
    ValueError
        If ``windows`` is empty.

    Returns
    -------
    callable
        A zero-arg sampler that returns one :class:`EmptyWindow` per call.
    """
    by_provider: dict[str, list[int]] = {}
    for i, w in enumerate(windows):
        by_provider.setdefault(w.provider, []).append(i)
    providers = sorted(by_provider.keys())
    if not providers:
        raise ValueError("No empty windows available.")
    provider_idx = {p: np.asarray(by_provider[p], dtype=np.int64) for p in providers}

    def sample() -> EmptyWindow:
        prov = providers[int(rng.integers(0, len(providers)))]
        choices = provider_idx[prov]
        return windows[int(choices[rng.integers(0, len(choices))])]

    return sample


# ── Main ──────────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watkins-split", default="train")
    parser.add_argument("--dclde-split", default="all_excl_beanszero")
    parser.add_argument("--min-clip-duration", type=float, default=2.0)
    parser.add_argument("--max-clip-duration", type=float, default=10.0)
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Number of negatives per positive (default: 1.0 → 50/50 balance).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-root",
        default="gs://foundation-model-data/synthetic/whales/v0.1.0",
        help="Destination root (writes `whales.csv` and `audio_32k/*.wav` here).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, build only this many Watkins-derived positives (smoke test).",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    out_root = args.out_root.rstrip("/")
    manifest_path = f"{out_root}/whales.csv"

    rng_master = np.random.default_rng(args.seed)

    logger.info("Loading DCLDE (split=%s) metadata", args.dclde_split)
    dclde = DCLDE2026(split=args.dclde_split, sample_rate=None)
    logger.info("DCLDE rows: %d", len(dclde))

    logger.info("Discovering empty windows (>= %.2fs)", args.max_clip_duration)
    windows = collect_empty_windows(dclde, min_gap_s=args.max_clip_duration)
    if not windows:
        raise RuntimeError("No DCLDE empty windows of sufficient length were found.")
    total_empty_s = sum(w.duration_s for w in windows)
    logger.info(
        "Found %d empty windows totaling %.1f hours across %d providers",
        len(windows),
        total_empty_s / 3600.0,
        len(set(w.provider for w in windows)),
    )

    logger.info("Loading Watkins (split=%s) metadata", args.watkins_split)
    watkins = Watkins(split=args.watkins_split, sample_rate=None)
    n_watkins = len(watkins)
    logger.info("Watkins rows: %d", n_watkins)

    watkins_rows: list[dict[str, object]] = [watkins._data[i] for i in range(n_watkins)]
    order = rng_master.permutation(n_watkins)
    if args.limit > 0:
        order = order[: args.limit]
    n_positives = len(order)
    n_negatives = int(round(n_positives * args.negative_ratio))
    logger.info("Planning %d positives + %d negatives", n_positives, n_negatives)

    sample_window = _provider_balanced_sampler(windows, rng_master)

    pos_seeds = rng_master.integers(0, 2**63 - 1, size=n_positives, dtype=np.int64)
    neg_seeds = rng_master.integers(0, 2**63 - 1, size=n_negatives, dtype=np.int64)

    def positive_job(local_idx: int) -> dict[str, str]:
        rng = np.random.default_rng(int(pos_seeds[local_idx]))
        watkins_row = watkins_rows[int(order[local_idx])]
        window = sample_window()
        return build_positive_row(
            idx=local_idx,
            watkins_row=watkins_row,
            window=window,
            out_audio_root=out_root,
            rng=rng,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            duration_min=args.min_clip_duration,
            duration_max=args.max_clip_duration,
        )

    def negative_job(local_idx: int) -> dict[str, str]:
        rng = np.random.default_rng(int(neg_seeds[local_idx]))
        window = sample_window()
        return build_negative_row(
            idx=local_idx,
            window=window,
            out_audio_root=out_root,
            rng=rng,
            duration_min=args.min_clip_duration,
            duration_max=args.max_clip_duration,
        )

    rows: list[dict[str, str]] = []
    failures = 0
    t0 = time.time()

    def _drain(
        executor: ThreadPoolExecutor,
        futures: list[Future[dict[str, str]]],
        label: str,
    ) -> None:
        nonlocal failures
        completed = 0
        total = len(futures)
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as exc:
                failures += 1
                logger.warning("%s job failed: %s", label, exc)
            completed += 1
            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1)
                logger.info(
                    "%s %d/%d  rate=%.1f/s  failures=%d",
                    label,
                    completed,
                    total,
                    rate,
                    failures,
                )

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        pos_futures = [executor.submit(positive_job, i) for i in range(n_positives)]
        _drain(executor, pos_futures, "positives")
        neg_futures = [executor.submit(negative_job, i) for i in range(n_negatives)]
        _drain(executor, neg_futures, "negatives")

    rows.sort(key=lambda r: r["clip_id"])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(rows)

    fs = filesystem_from_path(manifest_path)
    with fs.open(manifest_path, "w") as handle:
        handle.write(buf.getvalue())

    logger.info("Done. wrote %d rows (%d failures) to %s", len(rows), failures, manifest_path)


if __name__ == "__main__":
    main()

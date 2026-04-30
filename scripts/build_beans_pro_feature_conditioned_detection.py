#!/usr/bin/env python3
# ruff: noqa: DOC201
"""Build BEANS-Pro feature-conditioned detection splits from NBM and Birdeep.

The generated tasks mirror DRASDIC feature-conditioned detection prompts, but
replace reference audio exemplars with acoustic feature summaries derived from
annotated reference events. Query audio remains an audio prompt.

Usage
-----
uv run python scripts/build_beans_pro_feature_conditioned_detection.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.datasets.birdeep import Birdeep  # noqa: E402
from esp_data.datasets.nocturnal_bird_migration import NocturnalBirdMigration  # noqa: E402
from esp_data.io import anypath, audio_stereo_to_mono, read_audio  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SEED = 42
LABELS = ["A", "B", "C", "D"]
N_OPTIONS = 4
N_SUPPORT = 2
QUERY_WINDOW_SECONDS = 10.0
MIN_TRAIN_EVENTS = N_SUPPORT
UNKNOWN_LABEL = "Unknown"


@dataclass(frozen=True)
class SourceSpec:
    """Configuration for one annotated source dataset."""

    output_name: str
    dataset_name: str
    dataset_cls: type[NocturnalBirdMigration] | type[Birdeep]
    train_split: str
    test_split: str
    low_freq_column: str
    high_freq_column: str
    source_label: str
    license: str


@dataclass(frozen=True)
class Event:
    """Single annotated event with source audio provenance."""

    source_id: str
    audio_path: str
    species: str
    begin_time: float
    end_time: float
    low_freq_hz: float
    high_freq_hz: float

    @property
    def duration_seconds(self) -> float:
        """Return event duration in seconds."""

        return self.end_time - self.begin_time


@dataclass(frozen=True)
class FeatureSummary:
    """Acoustic feature summary for one option species."""

    peak_freq_hz: tuple[float, float]
    frequency_range_hz: tuple[float, float]
    duration_seconds: tuple[float, float]


SOURCES = [
    SourceSpec(
        output_name="nbm",
        dataset_name="nbm-feature-conditioned-detection-balanced",
        dataset_cls=NocturnalBirdMigration,
        train_split="train",
        test_split="test",
        low_freq_column="Low Frequency (Hz)",
        high_freq_column="High Frequency (Hz)",
        source_label="Nocturnal Bird Migration",
        license="CC BY-ND 3.0",
    ),
    SourceSpec(
        output_name="birdeep",
        dataset_name="birdeep-feature-conditioned-detection-balanced",
        dataset_cls=Birdeep,
        train_split="train",
        test_split="test",
        low_freq_column="Low Freq (Hz)",
        high_freq_column="High Freq (Hz)",
        source_label="Birdeep",
        license="MIT",
    ),
]


def _full_audio_path(data_root: object, rel_path: str) -> str:
    """Resolve an audio path to a string URI.

    Parameters
    ----------
    data_root : Any
        Source dataset audio root.
    rel_path : str
        Relative or absolute audio path from the source row.

    Returns
    -------
    str
        Fully qualified audio path.
    """

    if "://" in rel_path:
        return rel_path
    candidate = anypath(rel_path)
    if isinstance(candidate, Path) and candidate.is_absolute():
        return str(candidate)
    return str(data_root / rel_path)


def _coerce_float(value: object) -> float | None:
    """Convert a value to float, returning None for invalid values."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _extract_events(dataset: NocturnalBirdMigration | Birdeep, spec: SourceSpec) -> list[Event]:
    """Extract valid events from a dataset split.

    Parameters
    ----------
    dataset : Any
        Loaded source dataset.
    spec : SourceSpec
        Source configuration.

    Returns
    -------
    list[Event]
        Parsed valid events.
    """

    events: list[Event] = []
    for row_idx, row in enumerate(dataset._data):
        audio_path = _full_audio_path(dataset.data_root, str(row["audio_path"]))
        raw_st = row.get("selection_table")
        if not raw_st:
            continue
        st = pd.read_csv(StringIO(raw_st), sep="\t")
        for event_idx, event_row in st.iterrows():
            species = str(event_row.get("Species", "")).strip()
            if not species or species == UNKNOWN_LABEL:
                continue
            begin_time = _coerce_float(event_row.get("Begin Time (s)"))
            end_time = _coerce_float(event_row.get("End Time (s)"))
            low_freq = _coerce_float(event_row.get(spec.low_freq_column))
            high_freq = _coerce_float(event_row.get(spec.high_freq_column))
            if (
                begin_time is None
                or end_time is None
                or low_freq is None
                or high_freq is None
                or end_time <= begin_time
                or high_freq <= low_freq
            ):
                continue
            events.append(
                Event(
                    source_id=f"{spec.output_name}_{row_idx:05d}_{event_idx:03d}",
                    audio_path=audio_path,
                    species=species,
                    begin_time=begin_time,
                    end_time=end_time,
                    low_freq_hz=low_freq,
                    high_freq_hz=high_freq,
                )
            )
    return events


def _group_events_by_species(events: list[Event]) -> dict[str, list[Event]]:
    """Group events by species."""

    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.species].append(event)
    return dict(grouped)


def _query_window(event: Event, window_seconds: float) -> tuple[float, float]:
    """Return a fixed-width query window centered on an event."""

    midpoint = (event.begin_time + event.end_time) / 2.0
    start_time = max(0.0, midpoint - (window_seconds / 2.0))
    return start_time, start_time + window_seconds


def _events_overlapping_window(
    events: list[Event],
    start_time: float,
    end_time: float,
) -> list[Event]:
    """Return events overlapping a time window in the same recording."""

    return [
        event
        for event in events
        if event.begin_time < end_time and event.end_time > start_time
    ]


def _peak_frequency_hz(event: Event) -> float:
    """Estimate peak frequency from the bounded event audio.

    Parameters
    ----------
    event : Event
        Annotated event whose audio segment should be analyzed.

    Returns
    -------
    float
        Frequency bin with maximum average magnitude.
    """

    try:
        audio, sample_rate = read_audio(
            event.audio_path,
            start_time=event.begin_time,
            end_time=event.end_time,
        )
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
    except Exception as exc:
        logger.warning("Falling back to center frequency for %s: %s", event.source_id, exc)
        return (event.low_freq_hz + event.high_freq_hz) / 2.0
    if audio.size == 0:
        return (event.low_freq_hz + event.high_freq_hz) / 2.0
    windowed = audio * np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    mask = (freqs >= event.low_freq_hz) & (freqs <= event.high_freq_hz)
    if not mask.any():
        return (event.low_freq_hz + event.high_freq_hz) / 2.0
    masked_freqs = freqs[mask]
    masked_spectrum = spectrum[mask]
    return float(masked_freqs[int(np.argmax(masked_spectrum))])


def _feature_summary(events: list[Event]) -> FeatureSummary:
    """Summarize acoustic features over reference events.

    Parameters
    ----------
    events : list[Event]
        Reference events for one option species.

    Returns
    -------
    FeatureSummary
        Min/max feature ranges over the reference events.
    """

    peaks = [_peak_frequency_hz(event) for event in events]
    lows = [event.low_freq_hz for event in events]
    highs = [event.high_freq_hz for event in events]
    durations = [event.duration_seconds for event in events]
    return FeatureSummary(
        peak_freq_hz=(min(peaks), max(peaks)),
        frequency_range_hz=(min(lows), max(highs)),
        duration_seconds=(min(durations), max(durations)),
    )


def _format_freq_value(freq_hz: float, use_khz: bool) -> str:
    """Format one frequency value."""

    if use_khz:
        return f"{freq_hz / 1000.0:.1f}"
    return f"{int(round(freq_hz))}"


def _format_freq_range(freq_range: tuple[float, float]) -> str:
    """Format a frequency range in Hz or kHz."""

    low, high = freq_range
    use_khz = high >= 1000.0
    unit = "kHz" if use_khz else "Hz"
    return f"{_format_freq_value(low, use_khz)}-{_format_freq_value(high, use_khz)} {unit}"


def _format_duration_range(duration_range: tuple[float, float]) -> str:
    """Format a duration range in seconds."""

    low, high = duration_range
    if low < 0.1 or high < 0.1:
        return f"{low:.2f}-{high:.2f} seconds"
    return f"{low:.1f}-{high:.1f} seconds"


def _feature_text(label: str, summary: FeatureSummary) -> str:
    """Format one option's DRASDIC-style acoustic description."""

    return (
        f"Sound {label} typically has a peak frequency of "
        f"{_format_freq_range(summary.peak_freq_hz)}, frequency range "
        f"{_format_freq_range(summary.frequency_range_hz)}, and duration of "
        f"{_format_duration_range(summary.duration_seconds)}."
    )


def _build_instruction(
    option_features: dict[str, FeatureSummary],
    has_background: bool,
) -> str:
    """Build a feature-conditioned detection prompt."""

    lines = [
        _feature_text(label, option_features[label])
        for label in LABELS[: len(option_features)]
    ]
    if has_background:
        lines.extend(
            [
                "",
                "Here is the background environment: <Audio><AudioHere></Audio>",
            ]
        )
    lines.extend(
        [
            "",
            "Which of the above sounds are present in this recording, if any?",
            "<Audio><AudioHere></Audio>",
        ]
    )
    return "\n".join(lines)


def _audio_segment(path: str, start_time: float, end_time: float) -> dict[str, object]:
    """Build a manifest audio segment."""

    return {
        "path": path,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
    }


def _make_row(
    *,
    spec: SourceSpec,
    row_idx: int,
    option_species: list[str],
    option_features: dict[str, FeatureSummary],
    query_event: Event,
    query_window: tuple[float, float],
    answer: str,
    has_background: bool,
    background_event: Event | None,
    background_window: tuple[float, float] | None,
) -> dict[str, Any]:
    """Build one JSONL output row."""

    row_id = f"{spec.output_name}_fcdet_bal_{row_idx:06d}"
    option_types = {
        label: species for label, species in zip(LABELS, option_species, strict=True)
    }
    audio_segments = []
    if has_background and background_event is not None and background_window is not None:
        audio_segments.append(
            _audio_segment(
                background_event.audio_path,
                background_window[0],
                background_window[1],
            )
        )
    audio_segments.append(_audio_segment(query_event.audio_path, query_window[0], query_window[1]))
    audio_paths = [str(segment["path"]) for segment in audio_segments]
    metadata = {
        "option_types": option_types,
        "option_features": {
            label: {
                "species": species,
                "peak_freq_hz": option_features[label].peak_freq_hz,
                "frequency_range_hz": option_features[label].frequency_range_hz,
                "duration_seconds": option_features[label].duration_seconds,
            }
            for label, species in option_types.items()
        },
        "query_species": query_event.species,
        "query_event_id": query_event.source_id,
        "query_window_seconds": query_window,
        "has_background": has_background,
        "background_event_id": background_event.source_id if background_event else None,
        "background_window_seconds": background_window,
    }
    return {
        "id": row_id,
        "audio_paths": audio_paths,
        "audio_segments": audio_segments,
        "audio_ids": [row_id],
        "template_path": "audio_synth/feature_conditioned_detection",
        "skills": ["feature_conditioned_detection", "multilabel_detection"],
        "messages": [
            {"role": "user", "content": _build_instruction(option_features, has_background)},
            {"role": "assistant", "content": answer},
        ],
        "task": "feature_conditioned_detection",
        "source_dataset": spec.source_label,
        "dataset_name": spec.dataset_name,
        "license": spec.license,
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": query_event.audio_path,
    }


def _eligible_species(
    train_by_species: dict[str, list[Event]],
    test_by_species: dict[str, list[Event]],
) -> list[str]:
    """Return species with enough reference and query events."""

    return sorted(
        species
        for species, train_events in train_by_species.items()
        if len(train_events) >= MIN_TRAIN_EVENTS and test_by_species.get(species)
    )


def _build_support_features(
    train_by_species: dict[str, list[Event]],
    eligible_species: list[str],
    rng: random.Random,
) -> tuple[dict[str, list[Event]], dict[str, FeatureSummary]]:
    """Sample support events and compute feature summaries."""

    support_events: dict[str, list[Event]] = {}
    feature_summaries: dict[str, FeatureSummary] = {}
    for species in eligible_species:
        sampled = rng.sample(train_by_species[species], N_SUPPORT)
        support_events[species] = sampled
        feature_summaries[species] = _feature_summary(sampled)
    return support_events, feature_summaries


def _build_examples(
    *,
    spec: SourceSpec,
    test_events: list[Event],
    events_by_audio: dict[str, list[Event]],
    eligible_species: list[str],
    feature_summaries: dict[str, FeatureSummary],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build balanced present-vs-none rows for one source."""

    positives: list[tuple[Event, tuple[float, float], list[str], str]] = []
    none_candidates: list[tuple[Event, tuple[float, float], list[str]]] = []
    eligible_set = set(eligible_species)

    for event in test_events:
        if event.species not in eligible_set:
            continue
        window = _query_window(event, QUERY_WINDOW_SECONDS)
        overlapping = _events_overlapping_window(events_by_audio[event.audio_path], *window)
        window_species = {overlap.species for overlap in overlapping}
        absent_options = sorted(eligible_set - window_species)

        if len(absent_options) >= N_OPTIONS - 1:
            positives.append((event, window, absent_options, event.species))
        if len(absent_options) >= N_OPTIONS:
            none_candidates.append((event, window, absent_options))

    rng.shuffle(positives)
    rng.shuffle(none_candidates)
    n_rows_per_class = min(len(positives), len(none_candidates))
    rows: list[dict[str, Any]] = []
    for event, window, absent_options, target_species in positives[:n_rows_per_class]:
        option_species = [target_species, *rng.sample(absent_options, N_OPTIONS - 1)]
        rng.shuffle(option_species)
        answer = LABELS[option_species.index(target_species)]
        rows.append(
            _row_from_options(
                spec=spec,
                row_idx=len(rows),
                option_species=option_species,
                feature_summaries=feature_summaries,
                query_event=event,
                query_window=window,
                answer=answer,
                background_pool=none_candidates,
                rng=rng,
            )
        )

    for event, window, absent_options in none_candidates[:n_rows_per_class]:
        option_species = rng.sample(absent_options, N_OPTIONS)
        rows.append(
            _row_from_options(
                spec=spec,
                row_idx=len(rows),
                option_species=option_species,
                feature_summaries=feature_summaries,
                query_event=event,
                query_window=window,
                answer="None",
                background_pool=none_candidates,
                rng=rng,
            )
        )

    rng.shuffle(rows)
    for row_idx, row in enumerate(rows):
        row["id"] = f"{spec.output_name}_fcdet_bal_{row_idx:06d}"
        row["audio_ids"] = [row["id"]]
    return rows


def _row_from_options(
    *,
    spec: SourceSpec,
    row_idx: int,
    option_species: list[str],
    feature_summaries: dict[str, FeatureSummary],
    query_event: Event,
    query_window: tuple[float, float],
    answer: str,
    background_pool: list[tuple[Event, tuple[float, float], list[str]]],
    rng: random.Random,
) -> dict[str, Any]:
    """Build one row after option species have been selected."""

    option_features = {
        label: feature_summaries[species]
        for label, species in zip(LABELS, option_species, strict=True)
    }
    has_background = rng.random() < 0.5 and bool(background_pool)
    background_event = None
    background_window = None
    if has_background:
        background_candidates = [
            item for item in background_pool if item[0].source_id != query_event.source_id
        ]
        if background_candidates:
            background_event, background_window, _ = rng.choice(background_candidates)
        else:
            has_background = False
    return _make_row(
        spec=spec,
        row_idx=row_idx,
        option_species=option_species,
        option_features=option_features,
        query_event=query_event,
        query_window=query_window,
        answer=answer,
        has_background=has_background,
        background_event=background_event,
        background_window=background_window,
    )


def _load_source_events(spec: SourceSpec) -> tuple[list[Event], list[Event]]:
    """Load train and test events for a source specification."""

    train_ds = spec.dataset_cls(split=spec.train_split, sample_rate=None, backend="polars")
    test_ds = spec.dataset_cls(split=spec.test_split, sample_rate=None, backend="polars")
    return _extract_events(train_ds, spec), _extract_events(test_ds, spec)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize generated rows."""

    answers = Counter(row["messages"][1]["content"] for row in rows)
    background_count = 0
    for row in rows:
        metadata = json.loads(row["metadata"])
        if metadata["has_background"]:
            background_count += 1
    return {
        "rows": len(rows),
        "answers": dict(sorted(answers.items())),
        "background_rows": background_count,
    }


def build_split(spec: SourceSpec, output_dir: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Build and write one feature-conditioned split.

    Parameters
    ----------
    spec : SourceSpec
        Source dataset configuration.
    output_dir : Path
        Directory to write JSONL manifests.
    rng : random.Random
        Seeded random number generator.

    Returns
    -------
    list[dict[str, Any]]
        Generated rows.
    """

    logger.info("Loading %s source events", spec.output_name)
    train_events, test_events = _load_source_events(spec)
    train_by_species = _group_events_by_species(train_events)
    test_by_species = _group_events_by_species(test_events)
    species = _eligible_species(train_by_species, test_by_species)
    logger.info(
        "%s: train events=%d, test events=%d, eligible species=%d",
        spec.output_name,
        len(train_events),
        len(test_events),
        len(species),
    )

    _, feature_summaries = _build_support_features(train_by_species, species, rng)
    events_by_audio: dict[str, list[Event]] = defaultdict(list)
    for event in test_events:
        events_by_audio[event.audio_path].append(event)

    rows = _build_examples(
        spec=spec,
        test_events=test_events,
        events_by_audio=dict(events_by_audio),
        eligible_species=species,
        feature_summaries=feature_summaries,
        rng=rng,
    )
    output_path = output_dir / f"{spec.output_name}_feature_conditioned_detection_balanced.jsonl"
    _write_jsonl(rows, output_path)
    logger.info("%s summary: %s", spec.output_name, _summarize(rows))
    return rows


def main() -> None:
    """Build all feature-conditioned detection splits."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_feature_conditioned_detection",
        help="Directory for output JSONL files.",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    for spec in SOURCES:
        build_split(spec, args.output_dir, rng)


if __name__ == "__main__":
    main()

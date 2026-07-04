#!/usr/bin/env python3
"""Build a ROOTS-format synthetic *call-description* MCQ dataset.

Idea
----
Some animals have famously iconic, well-described calls (a loon's wail, a
cuckoo's two-note "cuck-oo", a kookaburra's laugh). For a curated list of such
species (``scripts/iconic_call_descriptions.py``) we:

1. Find Xeno-canto / iNaturalist recordings whose focal label is one of the
   target species.
2. Use BirdCODE sound-event detections to *precisely isolate* where the focal
   species vocalizes, and keep only windows where the detected species matches
   the recording's focal label (never background species) and no other species
   is detected inside the crop window.
3. Optionally gate on the Xeno-canto ``behavior`` metadata so a description of
   a *song* is not attached to a clip the recordist tagged purely as a *call*.
4. Build a 4-way multiple-choice question: the correct option is the iconic
   description of the focal species; the three distractors are real
   descriptions of *other* species that also appear in this dataset.

Output is a ROOTS-compatible JSONL (one row per crop) plus a sidecar
``*_extractions.parquet`` of the raw extractions for QA. Audio paths are
absolute ``gs://`` URIs so the single split can mix XC and iNat; with
``sample_rate=32000`` the ROOTS adapter auto-redirects to the 32 kHz mirrors
and the ``__crop_<start_ms>_<end_ms>`` token in ``audio_ids`` drives cropping.

Usage::

    uv run python scripts/build_roots_call_description_mcq.py \
        --xc-shard-dir /tmp/birdcode_preds/xc \
        --inat-shard-dir /tmp/birdcode_preds/inat \
        --output data/roots_call_description_mcq/call_description_mcq_iconic_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.iconic_call_descriptions import ICONIC_CALL_DESCRIPTIONS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Tunables ───────────────────────────────────────────────────────────────
SEED = 42
MIN_SCORE = 0.6  # minimum BirdCODE detection score to anchor a crop
PAD_SEC = 0.4  # padding added around the chosen detection
MIN_WIN_SEC = 3.0  # minimum crop duration
MAX_WIN_SEC = 8.0  # maximum crop duration
MAX_PER_SPECIES = 300  # cap extractions per species to keep the set balanced
N_OPTIONS = 4
OPTION_LETTERS = ["a", "b", "c", "d", "e", "f"][:N_OPTIONS]

_XC_CSV = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/all_20260203.csv"
_INAT_CSV = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw/all_20260201.csv"
_XC_AUDIO_ROOT = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio"
_INAT_AUDIO_ROOT = "gs://esp-ml-datasets/inaturalist/v0.1.0/raw"

_QUESTION_TEMPLATES = [
    "Which of the following best describes the vocalization in this recording?",
    "Which description best matches the sound in this clip?",
    "Listen to the animal in this recording. Which description fits its call best?",
    "Which of these best characterizes the vocalization you hear?",
]


# ── Data types ───────────────────────────────────────────────────────────


@dataclass
class Recording:
    """A source recording keyed by its BirdCODE ``file_id``."""

    source: str  # "xeno-canto" | "inaturalist"
    src_id: str  # xc_id / inat_id
    focal: str  # canonical_name
    audio_uri: str  # absolute gs:// path used in audio_paths
    behavior: str  # raw behavior metadata ("" if missing)
    license: str


@dataclass
class Extraction:
    """A single isolated focal-call crop ready to become an MCQ."""

    source: str
    src_id: str
    focal: str
    audio_uri: str
    start_ms: int
    end_ms: int
    det_score: float
    behavior: str
    license: str


# ── Selection-table parsing ────────────────────────────────────────────────


def parse_selection_table(tsv: str) -> list[tuple[float, float, str, float]]:
    """Parse a BirdCODE selection-table TSV string.

    Parameters
    ----------
    tsv : str
        Tab-separated table with columns
        ``Begin Time (s)``, ``End Time (s)``, ``Species``, ``Score``.

    Returns
    -------
    list[tuple[float, float, str, float]]
        ``(begin, end, species, score)`` rows.
    """
    rows: list[tuple[float, float, str, float]] = []
    reader = csv.reader(io.StringIO(tsv), delimiter="\t")
    header = next(reader, None)
    if header is None:
        return rows
    for r in reader:
        if len(r) < 4:
            continue
        try:
            begin = float(r[0])
            end = float(r[1])
            score = float(r[3])
        except ValueError:
            continue
        rows.append((begin, end, r[2].strip(), score))
    return rows


def choose_crop_window(
    dets: list[tuple[float, float, str, float]],
    focal: str,
) -> tuple[int, int, float] | None:
    """Choose one clean crop window isolating the focal species' call.

    Anchors on the highest-scoring focal detection, pads it, enforces a
    min/max duration, and rejects the window if any *non-focal* species is
    detected inside it (so the crop is single-species).

    Parameters
    ----------
    dets : list[tuple[float, float, str, float]]
        All detections ``(begin, end, species, score)`` for the recording.
    focal : str
        Focal canonical name that detections must match.

    Returns
    -------
    tuple[int, int, float] | None
        ``(start_ms, end_ms, anchor_score)`` or ``None`` if no clean window.
    """
    focal_dets = [d for d in dets if d[2] == focal and d[3] >= MIN_SCORE]
    if not focal_dets:
        return None
    other_dets = [d for d in dets if d[2] != focal]

    # Strongest focal detection first.
    for begin, end, _sp, score in sorted(focal_dets, key=lambda d: -d[3]):
        start = max(0.0, begin - PAD_SEC)
        stop = end + PAD_SEC
        dur = stop - start
        if dur < MIN_WIN_SEC:  # expand symmetrically
            center = (start + stop) / 2.0
            start = max(0.0, center - MIN_WIN_SEC / 2.0)
            stop = start + MIN_WIN_SEC
        elif dur > MAX_WIN_SEC:  # center-crop
            center = (begin + end) / 2.0
            start = max(0.0, center - MAX_WIN_SEC / 2.0)
            stop = start + MAX_WIN_SEC

        overlaps_other = any(o[0] < stop and o[1] > start for o in other_dets)
        if overlaps_other:
            continue
        return int(round(start * 1000)), int(round(stop * 1000)), score
    return None


# ── Source-metadata maps ───────────────────────────────────────────────────


def build_xc_map(targets: set[str]) -> dict[str, Recording]:
    """Map XC ``relative_path`` (== BirdCODE file_id) to recording metadata."""
    cols = ["xc_id", "relative_path", "canonical_name", "behavior", "license"]
    df = (
        pl.scan_csv(_XC_CSV, infer_schema_length=2000)
        .select(cols)
        .filter(pl.col("canonical_name").is_in(list(targets)))
        .collect(engine="streaming")
    )
    out: dict[str, Recording] = {}
    for row in df.iter_rows(named=True):
        rel = row["relative_path"]
        if not rel:
            continue
        out[rel] = Recording(
            source="xeno-canto",
            src_id=str(row["xc_id"]),
            focal=row["canonical_name"],
            audio_uri=f"{_XC_AUDIO_ROOT}/{rel}",
            behavior=(row["behavior"] or "").strip().lower(),
            license=row["license"] or "",
        )
    logger.info("XC: %d target recordings indexed", len(out))
    return out


def build_inat_map(targets: set[str]) -> dict[str, Recording]:
    """Map iNat ``originals_path`` (== BirdCODE file_id) to recording metadata."""
    cols = ["inat_id", "originals_path", "32khz_path", "canonical_name", "behavior", "license"]
    df = (
        pl.scan_csv(_INAT_CSV, infer_schema_length=2000)
        .select(cols)
        .filter(pl.col("canonical_name").is_in(list(targets)))
        .collect(engine="streaming")
    )
    out: dict[str, Recording] = {}
    for row in df.iter_rows(named=True):
        file_id = row["originals_path"]
        path32 = row["32khz_path"]
        if not file_id or not path32:
            continue
        out[file_id] = Recording(
            source="inaturalist",
            src_id=str(row["inat_id"]),
            focal=row["canonical_name"],
            audio_uri=f"{_INAT_AUDIO_ROOT}/{path32}",
            behavior=(row["behavior"] or "").strip().lower(),
            license=row["license"] or "",
        )
    logger.info("iNat: %d target recordings indexed", len(out))
    return out


def behavior_compatible(behavior: str, vocalization: str) -> bool:
    """Return True if the recordist's behavior tag is compatible with desc type.

    Missing behavior or ``vocalization == "any"`` always passes. Otherwise the
    behavior string must mention the required type (``song``/``call``). Note
    "flight call", "alarm call" etc. all contain "call".
    """
    if not behavior or behavior in {"uncertain", "unknown"}:
        return True
    if vocalization == "any":
        return True
    if vocalization == "song":
        return "song" in behavior
    if vocalization == "call":
        return "call" in behavior
    return True


# ── Extraction ─────────────────────────────────────────────────────────────


def extract_from_shards(
    shard_dir: Path,
    rec_map: dict[str, Recording],
    voc_by_species: dict[str, str],
) -> list[Extraction]:
    """Scan npz shards and emit one clean Extraction per matching recording."""
    extractions: list[Extraction] = []
    shards = sorted(shard_dir.glob("shard_*.npz"))
    n_seen = n_matched = n_clean = n_behavior_skip = 0
    for shard in shards:
        data = np.load(shard, allow_pickle=True)
        file_ids = data["file_ids"]
        for i, file_id in enumerate(file_ids):
            n_seen += 1
            rec = rec_map.get(str(file_id))
            if rec is None:
                continue
            n_matched += 1
            table = data[f"table_{i}"]
            tsv = table[0] if table.shape else str(table)
            dets = parse_selection_table(str(tsv))
            window = choose_crop_window(dets, rec.focal)
            if window is None:
                continue
            voc = voc_by_species.get(rec.focal, "any")
            if not behavior_compatible(rec.behavior, voc):
                n_behavior_skip += 1
                continue
            start_ms, end_ms, score = window
            n_clean += 1
            extractions.append(
                Extraction(
                    source=rec.source,
                    src_id=rec.src_id,
                    focal=rec.focal,
                    audio_uri=rec.audio_uri,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    det_score=score,
                    behavior=rec.behavior,
                    license=rec.license,
                )
            )
    logger.info(
        "%s: scanned %d file_ids, matched %d, clean crops %d, behavior-skipped %d",
        shard_dir.name, n_seen, n_matched, n_clean, n_behavior_skip,
    )
    return extractions


def cap_per_species(
    extractions: list[Extraction], cap: int, rng: random.Random
) -> list[Extraction]:
    """Down-sample to at most ``cap`` extractions per focal species."""
    by_sp: dict[str, list[Extraction]] = collections.defaultdict(list)
    for e in extractions:
        by_sp[e.focal].append(e)
    kept: list[Extraction] = []
    for sp, items in by_sp.items():
        if len(items) > cap:
            items = rng.sample(items, cap)
        kept.extend(items)
    return kept


# ── MCQ assembly ───────────────────────────────────────────────────────────


def build_rows(
    extractions: list[Extraction],
    desc_by_species: dict[str, dict[str, str]],
    split_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Assemble 4-way MCQ rows with distractors from other present species."""
    present = sorted({e.focal for e in extractions})
    if len(present) < N_OPTIONS:
        raise RuntimeError(f"Need >= {N_OPTIONS} species, have {len(present)}")

    rng.shuffle(extractions)
    rows: list[dict[str, Any]] = []
    for idx, e in enumerate(extractions):
        correct_desc = desc_by_species[e.focal]["description"]
        # distractor species: present, different focal, different description text
        pool = [s for s in present if s != e.focal]
        confuser_sp = rng.sample(pool, N_OPTIONS - 1)
        confuser_desc = [desc_by_species[s]["description"] for s in confuser_sp]

        correct_pos = idx % N_OPTIONS  # balanced rotation
        option_desc = list(confuser_desc)
        option_desc.insert(correct_pos, correct_desc)
        option_sp = list(confuser_sp)
        option_sp.insert(correct_pos, e.focal)

        q = _QUESTION_TEMPLATES[idx % len(_QUESTION_TEMPLATES)]
        opts = ", ".join(f"{OPTION_LETTERS[i]}) {d}" for i, d in enumerate(option_desc))
        content = f"<Audio><AudioHere></Audio>\n{q}\nOptions: {opts}"
        answer = OPTION_LETTERS[correct_pos]

        row_id = f"{split_name}_{idx:06d}"
        audio_id = f"{e.src_id}__crop_{e.start_ms}_{e.end_ms}"
        rows.append(
            {
                "id": row_id,
                "audio_paths": [e.audio_uri],
                "audio_ids": [audio_id],
                "template_path": "call_description_mcq_iconic_v1",
                "skills": ["multiple_choice", "vocalization_description"],
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": answer},
                ],
                "task": "roots_tier1_vocal_description_mcq",
                "source_dataset": e.source,
                "dataset_name": split_name,
                "license": e.license,
                "metadata": json.dumps(
                    {
                        "correct": answer,
                        "correct_species": e.focal,
                        "correct_common_name": desc_by_species[e.focal]["common_name"],
                        "option_species": dict(zip(OPTION_LETTERS, option_sp, strict=True)),
                        "crop_start_sec": e.start_ms / 1000.0,
                        "crop_end_sec": e.end_ms / 1000.0,
                        "detection_score": round(e.det_score, 4),
                        "behavior": e.behavior,
                        "vocalization_type": desc_by_species[e.focal]["vocalization"],
                    }
                ),
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a local JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), path)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Build the call-description MCQ dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc-shard-dir", type=Path, default=Path("/tmp/birdcode_preds/xc"))
    parser.add_argument("--inat-shard-dir", type=Path, default=Path("/tmp/birdcode_preds/inat"))
    parser.add_argument("--include-inat", action="store_true", default=True)
    parser.add_argument("--no-inat", dest="include_inat", action="store_false")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "roots_call_description_mcq"
        / "call_description_mcq_iconic_v1.jsonl",
    )
    parser.add_argument("--split-name", default="call_description_mcq_iconic_v1")
    args = parser.parse_args()

    rng = random.Random(SEED)

    desc_by_species = {e["scientific_name"]: e for e in ICONIC_CALL_DESCRIPTIONS}
    voc_by_species = {k: v["vocalization"] for k, v in desc_by_species.items()}
    targets = set(desc_by_species)
    logger.info("Target species: %d", len(targets))

    xc_map = build_xc_map(targets)
    extractions = extract_from_shards(args.xc_shard_dir, xc_map, voc_by_species)

    if args.include_inat:
        inat_map = build_inat_map(targets)
        extractions += extract_from_shards(args.inat_shard_dir, inat_map, voc_by_species)

    logger.info("Total clean extractions before capping: %d", len(extractions))
    extractions = cap_per_species(extractions, MAX_PER_SPECIES, rng)
    by_sp = collections.Counter(e.focal for e in extractions)
    logger.info(
        "After capping: %d extractions across %d species (median/spc=%d)",
        len(extractions), len(by_sp),
        int(np.median(list(by_sp.values()))) if by_sp else 0,
    )

    rows = build_rows(extractions, desc_by_species, args.split_name, rng)
    answer_dist = collections.Counter(r["messages"][1]["content"] for r in rows)
    logger.info("Answer distribution: %s", dict(sorted(answer_dist.items())))
    src_dist = collections.Counter(r["source_dataset"] for r in rows)
    logger.info("Source distribution: %s", dict(src_dist))

    write_jsonl(rows, args.output)

    # Sidecar parquet of raw extractions for QA / spectrogram review.
    ext_df = pl.DataFrame(
        [
            {
                "source": e.source, "src_id": e.src_id, "focal": e.focal,
                "audio_uri": e.audio_uri, "start_ms": e.start_ms, "end_ms": e.end_ms,
                "det_score": e.det_score, "behavior": e.behavior,
            }
            for e in extractions
        ]
    )
    ext_path = args.output.with_name(args.output.stem + "_extractions.parquet")
    ext_df.write_parquet(ext_path)
    logger.info("Wrote extractions sidecar to %s", ext_path)


if __name__ == "__main__":
    main()

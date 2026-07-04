#!/usr/bin/env python3
"""Within-species call-type description MCQ (ROOTS) — extraction, QA, build.

Three stages (``--stage``):

``extract``
    Scan cached BirdCODE XC shards, keep Xeno-canto recordings of the priority
    species whose ``behavior`` maps to exactly one *described* canonical call
    type, isolate the focal detection crop, and write a per-extraction parquet.

``montage``
    Render one mel-spectrogram montage per species (rows = call types, columns
    = example crops) so the metadata call-type label can be visually validated
    against the actual acoustic content. This is the primary validation for the
    initial ~20-species set.

``build``
    Assemble the within-species MCQ JSONL: the audio is one call type of
    species X; the correct option describes that call type; the distractors are
    descriptions of the *other* call types of the *same* species X.

Usage::

    uv run python scripts/build_roots_calltype_mcq.py --stage extract
    uv run python scripts/build_roots_calltype_mcq.py --stage montage --species "Strix aluco"
    uv run python scripts/build_roots_calltype_mcq.py --stage build
"""

from __future__ import annotations

import argparse
import collections
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

from scripts.build_roots_call_description_mcq import (  # noqa: E402
    choose_crop_window,
    parse_selection_table,
)
from scripts.calltype_descriptions import CALLTYPE_DESCRIPTIONS, CANON_CALLTYPE  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SEED = 42
_XC_CSV = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/all_20260203.csv"
_XC_AUDIO_ROOT = "gs://esp-ml-datasets/xeno-canto/v0.1.0/raw/audio"
_OUT_DIR = REPO_ROOT / "data" / "roots_calltype_mcq"
_EXT_PARQUET = _OUT_DIR / "calltype_extractions.parquet"

N_OPTIONS = 4  # used in build; falls back to per-species #types when fewer
OPTION_LETTERS = ["a", "b", "c", "d", "e", "f"]
MAX_PER_GROUP = 250  # cap extractions per (species, call_type)

_QUESTION_TEMPLATES = [
    "You are listening to one of several call types of {common}. "
    "Which description best matches the vocalization in this recording?",
    "This recording contains a {common}. Which description best matches the "
    "particular call type you hear?",
    "A {common} is vocalizing. Which of these best describes this specific call type?",
]


# ── Stage: extract ─────────────────────────────────────────────────────────


@dataclass
class Extraction:
    scientific_name: str
    common_name: str
    call_type: str
    xc_id: str
    audio_uri: str
    start_ms: int
    end_ms: int
    det_score: float


def _described_index() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Return (common_name_by_species, {(species, call_type): description})."""
    common = {e["scientific_name"]: e["common_name"] for e in CALLTYPE_DESCRIPTIONS}
    desc = {(e["scientific_name"], e["call_type"]): e["description"] for e in CALLTYPE_DESCRIPTIONS}
    return common, desc


def _canon_single_type(behavior: str) -> str | None:
    """Map a raw behavior string to a single canonical call type, else None."""
    if not behavior:
        return None
    toks = [t.strip().lower() for t in str(behavior).split(",") if t.strip()]
    canon = {CANON_CALLTYPE[t] for t in toks if t in CANON_CALLTYPE}
    # Reject if the record also carries a non-canonical token (ambiguous),
    # or if it does not collapse to exactly one canonical type.
    has_unknown = any(t not in CANON_CALLTYPE for t in toks if t not in {"uncertain", "unknown"})
    if has_unknown or len(canon) != 1:
        return None
    return next(iter(canon))


def stage_extract(shard_dir: Path) -> None:
    """Build the per-(species, call_type) extraction parquet."""
    common_by_sp, desc_by_key = _described_index()
    described_species = set(common_by_sp)

    cols = ["xc_id", "relative_path", "canonical_name", "behavior"]
    df = (
        pl.scan_csv(_XC_CSV, infer_schema_length=2000)
        .select(cols)
        .filter(pl.col("canonical_name").is_in(list(described_species)))
        .collect(engine="streaming")
    )
    # rel_path -> (xc_id, focal, call_type) only for single-type, described combos
    rec_map: dict[str, tuple[str, str, str]] = {}
    for row in df.iter_rows(named=True):
        rel = row["relative_path"]
        focal = row["canonical_name"]
        if not rel:
            continue
        ct = _canon_single_type(row["behavior"])
        if ct is None or (focal, ct) not in desc_by_key:
            continue
        rec_map[rel] = (str(row["xc_id"]), focal, ct)
    logger.info("Eligible single-type XC recordings: %d", len(rec_map))

    extractions: list[Extraction] = []
    n_clean = 0
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        data = np.load(shard, allow_pickle=True)
        file_ids = data["file_ids"]
        for i, file_id in enumerate(file_ids):
            hit = rec_map.get(str(file_id))
            if hit is None:
                continue
            xc_id, focal, ct = hit
            table = data[f"table_{i}"]
            tsv = table[0] if table.shape else str(table)
            window = choose_crop_window(parse_selection_table(str(tsv)), focal)
            if window is None:
                continue
            start_ms, end_ms, score = window
            n_clean += 1
            extractions.append(
                Extraction(
                    scientific_name=focal,
                    common_name=common_by_sp[focal],
                    call_type=ct,
                    xc_id=xc_id,
                    audio_uri=f"{_XC_AUDIO_ROOT}/{file_id}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    det_score=score,
                )
            )
    logger.info("Clean focal-matched crops: %d", n_clean)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext_df = pl.DataFrame([vars(e) for e in extractions])
    ext_df.write_parquet(_EXT_PARQUET)

    # Report yield per (species, call_type)
    counts = (
        ext_df.group_by(["common_name", "call_type"]).len().sort(["common_name", "call_type"])
    )
    logger.info("Yield per (species, call_type):\n%s", counts)
    logger.info("Wrote %d extractions to %s", ext_df.height, _EXT_PARQUET)


# ── Stage: montage ─────────────────────────────────────────────────────────


def stage_montage(species: list[str] | None, per_type: int, seed: int) -> None:
    """Render mel-spectrogram montages (rows = call types) per species."""
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from esp_data.io import anypath, audio_stereo_to_mono, read_audio

    rng = random.Random(seed)
    ext = pl.read_parquet(_EXT_PARQUET)
    sci_names = species or sorted(ext["scientific_name"].unique().to_list())

    def _redirect_32k(uri: str) -> str:
        return uri.replace("/raw/audio/", "/raw/audio_32k/").rsplit(".", 1)[0] + ".wav"

    montage_dir = _OUT_DIR / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)

    for sci in sci_names:
        sub = ext.filter(pl.col("scientific_name") == sci)
        if sub.height == 0:
            continue
        call_types = sorted(sub["call_type"].unique().to_list())
        common = sub["common_name"][0]
        nrow, ncol = len(call_types), per_type
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
        for r, ct in enumerate(call_types):
            rows = sub.filter(pl.col("call_type") == ct).to_dicts()
            picks = rng.sample(rows, min(per_type, len(rows)))
            for c in range(ncol):
                ax = axes[r][c]
                if c >= len(picks):
                    ax.axis("off")
                    continue
                e = picks[c]
                try:
                    audio, sr = read_audio(
                        anypath(_redirect_32k(e["audio_uri"])),
                        start_time=e["start_ms"] / 1000.0,
                        end_time=e["end_ms"] / 1000.0,
                    )
                    audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)
                    S = librosa.power_to_db(
                        librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr // 2),
                        ref=np.max,
                    )
                    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax)
                except Exception as exc:  # noqa: BLE001
                    ax.text(0.5, 0.5, f"err\n{exc}", fontsize=6, ha="center")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if c == 0:
                    ax.set_ylabel(ct, fontsize=10, fontweight="bold")
                ax.set_title(f"XC{e['xc_id']} det={e['det_score']:.2f}", fontsize=7)
        fig.suptitle(f"{common} ({sci}) — rows are metadata call types", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = montage_dir / f"{sci.replace(' ', '_')}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        logger.info("Wrote montage %s (%d types)", out, len(call_types))


# ── Stage: build ─────────────────────────────────────────────────────────


def stage_build(
    split_name: str,
    output: Path,
    valid_keys: set[tuple[str, str]] | None,
    min_types: int,
) -> None:
    """Build the within-species call-type MCQ JSONL."""
    _common, desc_by_key = _described_index()
    rng = random.Random(SEED)

    ext = pl.read_parquet(_EXT_PARQUET)
    if valid_keys is not None:
        ext = ext.filter(
            pl.struct(["scientific_name", "call_type"]).map_elements(
                lambda s: (s["scientific_name"], s["call_type"]) in valid_keys,
                return_dtype=pl.Boolean,
            )
        )

    # cap per (species, call_type)
    rows_by_key: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in ext.to_dicts():
        rows_by_key[(r["scientific_name"], r["call_type"])].append(r)
    for k, items in rows_by_key.items():
        if len(items) > MAX_PER_GROUP:
            rows_by_key[k] = rng.sample(items, MAX_PER_GROUP)

    # species -> set of available (described) call types
    types_by_sp: dict[str, list[str]] = collections.defaultdict(list)
    for (sp, ct) in rows_by_key:
        types_by_sp[sp].append(ct)
    eligible_sp = {sp for sp, ts in types_by_sp.items() if len(set(ts)) >= min_types}
    logger.info("Species eligible (>=%d call types): %d", min_types, len(eligible_sp))

    out_rows: list[dict[str, Any]] = []
    for (sp, ct), items in rows_by_key.items():
        if sp not in eligible_sp:
            continue
        other_types = [t for t in set(types_by_sp[sp]) if t != ct]
        for e in items:
            n_distract = min(N_OPTIONS - 1, len(other_types))
            distract_types = rng.sample(other_types, n_distract)
            option_types = [ct] + distract_types
            option_desc = [desc_by_key[(sp, t)] for t in option_types]
            order = list(range(len(option_types)))
            rng.shuffle(order)
            correct_pos = order.index(0)
            shuffled_desc = [option_desc[i] for i in order]
            shuffled_types = [option_types[i] for i in order]

            letters = OPTION_LETTERS[: len(shuffled_desc)]
            q = rng.choice(_QUESTION_TEMPLATES).format(common=e["common_name"])
            opts = ", ".join(f"{letters[i]}) {d}" for i, d in enumerate(shuffled_desc))
            content = f"<Audio><AudioHere></Audio>\n{q}\nOptions: {opts}"
            answer = letters[correct_pos]

            audio_id = f"{e['xc_id']}__crop_{e['start_ms']}_{e['end_ms']}"
            out_rows.append(
                {
                    "id": None,  # filled after shuffle
                    "audio_paths": [e["audio_uri"]],
                    "audio_ids": [audio_id],
                    "template_path": "calltype_description_mcq_within_species_v1",
                    "skills": ["multiple_choice", "vocalization_description", "within_species"],
                    "messages": [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": answer},
                    ],
                    "task": "roots_tier1_vocal_description_mcq",
                    "source_dataset": "xeno-canto",
                    "dataset_name": split_name,
                    "license": "",
                    "metadata": json.dumps(
                        {
                            "correct": answer,
                            "correct_species": sp,
                            "correct_common_name": e["common_name"],
                            "correct_call_type": ct,
                            "option_call_types": dict(zip(letters, shuffled_types, strict=True)),
                            "crop_start_sec": e["start_ms"] / 1000.0,
                            "crop_end_sec": e["end_ms"] / 1000.0,
                            "detection_score": round(e["det_score"], 4),
                        }
                    ),
                }
            )

    rng.shuffle(out_rows)
    for i, row in enumerate(out_rows):
        row["id"] = f"{split_name}_{i:06d}"
        row["audio_ids"] = [f"{row['audio_ids'][0]}"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    ans = collections.Counter(r["messages"][1]["content"] for r in out_rows)
    ct_dist = collections.Counter(json.loads(r["metadata"])["correct_call_type"] for r in out_rows)
    sp_dist = collections.Counter(
        json.loads(r["metadata"])["correct_common_name"] for r in out_rows
    )
    logger.info("Wrote %d MCQ rows to %s", len(out_rows), output)
    logger.info("Answer dist: %s", dict(sorted(ans.items())))
    logger.info("Call-type dist: %s", dict(ct_dist))
    logger.info("Species: %d, per-species rows: %s", len(sp_dist), dict(sp_dist))


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract", "montage", "build"], required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("/tmp/birdcode_preds/xc"))
    parser.add_argument("--species", nargs="*", default=None, help="montage: limit to species")
    parser.add_argument("--per-type", type=int, default=4, help="montage: examples per call type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-name", default="calltype_mcq_within_species_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUT_DIR / "calltype_mcq_within_species_v1.jsonl",
    )
    parser.add_argument("--min-types", type=int, default=3)
    parser.add_argument(
        "--valid-keys-json",
        type=Path,
        default=None,
        help="build: JSON list of [species, call_type] pairs validated to keep",
    )
    args = parser.parse_args()

    if args.stage == "extract":
        stage_extract(args.shard_dir)
    elif args.stage == "montage":
        stage_montage(args.species, args.per_type, args.seed)
    elif args.stage == "build":
        valid_keys = None
        if args.valid_keys_json is not None:
            pairs = json.loads(Path(args.valid_keys_json).read_text())
            valid_keys = {(p[0], p[1]) for p in pairs}
        stage_build(args.split_name, args.output, valid_keys, args.min_types)


if __name__ == "__main__":
    main()

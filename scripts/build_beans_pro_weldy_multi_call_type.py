#!/usr/bin/env python3
"""Build the BEANS-Pro ``weldy-multi-call-type`` evaluation split.

Turns Weldy NW Dawn Chorus per-window sonotype annotations into a
per-species N-way **call-type discrimination** task in BEANS-Pro JSONL format.
Unlike ``weldy-call-or-song`` (which strips the ``_N`` variant suffix and
classifies call vs song), this split KEEPS the ``call_N`` variant suffix and
asks the model to distinguish between distinct calls within a single species'
repertoire — leveraging the published call-type descriptions in
``metadata/annotation_metadata.tsv``.

Filtering / balancing pipeline:

1. Iterate ``WeldyDawnChorus(split="labeled")``'s underlying CSV; each row
   carries an embedded selection_table (TSV).
2. Load ``metadata/annotation_metadata.tsv`` and build a
   ``(eBird code, sonotype) → description`` map. Drop variants with empty
   description.
3. Within each clip, keep windows with ``Category == "species"``, non-empty
   ``Species``, ``Species Code`` matching the description map, and a
   ``Sonotype`` of the form ``call_<digit>``. Song / drum / chorus are
   excluded by design.
4. Per (file_id, begin, end, species): drop windows whose call_N set is
   ambiguous (more than one variant annotated for the same window).
5. Per species: require ≥ 2 distinct call_N variants, each with
   ≥ ``--min-per-class`` windows.
6. Per species: balance to ``min(n_variant_1, n_variant_2, ...)`` of each
   variant.
7. For each kept row, cut a 2-s mono 32-kHz PCM16 WAV segment.
8. Emit a JSONL row with a multi-choice instruction whose A/B/C/... options
   are the descriptions of all eligible call variants for that species,
   shuffled per clip (seeded by the row uuid). The species name is NOT
   mentioned in the prompt — the model must discriminate from audio.

Upload to GCS::

    gsutil -m cp -r <output_dir>/* \\
        gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_multi_call_type/

Usage::

    uv run python scripts/build_beans_pro_weldy_multi_call_type.py \\
        --output-dir data/beans_pro_weldy_multi_call_type \\
        --min-per-class 5
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import fsspec
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
WELDY_GCS_ROOT = "gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
WELDY_LABELED_CSV = f"{WELDY_GCS_ROOT}/labeled.csv"
WELDY_ANNOTATION_METADATA = f"{WELDY_GCS_ROOT}/metadata/annotation_metadata.tsv"
TARGET_SR = 32_000
LICENSE_STR = "CC-BY-4.0"
SOURCE_DATASET = "weldy_dawn_chorus"
DATASET_NAME = "weldy-multi-call-type"
TASK = "multi_call_type_classification"
PROMPT_PREAMBLE = "Which description best matches the vocalization in this clip?"
PROMPT_SUFFIX = "Answer with the letter only."
LETTERS = "ABCDEFGHIJ"  # supports up to 10 call variants per species

# Selection-table Sonotype column is the basic class ("call"/"song"/"drum"); the
# `_N` variant suffix lives on the Label column instead, formatted as
# ``<ebird_code>_<sonotype>_<N>`` (e.g. ``"mouchi_call_1"``).
_LABEL_RE = re.compile(r"^([a-z0-9]+)_(call_(\d+))$")


def _parse_label(raw: str) -> tuple[str, str] | None:
    """Return ``(ebird_code, "call_N")`` if `raw` is a species call variant.

    Returns
    -------
    tuple[str, str] | None
        ``(eBird code, "call_N")`` on a parse, ``None`` for songs / drums /
        non-species labels / malformed strings.
    """
    if not raw:
        return None
    m = _LABEL_RE.match(str(raw).strip().lower())
    if not m:
        return None
    return m.group(1), m.group(2)


def _slug(s: str) -> str:
    """Filesystem-safe slug from a species or file name.

    Returns
    -------
    str
        ``s`` with any non-alphanumeric characters collapsed to underscores.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


# ── Description map ──────────────────────────────────────────────────────


def _load_descriptions(url: str) -> dict[tuple[str, str], str]:
    """Build ``(eBird code, sonotype) → description`` from annotation_metadata.tsv.

    Returns
    -------
    dict[tuple[str, str], str]
        Keys ``(eBird code, "call_N")``; values are the human-readable
        descriptions sourced from the Weldy paper.
    """
    df = pd.read_csv(url, sep="\t", keep_default_na=False, na_values=[""])
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        ebird = str(row.get("eBird_2021", "")).strip()
        sound = str(row.get("sound", "")).strip()
        desc = str(row.get("description", "")).strip()
        if not ebird or not sound or not desc:
            continue
        # Only keep call_N rows (drop song_, drum_, etc).
        if not re.match(r"^call_\d+$", sound):
            continue
        out[(ebird, sound)] = desc
    return out


# ── Pass 1: gather candidate rows ────────────────────────────────────────


def _iter_candidate_rows(
    manifest_df: pd.DataFrame,
    desc_map: dict[tuple[str, str], str],
    limit_clips: int | None,
) -> Iterator[dict]:
    """Yield candidate annotation rows for `call_N` sonotypes with descriptions.

    Yields
    ------
    dict
        Keys: ``file_id``, ``recording_id``, ``audio_32k_path``, ``begin``,
        ``end``, ``species`` (scientific), ``species_common``, ``species_code``
        (eBird code), ``variant`` (``"call_N"``), ``description``.
    """
    n_clips = 0
    for _, row in manifest_df.iterrows():
        if limit_clips is not None and n_clips >= limit_clips:
            break
        n_clips += 1
        st_raw = row.get("selection_table")
        if not isinstance(st_raw, str) or not st_raw.strip():
            continue
        st = pd.read_csv(StringIO(st_raw), sep="\t")
        if st.empty:
            continue
        if "Category" not in st.columns or "Label" not in st.columns:
            continue
        sub = st[
            (st["Category"] == "species")
            & st["Species"].notna()
            & (st["Species"].astype(str).str.strip() != "")
        ].copy()
        if sub.empty:
            continue
        parsed = [_parse_label(lbl) for lbl in sub["Label"].astype(str)]
        sub["species_code"] = [p[0] if p else None for p in parsed]
        sub["variant"] = [p[1] if p else None for p in parsed]
        sub = sub[sub["variant"].notna()].copy()
        if sub.empty:
            continue
        # Keep only variants we have a description for.
        sub["description"] = [
            desc_map.get((sc, v), "")
            for sc, v in zip(sub["species_code"], sub["variant"], strict=True)
        ]
        sub = sub[sub["description"] != ""].copy()
        if sub.empty:
            continue

        # Per-window ambiguity: drop windows where the SAME species has more
        # than one call_N variant simultaneously annotated.
        sub["_key"] = list(
            zip(sub["Begin Time (s)"], sub["End Time (s)"], sub["Species"], strict=True)
        )
        per_key_variants: dict[tuple, set[str]] = defaultdict(set)
        for _, srow in sub.iterrows():
            per_key_variants[srow["_key"]].add(srow["variant"])
        keep_keys = {k for k, vs in per_key_variants.items() if len(vs) == 1}
        sub = sub[sub["_key"].isin(keep_keys)].copy()
        if sub.empty:
            continue
        sub = sub.drop_duplicates(subset=["_key"], keep="first")

        recording_id = row.get("fn") or Path(str(row.get("file", ""))).stem
        for _, srow in sub.iterrows():
            yield {
                "file_id": str(row.get("file", "")),
                "recording_id": str(recording_id),
                "audio_32k_path": str(row.get("32khz_path", "")),
                "begin": float(srow["Begin Time (s)"]),
                "end": float(srow["End Time (s)"]),
                "species": str(srow["Species"]),
                "species_common": str(srow.get("Common Name", "")),
                "species_code": srow["species_code"],
                "variant": srow["variant"],
                "description": srow["description"],
            }


# ── Pass 2: eligibility + balancing ──────────────────────────────────────


def _select_balanced(
    candidates: list[dict],
    min_per_class: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    """Per-species: ≥ 2 variants ≥ min_per_class, balance to min per variant.

    Returns
    -------
    selected : list[dict]
    counts_before : ``{species: {variant: n}}`` pre-balance
    species_variant_desc : ``{species: {variant: description}}`` for surviving species
    """
    by_species: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    variant_desc: dict[str, dict[str, str]] = defaultdict(dict)
    for c in candidates:
        by_species[c["species"]][c["variant"]].append(c)
        variant_desc[c["species"]].setdefault(c["variant"], c["description"])

    counts_before = {
        sp: {v: len(vs) for v, vs in vbuckets.items()} for sp, vbuckets in by_species.items()
    }
    rng = random.Random(seed)
    selected: list[dict] = []
    species_variant_desc: dict[str, dict[str, str]] = {}
    for sp in sorted(by_species):
        vbuckets = by_species[sp]
        eligible_variants = sorted(v for v, vs in vbuckets.items() if len(vs) >= min_per_class)
        if len(eligible_variants) < 2:
            continue
        n = min(len(vbuckets[v]) for v in eligible_variants)
        for v in eligible_variants:
            chosen = rng.sample(vbuckets[v], n)
            selected.extend(chosen)
        species_variant_desc[sp] = {v: variant_desc[sp][v] for v in eligible_variants}
    return selected, counts_before, species_variant_desc


# ── Prompt rendering ─────────────────────────────────────────────────────


def _render_multichoice_instruction(
    descriptions: list[str],
    *,
    correct_idx: int,
    shuffle_seed: int,
) -> tuple[str, str, list[str]]:
    """Build the multiple-choice prompt + return the correct letter.

    Returns
    -------
    instruction_text : str
    correct_letter : str  (e.g. "B")
    ordered_descriptions : list[str]  (the order shown to the model)
    """
    n = len(descriptions)
    assert n <= len(LETTERS), f"too many variants ({n}) — bump LETTERS string."
    order = list(range(n))
    random.Random(shuffle_seed).shuffle(order)
    ordered = [descriptions[i] for i in order]
    new_correct = order.index(correct_idx)
    correct_letter = LETTERS[new_correct]
    lines = [PROMPT_PREAMBLE, ""]
    for i, desc in enumerate(ordered):
        lines.append(f"{LETTERS[i]}) {desc}")
    lines.append("")
    lines.append(PROMPT_SUFFIX)
    return "\n".join(lines), correct_letter, ordered


# ── Audio cutting (verbatim from call_or_song builder) ───────────────────


def _load_weldy_audio(audio_32k_path: str, fs: fsspec.AbstractFileSystem) -> tuple[np.ndarray, int]:
    full_uri = f"{WELDY_GCS_ROOT}/{audio_32k_path}"
    proto, stripped = full_uri.split("://", 1)
    assert proto == "gs"
    with fs.open(stripped, "rb") as fh:
        audio, sr = librosa.load(io.BytesIO(fh.read()), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False), sr


def _cut_and_write(audio: np.ndarray, sr: int, begin: float, end: float, out_path: Path) -> None:
    a = max(0, int(round(begin * sr)))
    b = min(audio.shape[-1], int(round(end * sr)))
    seg = audio[a:b]
    peak = float(np.max(np.abs(seg)) or 1.0)
    seg = (seg / peak * 0.97).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, seg, sr, subtype="PCM_16")


def make_beans_pro_row(
    *,
    instruction_text: str,
    output_letter: str,
    audio_path: str,
    metadata: dict,
) -> dict:
    """Build a single BEANS-Pro JSONL row.

    Returns
    -------
    dict
        Row in the BEANS-Pro JSONL schema (mirrors the sibling
        ``build_beans_pro_weldy_call_or_song.py``).
    """
    return {
        "source_dataset": SOURCE_DATASET,
        "dataset_name": DATASET_NAME,
        "output": output_letter,
        "instruction_text": instruction_text,
        "instruction": f"<Audio><AudioHere></Audio> {instruction_text}",
        "task": TASK,
        "file_name": audio_path.split("/")[-1],
        "license": LICENSE_STR,
        "id": str(uuid.uuid4()),
        "metadata": json.dumps(metadata),
        "audio_path_original_sample_rate": audio_path,
    }


def _log_eligibility_table(
    counts_before: dict[str, dict[str, int]],
    species_variant_desc: dict[str, dict[str, str]],
    min_per_class: int,
) -> None:
    """Pretty-print which species made the cut and per-variant counts."""
    logger.info(
        "Per-species pre-balance counts (eligibility = ≥ 2 variants ≥ %d windows):",
        min_per_class,
    )
    logger.info(f"  {'species':<32s} variants...")
    elig = 0
    for sp, vcounts in sorted(counts_before.items()):
        eligible = sp in species_variant_desc
        if eligible:
            elig += 1
        flag = "Y" if eligible else "."
        parts = [f"{v}={n}" for v, n in sorted(vcounts.items())]
        logger.info(f"  {sp:<32s} {flag}  {', '.join(parts)}")
    logger.info("Species: %d total, %d eligible.", len(counts_before), elig)


# ── Main build ───────────────────────────────────────────────────────────


def build(
    *,
    output_dir: Path,
    min_per_class: int,
    limit_clips: int | None,
    limit_species: list[str] | None,
    max_per_class: int | None,
    seed: int,
) -> None:
    """Run the full build into ``output_dir/{test.jsonl,audio/*.wav}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(exist_ok=True)

    logger.info("Loading annotation metadata: %s", WELDY_ANNOTATION_METADATA)
    desc_map = _load_descriptions(WELDY_ANNOTATION_METADATA)
    logger.info("Description map: %d (ebird, call_N) entries", len(desc_map))

    logger.info("Reading Weldy labeled manifest: %s", WELDY_LABELED_CSV)
    manifest = pd.read_csv(WELDY_LABELED_CSV, keep_default_na=False, na_values=[""])
    logger.info("Weldy labeled: %d clips", len(manifest))

    candidates = list(_iter_candidate_rows(manifest, desc_map, limit_clips))
    logger.info("Candidate windows (call_N with description): %d", len(candidates))

    selected, counts_before, species_variant_desc = _select_balanced(
        candidates, min_per_class, seed
    )
    _log_eligibility_table(counts_before, species_variant_desc, min_per_class)

    if limit_species:
        sp_set = set(limit_species)
        selected = [r for r in selected if r["species"] in sp_set]
        species_variant_desc = {sp: vd for sp, vd in species_variant_desc.items() if sp in sp_set}
        logger.info(
            "After --limit-species filter: %d rows, %d species",
            len(selected),
            len(species_variant_desc),
        )

    if max_per_class is not None:
        rng = random.Random(seed)
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in selected:
            by_pair[(r["species"], r["variant"])].append(r)
        capped = []
        for _k, rs in by_pair.items():
            capped.extend(rs if len(rs) <= max_per_class else rng.sample(rs, max_per_class))
        selected = capped
        logger.info("After --max-per-class cap: %d rows", len(selected))

    logger.info(
        "Balanced selection: %d rows / %d species", len(selected), len(species_variant_desc)
    )
    if not selected:
        logger.warning("No rows survived eligibility/balancing.")
        return

    fs = fsspec.filesystem("gs")
    selected_by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        selected_by_clip[r["audio_32k_path"]].append(r)
    logger.info("Source clips needed: %d", len(selected_by_clip))

    jsonl_rows: list[dict] = []
    n_done = 0
    for path, rows in selected_by_clip.items():
        try:
            audio, sr = _load_weldy_audio(path, fs)
        except Exception as err:  # noqa: BLE001
            logger.warning("Audio load failed for %s: %s — skipping %d rows.", path, err, len(rows))
            continue
        for r in rows:
            stem = (
                f"{_slug(r['recording_id'])}__"
                f"{r['begin']:.3f}_{r['end']:.3f}__"
                f"{_slug(r['species'])}__{r['variant']}"
            )
            seg_filename = f"{stem}.wav"
            seg_out = audio_out_dir / seg_filename
            _cut_and_write(audio, sr, r["begin"], r["end"], seg_out)

            # Build the multi-choice prompt deterministically.
            row_uuid = uuid.uuid4()
            variants = sorted(species_variant_desc[r["species"]].keys())
            descs = [species_variant_desc[r["species"]][v] for v in variants]
            correct_idx = variants.index(r["variant"])
            instruction_text, correct_letter, ordered_descs = _render_multichoice_instruction(
                descs, correct_idx=correct_idx, shuffle_seed=row_uuid.int
            )

            metadata = {
                "species": r["species"],
                "species_common": r["species_common"],
                "species_code": r["species_code"],
                "source_dataset": SOURCE_DATASET,
                "variant": r["variant"],
                "description": r["description"],
                "choices": ordered_descs,  # in the order shown (A, B, C…)
                "choice_variants": [variants[descs.index(d)] for d in ordered_descs],
                "correct_letter": correct_letter,
                "n_choices": len(variants),
                "begin_time_s": r["begin"],
                "end_time_s": r["end"],
                "recording_id": r["recording_id"],
            }
            row = make_beans_pro_row(
                instruction_text=instruction_text,
                output_letter=correct_letter,
                audio_path=f"audio/{seg_filename}",
                metadata=metadata,
            )
            row["id"] = str(row_uuid)
            jsonl_rows.append(row)
            n_done += 1
        if n_done % 200 == 0:
            logger.info("Cut %d / %d segments", n_done, len(selected))

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")

    letter_counts = pd.Series([r["output"] for r in jsonl_rows]).value_counts().to_dict()
    species_counts = (
        pd.Series([json.loads(r["metadata"])["species"] for r in jsonl_rows])
        .value_counts()
        .to_dict()
    )
    n_choices_dist = (
        pd.Series([json.loads(r["metadata"])["n_choices"] for r in jsonl_rows])
        .value_counts()
        .to_dict()
    )
    logger.info("Wrote %d JSONL rows to %s", len(jsonl_rows), jsonl_path)
    logger.info("Wrote %d audio segments to %s", n_done, audio_out_dir)
    logger.info(
        "Correct-letter distribution (should be ~uniform if shuffle works): %s", letter_counts
    )
    logger.info("n_choices distribution: %s", n_choices_dist)
    logger.info(
        "Species count: %d (top 5: %s)", len(species_counts), dict(list(species_counts.items())[:5])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_pro_weldy_multi_call_type",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=5,
        help="Per-species floor on call_N counts to be eligible (variants below "
        "this are dropped). Species with < 2 variants surviving are skipped.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional per (species, variant) cap after balancing.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Cap on Weldy source clips processed (smoke).",
    )
    parser.add_argument(
        "--limit-species",
        type=str,
        nargs="*",
        default=None,
        help="Filter the build to these scientific names (smoke).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(
        output_dir=args.output_dir,
        min_per_class=args.min_per_class,
        limit_clips=args.limit_clips,
        limit_species=args.limit_species,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

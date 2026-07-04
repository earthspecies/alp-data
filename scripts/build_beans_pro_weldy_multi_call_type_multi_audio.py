#!/usr/bin/env python3
"""Build the BEANS-Pro ``weldy-multi-call-type-fewshot`` few-shot eval split.

Audio-grounded multi-choice variant of ``build_beans_pro_weldy_multi_call_type.py``.
Instead of presenting the candidate call types as **text descriptions**, this
build presents one **exemplar audio clip** per variant: the model must pick
the option whose exemplar audio matches the query clip — within a single
species' repertoire, no species name given.

Prompt template (training-aligned with DRASDIC species_mcq and the
avex-hard-neg-v2 call_type_multiple_choice_v2 task):

    Here are N call types.

    A: <Audio><AudioHere></Audio>
    B: <Audio><AudioHere></Audio>
    ...

    Which call type best matches the following recording?
    <Audio><AudioHere></Audio>

Filtering / balancing pipeline is identical to the text-only sibling:
1. Iterate ``WeldyDawnChorus(split="labeled")``'s underlying CSV.
2. Load ``metadata/annotation_metadata.tsv`` and build a
   ``(eBird code, sonotype) → description`` map (only used to gate which
   variants we keep — the description is NOT shown in the prompt here).
3. Keep windows where ``Category == "species"``, ``Sonotype`` is ``call_N``,
   and the (species_code, variant) pair has a description.
4. Per species: require ≥ 2 distinct call_N variants, each with
   ≥ ``--min-per-class`` windows. Balance to ``min(n_v1, n_v2, ...)``.
5. Per row, cut a 2-s mono 32-kHz WAV. Each cut is stored once and reused
   as either an option exemplar or a query.
6. For each query window, pick one exemplar from EACH eligible variant of
   that species (the exemplar must not be the query itself). Letter order
   is deterministically shuffled per row.

Upload to GCS::

    gsutil -m cp -r <output_dir>/* \\
        gs://esp-data-ingestion/beans-pro/v0.1.0/raw/weldy_multi_call_type_fewshot/

Usage::

    uv run python scripts/build_beans_pro_weldy_multi_call_type_multi_audio.py \\
        --output-dir data/beans_pro_weldy_multi_call_type_fewshot \\
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
DATASET_NAME_PREFIX = "weldy-multi-call-type-fewshot"
TASK = "call_type_multiple_choice"
TEMPLATE_PATH = "call_type_multiple_choice"
LETTERS = "ABCDEFGHIJ"  # supports up to 10 call variants per species

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
        descriptions sourced from the Weldy paper. Used only to gate which
        variants are admitted — descriptions are NOT shown to the model.
    """
    df = pd.read_csv(url, sep="\t", keep_default_na=False, na_values=[""])
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        ebird = str(row.get("eBird_2021", "")).strip()
        sound = str(row.get("sound", "")).strip()
        desc = str(row.get("description", "")).strip()
        if not ebird or not sound or not desc:
            continue
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
) -> tuple[list[dict], dict[str, dict[str, int]], dict[str, list[str]]]:
    """Per-species: ≥ 2 variants ≥ min_per_class, balance to min per variant.

    Returns
    -------
    selected : list[dict]
    counts_before : ``{species: {variant: n}}`` pre-balance
    species_variants : ``{species: [variants in sorted order]}``
    """
    by_species: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        by_species[c["species"]][c["variant"]].append(c)

    counts_before = {
        sp: {v: len(vs) for v, vs in vbuckets.items()} for sp, vbuckets in by_species.items()
    }
    rng = random.Random(seed)
    selected: list[dict] = []
    species_variants: dict[str, list[str]] = {}
    for sp in sorted(by_species):
        vbuckets = by_species[sp]
        eligible_variants = sorted(v for v, vs in vbuckets.items() if len(vs) >= min_per_class)
        if len(eligible_variants) < 2:
            continue
        n = min(len(vbuckets[v]) for v in eligible_variants)
        for v in eligible_variants:
            chosen = rng.sample(vbuckets[v], n)
            selected.extend(chosen)
        species_variants[sp] = eligible_variants
    return selected, counts_before, species_variants


# ── Few-shot row assembly ────────────────────────────────────────────────


def _render_fewshot_instruction(num_options: int, num_shots: int) -> str:
    """Build the few-shot multi-audio MCQ prompt.

    Matches the avex-hard-neg-v2 ``call_type_multiple_choice_v2`` training
    template (DRASDIC multi-audio) used as the bulk of training for the
    multi-audio MCQ task, and the BEANS-Pro ``crow-4way`` / ``giant-otter-4way``
    evaluation template. With more than one shot, multiple ``<Audio>...``
    blocks are concatenated on the same option line, separated by spaces —
    same format as the avex-hard-neg training data.

    Returns
    -------
    str
        Prompt with ``num_options * num_shots + 1`` ``<AudioHere>``
        placeholders (one per option exemplar plus one for the query).
    """
    lines = [f"Here are {_num_word(num_options)} call types.", ""]
    audio_block = "<Audio><AudioHere></Audio>"
    for i in range(num_options):
        per_option = " ".join([audio_block] * num_shots)
        lines.append(f"{LETTERS[i]}: {per_option}")
    lines.append("")
    lines.append("Which call type best matches the following recording?")
    lines.append("<Audio><AudioHere></Audio>")
    return "\n".join(lines)


def _num_word(n: int) -> str:
    """Render a small integer as the English word used by the training prompt.

    Returns
    -------
    str
        ``"two"``, ``"three"``, ``"four"``, ``"five"`` … falling back to the
        digit string for values outside the small-int range.
    """
    return {
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }.get(n, str(n))


# ── Audio cutting ────────────────────────────────────────────────────────


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


def _log_eligibility_table(
    counts_before: dict[str, dict[str, int]],
    species_variants: dict[str, list[str]],
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
        eligible = sp in species_variants
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
    num_shots: int,
) -> None:
    """Run the full few-shot multi-audio build into ``output_dir``.

    Parameters
    ----------
    num_shots
        Number of exemplar clips per call variant. Must be ≥ 1 and ≤
        ``min_per_class - 1`` (the query consumes one slot from the per-variant
        pool, the remaining slots are available as exemplars).

    Raises
    ------
    ValueError
        If ``num_shots`` is not in ``[1, min_per_class - 1]``.
    """
    if num_shots < 1 or num_shots > min_per_class - 1:
        raise ValueError(
            f"num_shots={num_shots} must be in [1, min_per_class-1={min_per_class - 1}]"
        )
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

    selected, counts_before, species_variants = _select_balanced(
        candidates, min_per_class, seed
    )
    _log_eligibility_table(counts_before, species_variants, min_per_class)

    if limit_species:
        sp_set = set(limit_species)
        selected = [r for r in selected if r["species"] in sp_set]
        species_variants = {sp: vs for sp, vs in species_variants.items() if sp in sp_set}
        logger.info(
            "After --limit-species filter: %d rows, %d species",
            len(selected),
            len(species_variants),
        )

    if max_per_class is not None:
        rng_cap = random.Random(seed)
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in selected:
            by_pair[(r["species"], r["variant"])].append(r)
        capped = []
        for _k, rs in by_pair.items():
            capped.extend(rs if len(rs) <= max_per_class else rng_cap.sample(rs, max_per_class))
        selected = capped
        logger.info("After --max-per-class cap: %d rows", len(selected))

    logger.info(
        "Balanced selection: %d rows / %d species", len(selected), len(species_variants)
    )
    if not selected:
        logger.warning("No rows survived eligibility/balancing.")
        return

    # ── Step 1: cut all unique 2-s windows to disk ────────────────────────
    fs = fsspec.filesystem("gs")
    selected_by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        selected_by_clip[r["audio_32k_path"]].append(r)
    logger.info("Source clips needed: %d", len(selected_by_clip))

    # Index each selected window by (species, variant) for exemplar sampling.
    # ``window_key`` is a stable identifier we can dedupe by.
    def _wkey(r: dict) -> str:
        return (
            f"{_slug(r['recording_id'])}__"
            f"{r['begin']:.3f}_{r['end']:.3f}__"
            f"{_slug(r['species'])}__{r['variant']}"
        )

    by_species_variant: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    window_meta: dict[str, dict] = {}
    audio_filename_by_window: dict[str, str] = {}

    n_done = 0
    for path, rows in selected_by_clip.items():
        try:
            audio, sr = _load_weldy_audio(path, fs)
        except Exception as err:  # noqa: BLE001
            logger.warning("Audio load failed for %s: %s — skipping %d rows.", path, err, len(rows))
            continue
        for r in rows:
            wkey = _wkey(r)
            seg_filename = f"{wkey}.wav"
            audio_filename_by_window[wkey] = seg_filename
            _cut_and_write(audio, sr, r["begin"], r["end"], audio_out_dir / seg_filename)
            by_species_variant[r["species"]][r["variant"]].append(wkey)
            window_meta[wkey] = r
            n_done += 1
        if n_done % 200 == 0:
            logger.info("Cut %d / %d segments", n_done, len(selected))
    logger.info("Cut %d segments total", n_done)

    # ── Step 2: assemble few-shot MCQ rows ────────────────────────────────
    jsonl_rows: list[dict] = []
    rng_assemble = random.Random(seed + 1)

    dataset_name = f"{DATASET_NAME_PREFIX}-{num_shots}shot"
    for sp in sorted(by_species_variant):
        variants = species_variants[sp]
        # Each query is each window in the species's selected pool; ``num_shots``
        # exemplars per variant are sampled from the OTHER windows of that
        # variant (never the query itself).
        for variant in variants:
            query_windows = by_species_variant[sp][variant]
            for q_wkey in query_windows:
                option_exemplars: dict[str, list[str]] = {}
                feasible = True
                for v in variants:
                    pool = [w for w in by_species_variant[sp][v] if w != q_wkey]
                    if len(pool) < num_shots:
                        # Not enough non-query exemplars for K-shot at this
                        # variant. With balancing to min_per_class >= num_shots+1
                        # this should never trigger, but guard anyway.
                        feasible = False
                        break
                    option_exemplars[v] = rng_assemble.sample(pool, num_shots)
                if not feasible:
                    continue

                # Deterministically shuffle option order keyed on a per-row uuid.
                row_uuid = uuid.UUID(int=rng_assemble.getrandbits(128))
                order = list(variants)
                random.Random(row_uuid.int).shuffle(order)
                correct_idx = order.index(variant)
                correct_letter = LETTERS[correct_idx]

                option_audio_paths: list[str] = []
                for v in order:
                    for exemplar_key in option_exemplars[v]:
                        option_audio_paths.append(
                            f"audio/{audio_filename_by_window[exemplar_key]}"
                        )
                query_audio_path = f"audio/{audio_filename_by_window[q_wkey]}"
                audio_paths = option_audio_paths + [query_audio_path]

                instruction_text = _render_fewshot_instruction(len(order), num_shots)

                metadata = {
                    "species": sp,
                    "species_common": window_meta[q_wkey]["species_common"],
                    "species_code": window_meta[q_wkey]["species_code"],
                    "source_dataset": SOURCE_DATASET,
                    "query_variant": variant,
                    "option_variants": list(order),
                    "option_types": {
                        LETTERS[i]: order[i] for i in range(len(order))
                    },
                    "correct": correct_letter,
                    "correct_variant": variant,
                    "n_choices": len(order),
                    "num_shots": num_shots,
                    "query_window_key": q_wkey,
                    "option_window_keys": {
                        LETTERS[i]: option_exemplars[order[i]] for i in range(len(order))
                    },
                    "recording_id": window_meta[q_wkey]["recording_id"],
                    "begin_time_s": window_meta[q_wkey]["begin"],
                    "end_time_s": window_meta[q_wkey]["end"],
                }

                row = {
                    "id": str(row_uuid),
                    "audio_paths": audio_paths,
                    "audio_ids": [Path(p).stem for p in audio_paths],
                    "template_path": TEMPLATE_PATH,
                    "skills": ["call_type_multiple_choice"],
                    "messages": [
                        {"role": "user", "content": instruction_text},
                        {"role": "assistant", "content": correct_letter},
                    ],
                    "task": TASK,
                    "source_dataset": SOURCE_DATASET,
                    "dataset_name": dataset_name,
                    "license": LICENSE_STR,
                    "metadata": json.dumps(metadata),
                    "audio_path_original_sample_rate": audio_paths,
                }
                jsonl_rows.append(row)

    jsonl_path = output_dir / "test.jsonl"
    with open(jsonl_path, "w") as fh:
        for row in jsonl_rows:
            fh.write(json.dumps(row) + "\n")

    letter_counts = (
        pd.Series([r["messages"][1]["content"] for r in jsonl_rows]).value_counts().to_dict()
    )
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
        default=REPO_ROOT / "data" / "beans_pro_weldy_multi_call_type_fewshot",
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
    parser.add_argument(
        "--num-shots",
        type=int,
        default=1,
        help="Exemplar clips per call variant per row (1-shot/2-shot/3-shot). "
        "Must be <= min-per-class - 1.",
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
        num_shots=args.num_shots,
    )


if __name__ == "__main__":
    main()

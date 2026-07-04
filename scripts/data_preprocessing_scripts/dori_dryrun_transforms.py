"""Metadata-only dry-run of the DORI stage3.5 transform chains.

Loads the DORI ``train`` split (CSV metadata only — no audio decode) and
applies the NON-audio transforms from the two stage3.5 DORI entries via
``apply_transformations``, then reports the resulting row counts +
species_list distribution. Verifies the long_tail_upsample / downsample
math before the block goes into a training run.

The ``chat`` transform is dropped from each chain here (it only renders
text from already-present columns and needs the prompt dir); everything
that changes row counts (filter / drop / long_tail_upsample / downsample /
rename / select / set_columns) is exercised.
"""

from __future__ import annotations

from collections import Counter

# Register the NatureLM-project transforms (drop_null_or_empty_string,
# select_columns, set_columns, chat, ...) into the shared transform
# registry BEFORE building the TypeAdapter. Requires running with the
# project root on PYTHONPATH (the sbatch wrapper cds into it).
import data.transforms  # noqa: F401  registers project transforms
from pydantic import TypeAdapter  # noqa: E402

from esp_data.datasets import DORI  # noqa: E402
from esp_data.transforms.registry import RegisteredTransformConfigs  # noqa: E402


def _validate(tfs: list[dict]) -> list:
    """Validate raw transform dicts into RegisteredTransformConfigs objects.

    Built lazily so the TypeAdapter captures the project-extended union.
    """
    adapter = TypeAdapter(RegisteredTransformConfigs)
    return [adapter.validate_python(t) for t in tfs]


def _counts(ds: DORI, col: str = "species_list") -> Counter:
    """Return a Counter of values in ``col`` across the backend rows."""
    c: Counter = Counter()
    for row in ds._data:
        v = row.get(col, "")
        c[v if (v is not None and str(v).strip()) else "<empty/None>"] += 1
    return c


POSITIVE_TFS = [
    {"type": "drop_null_or_empty_string", "columns": ["species"]},
    {"type": "filter", "property": "label_source",
     "values": ["DORI (pseudo-label)"], "mode": "exclude"},
    {"type": "long_tail_upsample", "property": "species",
     "sufficient_threshold": 4000, "max_repeats": 5, "seed": 42},
    {"type": "rename_columns", "mapping": {"species": "species_list"}},
    {"type": "select_columns",
     "columns": ["16khz_path", "32khz_path", "audio_fp", "species_list"]},
    {"type": "set_columns", "columns": {"mixup_group": "marine_mammal"}},
]

NEGATIVE_TFS = [
    {"type": "filter", "property": "is_negative",
     "values": ["True"], "mode": "include"},
    {"type": "downsample", "fraction": 0.10, "seed": 42},
    {"type": "rename_columns", "mapping": {"species": "species_list"}},
    {"type": "select_columns",
     "columns": ["16khz_path", "32khz_path", "audio_fp", "species_list"]},
    {"type": "set_columns", "columns": {"mixup_group": "marine_mammal"}},
]


def main() -> None:
    """Run both DORI chains and print resulting distributions."""
    print("=== DORI POSITIVES (drop pseudo + LTU 4000/5x) ===", flush=True)
    pos = DORI(split="train", sample_rate=None, backend="pandas")
    print(f"raw train rows: {len(pos)}", flush=True)
    pos.apply_transformations(_validate(POSITIVE_TFS))
    print(f"after transforms: {len(pos)} rows", flush=True)
    dist = _counts(pos)
    for k, v in dist.most_common():
        print(f"  {k:32s} {v:>7}")

    print("\n=== DORI NEGATIVES (is_negative + downsample 0.10) ===", flush=True)
    neg = DORI(split="train", sample_rate=None, backend="pandas")
    neg.apply_transformations(_validate(NEGATIVE_TFS))
    print(f"after transforms: {len(neg)} rows (-> all 'None')", flush=True)
    ndist = _counts(neg)
    for k, v in ndist.most_common(5):
        print(f"  {k:32s} {v:>7}")

    print(f"\nTOTAL DORI rows into chain: {len(pos) + len(neg):,}")


if __name__ == "__main__":
    main()

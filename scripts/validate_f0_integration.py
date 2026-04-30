"""Validate F0 Bioacoustic integration end-to-end.

Checks:
1. Dataset loads from updated GCS path
2. Taxa filtering works
3. f0_features transform produces correct columns
4. chat prompt templates render with templated values
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "esp-research" / "projects" / "NatureLM-audio-v1.5"))
import data.transforms  # noqa: F401 — registers custom transforms

from esp_data.datasets.f0_bioacoustic import F0Bioacoustic, TAXA
from esp_data.transforms import transform_from_config

from data.transforms.f0_features import F0Features, F0FeaturesConfig
from data.transforms.make_chat import MakeChat, MakeChatConfig
from data.transforms.drop_null_or_empty_string import DropNullOrEmptyString, DropNullOrEmptyStringConfig


TAXA_16K = [
    "canids", "hummingbirds", "La_Palma_chaffinches", "lions", "little_owls",
    "long-billed_hermits", "monk_parakeets", "orangutans",
    "Reunion_grey_white_eyes", "spotted_hyenas",
]


def main() -> None:
    # Step 1: Load dataset
    print("=" * 60)
    print("Step 1: Load F0 Bioacoustic dataset")
    print("=" * 60)
    ds = F0Bioacoustic(split="all", sample_rate=16000, taxa=TAXA_16K)
    print(f"  Loaded {len(ds)} rows (16kHz taxa)")
    print(f"  Columns: {ds.columns}")
    assert len(ds) > 0, "Dataset is empty!"

    # Peek at a row
    row = ds._data[0]
    print(f"  Sample row keys: {list(row.keys())}")
    f0_raw = row.get("f0_contour", "")
    print(f"  f0_contour type: {type(f0_raw).__name__}, length: {len(str(f0_raw))} chars")
    print(f"  f0_contour preview: {str(f0_raw)[:200]}...")
    print()

    # Step 2: Apply f0_features transform
    print("=" * 60)
    print("Step 2: Apply f0_features transform")
    print("=" * 60)
    cfg = F0FeaturesConfig(
        type="f0_features",
        f0_column="f0_contour",
        output_rate_hz=[5, 10, 20, 30],
        max_points=30,
        seed=42,
    )
    transform = F0Features.from_config(cfg)
    backend_copy = ds._data.copy()
    backend_out, meta = transform(backend_copy)
    print(f"  Output columns: {backend_out.columns}")

    expected_cols = {"f0_mean", "f0_range", "f0_contour_text", "f0_rate_hz", "f0_duration"}
    actual_cols = set(backend_out.columns)
    missing = expected_cols - actual_cols
    assert not missing, f"Missing columns: {missing}"
    print("  All expected columns present!")

    # Check a few rows
    nonempty = 0
    for i in range(min(20, len(backend_out))):
        r = backend_out[i]
        if r.get("f0_mean", ""):
            nonempty += 1
            if nonempty <= 3:
                print(f"  Row {i}: f0_mean={r['f0_mean']}, f0_range={r['f0_range']}, "
                      f"rate={r['f0_rate_hz']}Hz, duration={r['f0_duration']}")
                print(f"          contour (first 120 chars): {r['f0_contour_text'][:120]}")
    print(f"  Non-empty f0_mean in first 20 rows: {nonempty}")
    print()

    # Step 3: Apply drop_null_or_empty_string
    print("=" * 60)
    print("Step 3: Drop empty rows")
    print("=" * 60)
    drop_cfg = DropNullOrEmptyStringConfig(
        type="drop_null_or_empty_string",
        columns=["f0_mean"],
    )
    drop_transform = DropNullOrEmptyString.from_config(drop_cfg)
    backend_filtered, _ = drop_transform(backend_out)
    print(f"  Rows after filtering: {len(backend_filtered)} (was {len(backend_out)})")
    print()

    # Step 4: Apply chat template (f0_summary)
    print("=" * 60)
    print("Step 4: Render chat templates")
    print("=" * 60)

    for template_name in ["f0_summary", "f0_contour", "f0_species"]:
        chat_cfg = MakeChatConfig(
            type="chat",
            template_name=template_name,
            seed=42,
        )
        chat_transform = MakeChat.from_config(chat_cfg)
        chat_backend, _ = chat_transform(backend_filtered.copy())

        r = chat_backend[0]
        msgs = r.get("messages", [])
        print(f"  [{template_name}] messages type: {type(msgs).__name__}, length: {len(msgs)}")
        if msgs:
            print(f"    User: {msgs[0].get('content', '')[:120]}...")
            print(f"    Asst: {msgs[1].get('content', '')[:120]}...")
        print()

    # Step 5: Quick check on dolphins at 32kHz
    print("=" * 60)
    print("Step 5: Dolphins at 32kHz")
    print("=" * 60)
    ds_dolphin = F0Bioacoustic(split="all", sample_rate=32000, taxa=["dolphins"])
    print(f"  Loaded {len(ds_dolphin)} dolphin rows at 32kHz")
    assert len(ds_dolphin) > 0, "No dolphin rows!"

    print()
    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

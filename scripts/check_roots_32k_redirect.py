# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""End-to-end check that ROOTS at sample_rate=32000 hits 32 kHz mirrors.

For each ROOTS split that was previously live-resampled, instantiate the
dataset, fetch the first row, and check:
  - _resolve_audio_path returns a path containing 'audio_32k' / 'audio_32khz'
    / 'pseudovox_32k' / 'sed_scenes_32k' / 'sed_diarization_32k' / 'audio_32k'
  - row['sample_rate'] == 32000

Run via `uv run python scripts/check_roots_32k_redirect.py` from the project
root inside a slurm job (it needs the full esp-data dependency tree).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from esp_data.datasets.roots import ROOTS  # noqa: E402

EXPECTED_MARKERS: dict[str, tuple[str, ...]] = {
    # T1 v2 splits that previously live-resampled
    "tier1_v2_acoustic_caption_pseudovox": ("animalspeak_pseudovox_32k",),
    "tier1_v2_voc_desc_mcq_pseudovox": ("animalspeak_pseudovox_32k",),
    "tier1_v2_acoustic_caption_field_notes_inat": ("audio_32khz", "audio_32k"),
    "tier1_v2_acoustic_caption_field_notes_xc": ("audio_32k",),
    "tier1_v2_cat_snr_mcq_custom_bins_xc": ("audio_32k",),
    "tier1_v2_cat_snr_mcq_inat": ("audio_32khz", "audio_32k"),
    "tier1_v2_cat_snr_mcq_xc": ("audio_32k",),
    "tier1_v2_snr_binary_xc": ("audio_32k",),
    "tier1_v2_snr_oe_xc": ("audio_32k",),
    "tier1_v2_voc_desc_mcq_field_notes_inat": ("audio_32khz", "audio_32k"),
    "tier1_v2_voc_desc_mcq_field_notes_xc": ("audio_32k",),
    # F0-based splits are already 32 kHz native; redirect is a no-op
    "tier1_v2_acoustic_caption_f0bioacoustic": ("audio_32k",),
    "tier1_v2_f0_bioacoustic_summary": ("audio_32k",),
    "tier1_v2_voc_desc_f0_mcq_f0bioacoustic": ("audio_32k",),
    # Tier 3 splits
    "tier3_highest_pitch_species_mcq_wabad_v2": ("audio_32k",),
    "tier3_longest_voc_species_mcq_sed_v2": ("synthetic_sed_scenes_32k",),
    "tier3_longest_voc_species_mcq_sed_diarization_v2": ("synthetic_sed_diarization_32k",),
    "tier3_tier1_structural_caption_wabad_v1": ("audio_32k",),
    "tier3_wabad_cropped_templated_v1": ("audio_32k",),
}


def main() -> None:
    fails: list[tuple[str, str]] = []
    for split, markers in EXPECTED_MARKERS.items():
        try:
            ds = ROOTS(split=split, sample_rate=32000)
        except Exception as exc:
            fails.append((split, f"instantiation failed: {exc}"))
            continue

        try:
            raw_row = ds._data[0]
            resolved = str(ds._resolve_audio_path(raw_row))
        except Exception as exc:
            fails.append((split, f"resolve failed: {exc}"))
            continue

        hit_marker = any(m in resolved for m in markers)
        if not hit_marker:
            fails.append((split, f"resolved path missing markers {markers}: {resolved}"))
            continue

        # Now do a real read of that one row
        try:
            row = ds[0]
        except Exception as exc:
            fails.append((split, f"row read failed (resolved={resolved}): {exc}"))
            continue

        sr = row.get("sample_rate")
        if sr != 32000:
            fails.append((split, f"sample_rate={sr} not 32000 (resolved={resolved})"))
            continue

        print(f"OK  {split}\n    -> {resolved}", flush=True)

    print(f"\nfailures: {len(fails)}", flush=True)
    for split, msg in fails:
        print(f"  FAIL {split}: {msg}", flush=True)

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

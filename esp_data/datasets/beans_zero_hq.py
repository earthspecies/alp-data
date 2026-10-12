"""BEANS-Zero, lifestage/call-type only, with true-original (not pre-downsampled) audio.

BEANS-Zero's own "audio_path_original_sample_rate" column for the lifestage and
call-type splits is NOT the true original Xeno-Canto recording: empirically, it
is already capped at 16000 Hz for every clip checked (real spectral content only
reaches ~5-7 kHz), while the true XC recordings are natively ~44.1kHz with
content up to ~16-17 kHz. See scratch_ports/afnext_subseg/beans_zero_hq/ for the
fetch script that re-downloaded the true originals directly from xeno-canto.org
and built the corrected manifests this class points at.

Only lifestage and call-type are re-pointed here (the only two Xeno-Canto-derived
splits in BEANS-Zero); every other split's provenance was not re-examined.
"""

from esp_data import DatasetInfo, register_dataset

from .beans_zero import BeansZero


@register_dataset
class BeansZeroHQ(BeansZero):
    """BEANS-Zero lifestage/call-type with re-fetched, true-original-fidelity audio."""

    info = DatasetInfo(
        name="beans_zero_hq",
        owner=BeansZero.info.owner,
        split_paths={
            "lifestage": "gs://foundation-model-data/beans_zero_hq/lifestage_test_hq.jsonl",
            "call-type": "gs://foundation-model-data/beans_zero_hq/call-type_test_hq.jsonl",
        },
        version="0.1.0-hq",
        description="BEANS-Zero lifestage/call-type with true-original (re-fetched from "
        "xeno-canto.org) rather than pre-downsampled-to-16kHz audio.",
        sources=["Xeno-canto"],
        license=BeansZero.info.license,
    )

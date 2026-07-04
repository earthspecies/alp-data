"""Validate that every audio referenced by the synthetic sg_call manifest
exists on GCS under ``gs://foundation-model-data/synthetic/subsegmentation/sg_call/audio/``.

Streaming, memory-light: bounded sets, no full materialisation. Mirrors
``scripts/data_preprocessing_scripts/xeno_canto_strong/validate_audio_present.py``.

Outputs:
- JSON summary
- CSV of any missing audio_file_name with their manifest row index
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path

MANIFEST_URI = "gs://foundation-model-data/synthetic/subsegmentation/sg_call/manifest.csv"
AUDIO_PREFIX = "gs://foundation-model-data/synthetic/subsegmentation/sg_call/audio"


def _set_from_listing(prefix: str, suffix: str) -> set[str]:
    """Stream ``gsutil ls -r prefix/**``; return set of basename stems."""
    print(f"Listing {prefix}/ ...", flush=True)
    out: set[str] = set()
    proc = subprocess.Popen(
        ["gsutil", "ls", f"{prefix}/**"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    n = 0
    pat = re.compile(rf".*/(sg_call_\d+){re.escape(suffix)}$")
    for line in proc.stdout:
        m = pat.match(line.strip())
        if m:
            out.add(m.group(1))
            n += 1
            if n % 50_000 == 0:
                print(f"  {n:,} files indexed ...", flush=True)
    proc.wait()
    print(f"  {len(out):,} unique stems at {prefix}", flush=True)
    return out


def _manifest_stems() -> tuple[set[str], dict[str, int]]:
    """Stream manifest.csv; return set of audio stems + their row indices.

    Returns
    -------
    tuple[set[str], dict[str, int]]
        ``(set_of_stems, {stem: row_index})``.
    """
    print(f"Streaming manifest {MANIFEST_URI} ...", flush=True)
    out = subprocess.run(
        ["gsutil", "cat", MANIFEST_URI],
        check=True, capture_output=True, text=True, timeout=300,
    ).stdout
    csv.field_size_limit(100 * 1024 * 1024)
    stems: set[str] = set()
    row_idx: dict[str, int] = {}
    for i, r in enumerate(csv.DictReader(io.StringIO(out))):
        # `audio_file_name` is e.g. ``sg_call_00005678.wav``; strip extension.
        name = r.get("audio_file_name", "").strip()
        if name.endswith(".wav"):
            stem = name[:-4]
            stems.add(stem)
            row_idx[stem] = i
    print(f"  {len(stems):,} unique stems in manifest", flush=True)
    return stems, row_idx


def main() -> None:
    """Run the audit, write a JSON summary + missing-rows CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/home/david_earthspecies_org/subseg_synthetic_audio_audit"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_stems, row_idx = _manifest_stems()
    audio_present = _set_from_listing(AUDIO_PREFIX, ".wav")

    missing = manifest_stems - audio_present
    extra = audio_present - manifest_stems

    summary = {
        "manifest_rows": len(manifest_stems),
        "audio_present_total": len(audio_present),
        "missing_in_audio": len(missing),
        "extra_in_audio_not_in_manifest": len(extra),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (args.out_dir / "subseg_synthetic_audio_audit.json").write_text(
        json.dumps(summary, indent=2)
    )
    if missing:
        miss_csv = args.out_dir / "subseg_synthetic_missing_audio.csv"
        with miss_csv.open("w") as f:
            w = csv.writer(f)
            w.writerow(["audio_file_name", "manifest_row_index"])
            for stem in sorted(missing):
                w.writerow([f"{stem}.wav", row_idx.get(stem, -1)])
        print(f"Wrote {len(missing):,} missing -> {miss_csv}")
        print("First 10 missing stems:")
        for s in sorted(missing)[:10]:
            print(f"  {s}")
    if extra:
        extra_csv = args.out_dir / "subseg_synthetic_extra_audio.csv"
        with extra_csv.open("w") as f:
            w = csv.writer(f)
            w.writerow(["audio_file_name"])
            for stem in sorted(extra):
                w.writerow([f"{stem}.wav"])
        print(f"Wrote {len(extra):,} extra (unreferenced) -> {extra_csv}")

    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()

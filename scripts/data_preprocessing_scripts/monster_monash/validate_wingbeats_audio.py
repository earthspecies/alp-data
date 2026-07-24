"""Audit audio completeness for the monster-monash wingbeat datasets.

For MosquitoSound and InsectSound, stream the manifest's audio stems and
bulk-list each GCS audio shard (audio / audio_16k / audio_32k); report
any manifest rows whose FLAC is missing from a shard.

Streaming + bounded sets only. Designed for the Slurm cpu partition.
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

DATASETS = {
    "mosquito_sound": "gs://esp-data-ingestion/monster-monash-mosquito-sound/v0.1.0",
    "insect_sound": "gs://esp-data-ingestion/monster-monash-insect-sound/v0.1.0",
}
SHARDS = ("audio", "audio_16k", "audio_32k")


def _stems_in_listing(prefix: str) -> set[str]:
    """Return the set of ``clip_NNNNNN`` stems present under ``prefix``."""
    out: set[str] = set()
    proc = subprocess.Popen(
        ["gsutil", "ls", f"{prefix}/**"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    pat = re.compile(r".*/(clip_\d+)\.flac$")
    for line in proc.stdout:
        m = pat.match(line.strip())
        if m:
            out.add(m.group(1))
    proc.wait()
    return out


def _manifest_stems(manifest_uri: str) -> set[str]:
    """Return the ``clip_NNNNNN`` stems referenced by the manifest's audio_path."""
    raw = subprocess.run(
        ["gsutil", "cat", manifest_uri],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    csv.field_size_limit(100 * 1024 * 1024)
    stems: set[str] = set()
    pat = re.compile(r"(clip_\d+)\.flac$")
    for r in csv.DictReader(io.StringIO(raw)):
        m = pat.search(r.get("audio_path", ""))
        if m:
            stems.add(m.group(1))
    return stems


def main() -> None:
    """Audit each dataset; write a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/home/david_earthspecies_org/wingbeats_audio_audit")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, dict] = {}
    any_missing = False
    for name, root in DATASETS.items():
        print(f"\n=== {name} ===", flush=True)
        manifest = f"{root}/{name}_all.csv"
        print(f"Streaming manifest {manifest} ...", flush=True)
        man_stems = _manifest_stems(manifest)
        print(f"  {len(man_stems):,} manifest rows", flush=True)

        ds_summary: dict[str, int] = {"manifest_rows": len(man_stems)}
        for shard in SHARDS:
            print(f"Listing {root}/{shard}/ ...", flush=True)
            present = _stems_in_listing(f"{root}/{shard}")
            missing = man_stems - present
            ds_summary[f"{shard}_present"] = len(present)
            ds_summary[f"{shard}_missing"] = len(missing)
            print(f"  {shard}: {len(present):,} present, {len(missing):,} missing", flush=True)
            if missing:
                any_missing = True
                miss_path = args.out_dir / f"{name}_{shard}_missing.csv"
                miss_path.write_text("\n".join(sorted(missing)))
        overall[name] = ds_summary

    print("\n=== SUMMARY ===")
    print(json.dumps(overall, indent=2))
    (args.out_dir / "wingbeats_audio_audit.json").write_text(json.dumps(overall, indent=2))
    sys.exit(1 if any_missing else 0)


if __name__ == "__main__":
    main()

"""Validate that every audio referenced by xc_strong_with_selection_table.csv
exists on GCS at the expected 16k + 32k paths.

Streaming, memory-light: keeps two sets (~600 KB + ~30 MB peak), no full
materialisation of any DataFrame. Designed to run on Slurm CPU partition.

Outputs:
- A JSON summary with per-shard missing counts.
- A CSV of missing xc_ids with which shards they're missing from.
- Lines like ``XC1120736: missing in audio_16k, audio_32k, audio`` to stdout.
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

MANIFEST_URI = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw/xc_strong_with_selection_table.csv"
AUDIO_ROOT = "gs://esp-data-ingestion/xeno-canto/v0.1.0/raw"


def _set_from_listing(prefix: str, suffix: str) -> set[str]:
    """Stream a `gsutil ls -r prefix/**` listing; return set of basename stems.

    Parameters
    ----------
    prefix : str
        e.g. ``gs://.../audio_16k``
    suffix : str
        File extension to keep (``.wav`` / ``.mp3``).

    Returns
    -------
    set[str]
        Filename stems (e.g. ``XC65654``).
    """
    print(f"Listing {prefix}/ ...", flush=True)
    out = set()
    # Pipe gsutil's output and parse line-by-line so we never hold the
    # full listing in memory at once.
    proc = subprocess.Popen(
        ["gsutil", "ls", f"{prefix}/**"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    n = 0
    pat = re.compile(rf".*/(XC\d+){re.escape(suffix)}$")
    for line in proc.stdout:
        m = pat.match(line.strip())
        if m:
            out.add(m.group(1))
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} files indexed ...", flush=True)
    proc.wait()
    print(f"  {len(out):,} unique IDs at {prefix}", flush=True)
    return out


def _manifest_ids() -> tuple[set[str], dict[str, dict[str, str]]]:
    """Stream the XCStrong manifest; return its full XC-id set + per-row paths.

    Returns
    -------
    tuple
        ``(set_of_xc_ids, {xc_id: {'audio_fp': ..., '16khz_path': ..., '32khz_path': ...}})``.
    """
    print(f"Streaming manifest {MANIFEST_URI} ...", flush=True)
    out = subprocess.run(
        ["gsutil", "cat", MANIFEST_URI],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout
    ids: set[str] = set()
    rows: dict[str, dict[str, str]] = {}
    csv.field_size_limit(100 * 1024 * 1024)
    for r in csv.DictReader(io.StringIO(out)):
        xcid = f"XC{r['xc_id']}"
        ids.add(xcid)
        rows[xcid] = {
            "audio_fp": r.get("audio_fp", ""),
            "16khz_path": r.get("16khz_path", ""),
            "32khz_path": r.get("32khz_path", ""),
        }
    print(f"  {len(ids):,} unique XC IDs in manifest", flush=True)
    return ids, rows


def main() -> None:
    """Run the existence sweep and write a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/david_earthspecies_org/logs"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_ids, manifest_paths = _manifest_ids()

    audio_16k = _set_from_listing(f"{AUDIO_ROOT}/audio_16k", ".wav")
    audio_32k = _set_from_listing(f"{AUDIO_ROOT}/audio_32k", ".wav")
    audio_mp3 = _set_from_listing(f"{AUDIO_ROOT}/audio", ".mp3")

    missing_16k = manifest_ids - audio_16k
    missing_32k = manifest_ids - audio_32k
    missing_mp3 = manifest_ids - audio_mp3
    missing_any = missing_16k | missing_32k | missing_mp3
    missing_all = missing_16k & missing_32k & missing_mp3

    summary = {
        "manifest_rows": len(manifest_ids),
        "audio_16k_present_total": len(audio_16k),
        "audio_32k_present_total": len(audio_32k),
        "audio_mp3_present_total": len(audio_mp3),
        "missing_16k": len(missing_16k),
        "missing_32k": len(missing_32k),
        "missing_mp3": len(missing_mp3),
        "missing_in_any_shard": len(missing_any),
        "missing_in_all_three_shards": len(missing_all),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (args.out_dir / "xc_strong_audio_audit.json").write_text(json.dumps(summary, indent=2))

    missing_csv = args.out_dir / "xc_strong_missing_audio.csv"
    with missing_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["xc_id", "missing_16k", "missing_32k", "missing_mp3", "manifest_audio_fp"])
        for xcid in sorted(missing_any):
            w.writerow(
                [
                    xcid,
                    int(xcid in missing_16k),
                    int(xcid in missing_32k),
                    int(xcid in missing_mp3),
                    manifest_paths[xcid]["audio_fp"],
                ]
            )
    print(f"Wrote {len(missing_any):,} missing-row report -> {missing_csv}")

    if missing_any:
        print("\nFirst 20 IDs missing in any shard:")
        for xcid in sorted(missing_any)[:20]:
            shards = []
            if xcid in missing_16k:
                shards.append("16k")
            if xcid in missing_32k:
                shards.append("32k")
            if xcid in missing_mp3:
                shards.append("mp3")
            print(f"  {xcid}: missing in {','.join(shards)}")

    sys.exit(0 if not missing_any else 1)


if __name__ == "__main__":
    main()

"""Stage a normalized mode-A manifest for the subseg_v9_pool_25k corpus.

Mode A's native ``manifest.csv`` carries only a time+hierarchy selection
table (no frequency), while its per-scene ``meta/*.json`` files hold the
per-unit frequency bounds and the group structure. This script joins the two
into a single alp_data-style manifest with two JSON columns the training
transform can render directly:

* ``units_json``  — ``[{start_s, end_s, low_hz, high_hz, group_id}]``
* ``groups_json`` — ``[{id, type, start_s, end_s, low_hz, high_hz}]`` where a
  group's frequency span is the min/max over its member units (the meta groups
  store only time bounds).

Modes B and C already expose a frequency ``units_json`` column in their native
manifests, so only mode A needs this pre-join. Output is written next to the
corpus mirror under the esp-data ingestion bucket.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
csv.field_size_limit(1 << 30)

_CORPUS = "gs://foundation-model-data/synthetic/subseg_v9_pool_25k"
_OUT = "gs://esp-data-ingestion/subseg_v9_pool_25k/mode_A_normalized.csv"


def _round_units_and_groups(meta: dict) -> tuple[str, str]:
    """Return ``(units_json, groups_json)`` strings for one scene's meta."""
    units = []
    for u in meta.get("units", []):
        units.append({
            "start_s": round(float(u["onset_s"]), 3),
            "end_s": round(float(u["offset_s"]), 3),
            "low_hz": int(round(float(u["low_hz"]))),
            "high_hz": int(round(float(u["high_hz"]))),
            "group_id": u.get("group_id"),
        })
    # group freq = min/max over member units (meta groups store only time)
    groups = []
    for g in meta.get("groups", []):
        members = [u for u in units if u["group_id"] == g["id"]]
        low = min((u["low_hz"] for u in members), default=0)
        high = max((u["high_hz"] for u in members), default=0)
        groups.append({
            "id": g["id"],
            "type": g["type"],
            "start_s": round(float(g["start_s"]), 3),
            "end_s": round(float(g["end_s"]), 3),
            "low_hz": low,
            "high_hz": high,
        })
    return json.dumps(units, separators=(",", ":")), json.dumps(groups, separators=(",", ":"))


def _process(fs: fsspec.AbstractFileSystem, row: dict) -> dict | None:
    name = row["audio_file_name"]
    stem = name[:-4] if name.endswith(".wav") else name
    meta_path = f"{_CORPUS.replace('gs://', '')}/mode_A/meta/{stem}.json"
    try:
        with fs.open(meta_path, "rt") as f:
            meta = json.load(f)
    except FileNotFoundError:
        logger.warning(f"no meta for {stem}")
        return None
    if not meta.get("units"):
        return None
    units_json, groups_json = _round_units_and_groups(meta)
    return {
        "example_id": stem,
        "audio_path": f"mode_A/audio/{stem}.wav",
        "species": row.get("Species", "") or meta.get("species", ""),
        "duration_s": "",
        "units_json": units_json,
        "groups_json": groups_json,
    }


def main() -> None:
    fs = fsspec.filesystem("gs")
    with fs.open(f"{_CORPUS}/mode_A/manifest.csv", "rt") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"mode_A manifest: {len(rows)} rows; reading meta JSONs ...")

    out_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=48) as ex:
        for i, rec in enumerate(ex.map(lambda r: _process(fs, r), rows)):
            if rec is not None:
                out_rows.append(rec)
            if (i + 1) % 2000 == 0:
                logger.info(f"  {i + 1}/{len(rows)} ({len(out_rows)} kept)")

    logger.info(f"kept {len(out_rows)} rows; writing {_OUT}")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["example_id", "audio_path", "species",
                                        "duration_s", "units_json", "groups_json"])
    w.writeheader()
    w.writerows(out_rows)
    with fs.open(_OUT, "wt") as f:
        f.write(buf.getvalue())
    logger.info(f"wrote {len(out_rows)} rows -> {_OUT}")


if __name__ == "__main__":
    main()

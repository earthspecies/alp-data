"""Analyze the frequency-box distribution of the delphinid-whistle annotations.

Answers: after resampling to the NatureLM 32 kHz stack (Nyquist 16 kHz), how
much of the annotated whistle-box energy actually survives? Reads either the
Raven-style selection tables (``.txt``, tab-separated with ``Low Freq (Hz)`` /
``High Freq (Hz)``) or the DeepAcoustics/DeepSqueak ``.mat`` Calls export
(``Box = [onset_s, low_hz, dur_s, bandwidth_hz]``), whichever is present.

    uv run --with scipy --with numpy python analyze_freq_boxes.py <annotations_dir>

Prints, overall and per top-level subfolder (recording location):
  * count of boxes, Low/High Freq percentiles
  * % boxes fully below 16 kHz  (captured intact at 32 kHz)
  * % straddling 16 kHz         (fundamental likely retained, top clipped)
  * % fully above 16 kHz        (lost entirely at 32 kHz)
  * same split at an 8 kHz ceiling (relevant only if 16 kHz audio were used)
and a keep-32k / raise-rate / filter recommendation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CEILINGS_HZ = (16000.0, 8000.0)


def _boxes_from_raven(txt: Path) -> list[tuple[float, float]]:
    """Return ``(low_hz, high_hz)`` pairs from one Raven selection table.

    Returns
    -------
    list[tuple[float, float]]
        One pair per selection row that has both frequency bounds.
    """
    import csv

    out: list[tuple[float, float]] = []
    with open(txt, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        lo_c = next((cols[c] for c in cols if c.startswith("low freq")), None)
        hi_c = next((cols[c] for c in cols if c.startswith("high freq")), None)
        if not lo_c or not hi_c:
            return out
        for row in reader:
            try:
                lo, hi = float(row[lo_c]), float(row[hi_c])
            except (TypeError, ValueError):
                continue
            if hi > lo >= 0:
                out.append((lo, hi))
    return out


def _boxes_from_mat(mat: Path) -> list[tuple[float, float]]:
    """Return ``(low_hz, high_hz)`` pairs from a DeepSqueak/DeepAcoustics .mat.

    Calls(i).Box = [onset_s, low_hz, duration_s, bandwidth_hz]; high = low + bw.

    Returns
    -------
    list[tuple[float, float]]
        One pair per call with a valid frequency box.
    """
    import scipy.io as sio

    out: list[tuple[float, float]] = []
    try:
        m = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    except Exception:
        return out
    calls = m.get("Calls")
    if calls is None:
        return out
    calls = np.atleast_1d(calls)
    for c in calls:
        box = getattr(c, "Box", None)
        if box is None:
            continue
        box = np.array(box).ravel()
        if box.size >= 4:
            lo, bw = float(box[1]), float(box[3])
            if bw > 0 and lo >= 0:
                out.append((lo, lo + bw))
    return out


def _report(name: str, boxes: list[tuple[float, float]]) -> None:
    """Print distribution + Nyquist-retention stats for one group."""
    if not boxes:
        print(f"\n[{name}] no boxes found")
        return
    arr = np.array(boxes)
    lo, hi = arr[:, 0], arr[:, 1]
    n = len(boxes)
    print(f"\n[{name}] {n} boxes")
    for label, v in (("Low Freq (Hz)", lo), ("High Freq (Hz)", hi)):
        p = np.percentile(v, [1, 25, 50, 75, 95, 99, 100])
        print(f"  {label:16s} min={v.min():8.0f}  p25={p[1]:8.0f}  "
              f"p50={p[2]:8.0f}  p75={p[3]:8.0f}  p95={p[4]:8.0f}  max={p[6]:8.0f}")
    for ceil in CEILINGS_HZ:
        below = np.mean(hi <= ceil) * 100
        strad = np.mean((lo < ceil) & (hi > ceil)) * 100
        above = np.mean(lo >= ceil) * 100
        print(f"  @ {ceil / 1000:.0f} kHz ceiling: "
              f"{below:5.1f}% fully below | {strad:5.1f}% straddle | {above:5.1f}% fully above")


def main() -> None:
    """Walk the annotations dir; report overall + per-location box stats."""
    if len(sys.argv) != 2:
        sys.exit("usage: analyze_freq_boxes.py <annotations_dir>")
    root = Path(sys.argv[1])
    txts = list(root.rglob("*.txt"))
    mats = list(root.rglob("*.mat"))
    print(f"found {len(txts)} .txt selection tables, {len(mats)} .mat files under {root}")

    # Prefer Raven .txt; fall back to .mat if no .txt frequency data.
    per_group: dict[str, list[tuple[float, float]]] = {}
    files = txts if txts else mats
    parse = _boxes_from_raven if txts else _boxes_from_mat
    for f in files:
        try:
            rel = f.relative_to(root)
            group = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        except ValueError:
            group = "(root)"
        per_group.setdefault(group, []).extend(parse(f))

    all_boxes = [b for g in per_group.values() for b in g]
    _report("ALL", all_boxes)
    for group in sorted(per_group):
        _report(group, per_group[group])


if __name__ == "__main__":
    main()

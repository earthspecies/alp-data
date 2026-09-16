"""Build the FASD13 few-shot detection benchmark (Zenodo 15843741).

Hoffman et al. 2025, "Synthetic data enables context-aware bioacoustic sound
event detection" (arXiv:2503.00296). FASD13 is the *evaluation* benchmark
introduced by that paper: 13 bioacoustics sub-datasets, 109 recordings, ~143 h,
each recording paired with a DCASE-style CSV marking onsets/offsets of a single
predetermined target category.

DRASDIC ("Domain Randomization for Animal Sound Detection In-Context") is the
paper's *model*, not a dataset. It is trained on synthetic scenes from the same
work and evaluated on FASD13; FASD13 shares no audio with that training data.

Source layout (per sub-dataset zip)::

    <CODE>/<stem>.wav
    <CODE>/<stem>.csv     # Starttime,Endtime,Q,Audiofilename   Q in {POS,UNK,NEG}

Licensing is per sub-dataset (CC-BY-1.0 / 4.0, CC-BY-SA-4.0, CC-BY-NC-4.0,
CC-BY-NC-SA-4.0, public domain, and one custom citation-required entry for MS),
so it is carried as a per-row column rather than a single dataset-level string.

Stages, run in order. Sized for a CPU node: the download is ~16 GB, the
extracted originals ~21 GB, and resampling is streamed so memory stays flat
regardless of recording length (HG is 8 h of stereo per file)::

    STAGE=<scratch>            # ~110 GB free
    DEST=gs://<bucket>/fasd13/v0.1.0

    # 1. fetch the 15 Zenodo files, verify md5, unzip
    python fasd13.py download  --stage-dir "$STAGE/zenodo" --work "$STAGE/work"

    # 2. 16 kHz + 32 kHz mono mirrors + durations.csv (duration, rate, channels)
    python fasd13.py resample   --root "$STAGE/work" --out-root "$STAGE/mirrors" --workers 16

    # 3. per-file selection tables -> one manifest per sub-dataset + fasd13_all.csv
    python fasd13.py manifests  --root "$STAGE/work" \
        --durations-csv "$STAGE/mirrors/durations.csv" --out-dir "$STAGE/manifests"

    # 4. deterministic few-shot support clips (5 per recording, 10 s, both rates)
    python fasd13.py support    --root "$STAGE/work" --mirrors "$STAGE/mirrors" \
        --out-root "$STAGE/mirrors" --manifest-dir "$STAGE/manifests" --workers 16

Then upload the originals, both mirrors, the support clips and the manifests to
``$DEST``, keeping the tree layout-preserving: manifests carry audio paths
relative to their own directory, so they must sit alongside ``audio_native/``,
``audio_16k/`` and ``audio_32k/``.

The N-shot protocol follows ``drasdic/data/test.py`` of the (private)
earthspecies/drasdic repo: events are ordered by end time, ``UNK`` events are
excluded from shot selection, the support region is everything up to the end of
the Nth event, and the query region is everything after it. Support clips place
their event one third of the way into a 10 s window, mirroring
``subselect_support_fixed``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

SOURCE_DATASET = "fasd13"
ZENODO_RECORD = "15843741"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Support-clip geometry (mirrors drasdic subselect_support_fixed).
MAX_SHOTS = 5
SUPPORT_CLIP_SEC = 10.0
SUPPORT_LEFT_FRACTION = 1.0 / 3.0

# Resampling is streamed in blocks of this many seconds to bound peak memory.
_BLOCK_SEC = 30.0

ST_COLUMNS = ["Selection", "Begin Time (s)", "End Time (s)", "Q", "event_index"]

# FASD13's target class is nameless -- in few-shot detection it is defined by
# the support examples, not a name -- so there is no species-style label
# column. ``Q`` carries the per-event status (POS / UNK) and is the single
# source of truth; ``detection_target`` on the row records what *kind* of
# category the target is. An earlier revision also wrote a constant
# ``Label="target"`` on every row, which marked UNK events as positives.

# Per-sub-dataset metadata, transcribed from the Zenodo summary table and
# LICENSE.txt. ``n_files`` / ``duration_hr`` / ``n_events`` are the published
# figures and are cross-checked against what we actually ingest.
SUBDATASETS: dict[str, dict[str, object]] = {
    "AS": {
        "name": "AnuraSet",
        "n_files": 12,
        "duration_hr": 0.20,
        "n_events": 162,
        "recording_type": "TPAM",
        "location": "Brazil",
        "taxa": "Anura",
        "detection_target": "Species",
        "license": "CC-BY-1.0",
        "citation": (
            "Cañas et al. (2023). AnuraSet: A dataset for benchmarking neotropical anuran "
            "calls identification in passive acoustic monitoring. Zenodo. "
            "https://doi.org/10.5281/zenodo.8342596"
        ),
    },
    "CC": {
        "name": "Carrion Crow",
        "n_files": 10,
        "duration_hr": 10.00,
        "n_events": 2200,
        "recording_type": "On-body",
        "location": "Spain",
        "taxa": "Corvus corone + Clamator glandarius",
        "detection_target": "Species+Life Stage",
        "license": "CC-BY-SA-4.0",
        "citation": (
            "Hoffman et al. (2025). Synthetic data enables context-aware bioacoustic sound "
            "event detection. arXiv:2503.00296"
        ),
    },
    "GS": {
        "name": "Gunshot",
        "n_files": 7,
        "duration_hr": 38.33,
        "n_events": 85,
        "recording_type": "TPAM",
        "location": "Gabon",
        "taxa": "Homo sapiens",
        "detection_target": "Production Mechanism",
        "license": "CC-BY-NC-4.0",
        "citation": (
            "Gottesman, B. (2024). Dataset of Gunshot Sounds and Koogu Model related to "
            "Yoh et al. 2024. Zenodo. https://doi.org/10.5281/zenodo.11192704"
        ),
    },
    "HA": {
        "name": "Hawaiian Birds",
        "n_files": 12,
        "duration_hr": 1.10,
        "n_events": 628,
        "recording_type": "TPAM",
        "location": "Hawaii, USA",
        "taxa": "Aves",
        "detection_target": "Species",
        "license": "CC-BY-4.0",
        "citation": (
            "Navine, A., Kahl, S., Tanimoto-Johnson, A., Klinck, H., & Hart, P. (2022). A "
            "collection of fully-annotated soundscape recordings from the Island of Hawai'i. "
            "Zenodo. https://doi.org/10.5281/zenodo.7078499"
        ),
    },
    "HG": {
        "name": "Hainan Gibbons",
        "n_files": 9,
        "duration_hr": 72.00,
        "n_events": 483,
        "recording_type": "TPAM",
        "location": "Hainan, China",
        "taxa": "Nomascus hainanus",
        "detection_target": "Species",
        "license": "CC-BY-NC-SA-4.0",
        "citation": (
            "Dufourq et al. (2020). Automated detection of Hainan gibbon calls for passive "
            "acoustic monitoring. Zenodo. https://doi.org/10.5281/zenodo.3991714"
        ),
    },
    "HW": {
        "name": "Humpback Whale",
        "n_files": 10,
        "duration_hr": 2.79,
        "n_events": 1565,
        "recording_type": "UPAM",
        "location": "North Pacific Ocean",
        "taxa": "Megaptera novaeangliae",
        "detection_target": "Species",
        "license": "public-domain",
        "citation": (
            "Allen, A. N., Harvey, M., Harrell, L., Jansen, A., et al. (2021). A convolutional "
            "neural network for automated detection of humpback whale song in a diverse, "
            "long-term passive acoustic dataset. Front. Mar. Sci., 8."
        ),
    },
    "JS": {
        "name": "Jumping Spider",
        "n_files": 4,
        "duration_hr": 0.23,
        "n_events": 924,
        "recording_type": "Substrate",
        "location": "Laboratory",
        "taxa": "Habronattus",
        "detection_target": "Sound Type",
        "license": "CC-BY-SA-4.0",
        "citation": (
            "Hoffman et al. (2025). Synthetic data enables context-aware bioacoustic sound "
            "event detection. arXiv:2503.00296"
        ),
    },
    "KD": {
        "name": "Katydid",
        "n_files": 12,
        "duration_hr": 2.00,
        "n_events": 883,
        "recording_type": "TPAM",
        "location": "Panamá",
        "taxa": "Tettigoniidae",
        "detection_target": "Species",
        "license": "public-domain",
        "citation": (
            "Madhusudhana, S., Klinck, H., & Symes, L. B. (2024). Extensive data engineering "
            "to the rescue. Trans. R. Soc. B, 379(1904)."
        ),
    },
    "MS": {
        "name": "Marmoset",
        "n_files": 10,
        "duration_hr": 1.67,
        "n_events": 1369,
        "recording_type": "Laboratory",
        "location": "Laboratory",
        "taxa": "Callithrix jacchus",
        "detection_target": "Call Type",
        "license": "custom-citation-required",
        "citation": (
            "Sarkar, E. & Magimai.-Doss, M. (2023), Interspeech, pp. 1189-1193; recordings by "
            "Zhang et al. (2018), JASA 144(1):478-487. Both must be cited."
        ),
    },
    "PM": {
        "name": "Powdermill",
        "n_files": 4,
        "duration_hr": 6.42,
        "n_events": 2032,
        "recording_type": "TPAM",
        "location": "Pennsylvania, USA",
        "taxa": "Passeriformes",
        "detection_target": "Species",
        "license": "public-domain",
        "citation": (
            "Chronister, L. M., Rhinehart, T. A., Place, A., & Kitzes, J. (2021). An annotated "
            "set of audio recordings of eastern North American birds. Ecology, 102(6):e03329."
        ),
    },
    "RG": {
        "name": "Ruffed Grouse",
        "n_files": 2,
        "duration_hr": 1.50,
        "n_events": 34,
        "recording_type": "TPAM",
        "location": "Pennsylvania, USA",
        "taxa": "Bonasa umbellus",
        "detection_target": "Species",
        "license": "public-domain",
        "citation": (
            "Lapp, S., Parker, H., & Tett, C. (2022). ARU audio recordings with ruffed grouse "
            "annotations (Pennsylvania, 2020). Dryad. https://doi.org/10.5061/dryad.hdr7sqvmc"
        ),
    },
    "RS": {
        "name": "Rana Sierrae",
        "n_files": 7,
        "duration_hr": 1.87,
        "n_events": 552,
        "recording_type": "UPAM",
        "location": "California, USA",
        "taxa": "Rana sierrae",
        "detection_target": "Species",
        "license": "public-domain",
        "citation": (
            "Lapp, S. & Kitzes, J. (2023). Rana sierrae annotated aquatic soundscapes (2022). "
            "Dryad. https://doi.org/10.5061/dryad.9s4mw6mn3"
        ),
    },
    "RW": {
        "name": "Right Whale",
        "n_files": 10,
        "duration_hr": 5.00,
        "n_events": 398,
        "recording_type": "UPAM",
        "location": "Gulf of St. Lawrence",
        "taxa": "Eubalaena glacialis",
        "detection_target": "Species",
        "license": "CC-BY-4.0",
        "citation": (
            "Simard, Y., Kirsebom, O., Frazao, F., Roy, N., Matwin, S., & Giard, S. (2020). "
            "Acoustic recordings of North Atlantic right whale upcalls in the Gulf of St. "
            "Lawrence. FRDR. https://doi.org/10.20383/101.0241"
        ),
    },
}

CODES = list(SUBDATASETS)


def _codes(selected: str | None) -> list[str]:
    """Resolve a comma-separated ``--codes`` value to a list of sub-dataset codes.

    Returns
    -------
    list[str]

    Raises
    ------
    SystemExit
        If an unknown code is requested.
    """
    if not selected:
        return CODES
    out = [c.strip().upper() for c in selected.split(",") if c.strip()]
    unknown = [c for c in out if c not in SUBDATASETS]
    if unknown:
        raise SystemExit(f"unknown sub-dataset code(s): {unknown}; expected {CODES}")
    return out


# ── download ────────────────────────────────────────────────────────────────
def _md5(path: Path, chunk: int = 1 << 22) -> str:
    """Compute the md5 hex digest of a file.

    Returns
    -------
    str
    """
    h = hashlib.md5()  # noqa: S324 - matching Zenodo's published checksum, not security
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(stage_dir: Path, work_dir: Path) -> None:
    """Fetch every Zenodo file, verify its md5, and unzip the sub-datasets.

    Already-downloaded files with a matching checksum are skipped, so the stage
    is resumable after a timeout.

    Parameters
    ----------
    stage_dir : Path
        Where the raw ``.zip`` / ``.pdf`` / ``.txt`` downloads are kept.
    work_dir : Path
        Where the ``<CODE>/`` trees are extracted.

    Raises
    ------
    SystemExit
        If a download's checksum does not match Zenodo's.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API) as resp:  # noqa: S310 - fixed https literal
        record = json.load(resp)
    files = {f["key"]: f for f in record["files"]}
    print(f"Zenodo record {ZENODO_RECORD}: {len(files)} files", flush=True)

    for key, meta in sorted(files.items()):
        dest = stage_dir / key
        want = str(meta["checksum"]).split(":", 1)[-1]
        if dest.exists() and _md5(dest) == want:
            print(f"  {key}: cached, md5 ok", flush=True)
        else:
            url = meta["links"]["self"]
            print(f"  {key}: downloading {int(meta['size']) / 1e6:.1f} MB ...", flush=True)
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - url from the Zenodo API
            got = _md5(dest)
            if got != want:
                raise SystemExit(f"checksum mismatch for {key}: {got} != {want}")
            print(f"  {key}: md5 ok", flush=True)

        if key.endswith(".zip"):
            code = key[:-4]
            if code not in SUBDATASETS:
                print(f"  {key}: unexpected zip, skipping extraction", flush=True)
                continue
            out = work_dir / code
            if out.exists() and any(out.glob("*.wav")):
                print(f"  {code}: already extracted", flush=True)
                continue
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(work_dir)
            print(f"  {code}: extracted {len(list(out.glob('*.wav')))} wavs", flush=True)

    for code in CODES:
        wavs = sorted((work_dir / code).glob("*.wav"))
        annos = sorted((work_dir / code).glob("*.csv"))
        expected = SUBDATASETS[code]["n_files"]
        flag = "" if len(wavs) == expected else f"  !! expected {expected}"
        print(f"  {code}: {len(wavs)} wav / {len(annos)} csv{flag}", flush=True)


# ── resample ────────────────────────────────────────────────────────────────
def _stream_convert(src: str, out_root: str, rel: str) -> tuple[float, int, int]:
    """Stream one recording into 16 kHz + 32 kHz mono mirrors.

    Deliberately block-wise rather than whole-file. FASD13's long recordings
    make the naive approach untenable: HG is 8 h of stereo at 9.6 kHz per file,
    which expands to ~4 GB once decoded to float32 and again on resampling to
    32 kHz, so a process pool of any width exhausts node memory. Streaming
    bounds peak memory to one block regardless of recording length.

    Uses ``soxr.ResampleStream``, which carries filter state across blocks and
    so is sample-exact with a whole-file resample -- a plain per-block resample
    would inject a discontinuity at every block boundary.

    Parameters
    ----------
    src : str
        Source audio path.
    out_root : str
        Root to write ``audio_16k/`` and ``audio_32k/`` under.
    rel : str
        Relative key ``<CODE>/<stem>``.

    Returns
    -------
    tuple[float, int, int]
        ``(duration_sec, native_sample_rate, channels)``.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    targets = (16000, 32000)
    with sf.SoundFile(src) as fh:
        native_sr, channels = int(fh.samplerate), int(fh.channels)
        block = max(1, int(_BLOCK_SEC * native_sr))
        writers, streams = {}, {}
        for tgt in targets:
            out = Path(out_root) / f"audio_{tgt // 1000}k" / f"{rel}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            writers[tgt] = sf.SoundFile(str(out), "w", samplerate=tgt, channels=1, subtype="PCM_16")
            streams[tgt] = (
                None
                if native_sr == tgt
                else soxr.ResampleStream(native_sr, tgt, 1, dtype="float32", quality="VHQ")
            )
        total = 0
        try:
            flushed = False
            while True:
                data = fh.read(block, dtype="float32", always_2d=True)
                if data.shape[0] == 0:
                    break
                total += data.shape[0]
                mono = data.mean(axis=1).astype(np.float32)
                last = data.shape[0] < block
                for tgt in targets:
                    stream = streams[tgt]
                    y = mono if stream is None else stream.resample_chunk(mono, last=last)
                    if y.size:
                        writers[tgt].write(np.clip(y, -1.0, 1.0))
                if last:
                    flushed = True
                    break
            if not flushed:
                empty = np.zeros(0, dtype=np.float32)
                for tgt in targets:
                    stream = streams[tgt]
                    if stream is not None:
                        y = stream.resample_chunk(empty, last=True)
                        if y.size:
                            writers[tgt].write(np.clip(y, -1.0, 1.0))
        finally:
            for w in writers.values():
                w.close()
    return total / float(native_sr), native_sr, channels


def _ffmpeg_transcode(src: str, dest: str) -> None:
    """Transcode to PCM WAV with ffmpeg, preserving rate and channel count.

    Every HA recording is a FLAC bitstream carrying a ``.wav`` extension, and
    two of them hold frames libsndfile refuses ("unknown error in flac
    decoder", "flac decoder lost sync"). ffmpeg decodes both in full, to
    exactly their declared durations, so it is the fallback rather than a
    reason to drop the files.

    Raises
    ------
    RuntimeError
        If ffmpeg exits non-zero.
    """
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-c:a", "pcm_s16le", dest],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")


def _resample_one(args: tuple[str, str, str]) -> tuple[str, float, int, int, str]:
    """Write 16k + 32k mono mirrors of one recording.

    Parameters
    ----------
    args : tuple[str, str, str]
        ``(src_wav, out_root, rel)`` where ``rel`` is ``<CODE>/<stem>``.

    Returns
    -------
    tuple[str, float, int, int, str]
        ``(rel, duration_sec, native_sr, channels, status)``.
    """
    import tempfile

    src, out_root, rel = args
    try:
        dur, sr, ch = _stream_convert(src, out_root, rel)
        return rel, dur, sr, ch, "ok"
    except Exception as exc:  # noqa: BLE001 - fall back before giving up
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = str(Path(td) / "decoded.wav")
                _ffmpeg_transcode(src, tmp)
                dur, sr, ch = _stream_convert(tmp, out_root, rel)
            return rel, dur, sr, ch, "ok (ffmpeg fallback)"
        except Exception as exc2:  # noqa: BLE001
            return rel, 0.0, 0, 0, f"ERROR: {exc} | ffmpeg fallback: {exc2}"


def resample(root: Path, out_root: Path, workers: int, codes: list[str]) -> None:
    """Write 16k + 32k mirrors for every recording and emit ``durations.csv``.

    Parameters
    ----------
    root : Path
        Extracted tree root holding ``<CODE>/``.
    out_root : Path
        Destination for ``audio_16k/`` and ``audio_32k/``.
    workers : int
        Process-pool size.
    codes : list[str]
        Sub-dataset codes to process.

    Raises
    ------
    SystemExit
        If any file fails to decode or write.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(w), str(out_root), f"{code}/{w.stem}")
        for code in codes
        for w in sorted((root / code).glob("*.wav"))
    ]
    print(f"resampling {len(jobs)} recordings with {workers} workers ...", flush=True)
    rows, errors, fallbacks, done = [], 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, j) for j in jobs]):
            rel, dur, sr, ch, status = fut.result()
            done += 1
            if status.startswith("ERROR"):
                errors += 1
                print(f"  {status} ({rel})", flush=True)
            else:
                if status != "ok":
                    fallbacks += 1
                    print(f"  {rel}: {status}", flush=True)
                rows.append((rel, dur, sr, ch))
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} ...", flush=True)
    with open(out_root / "durations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rel", "duration_sec", "native_sample_rate", "channels"])
        w.writerows(sorted(rows))
    total_hr = sum(r[1] for r in rows) / 3600.0
    print(
        f"done: {done} written ({total_hr:.2f} h), {fallbacks} via ffmpeg fallback, "
        f"{errors} errors",
        flush=True,
    )
    if errors:
        raise SystemExit(f"{errors} files failed")


# ── shared annotation logic ─────────────────────────────────────────────────
def _read_events(anno_fp: Path) -> pd.DataFrame:
    """Read one FASD13 annotation CSV, sorted by end time.

    ``event_index`` numbers the non-UNK events in end-time order starting at 1;
    UNK events get ``-1``. This is the ordering the N-shot protocol uses.

    Returns
    -------
    pd.DataFrame
        Columns ``Starttime``, ``Endtime``, ``Q``, ``event_index``.
    """
    df = pd.read_csv(anno_fp)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["Q"] = df["Q"].astype(str).str.strip().str.upper()
    df = df.sort_values("Endtime", kind="stable").reset_index(drop=True)
    df["event_index"] = -1
    known = df["Q"] != "UNK"
    df.loc[known, "event_index"] = range(1, int(known.sum()) + 1)
    return df


def _shot_ends_from_selection_tsv(tsv: str, max_shots: int = MAX_SHOTS) -> list[float]:
    """Derive shot end times from a serialised selection table.

    The manifest does not store these: they are a pure function of the
    selection table, and a stored copy silently goes stale under windowed
    reads, where table times are re-based but a stored column would not be.

    Parameters
    ----------
    tsv : str
        Serialised selection table, as carried in the ``selection_table``
        column.
    max_shots : int
        Maximum number of shots to return.

    Returns
    -------
    list[float]
        Ascending, recording-absolute end times of the first ``max_shots``
        non-UNK events.
    """
    st = pd.read_csv(StringIO(tsv), sep="\t")
    known = st[st["Q"] != "UNK"]
    return [float(t) for t in sorted(known["End Time (s)"])[:max_shots]]


def _selection_tsv(df: pd.DataFrame) -> str:
    """Serialise events into a WABAD-shaped Raven TSV blob.

    Returns
    -------
    str
    """
    out = pd.DataFrame(
        {
            "Selection": range(1, len(df) + 1),
            "Begin Time (s)": [round(float(v), 4) for v in df["Starttime"]],
            "End Time (s)": [round(float(v), 4) for v in df["Endtime"]],
            "Q": list(df["Q"]),
            "event_index": [int(v) for v in df["event_index"]],
        }
    )[ST_COLUMNS]
    return out.to_csv(sep="\t", index=False)


# ── manifests ───────────────────────────────────────────────────────────────
def build_manifests(root: Path, durations_csv: Path, out_dir: Path, codes: list[str]) -> None:
    """Parse annotation CSVs into per-file selection tables and write manifests.

    Emits one CSV per sub-dataset plus ``fasd13_all.csv``.

    Parameters
    ----------
    root : Path
        Extracted tree root holding ``<CODE>/``.
    durations_csv : Path
        ``durations.csv`` from the resample stage.
    out_dir : Path
        Destination for the manifest CSVs.
    codes : list[str]
        Sub-dataset codes to process.

    Raises
    ------
    SystemExit
        If a recording has no matching annotation CSV, or a sub-dataset is empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        r["rel"]: (float(r["duration_sec"]), int(r["native_sample_rate"]), int(r["channels"]))
        for r in csv.DictReader(open(durations_csv))
    }

    all_rows: list[dict[str, object]] = []
    for code in codes:
        sd = SUBDATASETS[code]
        rows: list[dict[str, object]] = []
        for wav in sorted((root / code).glob("*.wav")):
            rel = f"{code}/{wav.stem}"
            anno_fp = wav.with_suffix(".csv")
            if not anno_fp.exists():
                raise SystemExit(f"missing annotation CSV for {rel}")
            dur, native_sr, channels = meta[rel]
            df = _read_events(anno_fp)
            n_pos = int((df["Q"] == "POS").sum())
            n_unk = int((df["Q"] == "UNK").sum())
            rows.append(
                {
                    "sound_name": wav.name,
                    "subdataset": code,
                    "subdataset_name": sd["name"],
                    "split": code,
                    "audio_duration": round(dur, 3),
                    # audio_fp is the originals pointer, as it is in WABAD: the
                    # file exactly as Zenodo ships it, at its native rate and
                    # channel count. The mirrors below are the mono-normalised
                    # convenience views.
                    "audio_fp": f"audio_native/{rel}.wav",
                    "16khz_path": f"audio_16k/{rel}.wav",
                    "32khz_path": f"audio_32k/{rel}.wav",
                    "native_sample_rate": native_sr,
                    "channels": channels,
                    "n_events": len(df),
                    "n_pos": n_pos,
                    "n_unk": n_unk,
                    "taxa": sd["taxa"],
                    "detection_target": sd["detection_target"],
                    "recording_type": sd["recording_type"],
                    "location": sd["location"],
                    "source_dataset": SOURCE_DATASET,
                    "license": sd["license"],
                    "citation": sd["citation"],
                    "selection_table": _selection_tsv(df),
                }
            )
        if not rows:
            raise SystemExit(f"no recordings found for sub-dataset {code} under {root}")
        df_code = pd.DataFrame(rows)
        df_code.to_csv(out_dir / f"fasd13_{code}.csv", index=False)
        n_shots = df_code["selection_table"].map(lambda t: len(_shot_ends_from_selection_tsv(t)))
        short = df_code[n_shots < MAX_SHOTS]
        warn = f"  !! {len(short)} file(s) with <{MAX_SHOTS} shots" if len(short) else ""
        print(
            f"  {code} ({sd['name']}): {len(df_code)} recs, "
            f"{int(df_code['n_pos'].sum())} POS (published {sd['n_events']}), "
            f"{int(df_code['n_unk'].sum())} UNK, "
            f"{df_code['audio_duration'].sum() / 3600:.2f} h "
            f"(published {sd['duration_hr']:.2f} h){warn}",
            flush=True,
        )
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    head = [
        "sound_name",
        "subdataset",
        "subdataset_name",
        "split",
        "audio_duration",
        "audio_fp",
        "16khz_path",
        "32khz_path",
        "n_events",
        "n_pos",
        "n_unk",
    ]
    df_all = df_all[head + [c for c in df_all.columns if c not in head]]
    df_all.to_csv(out_dir / "fasd13_all.csv", index=False)
    print(
        f"  all: {len(df_all)} recordings, {int(df_all['n_pos'].sum())} POS events, "
        f"{df_all['audio_duration'].sum() / 3600:.2f} h -> fasd13_all.csv",
        flush=True,
    )


# ── support clips ───────────────────────────────────────────────────────────
def _support_window(event_start: float, support_end: float) -> tuple[float, float]:
    """Return the ``SUPPORT_CLIP_SEC`` window placing an event 1/3 of the way in.

    Bounded by ``support_end`` -- the end of the Nth event -- **not** by the
    recording length. That bound is the whole point: DRASDIC draws its support
    from ``audio[:support_endsample]`` and never sees a sample beyond it, so a
    clip that ran past it would hand us audio the reference system is not
    allowed, and (because the clip's event list is rendered from whatever falls
    inside it) would print query-region onsets straight into the prompt.

    If the support region is shorter than a clip, the whole region is used --
    a shorter clip rather than DRASDIC's cyclic padding, which would fabricate
    repeated audio.

    Returns
    -------
    tuple[float, float]
        ``(window_start, window_end)`` in seconds, inside ``[0, support_end]``.
    """
    if support_end <= SUPPORT_CLIP_SEC:
        return 0.0, support_end
    start = event_start - SUPPORT_CLIP_SEC * SUPPORT_LEFT_FRACTION
    start = min(max(0.0, start), support_end - SUPPORT_CLIP_SEC)
    return start, start + SUPPORT_CLIP_SEC


def _event_times_str(df: pd.DataFrame, win_start: float, win_end: float) -> str:
    """Render in-window non-UNK events in the DRASDIC few-shot SED wording.

    Times are window-relative, clipped to the window, and formatted as
    ``"1.7s-2.9s, 5.8s-6.6s"`` -- matching the ``synthetic_sed_fewshot``
    conversations NatureLM was trained on.

    Returns
    -------
    str
    """
    parts = []
    for _, ev in df.iterrows():
        if str(ev["Q"]).upper() == "UNK":
            continue
        b, e = float(ev["Starttime"]), float(ev["Endtime"])
        if e <= win_start or b >= win_end:
            continue
        b = max(b, win_start) - win_start
        e = min(e, win_end) - win_start
        parts.append(f"{b:.1f}s-{e:.1f}s")
    return ", ".join(parts)


def _support_one(args: tuple[str, str, str, str, float, list[float]]) -> tuple[list[dict], str]:
    """Cut the support clips for one recording at both sample rates.

    ``audio_root`` may be local or a ``gs://`` mirror root; only the needed
    slice is read, so this can be re-run against GCS without re-staging the
    ~50 GB of mirrors locally.

    Parameters
    ----------
    args : tuple
        ``(anno_fp, audio_root, out_root, rel, support_end, shot_starts)``
        where ``shot_starts`` holds the start time of each shot event and
        ``support_end`` is the end of the Nth event.

    Returns
    -------
    tuple[list[dict], str]
        ``(support_rows, status)``.

    Raises
    ------
    RuntimeError
        If a mirror does not carry its expected sample rate.
    """
    import numpy as np
    import soundfile as sf

    from alp_data.io import anypath, read_audio

    anno_fp, audio_root, out_root, rel, support_end, shot_starts = args
    code, stem = rel.split("/", 1)
    try:
        df = _read_events(Path(anno_fp))
        rows = []
        for shot, ev_start in enumerate(shot_starts, start=1):
            win_start, win_end = _support_window(float(ev_start), float(support_end))
            paths = {}
            for tgt in (16000, 32000):
                src = anypath(audio_root) / f"audio_{tgt // 1000}k" / f"{rel}.wav"
                audio, sr = read_audio(src, start_time=win_start, end_time=win_end)
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1 if audio.shape[-1] <= 8 else 0)
                if sr != tgt:
                    raise RuntimeError(f"{src}: expected {tgt} Hz mirror, got {sr}")
                out_rel = f"{code}/{stem}__shot{shot}.wav"
                out = Path(out_root) / f"support_{tgt // 1000}k" / out_rel
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, np.clip(audio, -1.0, 1.0).astype("float32"), tgt, subtype="PCM_16")
                paths[tgt] = f"support_{tgt // 1000}k/{out_rel}"
            in_win = df[(df["Endtime"] > win_start) & (df["Starttime"] < win_end)]
            rows.append(
                {
                    "subdataset": code,
                    "sound_name": f"{stem}.wav",
                    "shot_index": shot,
                    "support_16khz_path": paths[16000],
                    "support_32khz_path": paths[32000],
                    "window_start_sec": round(win_start, 4),
                    "window_end_sec": round(win_end, 4),
                    "clip_duration": round(win_end - win_start, 4),
                    "event_times": _event_times_str(df, win_start, win_end),
                    "n_events_in_clip": int((in_win["Q"] != "UNK").sum()),
                    "n_unk_in_clip": int((in_win["Q"] == "UNK").sum()),
                }
            )
        return rows, "ok"
    except Exception as exc:  # noqa: BLE001
        return [], f"ERROR: {exc}"


def build_support(
    root: Path, mirrors: str, out_root: Path, manifest_dir: Path, workers: int
) -> None:
    """Materialise the ``MAX_SHOTS`` support clips per recording, at both rates.

    Support clip ``k`` is derived only from event ``k``, so an N-shot episode
    simply uses clips 1..N -- one build serves the 1-, 4- and 5-shot configs.

    Parameters
    ----------
    root : Path
        Extracted tree root holding ``<CODE>/`` (for the annotation CSVs).
    mirrors : str
        Root holding ``audio_16k/`` / ``audio_32k/``. May be a ``gs://`` URI:
        only the needed slice of each recording is read, so support clips can
        be rebuilt without re-staging the mirrors locally.
    out_root : Path
        Local destination for ``support_16k/`` / ``support_32k/``.
    manifest_dir : Path
        Directory holding ``fasd13_all.csv``; ``fasd13_support.csv`` is written here.
    workers : int
        Process-pool size.

    Raises
    ------
    SystemExit
        If any recording fails.
    """
    df_all = pd.read_csv(manifest_dir / "fasd13_all.csv", keep_default_na=False, na_values=[""])
    jobs = []
    for _, r in df_all.iterrows():
        code, stem = str(r["subdataset"]), Path(str(r["sound_name"])).stem
        anno_fp = root / code / f"{stem}.csv"
        df = _read_events(anno_fp)
        known = df[df["Q"] != "UNK"].sort_values("Endtime", kind="stable")
        starts = [float(v) for v in known["Starttime"].tolist()[:MAX_SHOTS]]
        if len(starts) < MAX_SHOTS:
            print(f"  !! {code}/{stem}: only {len(starts)} usable events", flush=True)
        shot_ends = _shot_ends_from_selection_tsv(str(r["selection_table"]))
        support_end = shot_ends[MAX_SHOTS - 1] if len(shot_ends) >= MAX_SHOTS else shot_ends[-1]
        jobs.append(
            (
                str(anno_fp),
                str(mirrors),
                str(out_root),
                f"{code}/{stem}",
                support_end,
                starts,
            )
        )

    print(f"cutting support clips for {len(jobs)} recordings ...", flush=True)
    rows, errors = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_support_one, j) for j in jobs]):
            got, status = fut.result()
            if status != "ok":
                errors += 1
                print(f"  {status}", flush=True)
            rows.extend(got)
    if errors:
        raise SystemExit(f"{errors} recordings failed")

    df_sup = pd.DataFrame(rows).sort_values(["subdataset", "sound_name", "shot_index"])
    out = manifest_dir / "fasd13_support.csv"
    df_sup.to_csv(out, index=False)
    empty = int((df_sup["event_times"] == "").sum())
    # Hard guard: no support clip may reach past the Nth event. Crossing that
    # line hands us audio DRASDIC is not allowed and prints query-region onsets
    # into the prompt, so it must fail the build rather than ship quietly.
    _ends = {
        (str(r["subdataset"]), str(r["sound_name"])): _shot_ends_from_selection_tsv(
            str(r["selection_table"])
        )
        for _, r in df_all.iterrows()
    }
    cutoff = {k: v[MAX_SHOTS - 1] for k, v in _ends.items() if len(v) >= MAX_SHOTS}
    over = [
        f"{r['subdataset']}/{r['sound_name']}#{r['shot_index']} "
        f"ends {r['window_end_sec']:.2f} > {cutoff[(r['subdataset'], r['sound_name'])]:.2f}"
        for _, r in df_sup.iterrows()
        if (r["subdataset"], r["sound_name"]) in cutoff
        and r["window_end_sec"] > cutoff[(r["subdataset"], r["sound_name"])] + 1e-6
    ]
    if over:
        raise SystemExit(
            f"{len(over)} support clip(s) extend past the {MAX_SHOTS}th event "
            f"(support-region leakage); first few: {over[:5]}"
        )
    print(
        f"  {len(df_sup)} support clips ({df_sup['clip_duration'].sum() / 60:.1f} min), "
        f"{int(df_sup['n_unk_in_clip'].sum())} clips-with-UNK events, "
        f"{empty} clips with no listed event -> {out.name}",
        flush=True,
    )
    if empty:
        raise SystemExit(f"{empty} support clips contain no event -- window placement is wrong")


def main() -> None:
    """Entry point -- see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)

    pd_ = sub.add_parser("download")
    pd_.add_argument("--stage-dir", type=Path, required=True)
    pd_.add_argument("--work", type=Path, required=True)

    pr = sub.add_parser("resample")
    pr.add_argument("--root", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)
    pr.add_argument("--codes", default=None, help="comma-separated subset, e.g. AS,JS")

    pm = sub.add_parser("manifests")
    pm.add_argument("--root", type=Path, required=True)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)
    pm.add_argument("--codes", default=None, help="comma-separated subset, e.g. AS,JS")

    ps = sub.add_parser("support")
    ps.add_argument("--root", type=Path, required=True)
    ps.add_argument("--mirrors", required=True, help="local dir or gs:// mirror root")
    ps.add_argument("--out-root", type=Path, required=True)
    ps.add_argument("--manifest-dir", type=Path, required=True)
    ps.add_argument("--workers", type=int, default=16)

    args = p.parse_args()
    if args.stage == "download":
        download(args.stage_dir, args.work)
    elif args.stage == "resample":
        resample(args.root, args.out_root, args.workers, _codes(args.codes))
    elif args.stage == "manifests":
        build_manifests(args.root, args.durations_csv, args.out_dir, _codes(args.codes))
    else:
        build_support(args.root, args.mirrors, args.out_root, args.manifest_dir, args.workers)


if __name__ == "__main__":
    main()

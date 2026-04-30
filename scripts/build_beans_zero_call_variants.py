#!/usr/bin/env python3
"""Build BEANS-Zero call-type linkage and binary call-variant manifests.

This script can either:
1. read the BEANS-Zero ``call-type`` split, parse candidate Xeno-canto
   recording identifiers from filenames and paths, and enrich rows with
   Xeno-canto API metadata when possible; or
2. consume a precomputed BEANS-to-XC mapping CSV directly.

In both cases it normalizes recovered behavior strings to the fixed call-type
vocabulary used elsewhere in the repo and writes balanced binary manifests for
``flight call``, ``alarm call``, and ``begging call``.

The script is intentionally robust to API failures. Missing API keys, blocked
responses, rate limiting, or unmatched rows are recorded in the linkage
manifest rather than causing a hard failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from esp_data.io import filesystem_from_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

XC_ID_PATTERN = re.compile(r"\bXC(\d+)\b", re.IGNORECASE)
XC_API_BASE_URL = "https://xeno-canto.org/api/3/recordings"
NOISY_BEHAVIOR_SUBSTRINGS = (
    "{",
    "background",
    "playback",
    " pb",
    "highway",
    "aircraft",
    "traffic",
    "people",
    "microphone",
    "http",
    "windmill",
    "tractor",
    "motor",
    "machine",
    "in training",
)
FIXED_VOCAB: list[tuple[str, re.Pattern[str]]] = [
    ("alarm call", re.compile(r"\balarm call\b", re.IGNORECASE)),
    ("flight call", re.compile(r"\bflight call\b", re.IGNORECASE)),
    ("begging call", re.compile(r"\bbegging call\b", re.IGNORECASE)),
    ("song", re.compile(r"\bsong\b", re.IGNORECASE)),
    ("call", re.compile(r"(^|,\s*)call(\s*,|$)", re.IGNORECASE)),
]
TARGET_LABELS = ("flight call", "alarm call", "begging call")
UNSEEN_TARGET_LABELS = ("alarm call", "begging call")
UNSEEN_MAPPING_CSV = "gs://esp-ml-datasets/beans-zero/v0.1.0/raw/unseen_xc_mapping.csv"


@dataclass(slots=True)
class CandidateRow:
    """BEANS-Zero row plus derived candidate XC linkage keys."""

    beans_zero_id: str
    source_dataset: str
    dataset_name: str
    task: str
    beans_zero_label: str
    file_name: str
    license: str
    metadata: str
    audio_path_original_sample_rate: str
    audio_path_16khz: str
    audio_path_32khz: str
    parsed_xc_id: str | None
    candidate_file_basename: str
    candidate_audio_url_tail: str


@dataclass(slots=True)
class LookupResult:
    """Normalized Xeno-canto API lookup result."""

    xc_lookup_status: str
    xc_lookup_method: str
    xc_api_blocked_or_rate_limited: bool
    xc_id: str | None
    xc_file_url: str | None
    xc_behavior_raw: str | None
    xc_species: str | None
    xc_recording_url: str | None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Build BEANS-Zero call-type XC linkage and binary manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "beans_zero_call_variants",
        help="Directory for manifests, diagnostics, and API cache.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Optional precomputed BEANS-to-XC mapping CSV for the call-type split.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Xeno-canto API key. Defaults to the XC_API_KEY environment variable.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of BEANS rows to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between uncached API requests.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached API responses and refetch all lookups.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for balanced negative sampling.",
    )
    return parser.parse_args()


def _basename(value: str) -> str:
    """Return a decoded basename from a path or URL fragment.

    Returns
    -------
    str
        Decoded basename.
    """

    if not value:
        return ""
    text = unquote(str(value))
    parsed = urlparse(text)
    candidate = parsed.path if parsed.scheme else text
    return Path(candidate).name


def _parse_xc_id(*values: str) -> str | None:
    """Extract the first ``XC<digits>`` identifier from candidate strings.

    Returns
    -------
    str | None
        Parsed numeric XC identifier, if present.
    """

    for value in values:
        if not value:
            continue
        match = XC_ID_PATTERN.search(unquote(value))
        if match:
            return match.group(1)
    return None


def load_candidate_rows(max_rows: int | None = None) -> list[CandidateRow]:
    """Load raw BEANS-Zero call-type rows and derive XC linkage keys.

    Returns
    -------
    list[CandidateRow]
        Candidate rows with parsed linkage keys.
    """

    from esp_data.datasets import BeansZero

    dataset = BeansZero(split="call-type", sample_rate=None, backend="polars", streaming=False)
    raw_rows = list(dataset._data)
    if max_rows is not None:
        raw_rows = raw_rows[:max_rows]

    candidates: list[CandidateRow] = []
    for row in raw_rows:
        file_name = str(row.get("file_name", ""))
        original_path = str(row.get("audio_path_original_sample_rate", ""))
        path_16khz = str(row.get("audio_path_16KHz", ""))
        path_32khz = str(row.get("audio_path_32KHz", ""))
        basename = (
            _basename(file_name)
            or _basename(original_path)
            or _basename(path_16khz)
            or _basename(path_32khz)
        )
        candidates.append(
            CandidateRow(
                beans_zero_id=str(row.get("id", "")),
                source_dataset=str(row.get("source_dataset", "")),
                dataset_name=str(row.get("dataset_name", "")),
                task=str(row.get("task", "")),
                beans_zero_label=str(row.get("output", "")),
                file_name=file_name,
                license=str(row.get("license", "")),
                metadata=str(row.get("metadata", "")),
                audio_path_original_sample_rate=original_path,
                audio_path_16khz=path_16khz,
                audio_path_32khz=path_32khz,
                parsed_xc_id=_parse_xc_id(file_name, original_path, path_16khz, path_32khz),
                candidate_file_basename=basename,
                candidate_audio_url_tail=basename,
            )
        )
    return candidates


def build_beans_zero_audio_lookup() -> dict[str, CandidateRow]:
    """Index BEANS-Zero call-type rows by file name for audio-path reuse."""

    return {row.file_name: row for row in load_candidate_rows()}


def build_beans_zero_unseen_audio_lookup() -> dict[str, CandidateRow]:
    """Index BEANS-Zero unseen-sci rows by BEANS id for audio-path reuse."""

    from esp_data.datasets import BeansZero

    rows_by_id: dict[str, CandidateRow] = {}
    for split in ("unseen-species-sci", "unseen-genus-sci", "unseen-family-sci"):
        dataset = BeansZero(split=split, sample_rate=None, backend="polars", streaming=False)
        for row in dataset._data:
            file_name = str(row.get("file_name", ""))
            original_path = str(row.get("audio_path_original_sample_rate", ""))
            path_16khz = str(row.get("audio_path_16KHz", ""))
            path_32khz = str(row.get("audio_path_32KHz", ""))
            rows_by_id[str(row.get("id", ""))] = CandidateRow(
                beans_zero_id=str(row.get("id", "")),
                source_dataset=str(row.get("source_dataset", "")),
                dataset_name=str(row.get("dataset_name", "")),
                task=str(row.get("task", "")),
                beans_zero_label=str(row.get("output", "")),
                file_name=file_name,
                license=str(row.get("license", "")),
                metadata=str(row.get("metadata", "")),
                audio_path_original_sample_rate=original_path,
                audio_path_16khz=path_16khz,
                audio_path_32khz=path_32khz,
                parsed_xc_id=_parse_xc_id(file_name, original_path, path_16khz, path_32khz),
                candidate_file_basename=(
                    _basename(file_name)
                    or _basename(original_path)
                    or _basename(path_16khz)
                    or _basename(path_32khz)
                ),
                candidate_audio_url_tail=(
                    _basename(file_name)
                    or _basename(original_path)
                    or _basename(path_16khz)
                    or _basename(path_32khz)
                ),
            )
    return rows_by_id


def load_rows_from_mapping_csv(
    mapping_csv: Path,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load normalized call-type rows from a precomputed mapping CSV.

    Returns
    -------
    list[dict[str, Any]]
        Call-type rows shaped like the API-enriched records used downstream.
    """

    rows: list[dict[str, Any]] = []
    beans_audio_lookup = build_beans_zero_audio_lookup()
    with mapping_csv.open() as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if raw_row.get("beans_zero_split") not in {"", "call-type", None}:
                continue

            file_name = str(raw_row.get("file_name", ""))
            beans_row = beans_audio_lookup.get(file_name)
            if beans_row is None:
                logger.warning("Skipping mapped row with no BEANS audio match: %s", file_name)
                continue
            gcs_path = str(raw_row.get("gcs_path", ""))
            xc_id = str(raw_row.get("xc_id", "")).strip() or None
            basename = _basename(file_name) or _basename(gcs_path)
            rows.append(
                {
                    "beans_zero_id": beans_row.beans_zero_id,
                    "source_dataset": beans_row.source_dataset,
                    "dataset_name": str(raw_row.get("dataset_name", beans_row.dataset_name)),
                    "task": str(raw_row.get("task", beans_row.task or "classification")),
                    "beans_zero_label": str(raw_row.get("output", beans_row.beans_zero_label)),
                    "file_name": file_name,
                    "license": str(raw_row.get("bz_license", beans_row.license)),
                    "metadata": beans_row.metadata,
                    "audio_path_original_sample_rate": beans_row.audio_path_original_sample_rate,
                    "audio_path_16khz": beans_row.audio_path_16khz,
                    "audio_path_32khz": beans_row.audio_path_32khz,
                    "parsed_xc_id": xc_id or beans_row.parsed_xc_id or _parse_xc_id(file_name, gcs_path),
                    "candidate_file_basename": basename,
                    "candidate_audio_url_tail": basename,
                    "xc_lookup_status": "success" if gcs_path else "mapping_missing_gcs_path",
                    "xc_lookup_method": "mapping_csv",
                    "xc_api_blocked_or_rate_limited": False,
                    "xc_id": xc_id,
                    "xc_file_url": str(raw_row.get("media_url", "")) or None,
                    "xc_behavior_raw": str(raw_row.get("behavior", "")) or None,
                    "xc_species": (
                        str(raw_row.get("canonical_name", ""))
                        or str(raw_row.get("scientificName", ""))
                        or None
                    ),
                    "xc_recording_url": str(raw_row.get("url", "")) or None,
                    "mapping_source": str(raw_row.get("_source", "")),
                    "mapping_audio_found": str(raw_row.get("xc_audio_found", "")),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def load_rows_from_unseen_mapping_csv(
    mapping_csv: str = UNSEEN_MAPPING_CSV,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load unseen rows with mapped behavior metadata and BEANS audio paths."""

    rows: list[dict[str, Any]] = []
    beans_audio_lookup = build_beans_zero_unseen_audio_lookup()
    with filesystem_from_path(mapping_csv).open(str(mapping_csv), "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            beans_zero_id = str(raw_row.get("bz_id", ""))
            beans_row = beans_audio_lookup.get(beans_zero_id)
            if beans_row is None:
                logger.warning("Skipping unseen mapped row with no BEANS audio match: %s", beans_zero_id)
                continue

            gcs_path = str(raw_row.get("gcs_path", ""))
            xc_id = str(raw_row.get("xc_id", "")).strip() or None
            raw_behavior = str(raw_row.get("behavior", "")).strip() or None
            rows.append(
                {
                    "beans_zero_id": beans_zero_id,
                    "source_dataset": beans_row.source_dataset,
                    "dataset_name": "beans_zero_unseen",
                    "task": "classification",
                    "beans_zero_label": beans_row.beans_zero_label,
                    "file_name": beans_row.file_name,
                    "license": str(raw_row.get("bz_license", beans_row.license)),
                    "metadata": beans_row.metadata,
                    "audio_path_original_sample_rate": beans_row.audio_path_original_sample_rate,
                    "audio_path_16khz": beans_row.audio_path_16khz,
                    "audio_path_32khz": beans_row.audio_path_32khz,
                    "parsed_xc_id": xc_id or beans_row.parsed_xc_id or _parse_xc_id(beans_row.file_name, gcs_path),
                    "candidate_file_basename": beans_row.candidate_file_basename,
                    "candidate_audio_url_tail": beans_row.candidate_audio_url_tail,
                    "xc_lookup_status": "success",
                    "xc_lookup_method": "unseen_mapping_csv",
                    "xc_api_blocked_or_rate_limited": False,
                    "xc_id": xc_id,
                    "xc_file_url": str(raw_row.get("media_url", "")) or None,
                    "xc_behavior_raw": raw_behavior,
                    "xc_species": (
                        str(raw_row.get("canonical_name", ""))
                        or str(raw_row.get("scientificName", ""))
                        or None
                    ),
                    "xc_recording_url": str(raw_row.get("url", "")) or None,
                    "mapping_source": str(raw_row.get("_source", "")),
                    "mapping_audio_found": str(raw_row.get("xc_audio_found", "")),
                    "unseen_level": str(raw_row.get("unseen_level", "")),
                    "label_sci": str(raw_row.get("label_sci", "")),
                    "label_cmn": str(raw_row.get("label_cmn", "")),
                    "label_tax": str(raw_row.get("label_tax", "")),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached XC API lookup results from JSONL.

    Returns
    -------
    dict[str, dict[str, Any]]
        Cached lookup results keyed by lookup method and value.
    """

    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[str(record["cache_key"])] = record["result"]
    logger.info("Loaded %d cached XC lookups from %s", len(cache), cache_path)
    return cache


def write_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """Persist the XC API lookup cache as JSONL."""

    with cache_path.open("w") as handle:
        for cache_key, result in cache.items():
            payload = {"cache_key": cache_key, "result": result}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def xc_lookup_by_id(
    xc_id: str,
    api_key: str,
    sleep_seconds: float,
) -> LookupResult:
    """Fetch a single XC recording by numeric XC ID.

    Returns
    -------
    LookupResult
        Normalized lookup result for the requested XC ID.
    """

    params = f"query={quote_plus(f'nr:{xc_id}')}&key={quote_plus(api_key)}"
    url = f"{XC_API_BASE_URL}?{params}"
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        blocked = exc.code in {401, 403, 429, 503}
        body = exc.read(300).decode("utf-8", errors="replace")
        status = "api_blocked" if blocked else f"http_error_{exc.code}"
        logger.warning("XC API lookup failed for XC%s: %s %s", xc_id, exc.code, body)
        return LookupResult(
            xc_lookup_status=status,
            xc_lookup_method="xc_id",
            xc_api_blocked_or_rate_limited=blocked,
            xc_id=xc_id,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )
    except URLError as exc:
        logger.warning("XC API lookup URL error for XC%s: %s", xc_id, exc)
        return LookupResult(
            xc_lookup_status="url_error",
            xc_lookup_method="xc_id",
            xc_api_blocked_or_rate_limited=False,
            xc_id=xc_id,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )

    recordings = payload.get("recordings") or []
    if not recordings:
        return LookupResult(
            xc_lookup_status="no_match",
            xc_lookup_method="xc_id",
            xc_api_blocked_or_rate_limited=False,
            xc_id=xc_id,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    record = recordings[0]
    scientific_name = " ".join(
        part for part in (record.get("gen"), record.get("sp")) if part
    ).strip() or None
    return LookupResult(
        xc_lookup_status="success",
        xc_lookup_method="xc_id",
        xc_api_blocked_or_rate_limited=False,
        xc_id=str(record.get("id") or xc_id),
        xc_file_url=record.get("file"),
        xc_behavior_raw=record.get("type"),
        xc_species=scientific_name,
        xc_recording_url=record.get("url"),
    )


def build_filename_query(file_basename: str) -> str | None:
    """Build a conservative XC search query from a filename stem.

    Returns
    -------
    str | None
        Search query text, or ``None`` when no reasonable query can be built.
    """

    stem = unquote(Path(file_basename).stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if len(stem) < 6:
        return None
    return stem[:120]


def xc_lookup_by_file_basename(
    file_basename: str,
    api_key: str,
    sleep_seconds: float,
) -> LookupResult:
    """Attempt an exact file-basename match through XC search results.

    Returns
    -------
    LookupResult
        Normalized lookup result for the filename-based search.
    """

    query_text = build_filename_query(file_basename)
    if query_text is None:
        return LookupResult(
            xc_lookup_status="no_lookup_key",
            xc_lookup_method="file_basename",
            xc_api_blocked_or_rate_limited=False,
            xc_id=None,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )

    params = f"query={quote_plus(query_text)}&key={quote_plus(api_key)}"
    url = f"{XC_API_BASE_URL}?{params}"
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        blocked = exc.code in {401, 403, 429, 503}
        body = exc.read(300).decode("utf-8", errors="replace")
        status = "api_blocked" if blocked else f"http_error_{exc.code}"
        logger.warning("XC filename lookup failed for %s: %s %s", file_basename, exc.code, body)
        return LookupResult(
            xc_lookup_status=status,
            xc_lookup_method="file_basename",
            xc_api_blocked_or_rate_limited=blocked,
            xc_id=None,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )
    except URLError as exc:
        logger.warning("XC filename lookup URL error for %s: %s", file_basename, exc)
        return LookupResult(
            xc_lookup_status="url_error",
            xc_lookup_method="file_basename",
            xc_api_blocked_or_rate_limited=False,
            xc_id=None,
            xc_file_url=None,
            xc_behavior_raw=None,
            xc_species=None,
            xc_recording_url=None,
        )

    recordings = payload.get("recordings") or []
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    normalized_basename = _basename(file_basename)
    for record in recordings:
        record_file_basename = _basename(str(record.get("file", "")))
        if record_file_basename != normalized_basename:
            continue
        scientific_name = " ".join(
            part for part in (record.get("gen"), record.get("sp")) if part
        ).strip() or None
        return LookupResult(
            xc_lookup_status="success",
            xc_lookup_method="file_basename",
            xc_api_blocked_or_rate_limited=False,
            xc_id=str(record.get("id")) if record.get("id") else None,
            xc_file_url=record.get("file"),
            xc_behavior_raw=record.get("type"),
            xc_species=scientific_name,
            xc_recording_url=record.get("url"),
        )

    return LookupResult(
        xc_lookup_status="no_match",
        xc_lookup_method="file_basename",
        xc_api_blocked_or_rate_limited=False,
        xc_id=None,
        xc_file_url=None,
        xc_behavior_raw=None,
        xc_species=None,
        xc_recording_url=None,
    )


def enrich_rows(
    rows: list[CandidateRow],
    *,
    api_key: str | None,
    cache_path: Path,
    sleep_seconds: float,
    force_refresh: bool,
) -> list[dict[str, Any]]:
    """Enrich candidate rows with XC API metadata and cache results.

    Returns
    -------
    list[dict[str, Any]]
        Input rows augmented with XC lookup fields.
    """

    cache = load_cache(cache_path)
    enriched_rows: list[dict[str, Any]] = []
    blocked_seen = False

    for index, row in enumerate(rows, start=1):
        base_record = asdict(row)
        if blocked_seen:
            lookup = LookupResult(
                xc_lookup_status="api_blocked_after_first_failure",
                xc_lookup_method="skipped",
                xc_api_blocked_or_rate_limited=True,
                xc_id=row.parsed_xc_id,
                xc_file_url=None,
                xc_behavior_raw=None,
                xc_species=None,
                xc_recording_url=None,
            )
        elif not row.parsed_xc_id:
            if not api_key:
                lookup = LookupResult(
                    xc_lookup_status="missing_api_key",
                    xc_lookup_method="file_basename",
                    xc_api_blocked_or_rate_limited=False,
                    xc_id=None,
                    xc_file_url=None,
                    xc_behavior_raw=None,
                    xc_species=None,
                    xc_recording_url=None,
                )
            else:
                cache_key = f"file_basename:{row.candidate_file_basename}"
                cached = None if force_refresh else cache.get(cache_key)
                if cached is not None:
                    lookup = LookupResult(**cached)
                else:
                    lookup = xc_lookup_by_file_basename(
                        row.candidate_file_basename,
                        api_key=api_key,
                        sleep_seconds=sleep_seconds,
                    )
                    cache[cache_key] = asdict(lookup)
                    if lookup.xc_api_blocked_or_rate_limited:
                        blocked_seen = True
        elif not api_key:
            lookup = LookupResult(
                xc_lookup_status="missing_api_key",
                xc_lookup_method="xc_id",
                xc_api_blocked_or_rate_limited=False,
                xc_id=row.parsed_xc_id,
                xc_file_url=None,
                xc_behavior_raw=None,
                xc_species=None,
                xc_recording_url=None,
            )
        else:
            cache_key = f"xc_id:{row.parsed_xc_id}"
            cached = None if force_refresh else cache.get(cache_key)
            if cached is not None:
                lookup = LookupResult(**cached)
            else:
                lookup = xc_lookup_by_id(
                    row.parsed_xc_id,
                    api_key=api_key,
                    sleep_seconds=sleep_seconds,
                )
                cache[cache_key] = asdict(lookup)
                if lookup.xc_api_blocked_or_rate_limited:
                    blocked_seen = True

        enriched_rows.append(base_record | asdict(lookup))
        if index % 100 == 0 or index == len(rows):
            logger.info("Processed %d/%d BEANS call-type rows", index, len(rows))

    write_cache(cache_path, cache)
    return enriched_rows


def is_noisy_behavior(raw_behavior: str | None) -> bool:
    """Return ``True`` when a raw XC behavior string is too noisy to trust.

    Returns
    -------
    bool
        Whether the behavior string should be excluded from clean supervision.
    """

    if raw_behavior is None:
        return True
    text = raw_behavior.strip().lower()
    if not text or text in {"?", "uncertain"}:
        return True
    return any(fragment in text for fragment in NOISY_BEHAVIOR_SUBSTRINGS)


def map_behavior_to_fixed_vocab(raw_behavior: str | None) -> list[str]:
    """Normalize a raw behavior string to the fixed call-type vocabulary.

    Returns
    -------
    list[str]
        Normalized call-type labels in fixed-vocabulary order.
    """

    if raw_behavior is None:
        return []
    return [label for label, pattern in FIXED_VOCAB if pattern.search(raw_behavior)]


def attach_normalized_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add normalized XC behavior columns used for manifest generation.

    Returns
    -------
    list[dict[str, Any]]
        Rows augmented with normalized behavior fields.
    """

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_behavior = row.get("xc_behavior_raw")
        labels = map_behavior_to_fixed_vocab(raw_behavior)
        normalized_rows.append(
            row | {
                "xc_behavior_noisy": is_noisy_behavior(raw_behavior),
                "behavior_fixed_vocab": ", ".join(labels) if labels else "",
                "behavior_fixed_vocab_count": len(labels),
            }
        )
    return normalized_rows


def build_balanced_binary_rows(
    rows: list[dict[str, Any]],
    target_label: str,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build a balanced binary manifest for one target label.

    Returns
    -------
    list[dict[str, Any]]
        Balanced rows for one target binary task.
    """

    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []

    for row in rows:
        if row["xc_lookup_status"] != "success":
            continue
        if row["xc_behavior_noisy"]:
            continue

        labels = [part.strip() for part in row["behavior_fixed_vocab"].split(",") if part.strip()]
        if len(labels) != 1:
            continue

        if labels[0] == target_label:
            positive_rows.append(row)
        elif target_label not in labels:
            negative_rows.append(row)

    sample_size = min(len(positive_rows), len(negative_rows))
    if sample_size == 0:
        return []

    positive_sample = (
        rng.sample(positive_rows, sample_size)
        if len(positive_rows) > sample_size
        else positive_rows
    )
    negative_sample = (
        rng.sample(negative_rows, sample_size)
        if len(negative_rows) > sample_size
        else negative_rows
    )

    manifest_rows: list[dict[str, Any]] = []
    output_column = f"{target_label.replace(' ', '_')}_present"
    for row in positive_sample:
        manifest_rows.append(
            row | {"target_call_type": target_label, output_column: "Yes"}
        )
    for row in negative_sample:
        manifest_rows.append(
            row | {"target_call_type": target_label, output_column: "No"}
        )
    rng.shuffle(manifest_rows)
    return manifest_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to JSONL."""

    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""

    if not rows:
        path.write_text("")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    random_seed: int,
    unseen_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write linkage manifest, summary diagnostics, and binary manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "linkage_manifest.jsonl", rows)
    write_csv(output_dir / "linkage_manifest.csv", rows)

    rng = random.Random(random_seed)
    binary_manifests: dict[str, list[dict[str, Any]]] = {}
    for target_label in TARGET_LABELS:
        rows_for_target = build_balanced_binary_rows(rows, target_label, rng=rng)
        binary_manifests[target_label] = rows_for_target
        slug = target_label.replace(" ", "_")
        write_jsonl(output_dir / f"{slug}_binary.jsonl", rows_for_target)
        write_csv(output_dir / f"{slug}_binary.csv", rows_for_target)

    unseen_binary_manifests: dict[str, list[dict[str, Any]]] = {}
    if unseen_rows is not None:
        write_jsonl(output_dir / "unseen_linkage_manifest.jsonl", unseen_rows)
        write_csv(output_dir / "unseen_linkage_manifest.csv", unseen_rows)
        for target_label in UNSEEN_TARGET_LABELS:
            rows_for_target = build_balanced_binary_rows(unseen_rows, target_label, rng=rng)
            split_name = f"{target_label.replace(' ', '_')}_unseen"
            unseen_binary_manifests[split_name] = rows_for_target
            write_jsonl(output_dir / f"{split_name}_binary.jsonl", rows_for_target)
            write_csv(output_dir / f"{split_name}_binary.csv", rows_for_target)

    status_counts = Counter(str(row["xc_lookup_status"]) for row in rows)
    method_counts = Counter(str(row["xc_lookup_method"]) for row in rows)
    fixed_vocab_counts = Counter(
        str(row["behavior_fixed_vocab"])
        for row in rows
        if row["behavior_fixed_vocab"]
    )
    summary = {
        "total_rows": len(rows),
        "rows_with_parsed_xc_id": sum(1 for row in rows if row["parsed_xc_id"]),
        "rows_without_parsed_xc_id": sum(1 for row in rows if not row["parsed_xc_id"]),
        "xc_lookup_status_counts": dict(status_counts),
        "xc_lookup_method_counts": dict(method_counts),
        "normalized_behavior_counts": dict(fixed_vocab_counts.most_common()),
        "binary_manifest_sizes": {
            target_label: len(manifest_rows)
            for target_label, manifest_rows in binary_manifests.items()
        },
        "binary_manifest_positive_counts": {
            target_label: sum(
                1
                for row in manifest_rows
                if row[f"{target_label.replace(' ', '_')}_present"] == "Yes"
            )
            for target_label, manifest_rows in binary_manifests.items()
        },
        "unseen_binary_manifest_sizes": {
            split_name: len(manifest_rows)
            for split_name, manifest_rows in unseen_binary_manifests.items()
        },
        "unseen_binary_manifest_positive_counts": {
            split_name: sum(
                1
                for row in manifest_rows
                if row[f"{split_name.removesuffix('_unseen')}_present"] == "Yes"
            )
            for split_name, manifest_rows in unseen_binary_manifests.items()
        },
    }
    with (output_dir / "linkage_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main() -> None:
    """Run the BEANS-Zero call-variant builder."""

    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mapping_csv is not None:
        logger.info("Loading precomputed BEANS-XC mapping rows from %s", args.mapping_csv)
        enriched_rows = load_rows_from_mapping_csv(
            args.mapping_csv,
            max_rows=args.max_rows,
        )
        logger.info(
            "Loaded %d mapped rows (%d successful XC mappings)",
            len(enriched_rows),
            sum(1 for row in enriched_rows if row["xc_lookup_status"] == "success"),
        )
    else:
        api_key = args.api_key or os.environ.get("XC_API_KEY")
        cache_path = output_dir / "xc_api_cache.jsonl"

        logger.info("Loading BEANS-Zero call-type rows")
        candidate_rows = load_candidate_rows(max_rows=args.max_rows)
        logger.info(
            "Loaded %d rows (%d with parsed XC ids)",
            len(candidate_rows),
            sum(1 for row in candidate_rows if row.parsed_xc_id),
        )

        logger.info("Enriching rows with Xeno-canto API metadata")
        enriched_rows = enrich_rows(
            candidate_rows,
            api_key=api_key,
            cache_path=cache_path,
            sleep_seconds=args.sleep_seconds,
            force_refresh=args.force_refresh,
        )

    logger.info("Normalizing recovered XC behavior labels")
    normalized_rows = attach_normalized_behavior(enriched_rows)
    logger.info("Loading unseen-species/genus/family mapping rows")
    unseen_normalized_rows = attach_normalized_behavior(
        load_rows_from_unseen_mapping_csv(max_rows=args.max_rows)
    )

    logger.info("Writing manifests and diagnostics to %s", output_dir)
    write_outputs(
        normalized_rows,
        output_dir=output_dir,
        random_seed=args.random_seed,
        unseen_rows=unseen_normalized_rows,
    )
    logger.info("Done")


if __name__ == "__main__":
    main()

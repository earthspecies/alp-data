"""Parse XWAV foreign-metadata embedded in FLAC files (Triton / Wiggins SIO).

PIFSC PIPAN FLAC files were compressed with ``flac --keep-foreign-metadata``,
which preserves the original WAV's RIFF chunks inside FLAC ``APPLICATION``
metadata blocks (one chunk per block, application ID ``riff``). Among those
chunks is a custom ``harp`` chunk listing per-subchunk metadata: UTC start
time, byte offset in the original WAV, byte length, sample rate, and gain
stage.

This module extracts the subchunk table from a (possibly remote) FLAC by
reading only the first ~256 KB of bytes via cloudpathlib, decoding the FLAC
metadata blocks, locating the ``harp`` RIFF chunk inside an APPLICATION
``riff`` block, and parsing its records.

Reference: Triton ``MTRead.m`` / ``ioReadXWAVHeader.m``; XWAV format spec at
https://github.com/MarineBioAcousticsRC/Triton/wiki/XWAV-File-Format.
"""

from __future__ import annotations

import datetime
import struct
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

# Bytes per audio sample in the source WAV. PIPAN 10kHz data is 16-bit PCM.
_BYTES_PER_SAMPLE = 2

# Static header inside a harp chunk (before the per-subchunk records).
# Total: 56 bytes (per Triton MTRead.m).
_HARP_STATIC_HEADER_FMT = "<B10s4s4s8sB8sHiihxxxxxxxx"
_HARP_STATIC_HEADER_LEN = struct.calcsize(_HARP_STATIC_HEADER_FMT)
assert _HARP_STATIC_HEADER_LEN == 56, _HARP_STATIC_HEADER_LEN

# Per-subchunk record format. 32 bytes each.
# Year(u8) Month(u8) Day(u8) Hour(u8) Min(u8) Sec(u8) Ticks(u16) ByteLoc(u32)
# ByteLength(u32) WriteLength(u32) SampleRate(u32) GainStage(u8) Reserved(7)
_SUBCHUNK_REC_FMT = "<BBBBBBHIIIIB7s"
_SUBCHUNK_REC_LEN = struct.calcsize(_SUBCHUNK_REC_FMT)
assert _SUBCHUNK_REC_LEN == 32, _SUBCHUNK_REC_LEN


@dataclass
class XWAVSubchunk:
    """One subchunk's metadata from the harp custom chunk."""

    index: int
    start_utc: datetime.datetime
    byte_loc: int  # offset in ORIGINAL WAV bytes
    byte_length: int  # length in bytes of the audio data in this subchunk
    sample_rate: int  # Hz
    gain_stage: int

    @property
    def duration_s(self) -> float:
        """Audio duration of this subchunk in seconds (assuming 16-bit PCM)."""
        return self.byte_length / (_BYTES_PER_SAMPLE * self.sample_rate)


@dataclass
class XWAVIndex:
    """The list of subchunks for one FLAC file, plus cumulative offsets.

    ``cum_offset_s[i]`` is the time (in seconds) of the start of subchunk
    ``i`` in the decoded continuous audio stream. ``cum_offset_s[-1]``
    equals the total decoded duration.
    """

    subchunks: list[XWAVSubchunk]
    cum_offset_s: list[float]  # length = len(subchunks) + 1

    @classmethod
    def from_subchunks(cls, subchunks: list[XWAVSubchunk]) -> XWAVIndex:
        """Build an XWAVIndex from a subchunks list, computing cumulative offsets."""
        cum = [0.0]
        for sc in subchunks:
            cum.append(cum[-1] + sc.duration_s)
        return cls(subchunks=subchunks, cum_offset_s=cum)

    def offset_in_file(self, subchunk_index: int, offset_in_subchunk_s: float) -> float:
        """Return absolute decoded-audio offset for ``offset_in_subchunk_s``."""
        if subchunk_index < 0 or subchunk_index >= len(self.subchunks):
            raise IndexError(
                f"subchunk_index {subchunk_index} out of range [0, {len(self.subchunks)})"
            )
        return self.cum_offset_s[subchunk_index] + offset_in_subchunk_s


def _iter_flac_metadata_blocks(buf: BytesIO):
    """Yield ``(block_type, payload_bytes, is_last)`` for each metadata block."""
    magic = buf.read(4)
    if magic != b"fLaC":
        raise ValueError(f"Not a FLAC file (magic {magic!r})")
    while True:
        header = buf.read(4)
        if len(header) < 4:
            return
        first = header[0]
        is_last = bool(first & 0x80)
        block_type = first & 0x7F
        block_len = int.from_bytes(header[1:4], "big")
        payload = buf.read(block_len)
        if len(payload) < block_len:
            return  # truncated read; caller can supply more bytes if needed
        yield block_type, payload, is_last
        if is_last:
            return


def parse_xwav_from_bytes(head: bytes) -> XWAVIndex:
    """Parse the harp chunk in a FLAC's foreign metadata; return the index.

    Parameters
    ----------
    head : bytes
        The first N bytes of a FLAC file. Must include the fLaC magic and
        all metadata blocks (typically ~64 KB suffices, occasionally more
        for files with many subchunks).

    Returns
    -------
    XWAVIndex
        Parsed subchunk table plus cumulative-offset cache.

    Raises
    ------
    ValueError
        If the file is not a FLAC, or no APPLICATION ``riff`` block contains
        a ``harp`` chunk.
    """
    buf = BytesIO(head)
    for block_type, payload, _ in _iter_flac_metadata_blocks(buf):
        if block_type != 2:  # APPLICATION block
            continue
        if len(payload) < 4:
            continue
        app_id = payload[:4]
        if app_id != b"riff":
            continue
        # The remaining payload is one raw RIFF chunk:
        #   ChunkID (4) + ChunkSize (4 LE) + chunk_data (ChunkSize bytes)
        riff = payload[4:]
        if len(riff) < 8:
            continue
        chunk_id = riff[:4]
        chunk_size = struct.unpack("<I", riff[4:8])[0]
        chunk_data = riff[8 : 8 + chunk_size]
        if chunk_id != b"harp":
            continue
        return _parse_harp(chunk_data)
    raise ValueError("No 'harp' APPLICATION/riff block found in FLAC metadata")


def _parse_harp(data: bytes) -> XWAVIndex:
    """Parse the body of a harp chunk into an XWAVIndex."""
    if len(data) < _HARP_STATIC_HEADER_LEN:
        raise ValueError(f"harp chunk too short ({len(data)} < {_HARP_STATIC_HEADER_LEN})")
    fields = struct.unpack(_HARP_STATIC_HEADER_FMT, data[:_HARP_STATIC_HEADER_LEN])
    (
        _wav_version,
        _firmware,
        _instrument,
        _site,
        _experiment,
        _disk_seq,
        _disk_serial,
        num_raw_files,
        _longitude_x1e6,
        _latitude_x1e6,
        _depth,
    ) = fields
    offset = _HARP_STATIC_HEADER_LEN
    expected = offset + num_raw_files * _SUBCHUNK_REC_LEN
    if len(data) < expected:
        raise ValueError(
            f"harp chunk too short for {num_raw_files} subchunks: {len(data)} < {expected}"
        )
    subchunks: list[XWAVSubchunk] = []
    for i in range(num_raw_files):
        record = data[offset : offset + _SUBCHUNK_REC_LEN]
        offset += _SUBCHUNK_REC_LEN
        (
            yy,
            mm,
            dd,
            hh,
            mn,
            ss,
            ticks,
            byte_loc,
            byte_length,
            _write_length,
            sample_rate,
            gain_stage,
            _reserved,
        ) = struct.unpack(_SUBCHUNK_REC_FMT, record)
        try:
            ts = datetime.datetime(
                year=2000 + yy if yy < 70 else 1900 + yy,
                month=mm,
                day=dd,
                hour=hh,
                minute=mn,
                second=ss,
                microsecond=ticks * 1000,
                tzinfo=datetime.timezone.utc,
            )
        except ValueError as e:
            raise ValueError(
                f"Bad timestamp in subchunk {i}: y={yy} m={mm} d={dd} "
                f"H={hh} M={mn} S={ss} ticks={ticks}: {e}"
            ) from e
        subchunks.append(
            XWAVSubchunk(
                index=i,
                start_utc=ts,
                byte_loc=byte_loc,
                byte_length=byte_length,
                sample_rate=sample_rate,
                gain_stage=gain_stage,
            )
        )
    return XWAVIndex.from_subchunks(subchunks)


def parse_xwav_from_gcs(flac_uri: str, head_bytes: int = 262144) -> XWAVIndex:
    """Stream the first ``head_bytes`` of ``flac_uri`` and parse the harp index.

    Parameters
    ----------
    flac_uri : str
        ``gs://...`` path (or local) to the FLAC file.
    head_bytes : int
        Number of bytes to fetch from the start of the file. The harp chunk
        in PIPAN 10kHz files is typically <128 KB. If parse_xwav_from_bytes
        raises, retry with a larger value.

    Returns
    -------
    XWAVIndex
        Parsed subchunk table.

    Raises
    ------
    ValueError
        If parsing fails — likely the harp chunk extends beyond the slice.
    """
    if flac_uri.startswith("gs://"):
        proc = subprocess.run(
            ["gsutil", "cat", "-r", f"0-{head_bytes - 1}", flac_uri],
            check=True,
            capture_output=True,
            timeout=60,
        )
        head = proc.stdout
    else:
        head = Path(flac_uri).read_bytes()[:head_bytes]
    return parse_xwav_from_bytes(head)


if __name__ == "__main__":
    import sys

    uri = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "gs://esp-data-ingestion/superwhale/v0.1.0/raw/pifsc/audio/pipan_10/"
            "crosssm/pipan_crosssm_01/audio/audio/"
            "Cross_A_01_050606_123845.d20.x.flac"
        )
    )
    idx = parse_xwav_from_gcs(uri)
    print(f"{len(idx.subchunks)} subchunks; total decoded duration {idx.cum_offset_s[-1]:.2f}s")
    for sc in idx.subchunks[:5]:
        print(
            f"  [{sc.index:3d}] start={sc.start_utc.isoformat()} "
            f"sr={sc.sample_rate} bytes={sc.byte_length} "
            f"dur={sc.duration_s:.2f}s"
        )
    if len(idx.subchunks) > 5:
        last = idx.subchunks[-1]
        print(f"  ... [{last.index}] start={last.start_utc.isoformat()} dur={last.duration_s:.2f}s")

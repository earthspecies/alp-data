# Combine PR 293 + PR 257 into a unified `read_audio`

## Goal
Integrate PR 257 (ffmpeg remote-segment streaming from GCS) into the audio read
path, on top of PR 293 (librosa fallback + dropped extension gating + quieter
warnings), and collapse `read_audio_by_time` into a single public `read_audio`.

## Approach
Hand-port onto a fresh branch off `main` (not a git merge): both PRs are on the
old `esp_data` package; `main` is now `alp_data`. Branch:
`combine-293-257-ffmpeg-readaudio`.

Main change: `alp_data/io/read_utils.py`. Plus `pyproject.toml`, `.github/workflows/ci.yml`,
`alp_data/io/__init__.py`, tests.

## Architecture — `read_audio` as a thin dispatcher over private helpers
- `read_audio(...)` public entry, resolves path, dispatches:
  - full-file read (no `start_time`) -> `_read_audio_from_file` (sf -> librosa -> tmpfile)
  - time-range read (`start_time` set):
    - `gs://` -> try `_read_audio_ffmpeg`; on failure warn (once per cause) + fall back to `_read_audio_by_time`
    - local/s3/r2 -> `_read_audio_by_time`
- Private helpers: `_read_audio_ffmpeg`, `_read_audio_by_time` (was public
  `read_audio_by_time`, now private), `_read_audio_from_file`,
  `_read_audio_with_librosa`, `_read_audio_from_tmpfile`.

## Decisions locked
1. Integration, not mechanical merge.
2. Collapse `read_audio_by_time` -> private `_read_audio_by_time` (not in `__all__`,
   no public API break). Rewrite the 3 tests that import it to call
   `read_audio(..., start_time=, end_time=)`.
3. ffmpeg trigger = GCS + time-range only. Full-file GCS stays on download.
4. Scope A: GCS-only, bearer token. No S3/R2 ffmpeg, no presigned URLs, no
   signBlob IAM dependency. (Presigned-URL generalization = future option.)
5. Token caching: module-level cached credentials, refresh only when
   `not credentials.valid` (fixes PR 257's unconditional per-call refresh).
6. `anonymous: bool = False` param: when True skip token, pass bare URL to ffmpeg.
   Unblocks public-dataset users without ADC. No auto-detection.
7. Array convention: ffmpeg branch normalized to soundfile convention
   `(frames,)` mono / `(frames, channels)` multi. Drop PR 257's `(channels, frames)`.
   `_read_audio_ffmpeg` is private; no public `read_audio_ffmpeg`.
8. Failure handling: missing ffmpeg/ffprobe, missing ADC, probe error -> warn +
   fall back to download. Never raise from the optimization.
9. Take PR 293 fully: librosa fallback in both paths; drop `_AUDIO_FORMATS`
   gating from `read_audio`, `_read_audio_by_time`, and `get_audio_info`; silence
   routine sf->librosa->tmpfile fallback warnings.
10. Warnings: only ffmpeg->download fallback warns, once per distinct cause via
    `lru_cache` helper keyed on short cause string; specific exception -> `logger.debug`.

## Supporting changes
- deps: add `google-auth` explicitly to `pyproject.toml`. `librosa` already present.
- CI: port PR 257's `apt-get install ffmpeg` step. CI has ambient GCS creds.
- `__init__.py`: no new public export. Don't carry PR 257's duplicate `paths` import.
- tests: rewrite `read_audio_by_time` tests -> `read_audio` w/ time range; add
  ffmpeg segment test (real `gs://esp-ci-cd-tests` file); add fallback test
  (ffmpeg unavailable -> warning + download path); keep a `get_gcs_token` test.

## Verify during implementation
- CI ambient creds satisfy `google.auth.default()`.
- `read_audio` retains `frames`/`start` frame-based params unchanged.
- Docstrings updated per AGENTS.md (NumPy style; remove the now-gone `ValueError`
  Raises sections).

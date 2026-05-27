#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage the PteroSet tropical-bird PAM dataset (Zenodo record 19137071) into
# gs://esp-data-ingestion/pteroset/v0.1.0/.
#
# Downloads audios.zip (~86 GB, 192 kHz WAV), labels.zip (Raven Pro selection
# tables), and the metadata/JSON files, extracts them, and uploads to GCS:
#
#   recordings/*.wav    (192 kHz originals, flat)
#   annotations/*.txt   (Raven Pro selection tables)
#   metadata/           (metadata.csv, species.csv, annotations_*.json)
#
# audios.zip is a single huge file and Zenodo throttles ~1 MB/s per
# connection, so it is fetched with a parallel byte-range downloader.
#
# Heavy on network/IO (not memory); run on the dev VM (needs external
# internet). Set WORK to a disk with ~180 GB free (zip + extracted) — on the
# dev VM that is the NFS share (/mnt/home/...), NOT local $HOME.
#
# USAGE:
#   WORK=/mnt/home/pteroset_staging \
#     nohup bash scripts/data_preprocessing_scripts/pteroset_stage.sh \
#       > /mnt/home/pteroset_staging/stage.log 2>&1 &
# ---------------------------------------------------------------------------
set -euo pipefail

ZENODO="https://zenodo.org/records/19137071/files"
GCS_ROOT="gs://esp-data-ingestion/pteroset/v0.1.0"
WORK="${WORK:-/mnt/home/pteroset_staging}"
NCONN="${NCONN:-24}"   # parallel connections for the big zip

ZIPS="$WORK/zips"; META="$WORK/meta"; AUD="$WORK/extract_audio"; LAB="$WORK/extract_lab"
mkdir -p "$ZIPS" "$META" "$AUD" "$LAB"

META_FILES=(species.csv metadata.csv annotations_species.json annotations_identification.json)

remote_size() { curl -sIL "$1" 2>/dev/null \
    | awk 'BEGIN{IGNORECASE=1}/^content-length:/{cl=$2} END{gsub(/\r/,"",cl); print cl}'; }

dl() {  # dl <filename> <dest_dir>   (skip if already complete; else resume)
    local fn="$1" dest="$2"
    local url="$ZENODO/$fn?download=1"
    local remote local_sz=0
    remote=$(remote_size "$url")
    [ -f "$dest/$fn" ] && local_sz=$(stat -c%s "$dest/$fn")
    if [ -n "$remote" ] && [ "$local_sz" = "$remote" ]; then
        echo "[$(date +%H:%M:%S)] skip $fn (complete: $remote bytes)"; return 0
    fi
    curl -fsSL -C - --retry 8 --retry-delay 10 -o "$dest/$fn" "$url"
}

download_part() {  # url start end part  — resume via manual offset across drops
    local url="$1" start="$2" end="$3" part="$4"
    local want=$(( end - start + 1 )) have a
    for ((a = 0; a < 50; a++)); do
        have=0; [ -f "$part" ] && have=$(stat -c%s "$part")
        [ "$have" -ge "$want" ] && return 0
        # one request for the remaining bytes, appended; a dropped connection
        # just means the next iteration resumes from the new (larger) offset.
        curl -fsS --connect-timeout 30 -r "$(( start + have ))-${end}" "$url" >> "$part" || true
        sleep 2
    done
    have=0; [ -f "$part" ] && have=$(stat -c%s "$part")
    [ "$have" -ge "$want" ]
}

parallel_dl() {  # parallel_dl <filename> <dest_dir>   (N byte-range parts, resumable)
    local fn="$1" dest="$2"
    local url="$ZENODO/$fn?download=1" out="$dest/$fn"
    local size; size=$(remote_size "$url")
    [ -n "$size" ] || { echo "ERROR: no content-length for $fn"; return 1; }
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
        echo "[$(date +%H:%M:%S)] skip $fn (complete: $size bytes)"; return 0
    fi
    local chunk=$(( (size + NCONN - 1) / NCONN ))
    echo "[$(date +%H:%M:%S)] $fn: $size bytes, $NCONN parts of ~$((chunk/1024/1024)) MB"
    local pids=()
    for ((i=0; i<NCONN; i++)); do
        local start=$(( i * chunk )); [ "$start" -ge "$size" ] && break
        local end=$(( start + chunk - 1 )); [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        local part="$out.part$i" want=$(( end - start + 1 )) have=0
        [ -f "$part" ] && have=$(stat -c%s "$part")
        if [ "$have" -ge "$want" ]; then continue; fi   # part already complete
        download_part "$url" "$start" "$end" "$part" & pids+=($!)
    done
    local fail=0
    for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || fail=1; }; done
    [ "$fail" -eq 0 ] || { echo "ERROR: a part of $fn failed; re-run to resume"; return 1; }
    # assemble parts in NUMERIC order (ls -v); plain glob sorts lexically
    # (part0, part1, part10, ...) which would scramble the file.
    cat $(ls -v "$out".part*) > "$out"
    if [ "$(stat -c%s "$out")" != "$size" ]; then
        echo "ERROR: $fn assembled size mismatch"; rm -f "$out"; return 1
    fi
    case "$fn" in *.zip)
        unzip -l "$out" >/dev/null 2>&1 || { echo "ERROR: $fn invalid zip after assembly"; return 1; } ;;
    esac
    rm -f "$out".part*
    echo "[$(date +%H:%M:%S)] $fn assembled OK ($size bytes)"
}

echo "=== 1. metadata ==="
for f in "${META_FILES[@]}"; do [ -f "$META/$f" ] || dl "$f" "$META"; done
gsutil -m -q cp "$META"/*.csv "$META"/*.json "$GCS_ROOT/metadata/"

echo "=== 2. labels (Raven tables) ==="
dl "labels.zip" "$ZIPS"
chmod -R u+w "$LAB" 2>/dev/null || true
unzip -o -q "$ZIPS/labels.zip" -d "$LAB"
labdir=$(find "$LAB" -maxdepth 2 -type d -name labels | head -1); labdir="${labdir:-$LAB}"
echo "[$(date +%H:%M:%S)] uploading $(find "$labdir" -name '*.txt' | wc -l) annotations -> GCS ..."
gsutil -m -q rsync -r "$labdir" "$GCS_ROOT/annotations"

echo "=== 3. audios (192 kHz WAV, ~86 GB, parallel) ==="
parallel_dl "audios.zip" "$ZIPS"
echo "[$(date +%H:%M:%S)] extracting audios.zip ..."
chmod -R u+w "$AUD" 2>/dev/null || true
unzip -o -q "$ZIPS/audios.zip" -d "$AUD"
# upload all WAVs flat to recordings/ (robust to internal folder layout)
for d in $(find "$AUD" -name '*.wav' -printf '%h\n' | sort -u); do
    echo "[$(date +%H:%M:%S)] uploading $(find "$d" -maxdepth 1 -name '*.wav' | wc -l) wav from $(basename "$d") -> recordings/ ..."
    gsutil -m -q rsync "$d" "$GCS_ROOT/recordings"
done

echo "=== 4. counts ==="
echo "wav uploaded: $(gsutil ls "$GCS_ROOT/recordings/**.wav" 2>/dev/null | wc -l)"
echo "txt uploaded: $(gsutil ls "$GCS_ROOT/annotations/**.txt" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] staging DONE"

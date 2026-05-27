#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage the Dartmouth "Acoustic Forest Soundscape Dataset of Avian
# Vocalizations from Eastern North America" (Zenodo record 20038954) into
# gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0/.
#
# Downloads the 3 recording zips, 3 annotation zips, and the metadata files
# from Zenodo, extracts them, and uploads to GCS in the standard layout:
#
#   recordings/<DatasetID>/*.flac   (lossless originals, 32 kHz)
#   annotations/<DatasetID>/*.txt   (Raven Pro selection tables)
#   metadata/*.csv, ReadMe.txt
#
# Heavy on network/IO (not memory); intended to run on the dev VM in the
# background (the Zenodo download needs external internet). Idempotent:
# re-running resumes downloads and re-syncs to GCS.
#
# Set WORK to a directory on a disk with ~60 GB free (zips + extracted). On the
# dev VM that is the NFS share (/mnt/home/...), not local $HOME.
#
# USAGE:
#   WORK=/mnt/home/dartmouth_staging \
#     nohup bash scripts/data_preprocessing_scripts/dartmouth_avian_soundscapes_stage.sh \
#       > /mnt/home/dartmouth_staging/stage.log 2>&1 &
# ---------------------------------------------------------------------------
set -euo pipefail

ZENODO="https://zenodo.org/records/20038954/files"
GCS_ROOT="gs://esp-data-ingestion/dartmouth-avian-soundscapes/v0.1.0"
WORK="${WORK:-$HOME/dartmouth_staging}"

ZIPS="$WORK/zips"
REC="$WORK/extract_rec"
ANN="$WORK/extract_ann"
META="$WORK/meta"
mkdir -p "$ZIPS" "$REC" "$ANN" "$META"

DATASETS=(DatasetACAD DatasetMABI DatasetSIMR)
META_FILES=(recording_metadata.csv site_metadata.csv species_metadata.csv data_dictionary.csv ReadMe.txt)

dl() {  # dl <filename> <dest_dir>   (skip if already complete; else resume)
    local fn="$1" dest="$2"
    local url="$ZENODO/$fn?download=1"
    local remote local_sz=0
    remote=$(curl -sIL "$url" 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1}/^content-length:/{cl=$2} END{gsub(/\r/,"",cl); print cl}')
    [ -f "$dest/$fn" ] && local_sz=$(stat -c%s "$dest/$fn")
    if [ -n "$remote" ] && [ "$local_sz" = "$remote" ]; then
        echo "[$(date +%H:%M:%S)] skip $fn (complete: $remote bytes)"
        return 0
    fi
    curl -fsSL -C - --retry 8 --retry-delay 10 -o "$dest/$fn" "$url"
}

echo "=== 1. metadata ==="
for f in "${META_FILES[@]}"; do
    [ -f "$META/$f" ] || dl "$f" "$META"
done
gsutil -m -q cp "$META"/*.csv "$META/ReadMe.txt" "$GCS_ROOT/metadata/"

echo "=== 2. download all zips in parallel (Zenodo throttles ~1MB/s per connection) ==="
pids=()
for ds in "${DATASETS[@]}"; do
    dl "${ds}_Recordings.zip"  "$ZIPS" & pids+=($!)
    dl "${ds}_Annotations.zip" "$ZIPS" & pids+=($!)
done
echo "[$(date +%H:%M:%S)] launched ${#pids[@]} downloads (pids: ${pids[*]})"
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "WARN: download pid $pid exited non-zero"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "ERROR: one or more downloads failed; re-run to resume"; exit 1; }
echo "[$(date +%H:%M:%S)] all downloads complete"

echo "=== 3. extract + upload per dataset ==="
# Recordings zips extract to <ds>_Recordings/ and annotations to <ds>/; both are
# normalised to recordings/<ds>/ and annotations/<ds>/ on GCS.
for ds in "${DATASETS[@]}"; do
    echo "[$(date +%H:%M:%S)] extracting $ds ..."
    # zip dir entries are stored read-only (mode 555); make writable so a
    # re-run's `unzip -o` can overwrite instead of failing on delete.
    chmod -R u+w "$REC" "$ANN" 2>/dev/null || true
    unzip -o -q "$ZIPS/${ds}_Recordings.zip"  -d "$REC"
    unzip -o -q "$ZIPS/${ds}_Annotations.zip" -d "$ANN"

    rec_dir=$(find "$REC" -maxdepth 1 -type d -name "${ds}*" | head -1)
    ann_dir=$(find "$ANN" -maxdepth 1 -type d -name "${ds}*" | head -1)
    echo "[$(date +%H:%M:%S)] uploading $ds recordings ($(basename "$rec_dir")) -> GCS ..."
    gsutil -m -q rsync -r "$rec_dir" "$GCS_ROOT/recordings/$ds"
    echo "[$(date +%H:%M:%S)] uploading $ds annotations ($(basename "$ann_dir")) -> GCS ..."
    gsutil -m -q rsync -r "$ann_dir" "$GCS_ROOT/annotations/$ds"
done

echo "=== 3. counts ==="
echo "flac uploaded: $(gsutil ls -r "$GCS_ROOT/recordings/**.flac" 2>/dev/null | wc -l)"
echo "txt  uploaded: $(gsutil ls -r "$GCS_ROOT/annotations/**.txt" 2>/dev/null | wc -l)"
echo "[$(date +%H:%M:%S)] staging DONE"

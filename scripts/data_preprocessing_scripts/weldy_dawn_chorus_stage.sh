#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage Weldy et al. 2024 NW dawn-chorus dataset (Zenodo record 10895837) into
# gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0/.
#
# Downloads:
#   - 12 recording zips (annotated_recordings.zip + additional_recordings_part_*.zip)
#     ~ 20 GB total, 1,575 5-min stereo 32 kHz WAVs
#   - small TSV / PDF metadata (annotations, files, metadata, env covariates)
#
# Uploads to GCS in the standard layout:
#   recordings/<file>.wav           (32 kHz stereo originals, flat)
#   annotations/                    (raw Weldy TSVs for provenance)
#   metadata/                       (file/env/data-dictionary)
#
# Heavy on network/IO, not memory. Run on the dev VM (needs external internet).
# WORK should be on NFS (/mnt/home/...), NOT the local $HOME on this VM.
#
# USAGE:
#   WORK=/mnt/home/weldy_staging \
#     nohup bash scripts/data_preprocessing_scripts/weldy_dawn_chorus_stage.sh \
#       > /mnt/home/weldy_staging/stage.log 2>&1 &
# ---------------------------------------------------------------------------
set -euo pipefail

ZENODO="https://zenodo.org/records/10895837/files"
GCS_ROOT="gs://esp-data-ingestion/weldy_dawn_chorus/v0.1.0"
WORK="${WORK:-/mnt/home/weldy_staging}"
NCONN="${NCONN:-8}"   # bounded parallelism: keeps NFS dirty pages safe

ZIPS="$WORK/zips"; META="$WORK/meta"; EXTRACT="$WORK/extract"
mkdir -p "$ZIPS" "$META" "$EXTRACT"

ZIP_FILES=(
    annotated_recordings.zip
    additional_recordings_part_1.zip additional_recordings_part_2.zip
    additional_recordings_part_3.zip additional_recordings_part_4.zip
    additional_recordings_part_5.zip additional_recordings_part_6.zip
    additional_recordings_part_7.zip additional_recordings_part_8.zip
    additional_recordings_part_9.zip additional_recordings_part_10.zip
    additional_recordings_part_11.zip
)
META_FILES=(
    acoustic_annotations.tsv acoustic_files.tsv annotation_metadata.tsv
    partial_annotations.tsv annotator_method.tsv
    environmental_characteristics.tsv environmental_characteristics_metadata.tsv
    data_dictionaries.pdf
)

dl() {  # dl <filename> <dest_dir>   (skip if already complete; else resume)
    local fn="$1" dest="$2"
    local url="$ZENODO/$fn?download=1"
    local remote local_sz=0
    remote=$(curl -sIL "$url" 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1}/^content-length:/{cl=$2} END{gsub(/\r/,"",cl); print cl}')
    [ -f "$dest/$fn" ] && local_sz=$(stat -c%s "$dest/$fn")
    if [ -n "$remote" ] && [ "$local_sz" = "$remote" ]; then
        echo "[$(date +%H:%M:%S)] skip $fn (complete: $remote bytes)"; return 0
    fi
    curl -fsSL -C - --retry 8 --retry-delay 10 -o "$dest/$fn" "$url"
}

echo "=== 1. metadata ==="
for f in "${META_FILES[@]}"; do dl "$f" "$META"; done
gsutil -m -q cp "$META"/*.tsv "$META"/*.pdf "$GCS_ROOT/metadata/"

echo "=== 2. recording zips (parallel, $NCONN streams) ==="
pids=()
for z in "${ZIP_FILES[@]}"; do dl "$z" "$ZIPS" & pids+=($!); done
echo "[$(date +%H:%M:%S)] launched ${#pids[@]} downloads"
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "WARN dl pid $p exited non-zero"; fail=1; }; done
[ "$fail" -eq 0 ] || { echo "ERROR: a download failed; re-run to resume"; exit 1; }
echo "[$(date +%H:%M:%S)] all downloads complete ($(du -sh "$ZIPS"|cut -f1))"

echo "=== 3. extract + upload (per zip) ==="
# zip dir entries can be stored read-only; chmod first so -o can overwrite on re-run
chmod -R u+w "$EXTRACT" 2>/dev/null || true
for z in "${ZIP_FILES[@]}"; do
    echo "[$(date +%H:%M:%S)] extracting $z ..."
    unzip -o -q "$ZIPS/$z" -d "$EXTRACT"
done

# audio likely extracts to flat or per-zip subdirs; find each wav dir and rsync
# flat to recordings/ (basenames are unique: Site_NNN_Rep_X.wav)
for d in $(find "$EXTRACT" -name '*.wav' -printf '%h\n' | sort -u); do
    n=$(find "$d" -maxdepth 1 -name '*.wav' | wc -l)
    echo "[$(date +%H:%M:%S)] uploading $n wav from $(basename "$d") -> recordings/ ..."
    gsutil -m -q rsync "$d" "$GCS_ROOT/recordings"
done

echo "=== 4. counts ==="
echo "recordings wav: $(gsutil ls "$GCS_ROOT/recordings/**.wav" 2>/dev/null | wc -l) (expect 1575)"
echo "[$(date +%H:%M:%S)] staging DONE"

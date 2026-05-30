#!/usr/bin/env bash
#
# Stage Zenodo 17282717 (Mediterranean Cetacean PAM, Jankauskaite et al. 2025):
# - Download 5 missing species zips from Zenodo to NFS (parallel, 5 streams)
# - Pull existing Physeter zip from GCS (already staged)
# - Extract each zip into /mnt/home/zenodo_17282717_staging/extract/<species>/
# - Upload extracted content to gs://esp-data-ingestion/.../zenodo_17282717/
#
# Designed to run on Slurm `cpu` partition (network + I/O bound, no GPU needed).
#
# Total download: ~4 GB; total extracted: ~5 GB.
#
# Usage:
#   sbatch jobs/zenodo_17282717_stage.sh
#   (or run locally:  bash scripts/.../zenodo_17282717_stage.sh)
#
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home/%u/logs/zenodo_17282717_stage_%j.log
#SBATCH --qos=naturelm

set -euo pipefail

# NFS path differs: /mnt/home on dev VM, $HOME on slurm-login (same NFS share).
if [[ -d /mnt/home ]]; then
    STAGING="/mnt/home/zenodo_17282717_staging"
else
    STAGING="${HOME}/zenodo_17282717_staging"
fi
ZIPS="${STAGING}/zips"
EXTRACT="${STAGING}/extract"
GCS_DST="gs://esp-data-ingestion/superwhale/v0.1.0/raw/zenodo_17282717"
GCS_PHYSETER_ZIP="gs://esp-data-ingestion/superwhale/v0.1.0/raw/zenodo_17282717/Physeter_macrocephalus.zip"

mkdir -p "${ZIPS}" "${EXTRACT}"

# Zenodo direct-download URL pattern: https://zenodo.org/records/<id>/files/<filename>?download=1
ZENODO_BASE="https://zenodo.org/records/17282717/files"

# Each remote URL is a separate species zip. The Physeter zip is already on GCS;
# we will pull it from GCS to save bandwidth.
declare -A REMOTE_ZIPS=(
    [Delphinidae]="${ZENODO_BASE}/Delphinidae.zip?download=1"
    [Globicephala_melas]="${ZENODO_BASE}/Globicephala_melas.zip?download=1"
    [Grampus_griseus]="${ZENODO_BASE}/Grampus_griseus.zip?download=1"
    [Stenella_coeruleoalba]="${ZENODO_BASE}/Stenella_coeruleoalba.zip?download=1"
    [Tursiops_truncatus]="${ZENODO_BASE}/Tursiops_truncatus.zip?download=1"
)

# Skip-complete download: only re-fetch if local size != remote Content-Length.
dl() {
    local species="$1"; local url="$2"
    local dst="${ZIPS}/${species}.zip"
    local remote_size
    remote_size=$(curl -sIL "${url}" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {n=$2} END{print n+0}' | tr -d '\r')
    if [[ -f "${dst}" ]]; then
        local local_size
        local_size=$(stat -c '%s' "${dst}")
        if [[ "${local_size}" == "${remote_size}" && "${remote_size}" -gt 0 ]]; then
            echo "[skip] ${species}.zip already complete (${local_size} bytes)"
            return 0
        fi
    fi
    echo "[get ] ${species}.zip -> ${dst}  (remote ${remote_size} bytes)"
    curl -sL --retry 5 --retry-delay 10 -o "${dst}.part" "${url}"
    mv "${dst}.part" "${dst}"
    local fetched
    fetched=$(stat -c '%s' "${dst}")
    if [[ "${fetched}" != "${remote_size}" && "${remote_size}" -gt 0 ]]; then
        echo "[warn] size mismatch on ${species}: got ${fetched}, expected ${remote_size}"
    fi
}

echo "=== Downloading 5 missing species in parallel ==="
PIDS=()
for sp in "${!REMOTE_ZIPS[@]}"; do
    dl "${sp}" "${REMOTE_ZIPS[$sp]}" &
    PIDS+=($!)
done
wait "${PIDS[@]}" || true

echo "=== Pulling Physeter zip from GCS (already staged) ==="
if [[ ! -f "${ZIPS}/Physeter_macrocephalus.zip" ]]; then
    gsutil -q cp "${GCS_PHYSETER_ZIP}" "${ZIPS}/Physeter_macrocephalus.zip"
fi

echo "=== Download summary ==="
ls -lh "${ZIPS}/"

echo "=== Extracting all 6 zips ==="
for zip in "${ZIPS}"/*.zip; do
    species=$(basename "${zip}" .zip)
    species_dir="${EXTRACT}/${species}"
    if [[ -d "${species_dir}/2-annotation-clips" ]] && \
       compgen -G "${species_dir}/2-annotation-clips/*.wav" > /dev/null; then
        echo "[skip-extract] ${species} already extracted (clips present)"
        continue
    fi
    echo "[extract] ${species}"
    rm -rf "${species_dir}"
    mkdir -p "${species_dir}"
    unzip -q -o "${zip}" -d "${species_dir}"
    # Some zips may have a single top-level dir matching the species name; flatten if so.
    if [[ -d "${species_dir}/${species}" && ! -d "${species_dir}/1-annotation-tables" ]]; then
        echo "  flattening ${species}/${species}/ -> ${species}/"
        shopt -s dotglob
        mv "${species_dir}/${species}/"* "${species_dir}/"
        rmdir "${species_dir}/${species}"
        shopt -u dotglob
    fi
done

echo "=== Extracted tree summary ==="
for d in "${EXTRACT}"/*/; do
    species=$(basename "${d}")
    n_tables=$(find "${d}/1-annotation-tables" -name '*.txt' 2>/dev/null | wc -l || echo 0)
    n_clips=$(find "${d}/2-annotation-clips" -name '*.wav' 2>/dev/null | wc -l || echo 0)
    sz=$(du -sh "${d}" 2>/dev/null | cut -f1)
    echo "  ${species}: ${n_tables} tables, ${n_clips} clips, ${sz}"
done

echo "=== Uploading extracted dirs to GCS ==="
# Upload each species' extracted tree to ${GCS_DST}/<species>/
# rsync is idempotent: only changed/new files are transferred.
for d in "${EXTRACT}"/*/; do
    species=$(basename "${d}")
    echo "[upload] ${species}/ -> ${GCS_DST}/${species}/"
    gsutil -m rsync -r "${d}" "${GCS_DST}/${species}/"
done

echo "=== Done. Final GCS size: ==="
gsutil du -sh "${GCS_DST}/" 2>&1 | tail -2

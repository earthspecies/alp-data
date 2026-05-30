#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home/%u/logs/clotho_stage_%j.log
#SBATCH --qos=naturelm
set -euo pipefail
if [[ -d /mnt/home ]]; then STAGING="/mnt/home/clotho_staging"; else STAGING="${HOME}/clotho_staging"; fi
ZIPS="${STAGING}/zips"; EXTRACT="${STAGING}/extract"
mkdir -p "${ZIPS}" "${EXTRACT}"
ZEN_BASE="https://zenodo.org/api/records/4783391/files"
declare -A FILES=(
  [clotho_audio_development.7z]="${ZEN_BASE}/clotho_audio_development.7z/content"
  [clotho_audio_validation.7z]="${ZEN_BASE}/clotho_audio_validation.7z/content"
  [clotho_audio_evaluation.7z]="${ZEN_BASE}/clotho_audio_evaluation.7z/content"
  [clotho_captions_development.csv]="${ZEN_BASE}/clotho_captions_development.csv/content"
  [clotho_captions_validation.csv]="${ZEN_BASE}/clotho_captions_validation.csv/content"
  [clotho_captions_evaluation.csv]="${ZEN_BASE}/clotho_captions_evaluation.csv/content"
  [clotho_metadata_development.csv]="${ZEN_BASE}/clotho_metadata_development.csv/content"
  [clotho_metadata_validation.csv]="${ZEN_BASE}/clotho_metadata_validation.csv/content"
  [clotho_metadata_evaluation.csv]="${ZEN_BASE}/clotho_metadata_evaluation.csv/content"
)
dl() {
  local name="$1"; local url="$2"; local dst="${ZIPS}/${name}"
  local remote_size
  remote_size=$(curl -sIL "${url}" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {n=$2} END{print n+0}' | tr -d '\r')
  if [[ -f "${dst}" ]]; then
    local local_size; local_size=$(stat -c '%s' "${dst}")
    if [[ "${local_size}" == "${remote_size}" && "${remote_size}" -gt 0 ]]; then
      echo "[skip] ${name} already complete (${local_size} bytes)"; return 0
    fi
  fi
  echo "[get ] ${name} <- ${remote_size} bytes"
  curl -sL --retry 5 --retry-delay 10 -o "${dst}.part" "${url}"
  mv "${dst}.part" "${dst}"
}
echo "=== Downloading 9 Clotho files in parallel ==="
PIDS=()
for f in "${!FILES[@]}"; do dl "$f" "${FILES[$f]}" & PIDS+=($!); done
wait "${PIDS[@]}" || true
echo "=== Download summary ==="
ls -lh "${ZIPS}/"
echo "=== Extract 7z archives ==="
for arch in "${ZIPS}"/*.7z; do
  name=$(basename "${arch}" .7z)
  out="${EXTRACT}/$(echo "${name}" | sed 's/clotho_audio_//')"
  if [[ -d "${out}" ]] && compgen -G "${out}/*.wav" > /dev/null; then
    echo "[skip-extract] ${name}"; continue
  fi
  echo "[extract] ${name} -> ${out}"
  mkdir -p "${out}"
  # Use py7zr via uv (7z CLI not installed on Slurm nodes)
  uv run --with py7zr python -c "
import py7zr, sys
with py7zr.SevenZipFile('${arch}', mode='r') as z:
    z.extractall(path='${out}')
print('extracted', '${arch}', '->', '${out}')
"
done
echo "=== Extraction summary ==="
for d in "${EXTRACT}"/*/; do
  s=$(basename "${d}")
  n=$(find "${d}" -name '*.wav' | wc -l)
  sz=$(du -sh "${d}" | cut -f1)
  echo "  ${s}: ${n} wav, ${sz}"
done
echo "=== Upload audio + caption/metadata CSVs to GCS ==="
GCS_RAW="gs://esp-data-ingestion/clotho/v0.1.0/raw"
for d in "${EXTRACT}"/*/; do
  s=$(basename "${d}")
  echo "[upload] ${s}/"
  # Some 7z archives may have a top-level dir; figure it out
  if [[ -d "${d}/${s}" ]]; then src="${d}/${s}/"; else src="${d}"; fi
  gsutil -m rsync -r "${src}" "${GCS_RAW}/audio/${s}/"
done
echo "[upload] CSVs"
for csv in "${ZIPS}"/*.csv; do
  gsutil -q cp "${csv}" "${GCS_RAW}/metadata/$(basename ${csv})"
done
echo "=== Done. Final GCS size ==="
gsutil du -sh "${GCS_RAW}" 2>&1 | tail -2

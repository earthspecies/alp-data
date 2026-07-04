#!/usr/bin/env bash
#SBATCH --job-name=reupload-insect-32k
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --qos=naturelm
# Re-upload InsectSound audio_32k shard — the original local job's upload
# was truncated by an OOM crash (25,228 / 50,000 present on GCS). rsync
# skips files already uploaded and fills the ~24,772 gap.

set -euo pipefail

SRC="/home/david_earthspecies_org/esp-data-dev/monster_monash_staging/InsectSound/audio_32k/"
DST="gs://esp-data-ingestion/monster-monash-insect-sound/v0.1.0/audio_32k/"

echo "rsync ${SRC} -> ${DST}"
gsutil -m rsync -r "${SRC}" "${DST}"
echo "Done. Re-listing:"
gsutil ls -r "${DST}**" | grep -c '\.flac$'

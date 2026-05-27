#!/usr/bin/env bash
#SBATCH --job-name=pteroset-assemble
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/home/%u/logs/pteroset_assemble_%j.log
#SBATCH --error=/home/%u/logs/pteroset_assemble_%j.err

# ───────────────────────────────────────────────────────────────────
# Assemble the parallel-downloaded audios.zip parts (on NFS), extract the
# 192 kHz WAVs, and upload them to GCS recordings/. Runs on a Slurm cpu node
# (384 GB RAM) to avoid the large NFS-write memory pressure that crashed the
# 14 GB dev VM. Needs GCS access only — no external internet.
#
# Prereq: all audios.zip.part* present + complete on NFS (download finished on
# the dev VM). Submit:
#   ssh slurm-login 'mkdir -p ~/logs && cd ~/esp-data-dev && sbatch jobs/pteroset_assemble_upload.sh'
# ───────────────────────────────────────────────────────────────────
set -euo pipefail

WORK="$HOME/pteroset_staging"
ROOT="gs://esp-data-ingestion/pteroset/v0.1.0"
ZIP="$WORK/zips/audios.zip"
SIZE=85757149872

cd "$WORK"
echo "=== assemble $(date) ==="
if [ ! -f "$ZIP" ] || [ "$(stat -c%s "$ZIP")" != "$SIZE" ]; then
    echo "concatenating parts in numeric order ..."
    cat $(ls -v zips/audios.zip.part*) > "$ZIP"
fi
[ "$(stat -c%s "$ZIP")" = "$SIZE" ] || { echo "ERROR: assembled size != $SIZE"; exit 1; }
echo "validating zip ..."; unzip -l "$ZIP" >/dev/null || { echo "ERROR: invalid zip"; exit 1; }

echo "=== extract $(date) ==="
chmod -R u+w extract_audio 2>/dev/null || true
mkdir -p extract_audio
unzip -o -q "$ZIP" -d extract_audio
echo "extracted wav: $(find extract_audio -name '*.wav' | wc -l)"

echo "=== upload $(date) ==="
for d in $(find extract_audio -name '*.wav' -printf '%h\n' | sort -u); do
    echo "uploading $(find "$d" -maxdepth 1 -name '*.wav' | wc -l) wav from $(basename "$d") ..."
    gcloud storage rsync --recursive "$d" "$ROOT/recordings"
done

echo "=== done $(date) ==="
echo "wav in GCS: $(gcloud storage ls "$ROOT/recordings/**.wav" 2>/dev/null | wc -l)"

#!/usr/bin/env bash
#SBATCH --job-name=resample-zenodo-17282717
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/%u/logs/resample_zenodo_17282717_%A_%a.log
#SBATCH --error=/home/%u/logs/resample_zenodo_17282717_%A_%a.err

# ───────────────────────────────────────────────────────────────────
# Resample Zenodo 17282717 (Mediterranean Cetacean PAM) annotation clips to
# mono 16-bit PCM WAV at 16 kHz and 32 kHz, mirroring the source layout.
#
# Source clips have heterogeneous sample rates (48/96/192 kHz observed) per
# the HydroMoth recorder configuration. resample_to_sr.py filters to audio
# extensions, so it will skip non-.wav files in the source tree.
#
# Array index → target rate:
#   0  -> audio_16k/zenodo_17282717/
#   1  -> audio_32k/zenodo_17282717/
#
# Submit:
#   ssh slurm-login 'sbatch /home/$USER/esp-data-dev/jobs/resample_zenodo_17282717.sh'
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/esp-data-dev"

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo)}"
echo "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

ROOT="gs://esp-data-ingestion/superwhale/v0.1.0/raw"

RATES=(16000 32000)
DESTS_PREFIX=("$ROOT/audio_16k/zenodo_17282717" "$ROOT/audio_32k/zenodo_17282717")

IDX=${SLURM_ARRAY_TASK_ID:-0}
SR=${RATES[$IDX]}
DST_BASE=${DESTS_PREFIX[$IDX]}

echo "=== Zenodo 17282717 resample array task $IDX ($SR Hz) at $(date) ==="
echo "Dest prefix: $DST_BASE/"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-?}"

# Iterate per-species. Source restricted to 2-annotation-clips/ so we skip
# the much larger 3-complete-WAV/ deployment recordings that the manifest
# never references.
SPECIES=(Delphinidae Globicephala_melas Grampus_griseus Physeter_macrocephalus Stenella_coeruleoalba Tursiops_truncatus)

for sp in "${SPECIES[@]}"; do
    SRC="$ROOT/zenodo_17282717/${sp}/2-annotation-clips/"
    DST="$DST_BASE/${sp}/2-annotation-clips/"
    echo ""
    echo "--- $sp @ $SR Hz ---"
    echo "  src: $SRC"
    echo "  dst: $DST"

    uv run --script scripts/resample_to_sr.py \
        --source-prefix "$SRC" \
        --dest-prefix "$DST" \
        --target-sr "$SR" \
        --workers "${SLURM_CPUS_PER_TASK:-16}" \
        --skip-existing
done

echo "=== Finished array task $IDX at $(date) ==="

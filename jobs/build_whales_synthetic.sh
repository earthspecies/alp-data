#!/bin/bash
#SBATCH --job-name=build-whales
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output="/home/%u/logs/build_whales_%j.log"
#SBATCH --error="/home/%u/logs/build_whales_%j.err"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the synthetic `whales` dataset.
#
# Watkins (train) + DCLDE 2026 (all_excl_beanszero) → 32 kHz mono WAV
# clips written to gs://foundation-model-data/synthetic/whales/v0.1.0/
# plus a manifest CSV consumed by `esp_data.datasets.Whales`.
#
# This is a pure CPU job — no GPU required. The CPU partition has plenty
# of memory and avoids contending for accelerator nodes. If the cpu
# partition is unavailable, the script also runs on `t4` unchanged
# (override with `--partition=t4` on resubmission).
#
# USAGE
#   ssh slurm-login
#   cd /home/${USER}/esp-data-dev
#   mkdir -p ~/logs
#   sbatch jobs/build_whales_synthetic.sh
#
# Smoke test (200 positives, custom out-root):
#   sbatch jobs/build_whales_synthetic.sh --limit 200 \
#       --out-root gs://foundation-model-data/synthetic/whales/v0.1.0-smoke
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/david_earthspecies_org/esp-data-dev

echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Args: $*"

uv run python scripts/build_whales_synthetic.py \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

#!/bin/bash
#SBATCH --job-name=build-whales
#SBATCH --partition=t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=14
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output="/home/%u/logs/build_whales_%j.log"
#SBATCH --error="/home/%u/logs/build_whales_%j.err"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the synthetic `whales` dataset.
#
# Watkins (train) + DCLDE 2026 (all) → 32 kHz mono WAV clips written to
# gs://foundation-model-data/synthetic/whales/v0.1.0/ plus a manifest
# CSV consumed by `esp_data.datasets.Whales`.
#
# Pure CPU job — no GPU. Sized to stay inside ONE t4 GPU slot
# (≤14 CPUs, ≤14 GB RAM, no --gres) so it does not block GPU jobs from
# co-scheduling on the same node (see `esp-research/CLAUDE.md` t4 rules).
# Override with `--partition=cpu` when cpu nodes are free for faster IO
# parallelism.
#
# USAGE
#   ssh slurm-login
#   cd /home/${USER}/esp-data-dev
#   mkdir -p ~/logs
#   sbatch jobs/build_whales_synthetic.sh
#
# Smoke test (50 positives, custom out-root):
#   sbatch jobs/build_whales_synthetic.sh --limit 50 \
#       --out-root gs://foundation-model-data/synthetic/whales/v0.1.0-smoke
# ───────────────────────────────────────────────────────────────────

set -euo pipefail
cd /home/david_earthspecies_org/esp-data-dev

echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Args: $*"

uv run python scripts/build_whales_synthetic.py \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

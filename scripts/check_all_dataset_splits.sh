#!/usr/bin/env bash
# Run check_dataset_splits_exist.py for every registered dataset, one at a time.
#
# Iterates `list_registered_datasets()` from esp_data.dataset and invokes the
# checker per-dataset so a failure in one does not mask others. Exits 1 if any
# dataset has a missing split.
#
# Usage: scripts/check_all_dataset_splits.sh [--workers N]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WORKERS=16
if [[ "${1:-}" == "--workers" ]]; then
  WORKERS="${2:?--workers requires a value}"
  shift 2
fi

DATASETS=()
while IFS= read -r line; do
  DATASETS+=("${line}")
done < <(uv run python -c "
import esp_data.datasets  # noqa: F401
from esp_data.dataset import list_registered_datasets
for n in list_registered_datasets():
    print(n)
")

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No registered datasets found."
  exit 1
fi

echo "Checking ${#DATASETS[@]} datasets..."
echo

FAILED=()
for name in "${DATASETS[@]}"; do
  echo "=== ${name} ==="
  if ! uv run python scripts/check_dataset_splits_exist.py \
        --dataset "${name}" --workers "${WORKERS}"; then
    FAILED+=("${name}")
  fi
  echo
done

echo "============================================"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAIL: ${#FAILED[@]} dataset(s) had missing splits:"
  for n in "${FAILED[@]}"; do
    echo "  - ${n}"
  done
  exit 1
fi

echo "OK: all ${#DATASETS[@]} datasets passed."

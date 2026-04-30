"""Export the two selected Stage-2 F0 task backends.

This script extracts only the two 16kHz `f0_bioacoustic` entries from
`stage2_train_v1.yml` (templates `f0_summary` and `f0_species`), materializes
them with `esp_data.dataset_from_config`, and saves CSV/JSONL backends.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from esp_data import dataset_from_config
from esp_data.dataset import DatasetConfig
from esp_data.io import read_yaml

ROOT = Path(__file__).resolve().parent.parent
NATURELM_PROJECT = ROOT / "esp-research" / "projects" / "NatureLM-audio-v1.5"
DEFAULT_CONFIG = NATURELM_PROJECT / "configs" / "datasets" / "stage2_train_v1.yml"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "exports" / "stage2_f0"

# Register custom transforms used in NatureLM config (e.g. chat, f0_features).
sys.path.insert(0, str(NATURELM_PROJECT))
import data.transforms  # noqa: F401

TARGET_TAXA = [
    "canids",
    "hummingbirds",
    "La_Palma_chaffinches",
    "lions",
    "little_owls",
    "long-billed_hermits",
    "monk_parakeets",
    "orangutans",
    "Reunion_grey_white_eyes",
]
TARGET_TEMPLATES = {"f0_summary", "f0_species"}
TEMPLATE_ORDER = {"f0_summary": 0, "f0_species": 1}


def _chat_template_name(entry: dict[str, Any]) -> str | None:
    """Get chat template name from a chain entry."""
    for transform in entry.get("transformations", []):
        if transform.get("type") == "chat":
            return transform.get("template_name")
    return None


def _is_target_entry(entry: dict[str, Any]) -> bool:
    """Return True only for the requested two 16kHz F0 entries."""
    if entry.get("dataset_name") != "f0_bioacoustic":
        return False
    if entry.get("split") != "train":
        return False
    if entry.get("sample_rate") != 16000:
        return False
    if entry.get("taxa") != TARGET_TAXA:
        return False
    return _chat_template_name(entry) in TARGET_TEMPLATES


def _load_selected_entries(config_path: Path) -> list[dict[str, Any]]:
    """Load and validate the selected F0 entries from chain config."""
    cfg = read_yaml(config_path)
    if not isinstance(cfg, dict) or "chain" not in cfg:
        raise ValueError(f"Expected top-level `chain` key in {config_path}")
    chain_cfg = cfg["chain"]
    if not isinstance(chain_cfg, dict) or "datasets" not in chain_cfg:
        raise ValueError(f"Expected `chain.datasets` in {config_path}")
    datasets = chain_cfg["datasets"]
    if not isinstance(datasets, list):
        raise ValueError(f"`chain.datasets` must be a list in {config_path}")

    selected = [entry for entry in datasets if _is_target_entry(entry)]
    found_templates = {_chat_template_name(entry) for entry in selected}
    if found_templates != TARGET_TEMPLATES:
        raise ValueError(
            "Could not find exactly the requested F0 entries. "
            f"Expected templates={sorted(TARGET_TEMPLATES)}, "
            f"found={sorted(found_templates)}"
        )
    selected.sort(key=lambda entry: TEMPLATE_ORDER.get(_chat_template_name(entry) or "", 999))
    return selected


def _export_entry(entry: dict[str, Any], output_dir: Path, fmt: str) -> None:
    """Materialize one selected entry and save outputs."""
    template = _chat_template_name(entry)
    if template is None:
        raise ValueError("Selected entry missing chat template")

    print(f"Loading dataset for template={template} ...", flush=True)
    dataset_cfg = DatasetConfig.model_validate(entry)
    dataset, metadata = dataset_from_config(dataset_cfg)
    rows = len(dataset)
    print(f"Loaded template={template}, rows={rows}", flush=True)

    base = output_dir / f"f0_bioacoustic_16khz_{template}"
    if fmt in {"csv", "both"}:
        dataset.save_data(str(base.with_suffix(".csv")), fmt="csv")
    if fmt in {"jsonl", "both"}:
        dataset.save_data(str(base.with_suffix(".jsonl")), fmt="jsonl")
    meta_path = base.with_name(f"{base.name}_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved {template}", flush=True)
    if fmt in {"csv", "both"}:
        print(f"  - {base.with_suffix('.csv')}", flush=True)
    if fmt in {"jsonl", "both"}:
        print(f"  - {base.with_suffix('.jsonl')}", flush=True)
    print(f"  - {meta_path}", flush=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export selected Stage-2 F0 backends.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to stage2 YAML.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write exported files.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl", "both"],
        default="both",
        help="Export format.",
    )
    args = parser.parse_args()

    print(f"Config: {args.config}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Format: {args.format}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_entries = _load_selected_entries(args.config)
    print(f"Selected entries: {len(selected_entries)}", flush=True)

    for entry in selected_entries:
        _export_entry(entry, args.output_dir, args.format)


if __name__ == "__main__":
    main()

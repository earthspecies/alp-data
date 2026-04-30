"""Evaluate NatureLM-audio on xeno-canto validation data.

Loads a sample of xeno-canto recordings via esp-data, sends them to a
running NatureLM-audio server, and records predictions alongside ground
truth for a range of tasks and prompt styles.

Usage
-----
    uv run python scripts/eval_naturelm.py \
        --server-url http://hostname:8001 \
        --n-samples 500 \
        --output results/naturelm_eval.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from esp_data import dataset_from_config
from esp_data.dataset import ChainedDatasetConfig, DatasetConfig

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------
# Each entry: (task_name, prompt_messages, ground_truth_key_or_callable)
#
# prompt_messages is a list of dicts with role/content.  Ground truth is
# either a string (column name) or a callable(row) -> str.

TRAINING_PROMPTS: list[tuple[str, list[dict], str | callable]] = [
    # --- Species (common) ---
    (
        "species_common_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What species is vocalizing in this audio recording? Common name?"}],
        "species_common",
    ),
    (
        "species_common_v2",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What animal can be heard in this recording?"}],
        "species_common",
    ),
    # --- Species (scientific) ---
    (
        "species_scientific_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What is the scientific name of the species in this recording?"}],
        "canonical_name",
    ),
    (
        "species_scientific_v2",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Provide the Latin name for the animal heard in this audio."}],
        "canonical_name",
    ),
    # --- Genus ---
    (
        "genus_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What is the genus of the focal species in the audio?"}],
        "genus",
    ),
    # --- Family ---
    (
        "family_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What is the family of the focal species in the audio?"}],
        "family",
    ),
    # --- Order ---
    (
        "order_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What is the order of the focal species in the audio?"}],
        "order",
    ),
    # --- Full taxonomy ---
    (
        "taxonomic_name_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Provide the full taxonomic classification for the species in this recording, starting with the phylum."}],
        lambda row: f"{row.get('phylum', '')}; {row.get('class', '')}; {row.get('order', '')}; {row.get('family', '')}; {row.get('canonical_name', '')}",
    ),
    # --- Call type ---
    (
        "call_type_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What type of vocalization or call is this?"}],
        "behavior",
    ),
    (
        "call_or_song_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Is this a call or a song?"}],
        "behavior",
    ),
    # --- Life stage ---
    (
        "life_stage_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> What life stage is the animal in this recording?"}],
        "lifeStage",
    ),
    # --- Captioning ---
    (
        "caption_v1",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Caption this audio with a rich, detailed description. Avoid specific species names."}],
        None,  # no ground truth for open-ended captioning
    ),
]

# Prompts NOT used during training — test generalization / instruction following
CUSTOM_PROMPTS: list[tuple[str, list[dict], str | callable | None]] = [
    # Open-ended description
    (
        "custom_describe",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Describe every sound you hear in this recording in detail."}],
        None,
    ),
    # Habitat inference
    (
        "custom_habitat",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Based on the sounds, what habitat or environment do you think this was recorded in?"}],
        None,
    ),
    # Multi-species detection
    (
        "custom_multi_species",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> List all species you can hear in this recording, including background species."}],
        None,
    ),
    # Counting
    (
        "custom_count",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> How many individual animals can you hear vocalizing?"}],
        None,
    ),
    # Quality / noise assessment
    (
        "custom_quality",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Rate the audio quality of this recording on a scale of 1-5 and explain your rating."}],
        None,
    ),
    # Species with system prompt
    (
        "custom_system_prompt_species",
        [
            {"role": "system", "content": "You are a world-class ornithologist. Be precise and concise."},
            {"role": "user", "content": "<Audio><AudioHere></Audio> What species is this?"},
        ],
        "species_common",
    ),
    # Yes/no question format
    (
        "custom_yes_no",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Is there a bird singing in this recording? Answer yes or no."}],
        None,
    ),
    # Comparison / reasoning (multi-turn)
    (
        "custom_multiturn",
        [
            {"role": "user", "content": "<Audio><AudioHere></Audio> What species is this?"},
            {"role": "assistant", "content": "I need to listen carefully to identify the species."},
            {"role": "user", "content": "Please provide your best identification with the common name."},
        ],
        "species_common",
    ),
    # Structured output request
    (
        "custom_json",
        [{"role": "user", "content": "<Audio><AudioHere></Audio> Identify this recording. Respond in the format: Species: <name>, Call type: <type>, Confidence: <high/medium/low>"}],
        None,
    ),
    # Geographic context (uses row metadata)
    (
        "custom_geo_context",
        "DYNAMIC",  # handled specially — see build_geo_prompt()
        "species_common",
    ),
]

ALL_PROMPTS = TRAINING_PROMPTS + CUSTOM_PROMPTS


def build_geo_prompt(row: dict) -> list[dict]:
    """Build a prompt that includes geographic context from the row metadata."""
    country = row.get("country_code", "unknown")
    lat = row.get("latitudeDecimal", "")
    lon = row.get("longitudeDecimal", "")
    context_parts = []
    if country and country != "unknown":
        context_parts.append(f"country: {country}")
    if lat and lon:
        context_parts.append(f"coordinates: {lat}, {lon}")
    context = ", ".join(context_parts) if context_parts else "no location data"
    return [
        {"role": "user", "content": f"<Audio><AudioHere></Audio> This recording was made at {context}. What species is vocalizing?"}
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xc_sample(n: int, seed: int = 42) -> list[dict]:
    """Load *n* random xeno-canto validation samples via esp-data.

    Returns a list of row dicts, each containing ``audio`` (np.ndarray),
    ``sample_rate`` (int), and metadata columns.
    """
    config = ChainedDatasetConfig(
        datasets=[
            DatasetConfig(
                dataset_name="xeno-canto",
                split="validation",
                sample_rate=16000,
            ),
        ]
    )
    ds, _ = dataset_from_config(config)
    total = len(ds)
    print(f"Xeno-canto validation set: {total} samples")

    rng = random.Random(seed)
    indices = rng.sample(range(total), min(n, total))

    samples = []
    for i, idx in enumerate(indices):
        try:
            row = ds[idx]
            samples.append(row)
        except Exception as exc:
            print(f"  Warning: failed to load index {idx}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(indices)} samples")

    print(f"Loaded {len(samples)} samples successfully")
    return samples


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

def audio_to_base64(audio: np.ndarray, sr: int) -> str:
    """Encode a numpy audio array as base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_server(
    server_url: str,
    audio_b64: str,
    messages: list[dict],
    max_new_tokens: int = 300,
    timeout: int = 60,
) -> dict:
    """Send a single request to the NatureLM server.

    Returns the parsed JSON response or an error dict.
    """
    payload = {
        "audio_base64": audio_b64,
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "num_beams": 4,
    }
    try:
        resp = requests.post(
            f"{server_url}/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def get_ground_truth(row: dict, gt_spec) -> str | None:
    """Extract ground truth from a row given a spec (column name, callable, or None)."""
    if gt_spec is None:
        return None
    if callable(gt_spec):
        return gt_spec(row)
    val = row.get(gt_spec, "")
    if val is None:
        return ""
    return str(val)


def run_eval(
    server_url: str,
    samples: list[dict],
    output_path: str,
    prompt_filter: str | None = None,
) -> None:
    """Run evaluation across all prompts and samples, writing results to CSV."""
    # Filter prompts if requested
    prompts = ALL_PROMPTS
    if prompt_filter:
        prompts = [(n, m, g) for n, m, g in prompts if prompt_filter in n]
        print(f"Filtered to {len(prompts)} prompts matching '{prompt_filter}'")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_idx",
        "task",
        "prompt_category",
        "species_common",
        "canonical_name",
        "ground_truth",
        "prediction",
        "audio_duration_seconds",
        "latency_seconds",
        "error",
    ]

    total_requests = len(samples) * len(prompts)
    print(f"\nStarting evaluation: {len(samples)} samples x {len(prompts)} prompts = {total_requests} requests")
    print(f"Writing results to {output_path}\n")

    # Check server health first
    try:
        health = requests.get(f"{server_url}/health", timeout=10).json()
        print(f"Server health: {health}")
    except Exception as exc:
        print(f"WARNING: Cannot reach server at {server_url}: {exc}")
        print("Results will contain errors. Start the server and re-run, or pass --server-url.")

    completed = 0
    errors = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_i, row in enumerate(samples):
            audio_b64 = audio_to_base64(row["audio"], row["sample_rate"])

            for task_name, messages_spec, gt_spec in prompts:
                # Handle dynamic prompts
                if messages_spec == "DYNAMIC":
                    messages = build_geo_prompt(row)
                else:
                    messages = messages_spec

                gt = get_ground_truth(row, gt_spec)

                # Skip prompts that need metadata that's missing
                if gt_spec is not None and gt == "":
                    completed += 1
                    continue

                prompt_category = "training" if (task_name, messages_spec, gt_spec) in TRAINING_PROMPTS else "custom"

                t0 = time.time()
                result = query_server(server_url, audio_b64, messages)
                latency = time.time() - t0

                error = result.get("error", "")
                prediction = result.get("response", "")
                if error:
                    errors += 1

                writer.writerow({
                    "sample_idx": sample_i,
                    "task": task_name,
                    "prompt_category": prompt_category,
                    "species_common": row.get("species_common", ""),
                    "canonical_name": row.get("canonical_name", ""),
                    "ground_truth": gt or "",
                    "prediction": prediction,
                    "audio_duration_seconds": result.get("audio_duration_seconds", ""),
                    "latency_seconds": f"{latency:.2f}",
                    "error": error,
                })

                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{total_requests} ({errors} errors)")

    print(f"\nDone. {completed} requests, {errors} errors.")
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate NatureLM-audio on xeno-canto data")
    parser.add_argument("--server-url", required=True, help="NatureLM server URL (e.g. http://host:8001)")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of xeno-canto samples to evaluate")
    parser.add_argument("--output", default="results/naturelm_eval.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--prompt-filter", default=None, help="Only run prompts matching this substring")
    args = parser.parse_args()

    print("Loading xeno-canto samples...")
    samples = load_xc_sample(args.n_samples, seed=args.seed)

    if not samples:
        print("No samples loaded. Exiting.")
        sys.exit(1)

    run_eval(args.server_url, samples, args.output, args.prompt_filter)


if __name__ == "__main__":
    main()

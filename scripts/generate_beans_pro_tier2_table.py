#!/usr/bin/env python3
"""Generate the BEANS-Pro tier-2 LaTeX result table."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "esp-research"
    / "projects"
    / "NatureLM-audio-v1.5"
    / "tables"
    / "beans_pro_tier2.tex"
)

TASKS = [
    ("Bird", "beans_pro_bird_presence/beans_pro/bird-presence/accuracy"),
    ("Mammal", "beans_pro_mammal_presence/beans_pro/mammal-presence/accuracy"),
    ("Insect", "beans_pro_insect_presence/beans_pro/insect-presence/accuracy"),
    ("Amphibian", "beans_pro_amphibian_presence/beans_pro/amphibian-presence/accuracy"),
    ("Alarm Call", "beans_pro_alarm_call_presence/beans_pro/alarm-call-presence/accuracy"),
    ("Flight Call", "beans_pro_flight_call_presence/beans_pro/flight-call-presence/accuracy"),
    ("Behavior", None),
]

RAW_TASK_FILES = [
    (
        "beans_pro_bird_presence/beans_pro/bird-presence/accuracy",
        "beans_pro_bird_presence__beans_pro_bird-presence.jsonl",
    ),
    (
        "beans_pro_mammal_presence/beans_pro/mammal-presence/accuracy",
        "beans_pro_mammal_presence__beans_pro_mammal-presence.jsonl",
    ),
    (
        "beans_pro_insect_presence/beans_pro/insect-presence/accuracy",
        "beans_pro_insect_presence__beans_pro_insect-presence.jsonl",
    ),
    (
        "beans_pro_amphibian_presence/beans_pro/amphibian-presence/accuracy",
        "beans_pro_amphibian_presence__beans_pro_amphibian-presence.jsonl",
    ),
    (
        "beans_pro_alarm_call_presence/beans_pro/alarm-call-presence/accuracy",
        "beans_pro_alarm_call_presence__beans_pro_alarm-call-presence.jsonl",
    ),
    (
        "beans_pro_flight_call_presence/beans_pro/flight-call-presence/accuracy",
        "beans_pro_flight_call_presence__beans_pro_flight-call-presence.jsonl",
    ),
]

MODEL_ROWS = [
    ("Random", "Rand.", None),
    ("GPT-4o-preview", "GPT", None),
    ("Audio-Flamingo-NeXT", "AF-N", None),
    ("Qwen-3-Omni", "Qwen", None),
    ("Gemini-3.1", "Gem.", None),
    ("NatureLM-audio-1.0", "NLM 1.0", None),
    (r"\model", r"\model", "model"),
    (r"\model-semantic", "Sem.", "semantic"),
]


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file."""

    return path.expanduser().read_text(encoding="utf-8")


def _extract_metrics_from_log(text: str) -> dict[str, float]:
    """Extract the chat-task metrics dictionary from a log file."""

    for line in reversed(text.splitlines()):
        if "'metrics':" not in line and '"metrics":' not in line:
            continue
        match = re.search(r"(\{.*\})", line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return {str(key): float(value) for key, value in metrics.items()}
    raise ValueError("Could not find a metrics dictionary in the log file.")


def _load_metrics(path: Path) -> dict[str, float]:
    """Load metrics from a result JSON or a chat-task log file."""

    text = _read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _extract_metrics_from_log(text)

    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if isinstance(metrics, dict):
        return {str(key): float(value) for key, value in metrics.items()}
    if isinstance(payload, dict):
        return {str(key): float(value) for key, value in payload.items()}
    raise ValueError(f"Unsupported metrics file format: {path}")


def _load_raw_rows(path: Path) -> tuple[list[str], list[str]]:
    """Load raw prediction/target strings from a raw predictions JSONL file."""

    predictions: list[str] = []
    targets: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            predictions.append(str(row["prediction"]).strip())
            targets.append(str(row["target"]).strip())
    return predictions, targets


def _load_metrics_from_raw_dir(path: Path) -> dict[str, float]:
    """Compute tier-2 accuracy metrics from a raw prediction directory."""

    metrics: dict[str, float] = {}
    for metric_key, filename in RAW_TASK_FILES:
        raw_path = path.expanduser() / filename
        predictions, targets = _load_raw_rows(raw_path)
        metrics[metric_key] = float(accuracy_score(targets, predictions))
    return metrics


def _format_score(value: float) -> str:
    """Format a metric value as a percentage with one decimal place."""

    return f"{100.0 * value:.1f}"


def _cells_for_row(
    placeholder: str,
    row_key: str | None,
    metrics_by_row: dict[str, dict[str, float]],
) -> list[str]:
    """Return table cells for one model row."""

    if row_key is None or row_key not in metrics_by_row:
        return [placeholder for _display, _metric_key in TASKS]

    metrics = metrics_by_row[row_key]
    return [
        _format_score(metrics[metric_key])
        if metric_key is not None and metric_key in metrics
        else placeholder
        for _display, metric_key in TASKS
    ]


def generate_table(metrics_by_row: dict[str, dict[str, float]]) -> str:
    """Generate the LaTeX table."""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Model",
        r"& Bird",
        r"& Mammal",
        r"& Insect",
        r"& Amphibian",
        r"& Alarm Call",
        r"& Flight Call",
        r"& Behavior \\",
        r"\midrule",
    ]
    for model_name, placeholder, row_key in MODEL_ROWS:
        cells = _cells_for_row(placeholder, row_key, metrics_by_row)
        lines.append(f"{model_name} & {' & '.join(cells)} \\\\")
    lines.extend(
        [
            "",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{BEANS-Pro tier-2 additional tasks. All reported tasks use accuracy.}",
            r"\label{tab:beans-pro-tier2}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-result",
        type=Path,
        help="JSON result or log file for the \\model row.",
    )
    parser.add_argument(
        "--model-raw-dir",
        type=Path,
        help="Raw prediction directory for recomputing the \\model row.",
    )
    parser.add_argument(
        "--semantic-result",
        type=Path,
        help="JSON result or log file for the \\model-semantic row.",
    )
    parser.add_argument(
        "--semantic-raw-dir",
        type=Path,
        help="Raw prediction directory for recomputing the \\model-semantic row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .tex path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the table file."""

    args = parse_args()
    metrics_by_row: dict[str, dict[str, float]] = {}
    if args.model_result is not None:
        metrics_by_row["model"] = _load_metrics(args.model_result)
    if args.model_raw_dir is not None:
        metrics_by_row["model"] = _load_metrics_from_raw_dir(args.model_raw_dir)
    if args.semantic_result is not None:
        metrics_by_row["semantic"] = _load_metrics(args.semantic_result)
    if args.semantic_raw_dir is not None:
        metrics_by_row["semantic"] = _load_metrics_from_raw_dir(args.semantic_raw_dir)

    table = generate_table(metrics_by_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

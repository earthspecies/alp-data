# BEANS-Pro: Evaluation Benchmark for Audio-Language Models

BEANS-Pro is a frozen, pre-computed evaluation benchmark for multimodal audio-language models. Each split is a self-contained JSONL file with pre-built instruction/output pairs; the dataset class loads the JSONL and attaches audio at inference time. No dynamic transforms or prompt templates are applied at evaluation.

## Splits

### Acoustic description matching (original)

| Split | Examples | Task | Output format |
|-------|----------|------|---------------|
| `crow-description` | 200 | 4-choice description matching | A/B/C/D |
| `zebra-description` | 40 | 4-choice description matching | A/B/C/D |

Given an audio clip, the model must select the correct expert acoustic description from four candidates (one correct, three distractors from the same species). Descriptions are sourced verbatim from published bioacoustics papers.

### Mean F0 prediction

| Split | Examples | Task | Output format |
|-------|----------|------|---------------|
| `f0-mean-seen-taxa` | 2,086 | Mean fundamental frequency | e.g. "1240 Hz" |
| `f0-mean-heldout-taxa` | 571 | Mean fundamental frequency | e.g. "330 Hz" |

Given a vocalization, predict the mean F0 rounded to the nearest 10 Hz. Ground truth comes from the F0 Bioacoustic Benchmark (Musikhin et al. 2025).

**Seen taxa** (9): canids, hummingbirds, La Palma chaffinches, lions, little owls, long-billed hermits, monk parakeets, orangutans, Reunion grey white-eyes. Sourced from the `val` split.

**Heldout taxon**: spotted hyenas. Sourced from the `all` split to test cross-taxon generalization.

**Processing:** The `f0_contour` TSV is parsed, NaN values dropped (minimum 2 valid points required), the mean frequency computed, and rounded via `int(round(mean / 10) * 10)`. This matches the `_round_freq` function in the training pipeline (`data/transforms/f0_features.py`).

### Binary taxonomic presence

| Split | Examples | Task | Output format |
|-------|----------|------|---------------|
| `bird-presence` | 3,478 | Bird vocalization detection | Yes/No |
| `mammal-presence` | 468 | Mammal vocalization detection | Yes/No |
| `insect-presence` | 1,176 | Insect sound detection | Yes/No |
| `amphibian-presence` | 1,818 | Amphibian vocalization detection | Yes/No |

Binary yes/no detection: does this recording contain a vocalization from the target taxon?

**Data source:** Xeno-canto `val_unseen` + iNaturalist `val_unseen` splits. These splits are held out from all training data and exclude BEANS-Zero benchmark species.

**Negative construction:** Cross-taxonomic negatives from the same validation pools. For example, bird-presence negatives are recordings of mammals, insects, and amphibians.

**Balance:** All splits are balanced 50/50 by downsampling the majority class (seed=42).

**Audio:** 32 kHz pre-resampled files from both Xeno-canto and iNaturalist. Audio paths are relative to `gs://esp-ml-datasets/`.

**Prompt** (variant 0 from `taxon_presence.yml`):
- Bird: "Is there a bird vocalizing in this recording? Answer Yes or No."
- Mammal: "Does this recording contain mammal vocalizations? Answer Yes or No."
- Insect: "Does this recording contain insect sounds? Answer Yes or No."
- Amphibian: "Is there a frog or amphibian vocalizing in this recording? Answer Yes or No."

### Call-type tasks

| Split | Examples | Task | Output format |
|-------|----------|------|---------------|
| `alarm-call-presence` | 36 | Alarm call binary detection | Yes/No |
| `flight-call-presence` | 192 | Flight call binary detection | Yes/No |
| `call-type-fixed-vocab` | 999 | 5-label multilabel classification | comma-separated labels |

**Data source:** BEANS-Zero call-type split, linked to Xeno-canto metadata via the `beans_zero_call_variants` manifests. Audio paths are relative to `gs://esp-ml-datasets/`.

**Binary presence** asks whether a specific call type is present (balanced 50/50). The instruction fills in the target call type: "Is a {call type} present in this recording? Answer Yes or No."

**Fixed-vocab multilabel** asks the model to identify all vocalization types from a closed set of 5 labels: `alarm call`, `flight call`, `begging call`, `song`, `call`. The `call` label matches only when it appears as a standalone token (not as part of "alarm call", etc.). Output is a comma-separated list.

## JSONL schema

Every row across all splits follows this schema:

```json
{
  "source_dataset": "string (DOI, dataset name, or source identifier)",
  "dataset_name": "string (split name)",
  "output": "string (ground-truth answer)",
  "instruction_text": "string (question without audio tag)",
  "instruction": "string (<Audio><AudioHere></Audio> + question)",
  "task": "string (task identifier)",
  "file_name": "string (audio filename)",
  "license": "string",
  "id": "string (UUID)",
  "metadata": "string (JSON-encoded dict with species, taxon, etc.)",
  "audio_path_original_sample_rate": "string (relative path to audio)"
}
```

## BEANS-Pro Multi-Audio (`beans_pro_multi_audio`)

A separate dataset class for multi-audio evaluation tasks. Each example
has 2+ audio files referenced via the `audios` field, with multiple
`<AudioHere>` placeholders in the prompt.

### Few-shot gibbon call-type detection

| Split | Examples | Balance | Support clips | Task |
|-------|----------|---------|---------------|------|
| `gibbon-fewshot-multipulse` | 740 | 370/370 | 2 | Multiple pulse gibbon call |
| `gibbon-fewshot-singlepulse` | 84 | 42/42 | 2 | Single pulse gibbon call |
| `gibbon-fewshot-duet` | 44 | 22/22 | 2 | Gibbon duet |
| `gibbon-fewshot-tiny` | 24 | 12/12 | 2 | Mixed (pipeline testing) |

Given 2 support clips of a target gibbon call type, determine whether a
query clip contains that call type. Output: `Yes` or `No`.

**Data source:** BEANS-Zero gibbons split (Hainan Gibbons, 18,560 clips at 9.6 kHz). 2 positive clips held out as fixed support examples per call type; remaining positives + balanced negatives form the query set.

**Prompt** (matches DRASDIC `binary_audio` training format):
```
Here are example(s) of a target call type.
<Audio><AudioHere></Audio>
<Audio><AudioHere></Audio>

Does the following recording contain this target call type?
<Audio><AudioHere></Audio>
```

**Audio:** 32 kHz resampled. Each row has 3 audio paths (2 support + 1 query) in `audio_paths`, ordered to match `<AudioHere>` positions.

### Few-shot gibbon 3-way detection

| Split | Examples | Answer space | Support clips | Background env |
|-------|----------|--------------|---------------|----------------|
| `gibbon-fewshot-detection` | 18,554 | `A`, `B`, `C`, `None` | 3 (one per class) | ~50.5% |
| `gibbon-fewshot-detection-balanced` | 868 | `A`, `B`, `C`, `None` | 3 (one per class) | ~51.4% |

Fixed labels:
- `A`: Multiple pulse gibbon call
- `B`: Single pulse gibbon call
- `C`: Gibbon duet

Given support examples for all 3 gibbon call types, determine which of the
above sounds is present in the query recording, if any. Output: `A`, `B`,
`C`, or `None`.

**Data source:** BEANS-Zero gibbons split (Hainan Gibbons, 18,560 clips at
9.6 kHz). 2 clips per call type are held out as the support pool; the
remaining 434 positive clips and 18,120 `None` clips form the full query
set.

**Prompt** (matches the DRASDIC v2 few-shot detection format):
```
Here are examples of 3 sounds.

A: <Audio><AudioHere></Audio>
B: <Audio><AudioHere></Audio>
C: <Audio><AudioHere></Audio>

Which of the above sounds are present in this recording, if any?
<Audio><AudioHere></Audio>
```

Optional background-environment variant:
```
Here are examples of 3 sounds.

A: <Audio><AudioHere></Audio>
B: <Audio><AudioHere></Audio>
C: <Audio><AudioHere></Audio>

Here is the background environment: <Audio><AudioHere></Audio>

Which of the above sounds are present in this recording, if any?
<Audio><AudioHere></Audio>
```

**Audio:** 32 kHz resampled. Each row has 4 or 5 audio paths in
`audio_paths`, ordered to match `<AudioHere>` positions.

### Multi-audio JSONL schema

```json
{
  "id": "string",
  "audio_paths": ["support_1.wav", "support_2.wav", "query.wav"],
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "Yes"}],
  "task": "call_type_binary",
  "source_dataset": "string",
  "dataset_name": "string (split name)",
  "license": "string",
  "metadata": "string (JSON-encoded)",
  "audio_path_original_sample_rate": "string (query audio path)"
}
```

## Statistical baselines

### Binary tasks (all balanced 50/50)

All binary presence and few-shot detection splits are balanced, so:

| Baseline | Accuracy | F1 (macro) |
|----------|----------|------------|
| Random (50/50) | 0.500 | 0.500 |
| Always-Yes or Always-No | 0.500 | 0.333 |

Applies to: `bird-presence`, `mammal-presence`, `insect-presence`, `amphibian-presence`, `alarm-call-presence`, `flight-call-presence`, `gibbon-fewshot-multipulse`, `gibbon-fewshot-singlepulse`, `gibbon-fewshot-duet`, `gibbon-fewshot-tiny`.

### 3-way detection with abstention

| Split | Majority baseline | Macro F1 |
|-------|-------------------|----------|
| `gibbon-fewshot-detection` | Always `None` = 0.9766 accuracy | 0.247 |
| `gibbon-fewshot-detection-balanced` | Always `None` = 0.500 accuracy | 0.167 |

The full split is intentionally dominated by `None`, so the balanced split is
the more informative headline metric for model comparison.

### Mean F0 prediction

| Split | Examples | F0 range | Predict-mean baseline | Predict-median baseline |
|-------|----------|----------|----------------------|------------------------|
| `f0-mean-seen-taxa` | 2,086 | 120-11,700 Hz | MAE=2,434 Hz, MedianAE=2,443 Hz | MAE=2,401 Hz, MedianAE=2,015 Hz |
| `f0-mean-heldout-taxa` | 571 | 150-750 Hz | MAE=48 Hz, MedianAE=37 Hz | MAE=48 Hz, MedianAE=40 Hz |

The seen-taxa split has high variance (hummingbirds ~8-11 kHz vs lions ~120 Hz), making constant prediction a weak baseline. The heldout split (spotted hyenas only, 150-750 Hz) has low variance, so the constant baseline is much stronger.

### Call-type fixed vocabulary (multilabel)

| Baseline | Exact match |
|----------|-------------|
| Random (uniform over 9 observed outputs) | 0.111 |
| Always predict "song" (50.1% of labels) | 0.501 |

The 5-label vocabulary produces 9 distinct output combinations in the data. "song" alone is the majority output.

### Acoustic description matching

| Baseline | Accuracy |
|----------|----------|
| Random (4 choices) | 0.250 |

## Generation scripts

All JSONL files are generated by deterministic scripts and uploaded to GCS:

| Script | Generates |
|--------|-----------|
| `scripts/build_beans_pro_f0_mean.py` | f0-mean-seen-taxa, f0-mean-heldout-taxa |
| `scripts/build_beans_pro_presence.py` | bird/mammal/insect/amphibian-presence, alarm/flight-call-presence, call-type-fixed-vocab |
| `scripts/build_beans_pro_gibbon_fewshot.py` | gibbon-fewshot-multipulse/singlepulse/duet/tiny |
| `scripts/build_beans_pro_gibbon_detection.py` | gibbon-fewshot-detection, gibbon-fewshot-detection-balanced |

Re-running these scripts with the same seed reproduces identical outputs.

## Not yet included

- **Marine mammal presence**: No Watkins validation split exists; only 1 cetacean in iNat val_unseen.
- **Animal presence**: Requires non-animal audio negatives (e.g. AudioSet) with no clear held-out split available.
- **Begging call presence**: Only 14 examples (7 per class) — too small for reliable evaluation.

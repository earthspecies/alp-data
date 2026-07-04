# BEANS-Pro Multi-Audio Splits

This document describes the splits currently registered by
`BeansProMultiAudio` in `esp_data/datasets/beans_pro_multi_audio.py`.

Rows are pre-chat-formatted JSONL records with:

- `audio_paths`: ordered to match `<AudioHere>` placeholders in the prompt
- `messages`: user prompt plus assistant target
- `task` and `template_path`: task family metadata
- `metadata`: JSON-encoded split-specific provenance

## Registered Splits

| Split | Rows | Audio count | Task | Builder |
| --- | ---: | --- | --- | --- |
| `gibbon-fewshot-detection` | 18,554 | 4 or 5 | fixed-option few-shot detection | `scripts/build_beans_pro_gibbon_detection.py` |
| `gibbon-fewshot-detection-balanced` | 868 | 4 or 5 | balanced subset of above | `scripts/build_beans_pro_gibbon_detection.py` |
| `giant-otter-4way` | 500 | 5 | call-type 4-way MCQ | `scripts/build_beans_pro_giant_otter_calltype.py` |
| `dcase-fewshot-detection-balanced` | 3,158 | 5 or 6 | balanced few-shot multi-label detection | `scripts/build_beans_pro_dcase_detection.py` |
| `crow-4way` | 200 | 5 | call-type 4-way MCQ | `scripts/build_beans_pro_crow_zebra_calltype.py` |
| `zebra-4way` | 40 | 5 | call-type 4-way MCQ | `scripts/build_beans_pro_crow_zebra_calltype.py` |
| `unseen-species-4way` | 1,227 | 5 | species MCQ, random confusers | `scripts/build_beans_pro_unseen_species_mcq.py` |

## Prompt Families

### Fixed-Option Few-Shot Detection

Used by `gibbon-fewshot-detection*` and
`dcase-fewshot-detection-balanced`.

```text
Here are examples of N sounds.

A: <Audio><AudioHere></Audio>
B: <Audio><AudioHere></Audio>
...

[optional] Here is the background environment: <Audio><AudioHere></Audio>

Which of the above sounds are present in this recording, if any?
<Audio><AudioHere></Audio>
```

Answers are option labels (`A`, `B`, `C`, `D`), comma-separated label sets
such as `A, C`, or `None`.

### 4-Way Audio MCQ

Used by giant otter, crow/zebra, and the unseen species split.

```text
Here are four call types/species.

A: <Audio><AudioHere></Audio>
B: <Audio><AudioHere></Audio>
C: <Audio><AudioHere></Audio>
D: <Audio><AudioHere></Audio>

Which call type/species best matches the following recording?
<Audio><AudioHere></Audio>
```

Answers are one of `A`, `B`, `C`, or `D`.

## Dataset Families

### Gibbon Few-Shot Detection

Builder: `scripts/build_beans_pro_gibbon_detection.py`.

Source: BEANS-Zero `gibbons_test.jsonl` from Hainan gibbon data.

The builder holds out two support clips for each of three fixed labels:

- `A`: Multiple pulse gibbon call
- `B`: Single pulse gibbon call
- `C`: Gibbon duet

Every row shows the same A/B/C label meanings. The query target may be one of
the three call types or `None`. Half the rows include an optional background
environment clip drawn from the `None` pool.

| Split | Rows | Answer distribution | Background rows | Audio count |
| --- | ---: | --- | ---: | --- |
| `gibbon-fewshot-detection` | 18,554 | A 370, B 42, C 22, None 18,120 | 9,366 | 4 without background, 5 with background |
| `gibbon-fewshot-detection-balanced` | 868 | A 370, B 42, C 22, None 434 | 446 | 4 without background, 5 with background |

### Giant Otter 4-Way Call-Type Matching

Builder: `scripts/build_beans_pro_giant_otter_calltype.py`.

Source: giant otter vocal repertoire annotations
(`giant_otters_annotations_test.csv`). Rows are restricted to call types with
at least five existing audio files.

`giant-otter-4way` samples four distinct call types per row, picks one support
clip for each option, and uses a different clip from the correct type as the
query when possible. Correct labels are balanced by cycling A/B/C/D.

| Split | Rows | Answer distribution | Unique correct call types |
| --- | ---: | --- | ---: |
| `giant-otter-4way` | 500 | 125 per A/B/C/D | 21 |

### DCASE Few-Shot Detection

Builder: `scripts/build_beans_pro_dcase_detection.py`.

Source: BEANS-Zero DCASE Task 5 split.

The registered DCASE split is the balanced variant. The generator uses all
13,688 DCASE rows internally, including `None` and multi-label rows, then
keeps all present rows and samples an equal number of `None` rows.

Support pools are fixed with two clips per sound type, built from all clips
containing that type. Each target row presents four option types: all present
types plus random absent types. `None` rows contain only absent options. Half
the rows include a background environment clip from the `None` pool.

| Split | Rows | Label-count distribution | Background rows | Optionable sound types |
| --- | ---: | --- | ---: | ---: |
| `dcase-fewshot-detection-balanced` | 3,158 | None 1,579; 1-label 1,414; 2-label 153; 3-label 12 | 1,579 | 18 |

### Crow and Zebra Call-Type MCQ

Builder: `scripts/build_beans_pro_crow_zebra_calltype.py`.

These splits are aligned 1:1 with the single-audio BEANS-Pro acoustic
description tasks. The builder reads the existing text-description JSONLs,
parses the four original description options, maps each option back to a call
type, and replaces descriptions with audio exemplars. The target clip and
answer position remain aligned with the single-audio task.

| Split | Rows | Source | Answer distribution | Call types |
| --- | ---: | --- | --- | ---: |
| `crow-4way` | 200 | carrion crow descriptions | 50 per A/B/C/D | 25 |
| `zebra-4way` | 40 | plains zebra descriptions | 10 per A/B/C/D | 4 |

### Unseen Species MCQ

Builder: `scripts/build_beans_pro_unseen_species_mcq.py`.

Source: BEANS-Zero unseen species holdout plus `unseen_xc_mapping.csv`.
The task asks for species matching: four species exemplars are shown, then a
query recording must be matched to the correct species.

For every target clip from species with at least two recordings, one different
recording from the same species is used as the correct option. The other three
options are confuser species. Correct answer position cycles through A/B/C/D
before shuffling rows.

| Split | Rows | Correct species | Confusers | Hard-negative rows |
| --- | ---: | ---: | --- | ---: |
| `unseen-species-4way` | 1,227 | 172 | random held-out species | 0 |

Answer positions are balanced to within one row.

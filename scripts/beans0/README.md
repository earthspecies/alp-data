---
language:
- en
license: various
license_name: "Various (see details below)"
license_link: "LICENSE.md"
license_details: >
  This dataset aggregates multiple component datasets—each with its own license. For example, ESC-50 is CC-BY-NC, RFCX is for academic/research/non-commercial use, CBI is CC-BY-NC-SA, HumBugDB is CC-BY, Enabirds is CC0, HICEAS is free to use without restriction, Watkins is free for personal/academic use, Hainan Gibbons is CC-BY-NC-SA, ZF-Indiv is CC-BY, BirdVox70k_pitch is CC-BY, and several subsets use per-file licenses.
tags:
- audio
- bioacoustics
- benchmark
- zero-shot
- classification
- detection
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language_details:
- en-US
pretty_name: Beans0
size_categories:
- n/a
source_datasets:
- esc50
- Watkins
- CBI
- HumBugDB
- Enabirds
- HICEAS
- RFCX
- Hainan Gibbons
task_categories:
- audio-classification
- audio-detection
- pitch-estimation
- captioning
task_ids:
- classification
- detection
- pitch-estimation
- caption-common
paperswithcode_id: naturelm-audio
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/  # Placeholder – update with actual file paths
  - split: test
    path: data/test/   # Placeholder – update with actual file paths
dataset_info:
  features:
    - name: audio
      dtype: audio
    - name: label
      dtype: string
  config_name: default
  splits:
    - name: train
      num_bytes: n/a
      num_examples: n/a
  download_size: n/a
  dataset_size: n/a
extra_gated_fields: []
extra_gated_prompt: ""
train-eval-index:
- config: default
  task: audio-classification
  task_id: classification
  splits:
    train_split: train
    eval_split: test
  col_mapping: {}
  metrics:
    - type: accuracy
      name: Accuracy
---
Below is a proposed README.md that incorporates details from your dataset_config.json while following the dataset card template. You can adjust file paths and any placeholders as needed.

---

```yaml
---
language:
- en
license: various
license_name: "Various (see details below)"
license_link: "LICENSE.md"
license_details: >
  This dataset aggregates multiple component datasets—each with its own license. For example, ESC-50 is CC-BY-NC, RFCX is for academic/research/non-commercial use, CBI is CC-BY-NC-SA, HumBugDB is CC-BY, Enabirds is CC0, HICEAS is free to use without restriction, Watkins is free for personal/academic use, Hainan Gibbons is CC-BY-NC-SA, ZF-Indiv is CC-BY, BirdVox70k_pitch is CC-BY, and several subsets use per-file licenses.
tags:
- audio
- bioacoustics
- benchmark
- zero-shot
- classification
- detection
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language_details:
- en-US
pretty_name: Beans0
size_categories:
- n/a
source_datasets:
- esc50
- Watkins
- CBI
- HumBugDB
- Enabirds
- HICEAS
- RFCX
- Hainan Gibbons
task_categories:
- audio-classification
- audio-detection
- pitch-estimation
- captioning
task_ids:
- classification
- detection
- pitch-estimation
- caption-common
paperswithcode_id: naturelm-audio
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test/   # Placeholder – update with actual file paths
dataset_info:
  features:
    - name: audio
      dtype: audio
    - name: label
      dtype: string
  config_name: default
  splits:
    - name: train
      num_bytes: n/a
      num_examples: n/a
  download_size: n/a
  dataset_size: n/a
extra_gated_fields: []
extra_gated_prompt: ""
train-eval-index:
- config: default
  task: audio-classification
  task_id: classification
  splits:
    train_split: train
    eval_split: test
  col_mapping: {}
  metrics:
    - type: accuracy
      name: Accuracy
---
```

# Beans0

**Version:** 0.1.0
**Created on:** 2025-02-25
**Creators:**
- M. Hagiwara (masato at earthspecies dot org)
- D. Robinson (david at earthspecies dot org)
- M. Miron (marius at earthspecies dot org)
- S. Keen (sara at earthspecies dot org)
- G. Narula (gagan at earthspecies dot org)
- M. Alizadeh (milad at earthspecies dot org)
- O. Pietquin (olivier at earthspecies dot org)

## Overview

Beans0 is a **benchmark bioacoustic dataset** designed for zero-shot evaluation tasks. Introduced in the paper [NATURELM-AUDIO: AN AUDIO-LANGUAGE FOUNDATION MODEL FOR BIOACOUSTICS](https://arxiv.org/pdf/2411.07186), this dataset aggregates multiple subsets from both established component datasets and newly curated ones. It serves as a valuable resource for research in audio classification, detection, pitch estimation, and captioning within bioacoustics.

## Dataset Composition

Beans0 combines data from several well-known sources. It consists of two main groups:

### Already Part of the BEANS Benchmark
- **ESC-50:** A labeled collection of 2000 environmental audio recordings (5 seconds each) spanning 50 classes. ([CC-BY-NC](http://dx.doi.org/10.1145/2733373.2806390))
- **Watkins:** Marine mammal sound recordings.
- **CBI:** Cornell Birdcall Identification dataset. ([CC-BY-NC-SA](https://www.kaggle.com/competitions/birdsong-recognition/overview))
- **HumBugDB:** A large-scale, multi-species dataset of mosquito sounds. ([CC-BY](https://openreview.net/forum?id=vhjsBtq9OxO))
- **Enabirds:** Bird dawn chorus detection data. ([CC0](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecy.3329))
- **HICEAS:** Marine mammal vocalizations from the Hawaiian Islands.
- **RFCX:** Recordings for bird and frog vocalizations in soundscape recordings (academic/research & non-commercial use).
- **Hainan Gibbons:** Automated detection of Hainan gibbon calls. ([CC-BY-NC-SA](https://doi.org/10.1002/rse2.201))

### Newly Added Subsets
- **Lifestage:** Annotations for classifying animal lifestages (adult, juvenile, nestling).
- **Call-type:** Labels distinguishing between call and song in animal sounds.
- **Unseen-species-cmn / sci / tax:** Tasks for classifying species common names, scientific names, and taxonomic names not seen during training.
- **Unseen-genus-cmn / sci / tax:** Tasks for classifying genus-level names (common, scientific, taxonomic) of unseen species.
- **Unseen-family-cmn / sci / tax:** Tasks for classifying family-level names (common, scientific, taxonomic) of unseen species.
- **Captioning:** English captions for bioacoustic recordings.
- **ZF-Indiv:** A zebra finch dataset with individual and call-type annotations.
- **BirdVox70k_pitch:** A pitch estimation task derived from BirdVox-70k clips.

Each subset carries its own metadata and licensing details.

## Sources

Beans0 was assembled using data from:
- **Xeno-canto**
- **iNaturalist**
- **Animal Sound Archive**
- **Elie et al 2020**
- **ESC-50**
- **RFCX**
- **CBI**
- **HumBugDB**
- **Enabirds**
- **HICEAS**
- **Watkins**
- **Hainan Gibbons**
- **BirdVox70k**

## Tasks and Applications

Beans0 supports various research tasks:
- **Audio Classification:** Identify and categorize animal sounds.
- **Audio Detection:** Detect specific sound events in recordings.
- **Pitch Estimation:** Analyze and estimate pitch in audio signals.
- **Audio Captioning:** Generate natural language descriptions of bioacoustic recordings.

These tasks are particularly useful for exploring zero-shot learning applications in bioacoustics.

## Licensing

Due to its composite nature, Beans0 is subject to multiple licenses. Please see the [license details](#license) above in the YAML header. Individual subsets may have per-file licenses; consult the corresponding documentation for each component.

## Citation

If you use Beans0, please cite the following:

```bibtex
@misc{naturelm-audio,
  title={NATURELM-AUDIO: AN AUDIO-LANGUAGE FOUNDATION MODEL FOR BIOACOUSTICS},
  url={https://arxiv.org/pdf/2411.07186},
  note={Preprint},
  year={2024}
}
```

## How to Use

Beans0 is provided with a default configuration. Data files are organized into splits (e.g., train/test). Adjust the `data_files` paths in your dataset loader configuration as necessary to match your local setup.

## Contact

For questions, comments, or contributions, please contact:
- M. Hagiwara (masato at earthspecies dot org)
- D. Robinson (david at earthspecies dot org)
- M. Miron (marius at earthspecies dot org)
- S. Keen (sara at earthspecies dot org)
- G. Narula (gagan at earthspecies dot org)
- M. Alizadeh (milad at earthspecies dot org)
- O. Pietquin (olivier at earthspecies dot org)

---

*Happy benchmarking and exploring the rich field of bioacoustics with Beans0!*
```

---

Feel free to tweak the wording, update placeholder file paths, or add any further details as needed.

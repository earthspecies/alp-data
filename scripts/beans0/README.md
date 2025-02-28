---
license: other
task_categories:
- audio-classification
language:
- en
tags:
- biology
pretty_name: Beans0
size_categories:
- 100K<n<1M
---

# Beans0

**Version:** 0.1.0
**Created on:** 2025-02-25
**Creators:**
- EarthSpeciesProject (https://www.earthspecies.org)

## Overview

Beans0 is a **benchmark bioacoustic dataset** designed for zero-shot bioacoustic evaluation tasks. Introduced in the paper [NATURELM-AUDIO: AN AUDIO-LANGUAGE FOUNDATION MODEL FOR BIOACOUSTICS](https://arxiv.org/pdf/2411.07186), this dataset aggregates multiple subsets from both established component datasets and newly curated ones. It is a benchmark dataset only to be used to evaluate audio-text multimodal models that accept a bioacoustic audio input (query audio) and a text prompt (e.g. 'What species is in this audio?') and generate text as output (e.g. "Taeniopygia guttata").

<div class="alert alert-block alert-warning">
<b>NOTE:</b> Some of the examples (27,899) were sourced from the iNaturalist database
  https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7
  At this point we are not able to share the audio from the examples, just the metadata.
</div>

## Dataset Composition

Beans0 combines data from several well-known sources. There are total of 113,679 samples (examples). It consists of two main groups:

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

## Data Fields
The following fields are present in each example:
- **source_dataset** (str): One of the source datasets mentioned above
- **id** (str): Sample uuid.
- **created_at** (str): Sample creation datetime in utc
- **license** (str): Each sample can have a different license
- **file_name** (str): Sample file_name
- **instruction** (str): A prompt for created with a placeholder for audio tokens. E.g. '<Audio><AudioHere></Audio> What is the scientific name for the focal species in the audio?'
- **output** (str): The expected output from the model
- **task** (str): The task type e.g. classification / detection / pitch estimation / captioning.
- **dataset_name** (str): Names corresponding to the evaluation tasks, e.g. 'esc50' or 'unseen-family-sci'.

## Licensing

Due to its composite nature, Beans0 is subject to multiple licenses. Individual subsets may have per-file licenses; consult the corresponding documentation for each component.

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

Beans0 is provided with a default configuration. Data files are belong to a sing

## Contact

For questions, comments, or contributions, please contact:
- M. Hagiwara (masato at earthspecies dot org)
- D. Robinson (david at earthspecies dot org)
- M. Miron (marius at earthspecies dot org)
- G. Narula (gagan at earthspecies dot org)
- M. Alizadeh (milad at earthspecies dot org)

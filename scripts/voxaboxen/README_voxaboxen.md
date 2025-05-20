---
license: other
task_categories:
- audio-classification
language:
- en
tags:
- biology
- bioacoustics
- sound event detection
- birds
pretty_name: VoxaboxenData
size_categories:
- 100K<n<1M

annotations_creators:
  - crowdsourced
  - machine-generated
language_creators:
  - crowdsourced
  - machine-generated
language:
  - en
license:
  - cc-by-4.0
multilinguality:
  - monolingual
task_ids: []
configs:
  - config_name: Anuraset
    features:
      - name: audio
        dtype:
          audio:
            sample_rate: 16000
            decode: false
      - name: source_dataset
        dtype: string
      - name: file_name
        dtype: string
      - name: id
        dtype: string
      - name: selection_table
        dtype: string
      - name: metadata
        dtype: string
    data_files:
      - split: train
        path: train/shard*
      - split: val
        path: val/shard*
      - split: test
        path: test/shard*


---

# Overview

Detecting the sounds produced by animals is the foundation of bioacoustics research. This task must often be performed using noisy recordings that include many overlapping sounds from multiple individuals. Identifying each individual acoustic unit is necessary for a diversity of tasks, including species recognition and population estimation, which are critical to research on topics such as ecology and conservation.

This dataset consists of eight real component datasets for evaluating bioacoustic sound event detection performance, as well as six synthetic component datasets. Each dataset consists of several audio recordings. Annotations consist of the start- and stop-times of each event of interest, as well a class label.

# Component datasets

This dataset consists of eight real component datasets which are used to evaluate bioacoustic sound event detection performance. Seven of these datasets are derived from data that appeared in previous publications. For the license and original citation for each component dataset, please see `LICENSE.txt`. If you are using a component dataset, please cite our paper (see below) in addition to the original work. Dataset characteristics are summarized below:

|Dataset | N. Files (train/val/test) | N. Classes | Dur. (hr) (train/val/test) | N. Events (train/val/test) | Mean event dur. (sec) | Location | Taxa |
|--- | --- | --- | --- | --- | --- | --- | --- |
|Anuraset (AnSet) | 967/322/323 | 10 | 16.09/5.37/5.37 | 4279/1893/1635 | 6.23 | Brazil | Anura |
|BirdVox-10h  (BV10) | 5/5/5 | 1 | 6.00/2.00/2.00 | 4196/1064/3764 | 0.15 |  New York, USA | Passeriformes |
|Hawaii Birds  (HawB) | 379/126/130 | 9 | 30.48/10.05/10.35 | 33372/11209/11132 | 1.11 | Hawaii, USA | Aves |
|Humpback  (HbW) | 388/125/129 | 1 | 8.08/2.60/2.69 | 2952/959/865 | 0.99 | North Pacific Ocean | Megaptera novaeangliae |
|Katydids  (Katy)| 16/5/6 | 1 | 2.66/0.83/1.00 | 7434/1550/2977 | 0.17  | Panam\'a | Tettigoniidae |
|Meerkat (MT)  | 2/2/2 | 1 | 0.76/0.25/0.25 | 773/269/252 | 0.15  | South Africa | Suricata suricatta |
|Powdermill  (Pow)   | 44/14/19 | 6 | 3.67/1.17/1.58 | 5138/2276/2505 | 1.11  | Pennsylvania, USA | Passeriformes |
|Overlapping  Zebra Finch  (OZF) | 46/6/13 | 1 | 0.77/0.10/0.22 | 5514/1246/1744 | 0.11 | Laboratory | Taeniopygia  castanotis |


This dataset also consists of six synthetic component datasets Overlapping Zebra Finch Synthetic (OZF Synthetic) `x`, where `x` can be any of `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. The value of `x` is the ratio of the number of overlapping call pairs, to the number of calls. Each of these synthetic component datasets has the same characteristics. These were designed to mirror those of the real OZF dataset:

|Dataset | N. Files (train/val/test) | N. Classes | Dur. (hr) (train/val/test) | N. Events (train/val/test) | Mean event dur. (sec) | Location | Taxa |
|--- | --- | --- | --- | --- | --- | --- | --- |
|OZF Synthetic `x` | 65 | 1 | 1.08 | 8504 | 0.13 | Synthetic | Taeniopygia  castanotis |

# Example code for loading the data

```python
from datasets import load_dataset

dataset = load_dataset("EarthSpeciesProject/VoxaboxenData", split="train")

```

# Dataset details

## Data Fields
The following fields are present in each example:
- `source_dataset` (str): One of the source datasets mentioned above
- `audio` (Audio): The audio data in float32 format.
- `id` (str): Sample uuid.
- `metadata` (str): Each sample can have some extra data such as the url of the original audio, the recordist, duration,
sample_rate and taxonomic information such as family, genus, species, common name, etc. The metadata is a JSON string.
- `file_name` (str): Sample file_name
- `instruction` (str): A prompt (a query) corresponding to the audio for your audio-text model with a placeholder for audio tokens. E.g. '<Audio><AudioHere></Audio> What is the scientific name for the focal species in the audio?'
- `instruction_text` (str): Same as **instruction** but without the placeholder for audio tokens.
- `output` (str): The expected output from the model
- `task` (str): The task type e.g. 'taxonomic-classification', 'caption-common', 'lifestage', 'speech-nspeakers'
- `license` (str): The license of the dataset. For example, 'CC-BY-NC' or 'CC0'.

# Associated papers and code

Please cite our [paper](LINK) as:

{}

The code associated with the paper can be found [here](LINK).

If you use any of the datasets besides OZF and OZF synthetic, please also cite the original work (found in `LICENSE.txt`).

#!/usr/bin/env python3
"""Generate LaTeX tables and BibTeX for Stage 2 training config.

Parses stage2_train_v1.yml to extract (dataset, template_name) pairs,
then reads prompt templates to build:
  - tables_stage2.tex  (two longtables: by-dataset and by-task)
  - datasets.bib       (BibTeX citations for every source dataset)
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "esp-research" / "projects" / "NatureLM-audio-v1.5"
DATA_CONFIG = RESEARCH / "configs" / "datasets" / "stage2_train_v1.yml"
PROMPTS_DIR = RESEARCH / "configs" / "prompts"
OUTPUT_DIR = ROOT / "esp-research" / "projects" / "NatureLM-audio-v1.5"

DATASET_DISPLAY = {
    "xeno-canto": "Xeno-Canto",
    "inaturalist": "iNaturalist",
    "watkins": "Watkins",
    "animal-sound-archive": "Animal Sound Archive",
    "audioset": "AudioSet",
    "wabad": "WABAD",
    "birdset": "BirdSet",
    "dclde2026": "DCLDE 2026",
    "superwhale_detection": "SuperWhales",
    "audio_skills_xl": "AudioSkillsXL",
    "f0_bioacoustic": r"F0 Bioacoustic",
}

DS_ORDER = [
    "xeno-canto",
    "inaturalist",
    "watkins",
    "animal-sound-archive",
    "audioset",
    "wabad",
    "birdset",
    "dclde2026",
    "superwhale_detection",
    "audio_skills_xl",
    "f0_bioacoustic",
]

DATASET_CITE_KEY = {
    "xeno-canto": "xenocanto",
    "inaturalist": "inaturalist",
    "watkins": "watkins2018",
    "animal-sound-archive": "tierstimmenarchiv",
    "audioset": "gemmeke2017audioset",
    "wabad": "wabad",
    "birdset": "rauch2024birdset",
    "dclde2026": "palmer2025dclde",
    "superwhale_detection": "superwhales",
    "audio_skills_xl": "audioskills",
    "f0_bioacoustic": "musikhin2025f0",
}

DATASET_DESCRIPTION = {
    "xeno-canto": "Community bird/amphibian/insect/mammal sound archive",
    "inaturalist": "Multi-taxon citizen-science biodiversity platform",
    "watkins": "Marine mammal sound database (5$\\times$ upsampled)",
    "animal-sound-archive": "Tierstimmenarchiv, Museum f\\\"ur Naturkunde Berlin",
    "audioset": "General-domain audio events (Google)",
    "wabad": "Strongly-annotated bird soundscapes",
    "birdset": "Multi-task avian bioacoustics benchmark (SSW split)",
    "dclde2026": "Multi-species cetacean passive acoustic monitoring",
    "superwhale_detection": "Aggregated marine mammal detection datasets",
    "audio_skills_xl": "Audio understanding benchmark (counting QA, MCQ)",
    "f0_bioacoustic": "Fundamental frequency contours from animal vocalizations",
}

# Placeholder: set to None to emit "---" in the table.
# Fill in actual counts when available.
DATASET_NUM_RECORDINGS: dict[str, int | None] = {
    "xeno-canto": None,
    "inaturalist": None,
    "watkins": None,
    "animal-sound-archive": None,
    "audioset": None,
    "wabad": None,
    "birdset": None,
    "dclde2026": None,
    "superwhale_detection": None,
    "audio_skills_xl": None,
    "f0_bioacoustic": None,
}

DATASET_HOURS: dict[str, float | None] = {
    "xeno-canto": None,
    "inaturalist": None,
    "watkins": None,
    "animal-sound-archive": None,
    "audioset": None,
    "wabad": None,
    "birdset": None,
    "dclde2026": None,
    "superwhale_detection": None,
    "audio_skills_xl": None,
    "f0_bioacoustic": None,
}

TASK_DISPLAY: dict[str, str] = {
    "species_common": "Species ID (common name)",
    "species_scientific": "Species ID (scientific name)",
    "species_common_with_context": "Species ID + context (common)",
    "species_scientific_with_context": "Species ID + context (scientific)",
    "taxonomic_name_with_context": "Full taxonomy + context",
    "genus": "Genus classification",
    "family": "Family classification",
    "order": "Order classification",
    "taxonomic_name": "Full taxonomy classification",
    "call_type": "Call type classification",
    "call_type_with_species": "Call type (species-conditioned)",
    "call_or_song": "Call vs.~song (binary)",
    "alarm_call_presence": "Alarm call presence (binary)",
    "alarm_call_presence_with_species": "Alarm call + species (binary)",
    "alarm_call_3way": "Alarm / other call / song (3-way)",
    "call_types_present_multilabel": "Call types present (multi-label)",
    "call_type_fixed_vocab": "Call type (fixed 5-label vocab)",
    "fine_call_type": "Fine-grained call type",
    "fine_call_type_with_species": "Fine call type (species-cond.)",
    "life_stage": "Life stage classification",
    "life_stage_with_species": "Life stage (species-conditioned)",
    "life_stage_binary": "Adult vs.~juvenile (binary)",
    "sex": "Sex prediction",
    "sex_with_species": "Sex prediction (species-cond.)",
    "species_and_behavior": "Species + behavior (common)",
    "species_scientific_and_behavior": "Species + behavior (scientific)",
    "species_and_fine_call_type": "Species + fine call type",
    "species_and_lifestage": "Species + life stage (common)",
    "species_scientific_and_lifestage": "Species + life stage (scientific)",
    "species_behavior_lifestage": "Species + behavior + life stage",
    "detection_common": "Open detection (common name)",
    "detection_scientific": "Open detection (scientific name)",
    "detection_taxonomic": "Open detection (full taxonomy)",
    "species_common_options": "Species options (common, classif.)",
    "species_common_detection": "Species options (common, detect.)",
    "species_scientific_options": "Species options (scientific, classif.)",
    "species_scientific_detection": "Species options (scientific, detect.)",
    "species_common_context_options": "Context + options (common, classif.)",
    "species_common_context_detection": "Context + options (common, detect.)",
    "species_scientific_context_options": "Context + options (scientific, classif.)",
    "species_scientific_context_detection": "Context + options (scientific, detect.)",
    "genus_options": "Genus options (classification)",
    "genus_detection": "Genus options (detection)",
    "family_options": "Family options (classification)",
    "family_detection": "Family options (detection)",
    "taxonomic_options": "Taxonomic options (classification)",
    "taxonomic_detection": "Taxonomic options (detection)",
    "multilabel_species": "Multi-label species (scientific)",
    "multilabel_species_common": "Multi-label species (common)",
    "multilabel_species_count": "Multi-label species + count",
    "multilabel_species_soundscape": "Multi-label species (soundscape)",
    "general_caption": "Audio captioning (general)",
    "xc_caption": "Bioacoustic captioning (XC synth.)",
    "inat_caption_synth": "Bioacoustic captioning (iNat synth.)",
    "inat_caption_simple": "Bioacoustic captioning (iNat short)",
    "inat_caption_scientific_simple": "Bioacoustic captioning (iNat sci.)",
    "ecotype": "Killer whale ecotype",
    "call_count_per_species": "Call count per species",
    "total_call_count": "Total call count",
    "temporal_species_order": "Temporal species ordering",
    "frequency_range": "Frequency range prediction",
    "species_frequency_ranges": "Per-species frequency ranges",
    "species_summary": "Species summary (count + freq.)",
    "audioset_options": "AudioSet non-bio multilabel detection",
    "f0_summary": "F0 summary statistics",
    "f0_species": "F0 species-conditioned description",
    "bird_presence": "Bird presence (binary)",
    "marine_mammal_presence": "Marine mammal presence (binary)",
    "mammal_presence": "Mammal presence (binary)",
    "insect_presence": "Insect presence (binary)",
    "amphibian_presence": "Amphibian presence (binary)",
    "animal_presence": "Animal presence (binary)",
    "counting_qa": "Sound event counting QA",
    "multiple_choice_qa": "Multiple-choice sound QA",
}

EXAMPLE_PROMPTS: dict[str, tuple[str, str]] = {
    "species_common": (
        "What species is vocalizing in this audio recording? Common name?",
        "\\{\\{species\\_common\\}\\}",
    ),
    "species_scientific": (
        "What is the scientific name of the focal species in the audio?",
        "\\{\\{canonical\\_name\\}\\}",
    ),
    "species_common_with_context": (
        "Given the context: `\\{\\{context\\}\\}', what is the common name for the focal species?",
        "\\{\\{species\\_common\\}\\}",
    ),
    "species_scientific_with_context": (
        "Given the context: `\\{\\{context\\}\\}', what is the scientific name for the focal species?",
        "\\{\\{canonical\\_name\\}\\}",
    ),
    "taxonomic_name_with_context": (
        "Given the context: `\\{\\{context\\}\\}', provide the full taxonomic classification.",
        "\\{\\{phylum\\}\\} \\{\\{class\\}\\} \\{\\{order\\}\\} \\{\\{family\\}\\} \\{\\{canonical\\_name\\}\\}",
    ),
    "genus": (
        "What is the genus of the focal species in the audio?",
        "\\{\\{genus\\}\\}",
    ),
    "family": (
        "What is the family of the focal species in the audio?",
        "\\{\\{family\\}\\}",
    ),
    "order": (
        "What is the order of the focal species in the audio?",
        "\\{\\{order\\}\\}",
    ),
    "taxonomic_name": (
        "Provide the full taxonomic classification for the species, starting with the phylum.",
        "\\{\\{phylum\\}\\} \\{\\{class\\}\\} \\{\\{order\\}\\} \\{\\{family\\}\\} \\{\\{canonical\\_name\\}\\}",
    ),
    "call_type": (
        "What type of vocalization or call is this?",
        "\\{\\{behavior\\}\\}",
    ),
    "call_type_with_species": (
        "What type of call is the \\{\\{species\\_common\\}\\} making?",
        "\\{\\{behavior\\}\\}",
    ),
    "call_or_song": (
        "Is this a call or a song?",
        "\\{\\{behavior\\}\\}",
    ),
    "alarm_call_presence": (
        "Is an alarm call present in this recording? Answer Yes or No.",
        "\\{\\{alarm\\_present\\}\\}",
    ),
    "alarm_call_presence_with_species": (
        "Is the \\{\\{species\\_common\\}\\} making an alarm call? Yes or No.",
        "\\{\\{alarm\\_present\\}\\}",
    ),
    "alarm_call_3way": (
        "Classify this recording as one of: alarm, other\\_call, song.",
        "\\{\\{alarm\\_call\\_3way\\_label\\}\\}",
    ),
    "call_types_present_multilabel": (
        "List all call types present as a comma-separated list.",
        "\\{\\{behavior\\}\\}",
    ),
    "call_type_fixed_vocab": (
        "Which are present? alarm call, flight call, begging call, song, call.",
        "\\{\\{behavior\\_fixed\\_vocab\\}\\}",
    ),
    "fine_call_type": (
        "What are the fine-grained vocalization types in this recording?",
        "\\{\\{fine\\_call\\_type\\}\\}",
    ),
    "fine_call_type_with_species": (
        "What fine-grained vocalization types is the \\{\\{species\\_common\\}\\} producing?",
        "\\{\\{fine\\_call\\_type\\}\\}",
    ),
    "life_stage": (
        "What life stage is the animal in this recording?",
        "\\{\\{lifeStage\\}\\}",
    ),
    "life_stage_with_species": (
        "What is the life stage of the \\{\\{species\\_common\\}\\}?",
        "\\{\\{lifeStage\\}\\}",
    ),
    "life_stage_binary": (
        "Is the focal species an adult or juvenile?",
        "\\{\\{lifeStage\\}\\}",
    ),
    "sex": (
        "Is the animal vocalizing male or female?",
        "\\{\\{sex\\}\\}",
    ),
    "sex_with_species": (
        "What is the sex of the \\{\\{species\\_common\\}\\}?",
        "\\{\\{sex\\}\\}",
    ),
    "species_and_behavior": (
        "What species is vocalizing and what type of vocalization?",
        "\\{\\{species\\_common\\}\\}, \\{\\{behavior\\}\\}",
    ),
    "species_scientific_and_behavior": (
        "Give the scientific name and vocalization type.",
        "\\{\\{canonical\\_name\\}\\}, \\{\\{behavior\\}\\}",
    ),
    "species_and_fine_call_type": (
        "Identify the species and list the detailed vocalization types.",
        "\\{\\{species\\_common\\}\\}, \\{\\{fine\\_call\\_type\\}\\}",
    ),
    "species_and_lifestage": (
        "What species is in this recording and what is its life stage?",
        "\\{\\{species\\_common\\}\\}, \\{\\{lifeStage\\}\\}",
    ),
    "species_scientific_and_lifestage": (
        "Give the scientific name and life stage.",
        "\\{\\{canonical\\_name\\}\\}, \\{\\{lifeStage\\}\\}",
    ),
    "species_behavior_lifestage": (
        "Identify the species, call type, and life stage.",
        "\\{\\{species\\_common\\}\\}, \\{\\{behavior\\}\\}, \\{\\{lifeStage\\}\\}",
    ),
    "detection_common": (
        "Identify any species present by their common name.",
        "\\{\\{species\\_common\\}\\}",
    ),
    "detection_scientific": (
        "Identify any species present by their scientific name.",
        "\\{\\{canonical\\_name\\}\\}",
    ),
    "detection_taxonomic": (
        "Provide the full taxonomic classification for any species heard.",
        "\\{\\{phylum\\}\\} \\{\\{class\\}\\} ... \\{\\{canonical\\_name\\}\\}",
    ),
    "species_common_options": (
        "Which of these is the focal species? Options: \\{\\{species\\_choices\\}\\}",
        "\\{\\{species\\_common\\}\\}",
    ),
    "species_common_detection": (
        "Which of these species, if any, are present? \\{\\{species\\_choices\\}\\}",
        "\\{\\{species\\_common\\}\\} or None",
    ),
    "species_scientific_options": (
        "Which of these species (scientific name) is in the audio? \\{\\{species\\_choices\\}\\}",
        "\\{\\{canonical\\_name\\}\\}",
    ),
    "species_scientific_detection": (
        "Which of these species, if any, are present? \\{\\{species\\_choices\\}\\}",
        "\\{\\{canonical\\_name\\}\\} or None",
    ),
    "species_common_context_options": (
        "Context: \\{\\{context\\}\\}. Which species? \\{\\{species\\_choices\\}\\}",
        "\\{\\{species\\_common\\}\\}",
    ),
    "species_common_context_detection": (
        "Context: \\{\\{context\\}\\}. Which, if any? \\{\\{species\\_choices\\}\\}",
        "\\{\\{species\\_common\\}\\} or None",
    ),
    "species_scientific_context_options": (
        "Context: \\{\\{context\\}\\}. Which species (scientific)? \\{\\{species\\_choices\\}\\}",
        "\\{\\{canonical\\_name\\}\\}",
    ),
    "species_scientific_context_detection": (
        "Context: \\{\\{context\\}\\}. Which, if any (scientific)? \\{\\{species\\_choices\\}\\}",
        "\\{\\{canonical\\_name\\}\\} or None",
    ),
    "genus_options": (
        "Which of these genera matches the species? \\{\\{species\\_choices\\}\\}",
        "\\{\\{genus\\}\\}",
    ),
    "genus_detection": (
        "Which of these genera, if any, matches? \\{\\{species\\_choices\\}\\}",
        "\\{\\{genus\\}\\} or None",
    ),
    "family_options": (
        "Which of these families matches the species? \\{\\{species\\_choices\\}\\}",
        "\\{\\{family\\}\\}",
    ),
    "family_detection": (
        "Which of these families, if any, matches? \\{\\{species\\_choices\\}\\}",
        "\\{\\{family\\}\\} or None",
    ),
    "taxonomic_options": (
        "Which taxonomic classification matches? \\{\\{species\\_choices\\}\\}",
        "\\{\\{taxonomic\\_name\\}\\}",
    ),
    "taxonomic_detection": (
        "Which classification, if any, matches? \\{\\{species\\_choices\\}\\}",
        "\\{\\{taxonomic\\_name\\}\\} or None",
    ),
    "multilabel_species": (
        "List the scientific names of all species vocalizing in this clip.",
        "\\{\\{species\\_list\\}\\} or None",
    ),
    "multilabel_species_common": (
        "List the common names of all species vocalizing in this clip.",
        "\\{\\{species\\_list\\}\\} or None",
    ),
    "multilabel_species_count": (
        "How many species are vocalizing, and what are they (scientific)?",
        "\\{\\{species\\_count\\}\\}: \\{\\{species\\_list\\}\\}",
    ),
    "multilabel_species_soundscape": (
        "List the scientific names of all species in this soundscape.",
        "\\{\\{species\\_list\\}\\} or None",
    ),
    "general_caption": (
        "Caption this audio with a rich, detailed description.",
        "\\{\\{audiosetcaps\\_caption\\}\\}",
    ),
    "xc_caption": (
        "Caption the audio, using common names for any animal species.",
        "\\{\\{xc\\_caption\\}\\}",
    ),
    "inat_caption_synth": (
        "Caption the audio, using common names for any animal species.",
        "\\{\\{inat\\_caption\\_synth\\}\\}",
    ),
    "inat_caption_simple": (
        "Give a short caption for this recording, using common names.",
        "\\{\\{caption\\}\\}",
    ),
    "inat_caption_scientific_simple": (
        "Give a short caption, using scientific names for any animal species.",
        "\\{\\{caption2\\}\\}",
    ),
    "ecotype": (
        "What killer whale ecotype(s) are vocalizing? Use abbreviations.",
        "\\{\\{species\\_list\\}\\}",
    ),
    "call_count_per_species": (
        "How many calls from each species? Give scientific names.",
        "\\{\\{call\\_counts\\}\\}",
    ),
    "total_call_count": (
        "How many individual vocalizations can you detect in this audio?",
        "\\{\\{total\\_call\\_count\\}\\}",
    ),
    "temporal_species_order": (
        "List the species in the order they first vocalize (scientific).",
        "\\{\\{temporal\\_species\\_order\\}\\}",
    ),
    "frequency_range": (
        "What is the overall frequency range of the vocalizations?",
        "\\{\\{frequency\\_range\\}\\}",
    ),
    "species_frequency_ranges": (
        "What is the frequency range of each species' vocalizations?",
        "\\{\\{species\\_frequency\\_ranges\\}\\}",
    ),
    "species_summary": (
        "Describe each species with call counts and frequency ranges.",
        "\\{\\{species\\_summary\\}\\}",
    ),
    "audioset_options": (
        "Which of these non-animal sounds are present? \\{\\{audioset\\_option\\_choices\\}\\}",
        "\\{\\{audioset\\_option\\_targets\\}\\}",
    ),
    "f0_summary": (
        "What is the mean fundamental frequency of this vocalization?",
        "\\{\\{f0\\_mean\\}\\}",
    ),
    "f0_species": (
        "Identify the species and describe the F0 range and duration.",
        "\\{\\{canonical\\_name\\}\\}, F0: \\{\\{f0\\_range\\}\\}, mean \\{\\{f0\\_mean\\}\\}",
    ),
    "bird_presence": (
        "Is there a bird vocalizing in this recording? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "marine_mammal_presence": (
        "Are there whale or dolphin sounds in this recording? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "mammal_presence": (
        "Does this recording contain mammal vocalizations? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "insect_presence": (
        "Does this recording contain insect sounds? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "amphibian_presence": (
        "Does this recording contain amphibian vocalizations? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "animal_presence": (
        "Does this recording contain animal vocalizations? Yes or No.",
        "\\{\\{taxon\\_present\\}\\}",
    ),
    "counting_qa": (
        "How many [sound events] can you hear? (pre-formatted QA)",
        "[count]",
    ),
    "multiple_choice_qa": (
        "Which sound source produced this audio? (A) ... (B) ... (pre-formatted)",
        "[selected option]",
    ),
}


def _tex_escape(s: str) -> str:
    """Minimal LaTeX escaping for text inside table cells."""
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    return s


def parse_data_config(path: Path) -> list[tuple[str, str]]:
    """Extract active (dataset_name, template_name) pairs from config."""
    raw = path.read_text()

    # Remove commented lines before parsing
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    data = yaml.safe_load(cleaned)
    pairs: list[tuple[str, str]] = []
    for entry in data["chain"]["datasets"]:
        ds = entry["dataset_name"]
        transforms = entry.get("transformations", [])
        for t in transforms:
            if t.get("type") == "chat":
                tpl = t["template_name"]
                pairs.append((ds, tpl))
                break
        else:
            # AudioSkillsXL entries have no chat transform; use set_columns task
            for t in transforms:
                if t.get("type") == "set_columns":
                    task_val = t.get("columns", {}).get("task")
                    if task_val:
                        pairs.append((ds, task_val))
                        break
    return pairs


def build_dataset_task_map(
    pairs: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Map dataset -> ordered unique list of template names."""
    result: dict[str, list[str]] = defaultdict(list)
    for ds, tpl in pairs:
        if tpl not in result[ds]:
            result[ds].append(tpl)
    return dict(result)


def build_task_dataset_map(
    pairs: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Map template_name -> ordered unique list of datasets."""
    result: dict[str, list[str]] = defaultdict(list)
    for ds, tpl in pairs:
        if ds not in result[tpl]:
            result[tpl].append(ds)
    return dict(result)


def _fmt_count(v: int | None) -> str:
    if v is None:
        return "---"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return str(v)


def _fmt_hours(v: float | None) -> str:
    if v is None:
        return "---"
    if v >= 1_000:
        return f"{v / 1_000:.1f}k"
    return f"{v:.0f}"


def generate_dataset_table(
    ds_task_map: dict[str, list[str]],
) -> str:
    """Generate the dataset overview table with recording counts and hours."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Source datasets used in Stage~2 training.}",
        r"\label{tab:dataset-overview}",
        r"\footnotesize",
        r"\begin{tabular}{@{}l l r r l@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Description} & \textbf{Rec.} & \textbf{Hours} & \\",
        r"\midrule",
    ]
    for ds in DS_ORDER:
        if ds not in ds_task_map:
            continue
        display = DATASET_DISPLAY.get(ds, ds)
        desc = DATASET_DESCRIPTION.get(ds, "")
        n_rec = _fmt_count(DATASET_NUM_RECORDINGS.get(ds))
        hours = _fmt_hours(DATASET_HOURS.get(ds))
        cite_key = DATASET_CITE_KEY.get(ds, "")
        cite = f"\\citep{{{cite_key}}}" if cite_key else ""
        lines.append(f"  {display} & {desc} & {n_rec} & {hours} & {cite} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_table1(
    ds_task_map: dict[str, list[str]],
) -> str:
    """Generate Table 1: by dataset (tasks with example prompts)."""
    hdr = (
        r"\textbf{Dataset} & \textbf{Task} "
        r"& \textbf{Example Prompt} & \textbf{Example Response} \\"
    )
    lines = [
        r"\begin{longtable}{@{}p{1.6cm} p{2.8cm} p{4.6cm} p{3.5cm}@{}}",
        r"\caption{Training tasks by dataset, with a representative prompt and response template.}",
        r"\label{tab:tasks-by-dataset} \\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{4}{r@{}}{\footnotesize\itshape Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for ds in DS_ORDER:
        tasks = ds_task_map.get(ds, [])
        if not tasks:
            continue
        ds_display = DATASET_DISPLAY.get(ds, ds)
        for i, tpl in enumerate(tasks):
            task_name = TASK_DISPLAY.get(tpl, _tex_escape(tpl))
            prompt, response = EXAMPLE_PROMPTS.get(tpl, (_tex_escape(tpl), "---"))
            ds_cell = f"\\textbf{{{ds_display}}}" if i == 0 else ""
            lines.append(
                f"  {ds_cell} & \\footnotesize {task_name} "
                f"& \\footnotesize {prompt} "
                f"& \\footnotesize \\texttt{{{response}}} \\\\"
            )
        lines.append(r"  \addlinespace[3pt]\midrule\addlinespace[2pt]")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_table2(
    task_ds_map: dict[str, list[str]],
) -> str:
    """Generate Table 2: by task."""
    task_order = list(TASK_DISPLAY.keys())
    seen: set[str] = set()
    hdr = (
        r"\textbf{Task} & \textbf{Example Prompt} & \textbf{Datasets} \\"
    )
    ncols = "3"

    lines = [
        r"\begin{longtable}{@{}p{2.8cm} p{4.8cm} p{4.9cm}@{}}",
        r"\caption{Datasets used for each training task, with a representative prompt.}",
        r"\label{tab:datasets-by-task} \\",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        hdr,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{ncols}}}{{r@{{}}}}{{\footnotesize\itshape Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    def _row(tpl: str, datasets: list[str]) -> str:
        task_name = TASK_DISPLAY.get(tpl, _tex_escape(tpl))
        prompt, _ = EXAMPLE_PROMPTS.get(tpl, (_tex_escape(tpl), "---"))
        ds_names = ", ".join(DATASET_DISPLAY.get(d, d) for d in datasets)
        return (
            f"  \\footnotesize {task_name} "
            f"& \\footnotesize {prompt} "
            f"& \\footnotesize {ds_names} \\\\"
        )

    for tpl in task_order:
        datasets = task_ds_map.get(tpl)
        if datasets is None:
            continue
        seen.add(tpl)
        lines.append(_row(tpl, datasets))

    for tpl in sorted(task_ds_map.keys()):
        if tpl in seen:
            continue
        lines.append(_row(tpl, task_ds_map[tpl]))

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def generate_bibtex() -> str:
    """Hard-coded BibTeX entries from esp_data dataset docstrings."""
    return textwrap.dedent(r"""
    @misc{xenocanto,
      title   = {Xeno-canto: Sharing Bird Sounds from Around the World},
      url     = {https://www.xeno-canto.org/},
      year    = {2024},
      note    = {Community database of bird, grasshopper, and bat sounds}
    }

    @misc{inaturalist,
      title   = {{iNaturalist}},
      url     = {https://www.inaturalist.org/},
      year    = {2024},
      note    = {Citizen-science biodiversity observation platform}
    }

    @misc{watkins2018,
      title   = {Watkins Marine Mammal Sound Database},
      author  = {{Woods Hole Oceanographic Institution}},
      url     = {https://cis.whoi.edu/science/B/whalesounds/index.cfm},
      doi     = {10.1575/1912/7270},
      year    = {2018},
      note    = {2018 remaster with GBIF taxonomy}
    }

    @misc{tierstimmenarchiv,
      title   = {{Tierstimmenarchiv (Animal Sound Archive)}},
      author  = {{Museum f\"{u}r Naturkunde Berlin}},
      url     = {https://www.tierstimmenarchiv.de/},
      year    = {2024},
      note    = {World's oldest and most comprehensive animal sound archive}
    }

    @inproceedings{gemmeke2017audioset,
      title     = {Audio Set: An Ontology and Human-Labeled Dataset for Audio Events},
      author    = {Gemmeke, Jort F. and Ellis, Daniel P. W. and Freedman, Dylan and Jansen, Aren and Lawrence, Wade and Moore, R. Channing and Plakal, Manoj and Ritter, Marvin},
      booktitle = {Proc.~IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      = {2017},
      doi       = {10.1109/ICASSP.2017.7952261}
    }

    @article{baij2024audiosetcaps,
      title   = {{AudioSetCaps}: Enriched Audio Captioning for AudioSet},
      author  = {Baij, S.},
      journal = {arXiv preprint arXiv:2411.18953},
      year    = {2024},
      url     = {https://arxiv.org/abs/2411.18953},
      note    = {Captions used for general-domain audio captioning supervision}
    }

    @misc{wabad,
      title   = {{WABAD}: A World Annotated Bird Acoustic Dataset for Passive Acoustic Monitoring},
      url     = {https://zenodo.org/records/15629388},
      year    = {2024},
      doi     = {10.5281/zenodo.15629388},
      note    = {Strongly-annotated bird recordings from multiple biomes}
    }

    @article{rauch2024birdset,
      title   = {{BirdSet}: A Multi-Task Benchmark for Classification in Avian Bioacoustics},
      author  = {Rauch, Lukas and others},
      journal = {arXiv preprint arXiv:2403.10380},
      year    = {2024},
      url     = {https://arxiv.org/abs/2403.10380}
    }

    @article{palmer2025dclde,
      title   = {A multi-species cetacean passive acoustic monitoring dataset from the Northeast Pacific},
      author  = {Palmer, K. J. and others},
      journal = {Scientific Data},
      year    = {2025},
      doi     = {10.1038/s41597-025-05281-5}
    }

    @misc{superwhales,
      title   = {{SuperWhales Detection}: Aggregated Marine Mammal Detection Datasets},
      year    = {2025},
      note    = {Composite dataset; component data from Zenodo (\url{https://zenodo.org/records/14887842}),
                 Figshare (\url{https://figshare.com/articles/dataset/6313308}),
                 DCLDE (\url{https://www.cetus.ucsd.edu/dclde/datasetDocumentation.html}),
                 Dryad (\url{https://doi.org/10.5061/dryad.v15dv422r}),
                 and NOAA NCEI passive acoustic archives.
                 Licenses: CC-BY-4.0 and CC0-1.0}
    }

    @misc{audioskills,
      title   = {{AudioSkills}},
      author  = {{NVIDIA}},
      url     = {https://huggingface.co/datasets/nvidia/AudioSkills},
      year    = {2024},
      note    = {Audio understanding benchmark: counting QA, multiple-choice QA, captioning}
    }

    @article{musikhin2025f0,
      title   = {A dataset of fundamental frequency contours from animal vocalizations},
      author  = {Musikhin, V. and others},
      journal = {Bioacoustics},
      year    = {2025},
      doi     = {10.1080/09524622.2025.2500380},
      note    = {Data available at \url{https://doi.org/10.5061/dryad.prr4xgxw8}}
    }
    """).strip() + "\n"


HEADER = (
    "%% Auto-generated by scripts/generate_stage2_tables.py — do not edit.\n"
    "%% Required packages: booktabs, longtable, multirow, url\n"
)


def _write(path: Path, content: str) -> None:
    path.write_text(HEADER + "\n" + content + "\n")
    print(f"  {path}")


def main() -> None:
    pairs = parse_data_config(DATA_CONFIG)
    ds_task_map = build_dataset_task_map(pairs)
    task_ds_map = build_task_dataset_map(pairs)

    out = OUTPUT_DIR / "tables"
    out.mkdir(exist_ok=True)

    _write(out / "dataset_overview.tex", generate_dataset_table(ds_task_map))
    _write(out / "tasks_by_dataset.tex", generate_table1(ds_task_map))
    _write(out / "datasets_by_task.tex", generate_table2(task_ds_map))

    bib_path = out / "datasets.bib"
    bib_path.write_text(generate_bibtex())
    print(f"  {bib_path}")

    print(f"\nDatasets: {len(ds_task_map)}")
    print(f"Unique tasks: {len(task_ds_map)}")
    print(f"Total (dataset, task) pairs: {len(pairs)}")

    missing_display = {t for t in task_ds_map if t not in TASK_DISPLAY}
    if missing_display:
        print(f"\nWARNING: tasks missing display names: {missing_display}")

    missing_prompt = {t for t in task_ds_map if t not in EXAMPLE_PROMPTS}
    if missing_prompt:
        print(f"WARNING: tasks missing example prompts: {missing_prompt}")


if __name__ == "__main__":
    main()

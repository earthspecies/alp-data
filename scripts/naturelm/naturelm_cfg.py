"""NatureLM metadata samples look like

{'prompt': '<Audio><AudioHere></Audio> What are the common names for the species in the audio, if any?',
 'path': '/home/marius_miron_earthspecies_org/data/foundation-model-data/audio_16k/animalspeak2/16khz/Xeno-canto/XC188355-JUNCO_Oregon_s%204x_c_twitter%20above%20Maligne%20L%201900m%20Jasper%20NP%20061214%201148.flac',
 'task': 'species-common-multiple-detection',
 'license': 'CC BY-NC',
 'recordist': 'Richard E. Webster',
 'url': 'https://xeno-canto.org/sounds/uploaded/KZYUWIRZVH/XC188355-JUNCO_Oregon_s%204x_c_twitter%20above%20Maligne%20L%201900m%20Jasper%20NP%20061214%201148.mp3',
 'source': 'Xeno-canto',
 'duration': 21.0546875,
 'class': 'Aves',
 'family': 'Passerellidae',
 'genus': 'Junco',
 'species': 'Junco hyemalis',
 'phylum': 'Chordata',
 'order': 'Passeriformes',
 'subspecies': 'Junco hyemalis montanus',
 'output': 'Dark-eyed Junco'}

"""

from typing import Optional

from pydantic import Field

from esp_data.config import DataSample, DatasetConfig

LICENSES = {
    "esc50": "CC-BY-NC",
    "rfcx": "academic, research & non-commercial use",
    "cbi": "CC-BY-NC-SA",
    "humbugdb": "CC-BY",
    "enabirds": "CC0",
    "hiceas": "data are free to use without restriction",
    "watkins": "free for personal/academic uses",
    "gibbons": "CC-BY-NC-SA",
    "lifestage": "per file licenses, please see individual files",
    "call-type": "per file licenses, please see individual files",
    "unseen-species-cmn": "per file licenses, please see individual files",
    "unseen-species-sci": "per file licenses, please see individual files",
    "unseen-species-tax": "per file licenses, please see individual files",
    "unseen-genus-cmn": "per file licenses, please see individual files",
    "unseen-genus-sci": "per file licenses, please see individual files",
    "unseen-genus-tax": "per file licenses, please see individual files",
    "unseen-family-cmn": "per file licenses, please see individual files",
    "unseen-family-sci": "per file licenses, please see individual files",
    "unseen-family-tax": "per file licenses, please see individual files",
    "captioning": "per file licenses, please see individual files",
    "zf-indiv": "CC-BY",
    "birdvox70k_pitch": "CC-BY",
}


class NatureLMSample(DataSample):
    """Defines the structure of a Beans0 sample.

    Fields inherited from DataSample:
        - source_dataset: str
        - license: str | None
        - metadata: dict | None
        - created_at: datetime
        - id: str
        - derived_from: str | None (will be dropped)
        - version: str | None (will be dropped)
    """

    file_name: str = Field(description="Audio filename, could be a url")
    instruction: str = Field(min_length=1, description="Prompt for naturelm")
    instruction_text: str = Field(min_length=1, description="Prompt for naturelm without Audio token placeholder")
    output: str = Field(min_length=1, description="Some kind of expected output: text caption / label / answer")
    task: Optional[str] = Field(default=None, description="The task the model is trying to solve")
    # class_: Optional[str] = Field(description="The class of the output")
    # family: Optional[str] = Field(description="The family of the output")
    # genus: Optional[str] = Field(description="The genus of the output")
    # species: Optional[str] = Field(description="The species of the output")
    # subspecies: Optional[str] = Field(description="The subspecies of the output")
    # phylum: Optional[str] = Field(description="The phylum of the output")
    # order: Optional[str] = Field(description="The order of the output")
    # url: Optional[str] = Field(description="The url of the source")
    # recordist: Optional[str] = Field(description="The recordist of the source")
    # duration: Optional[float] = Field(description="The duration of the audio file")

    # overwrite the metadata field
    metadata: str = Field(description="Metadata for the sample")


class NatureLMDatasetConfig(DatasetConfig):
    """Defines the structure of a Beans0 dataset config.

    Fields inherited from DatasetConfig:
        - name: str
        - description: str
        - license: str
        - creator: str
    """

    # required
    name: str = Field(description="The name of the dataset")
    description: str = Field(min_length=1, description="A description of the dataset")
    license: str = Field(min_length=1, description="The license for the dataset")
    creator: str = Field(min_length=1, sdescription="The creator of the dataset")
    metadata: dict = Field(min_length=1, description="The metadata for the whole dataset")
    changelog: Optional[str] = Field(default=None, description="A description of the changes made to the dataset")


naturelm_cfg = NatureLMDatasetConfig(
    name="naturelm",
    description="NatureLM training dataset",
    license="CC BY-NC-SA 4.0",
    creator="Earth Species Project",
    metadata={"sample_rate": 16000, "language": "en"},
    version="0.1.0",
    sources=["Xeno-canto", "AnimalSpeak", "iNaturalist"],
)

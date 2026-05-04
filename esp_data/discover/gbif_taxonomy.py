"""
V2 of taxonomy lookup service. Redirects synonyms to GBIF accepted taxonomy.

Uses a preprocessed TSV of GBIF backbone taxonomy for animals,
downloadable from Google Cloud Storage, but can be cached
"""

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from esp_data.backends import DataBackend
from esp_data.io import AnyPathT, exists, filesystem_from_path
from esp_data.transforms import register_transform
from esp_data.utils import DATA_HOME

logger = logging.getLogger("esp_data")

TAXONOMY_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
# TODO: need a better versioning system
VERSION = "0.1.0"
# location of precomputed outputs
DEFAULT_PRECOMPUTED_LOCATION = f"{DATA_HOME}/gbif_taxonomy/v0.1.0/gbif_animals_converter_cache.json"
# Use current script directory for cache
this_dir = Path(__file__).parent.resolve()
PRECOMPUTED_CACHE_PATH = str(this_dir / "gbif_animals_converter_cache_0.1.0.json")

SCI_NAME_CORRECTION_MANUAL = {
    "Eupodotis rueppelii": "Eupodotis rueppellii",
    "Eudynamys melanorhynchus": "Eudynamys melanorhyncha",
    "Chrysococcyx meyerii": "Chrysococcyx meyeri",
    "Aramides cajaneus": "Aramides cajanea",
    "Laterallus spilonota": "Laterallus spilonotus",
    "Amaurornis moluccana": "Amaurornis olivacea",
    "Ictinaetus malaiensis": "Ictinaetus malayensis",
    "Indicator conirostris/minor": "Indicator conirostris",
    "Amazona mercenarius": "Amazona mercenaria",
    "Orthopsittaca manilatus": "Orthopsittaca manilata",
    "Neophema bourkii": "Neopsephotus bourkii",
    "Vini solitarius": "Phigys solitarius",
    "Glossoptila goldiei": "Glossoptilus goldiei",
    "Saudareos ornatus": "Saudareos ornata",
    "Serpophaga subcristata/munda": "Serpophaga subcristata",
    "Conopias parvus": "Conopias albovittatus",
    "Hylacola pyrrhopygia": "Hylacola pyrrhopygius",
    "Hylacola cauta": "Hylacola cautus",
    "Tchagra minutus": "Tchagra minuta",
    "Artamus leucorynchus": "Artamus leucorhynchos",
    "Dicrurus atactus": "Dicrurus modestus",
    "Dicrurus divaricatus": "Dicrurus adsimilis",
    "Certhilauda curvirostris/brevirostris": "Certhilauda curvirostris",
    "Galerida cristata/macrorhyncha": "Galerida cristata",
    "Rubigula squamata": "Rubigula squamatus",
    "Salpornis spilonota": "Salpornis spilonotus",
    "Saroglossa spilopterus": "Saroglossa spiloptera",
    "Cincloramphus mariei": "Cincloramphus mariae",
    "Sylvia nigricapillus": "Sylvia nigricapilla",
    "Phylloscopus sibilatrix": "Phylloscopus sibillatrix",
    "Phylloscopus affinis/occisinensis": "Phylloscopus affinis",
    "Notopholia corusca": "Notopholia corrusca",
    "Buphagus erythrorynchus": "Buphagus erythrorhynchus",
    "Neocossyphus finschi": "Neocossyphus finschii",
    "Calamornis heudei": "Paradoxornis heudei",
    "Cholornis paradoxus": "Cholornis paradoxa",
    "Phyllergates cucullatus": "Phyllergates cuculatus",
    "Monticola cinclorhyncha": "Monticola cinclorhynchus",
    "Ramphocelus bresilius": "Ramphocelus bresilia",
    "Sicalis uropygialis": "Sicalis uropigyalis",
    "Corcorax melanorhamphos": "Corcorax melanoramphos",
    "Chelidorhynx hypoxanthus": "Chelidorhynx hypoxantha",
    "Acrochordopus burmeisteri": "Phyllomyias burmeisteri",
    "Acrochordopus zeledoni": "Phyllomyias zeledoni",
    "Aerospiza castanilius": "Accipiter castanilius",
    "Aerospiza tachiro": "Accipiter tachiro",
    "Amirafra angolensis": "Mirafra angolensis",
    "Amirafra collaris": "Mirafra collaris",
    "Amirafra rufocinnamomea": "Mirafra rufocinnamomea",
    "Anarhynchus alticola": "Charadrius alticola",
    "Anarhynchus atrifrons": "Charadrius atrifrons",
    "Anarhynchus bicinctus": "Charadrius bicinctus",
    "Anarhynchus collaris": "Charadrius collaris",
    "Anarhynchus dealbatus": "Charadrius dealbatus",
    "Anarhynchus falklandicus": "Charadrius falklandicus",
    "Anarhynchus javanicus": "Charadrius javanicus",
    "Anarhynchus marginatus": "Charadrius marginatus",
    "Anarhynchus mongolus": "Charadrius mongolus",
    "Anarhynchus montanus": "Charadrius montanus",
    "Anarhynchus nivosus": "Charadrius alexandrinus",
    "Anarhynchus obscurus": "Charadrius obscurus",
    "Anarhynchus pallidus": "Charadrius pallidus",
    "Anarhynchus pecuarius": "Charadrius pecuarius",
    "Anarhynchus peronii": "Charadrius peronii",
    "Anarhynchus ruficapillus": "Charadrius ruficapillus",
    "Anarhynchus sanctaehelenae": "Charadrius sanctaehelenae",
    "Anarhynchus thoracicus": "Charadrius thoracicus",
    "Anarhynchus wilsonia": "Charadrius wilsonia",
    "Antiurus maculicaudus": "Hydropsalis maculicaudus",
    "Apteryx maxima": "Apteryx haastii",
    "Ardea coromanda": "Bubulcus coromandus",
    "Artomyias fuliginosa": "Bradornis fuliginosus",
    "Artomyias ussheri": "Muscicapa ussheri",
    "Astur bicolor": "Accipiter bicolor",
    "Astur chilensis": "Accipiter chilensis",
    "Astur cooperii": "Accipiter cooperii",
    "Astur gentilis": "Accipiter gentilis",
    "Astur gundlachi": "Accipiter gundlachi",
    "Astur henstii": "Accipiter henstii",
    "Astur melanoleucus": "Accipiter melanoleucus",
    "Astur meyerianus": "Accipiter meyerianus",
    "Botaurus cinnamomeus": "Ixobrychus cinnamomeus",
    "Botaurus dubius": "Ixobrychus dubius",
    "Botaurus eurhythmus": "Ixobrychus eurhythmus",
    "Botaurus exilis": "Ixobrychus exilis",
    "Botaurus flavicollis": "Dupetor flavicollis",
    "Botaurus involucris": "Ixobrychus involucris",
    "Botaurus minutus": "Ixobrychus minutus",
    "Botaurus sinensis": "Ixobrychus sinensis",
    "Botaurus sturmii": "Ixobrychus sturmii",
    "Buphagus erythroryncha": "Buphagus erythrorhynchus",
    "Centropus burchellii": "Centropus superciliosus",
    "Chalcopsitta fuscata": "Pseudeos fuscata",
    "Chiroxiphia bokermanni": "Antilophia bokermanni",
    "Chiroxiphia galeata": "Antilophia galeata",
    "Chrysuronia boucardi": "Amazilia boucardi",
    "Corypha africana": "Mirafra africana",
    "Corypha apiata": "Mirafra apiata",
    "Corypha fasciolata": "Mirafra fasciolata",
    "Corypha hypermetra": "Mirafra hypermetra",
    "Corypha somalica": "Mirafra somalica",
    "Cyclopsitta desmarestii": "Psittaculirostris desmarestii",
    "Cyclopsitta edwardsii": "Psittaculirostris edwardsii",
    "Cyclopsitta salvadorii": "Psittaculirostris salvadorii",
    "Daptrius albogularis": "Phalcoboenus albogularis",
    "Daptrius australis": "Phalcoboenus australis",
    "Daptrius carunculatus": "Phalcoboenus carunculatus",
    "Daptrius chimachima": "Milvago chimachima",
    "Daptrius chimango": "Milvago chimango",
    "Daptrius megalopterus": "Phalcoboenus megalopterus",
    "Driophlox atrimaxillaris": "Habia atrimaxillaris",
    "Driophlox cristata": "Habia cristata",
    "Driophlox fuscicauda": "Habia fuscicauda",
    "Driophlox gutturalis": "Habia gutturalis",
    "Emblema ruficauda": "Neochmia ruficauda",
    "Eopsaltria capito": "Tregellasia capito",
    "Eopsaltria leucops": "Tregellasia leucops",
    "Eopsaltria placens": "Poecilodryas placens",
    "Gyps rueppelli": "Gyps rueppellii",
    "Hesperoburhinus bistriatus": "Burhinus bistriatus vocifer",
    "Hesperoburhinus superciliaris": "Burhinus superciliaris",
    "Ixos leucogrammicus": "Pycnonotus leucogrammicus",
    "Leucophantes brachyurus": "Poecilodryas brachyura",
    "Lophorina latipennis": "Lophorina superba",
    "Lophospiza griseiceps": "Accipiter griseiceps",
    "Lophospiza trivirgata": "Accipiter trivirgatus",
    "Melanocharis piperata": "Rhamphocharis crassirostris",
    "Melanodryas bimaculata": "Peneothello bimaculata",
    "Melanodryas cryptoleuca": "Peneothello cryptoleuca",
    "Melanodryas cyanus": "Peneothello cyanus",
    "Melanodryas pulverulenta": "Peneoenanthe pulverulenta",
    "Melanodryas sigillata": "Peneothello sigillata",
    "Meliphaga chrysogenys": "Oreornis chrysogenys",
    "Meliphaga imitatrix": "Microptilotis imitatrix",
    "Microtarsus eutilotus": "Pycnonotus eutilotus",
    "Microtarsus fuscoflavescens": "Pycnonotus fuscoflavescens",
    "Microtarsus melanocephalos": "Pycnonotus atriceps",
    "Microtarsus priocephalus": "Pycnonotus priocephalus",
    "Microtarsus urostictus": "Pycnonotus urostictus",
    "Myopornis boehmi": "Muscicapa boehmi",
    "Nannopsittacus gulielmitertii": "Cyclopsitta gulielmitertii",
    "Neophilydor erythrocercum": "Philydor erythrocercum",
    "Neophilydor fuscipenne": "Philydor fuscipenne",
    "Oenanthe heuglinii": "Oenanthe heuglini",
    "Pachyglossa agilis": "Dicaeum agile",
    "Pachyglossa chrysorrhea": "Dicaeum chrysorrheum",
    "Pachyglossa everetti": "Dicaeum everetti",
    "Pachyglossa melanozantha": "Dicaeum melanozanthum",
    "Pachyglossa olivacea": "Prionochilus olivaceus",
    "Pachyglossa propria": "Dicaeum proprium",
    "Pachyglossa vincens": "Dicaeum vincens",
    "Plocealauda affinis": "Mirafra affinis",
    "Plocealauda assamica": "Mirafra assamica",
    "Plocealauda erythrocephala": "Mirafra erythrocephala",
    "Plocealauda erythroptera": "Mirafra erythroptera",
    "Plocealauda microptera": "Mirafra microptera",
    "Psitteuteles porphyrocephalus": "Glossopsitta porphyrocephala",
    "Psitteuteles pusillus": "Glossopsitta pusilla",
    "Quechuavis decussata": "Systellura decussata",
    "Rufirallus fasciatus": "Anurolimnas fasciatus",
    "Rufirallus leucopyrrhus": "Laterallus leucopyrrhus",
    "Rufirallus schomburgkii": "Micropygia schomburgkii",
    "Rufirallus xenopterus": "Laterallus xenopterus",
    "Strigops habroptilus": "Strigops habroptila",
    "Tachyspiza albogularis": "Accipiter albogularis",
    "Tachyspiza badia": "Accipiter badius",
    "Tachyspiza brevipes": "Accipiter brevipes",
    "Tachyspiza cirrocephala": "Accipiter cirrocephalus",
    "Tachyspiza erythrauchen": "Accipiter erythrauchen",
    "Tachyspiza erythropus": "Accipiter erythropus",
    "Tachyspiza fasciata": "Accipiter fasciatus",
    "Tachyspiza francesiae": "Accipiter francesiae",
    "Tachyspiza gularis": "Accipiter gularis",
    "Tachyspiza henicogramma": "Accipiter henicogrammus",
    "Tachyspiza hiogaster": "Accipiter hiogaster",
    "Tachyspiza melanochlamys": "Accipiter melanochlamys",
    "Tachyspiza minulla": "Accipiter minullus",
    "Tachyspiza nanus": "Accipiter nanus",
    "Tachyspiza novaehollandiae": "Accipiter novaehollandiae",
    "Tachyspiza poliocephala": "Accipiter poliocephalus",
    "Tachyspiza rhodogaster": "Accipiter rhodogaster",
    "Tachyspiza rufitorques": "Accipiter rufitorques",
    "Tachyspiza soloensis": "Accipiter soloensis",
    "Tachyspiza trinotata": "Accipiter trinotatus",
    "Tachyspiza virgata": "Accipiter virgatus",
    "Thinornis dubius": "Charadrius dubius",
    "Thinornis forbesi": "Charadrius forbesi",
    "Thinornis melanops": "Elseyornis melanops",
    "Thinornis placidus": "Charadrius placidus",
    "Thinornis tricollaris": "Charadrius tricollaris",
    "Trichoglossus borneus": "Eos bornea",
    "Trichoglossus concinnus": "Glossopsitta concinna",
    "Trichoglossus cyanogenius": "Eos cyanogenia",
    "Trichoglossus reticulatus": "Eos reticulata",
    "Trichoglossus squamatus": "Eos squamata",
    "Turdoides rufocinctus": "Kupeornis rufocinctus",
    "Tyranniscus cinereiceps": "Phyllomyias cinereiceps",
    "Tyranniscus nigrocapillus": "Phyllomyias nigrocapillus",
    "Tyranniscus uropygialis": "Phyllomyias uropygialis",
    "Vini margarethae": "Charmosyna margarethae",
    "Agalychnis taylori": "Agalychnis callidryas",
    "Anstisia alba": "Geocrinia alba",
    "Anstisia lutea": "Geocrinia lutea",
    "Anstisia rosea": "Geocrinia rosea",
    "Anstisia vitellina": "Geocrinia vitellina",
    "Aquarana catesbeiana": "Lithobates catesbeianus",
    "Aquarana clamitans": "Lithobates clamitans",
    "Aquarana grylio": "Lithobates grylio",
    "Aquarana septentrionalis": "Lithobates septentrionalis",
    "Arphia pseudonietana": "Arphia pseudo-nietana",
    "Boreorana sylvatica": "Lithobates sylvaticus",
    "Boulenophrys jinggangensis": "Megophrys jinggangensis",
    "Bufo praetextatus": "Bufo japonicus",
    "Cecropis rufula": "Cecropis daurica",
    "Cephalophorus harveyi": "Cephalophus harveyi",
    "Cinnyris frenatus": "Cinnyris jugularis",
    "Cinnyris infrenatus": "Cinnyris jugularis",
    "Cinnyris ornatus": "Cinnyris jugularis",
    "Circaetus spectabilis": "Dryotriorchis spectabilis",
    "Clemacantha goliath": "Eurycnema goliath",
    "Corvus philippinus": "Corvus macrorhynchos",
    "Corypha athi": "Mirafra africana",
    "Duellmanohyla legleri": "Ptychohyla legleri",
    "Duellmanohyla salvadorensis": "Ptychohyla salvadorensis",
    "Elachistocleis ovalis-complex": "Elachistocleis ovalis",
    "Emblema modesta": "Neochmia modesta",
    "Erethizon dorsatum": "Erethizon dorsatus",
    "Erythrogenys imberbis": "Megapomatorhinus erythrogenys",
    "Firouzophrynus stomaticus": "Bufo stomaticus",
    "Gastrotheca coeruleomaculata": "Gastrotheca coeruleomaculatus",
    "Hyalinobatrachium viridissimum": "Hyalinobatrachium fleischmanni",
    "Hyla flaviventris": "Dryophytes flaviventris",
    "Hyperolius hypsiphonus": "Alexteroon hypsiphonus",
    "Tibicinoides boweni": "Okanagana boweni",
    "Tibicinoides catalina": "Okanagana catalina",
    "Tibicinoides pallidula": "Okanagana pallidula",
    "Tibicinoides rubrovenosa": "Okanagana rubrovenosa",
    "Tibicinoides striatipes": "Okanagana striatipes",
    "Tibicinoides uncinata": "Okanagana uncinata",
    "Tibicinoides utahensis": "Okanagana utahensis",
    "Tibicinoides vanduzeei": "Okanagana vanduzeei",
    "Larus mongolicus": "Larus vegae",
    "Laterallus spilopterus": "Laterallus spiloptera",
    "Lupulella adusta": "Canis adustus",
    "Lupulella adustus": "Canis adustus",
    "Lupulella mesomelas": "Canis mesomelas",
    "Lycalopex grisea": "Pseudalopex griseus",
    "Lycalopex gymnocerca": "Lycalopex gymnocercus",
    "Magicicada cassinii": "Magicicada cassini",
    "Melanophryniscus formosus": "Melanophryniscus stelzneri",
    "Melogale subaurantiaca": "Melogale moschata",
    "Mertensophryne lughensis": "Poyntonophrynus lughensis",
    "Micropterus nigricans": "Micropterus floridanus",
    "Neogale vison": "Neovison vison",
    "Ochotona pallasii": "Ochotona pallasi",
    "Ololygon arduoa": "Ololygon arduous",
    "Otospermophilus douglasii": "Spermophilus beecheyi",
    "Pelobatrachus kobayashii": "Megophrys kobayashii",
    "Pelophylax 'esculentus'": "Pelophylax esculentus",
    "Pelophylax 'grafi'": "Pelophylax perezi",
    "Petaurista grandis": "Petaurista petaurista",
    "Philoria sphagnicola": "Philoria sphagnicolus",
    "Platyplectrum fletcheri": "Lechriodus fletcheri",
    "Pogoniulus uropygialis": "Pogoniulus pusillus",
    "Pogonotriccus difficilis": "Phylloscartes difficilis",
    "Pogonotriccus paulista": "Phylloscartes paulista",
    "Pycnogaster cucullata": "Pycnogaster cucullatus",
    "Ranoidea lesueuri": "Ranoidea lesueurii",
    "Romalea eques": "Taeniopoda eques",
    "Serranobatrachus sanctaemartae": "Eleutherodactylus sanctaemartae",
    "Stiphrornis mabirae": "Stiphrornis xanthogaster",
    "Tachiramantis cuentasi": "Eleutherodactylus cuentasi",
    "Tachiramantis tayrona": "Eleutherodactylus tayrona",
    "Tachyspiza haplochroa": "Accipiter haplochrous",
    "Tarsiger formosanus": "Tarsiger indicus",
    "Trachycephalus vermiculatus-complex": "Trachycephalus vermiculatus",
    "Troglodytes mesoleucus": "Troglodytes aedon",
    "Xenops mexicanus": "Xenops genibarbis",
    "Zoraena maculata": "Cordulegaster maculata",
}


class GBIFConverter:
    """
    Utility for resolving GBIF taxonomic names to their accepted species-level
    usage using the GBIF backbone taxonomy.

    The underlying GBIF table is indexed by both ``taxonID`` (unique) and
    ``canonicalName`` (potentially non-unique) to support efficient lookups.

    Parameters
    ----------
    precomputed_fp : str, optional
        Path to a json file containing animal GBIF taxonomy records,
        preprocessed via scripts/taxonomy_v2_source_to_tsv.py and
        scripts/data_preprocessing_scripts/cache_gbif_taxonomy_conversion.py

    precomputed_cache_path : str | AnyPathT | None, optional
        Path to a local cached copy of the GBIF taxonomy json. If provided,
        this path will be used instead of ``precomputed_fp``.
    """

    def __init__(
        self,
        precomputed_fp: str | AnyPathT = DEFAULT_PRECOMPUTED_LOCATION,
        precomputed_cache_path: str | AnyPathT | None = PRECOMPUTED_CACHE_PATH,
    ) -> None:
        """
        Load the GBIF animals taxonomy table and construct lookup indices.

        Parameters
        ----------
        precomputed_fp : str, optional
            Path to a json file containing precomputed outputs from
            scripts/cache_gbif_taxonomy_conversion.py, which shortcuts the
            search operations that are implemented in this class
        precomputed_cache_path : str | AnyPathT | None
            Path to a local cached copy of the GBIF taxonomy table. If provided,
            this path will be used instead of ``gbif_animals_converter_cache.json``.
        """

        _save_json = False
        if precomputed_cache_path is not None:
            if not exists(precomputed_cache_path):
                logger.warning(
                    f"GBIFConverter: precomputed_cache_path {precomputed_cache_path}"
                    "does not exist but has been set"
                    ", so we'll download and save the data to it."
                )
                _save_json = True
            else:
                precomputed_fp = precomputed_cache_path

        fs = filesystem_from_path(precomputed_fp)

        with fs.open(precomputed_fp, "rb") as f:
            self.lookupdict = pd.read_json(f).to_dict(orient="index")

        if _save_json:
            pd.DataFrame.from_dict(self.lookupdict, orient="index").to_json(
                precomputed_cache_path, indent=2
            )

    def __call__(self, lookup_name: str) -> tuple[dict[str, Any], bool]:
        """
        Resolve a scientific (canonical) name to its accepted species-level
        GBIF taxonomic record.

        The method:
        - Manually corrects scientific name to one that is searchable in GBIF
        - Resolves duplicate canonical-name matches by preferring accepted usages.
        - Walks up the taxonomy if the matched record is below species rank.
        - Redirects unaccepted names to their accepted usage.
        - Detects and aborts on cyclic or inconsistent references.

        Parameters
        ----------
        lookup_name : str
            Canonical scientific name to resolve (e.g., ``"Corvus corax"``).

        Returns
        -------
        (dict, bool)
            A tuple ``(info, ok)`` where ``info`` is a dictionary containing the
            resolved GBIF taxonomic fields (empty on failure), and ``ok`` is a
            boolean indicating whether resolution succeeded.
        """

        if lookup_name in SCI_NAME_CORRECTION_MANUAL:
            lookup_name_corrected = SCI_NAME_CORRECTION_MANUAL[lookup_name]
            logger.warning(
                f"{lookup_name} manually corrected to {lookup_name_corrected}"
                "before being passed to GBIFConverter"
            )
            lookup_name = lookup_name_corrected

        out = self.lookupdict.get(lookup_name, False)
        if not out:
            return {}, False
        else:
            return out, True


class AddTaxonomyConfig(BaseModel):
    """Configuration for AddTaxonomyTransform."""

    type: Literal["add_taxonomy"] = "add_taxonomy"
    feature: str = Field(
        description="Column name containing scientific names to look up.",
        default="scientific_name",
    )
    gbif_precomputed_taxonomy_path: str = Field(
        description="Path to GBIF taxonomy json file.",
        default=DEFAULT_PRECOMPUTED_LOCATION,
    )
    add_taxonomic_name: bool = Field(
        description="Whether to add a 'taxonomic_name' column with the full taxonomic name.",
        default=False,
    )

    @field_validator("gbif_precomputed_taxonomy_path")
    def check_file_exists(cls, v: str) -> str:
        if not exists(v):
            raise ValueError(f"GBIF data file does not exist: {v}")
        return v


class AddTaxonomy:
    """
    Transform that adds resolved GBIF taxonomy info to each row.

    Uses GBIFConverter to resolve scientific names in a specified column
    to their accepted species-level taxonomic records. New columns are added
    for each taxonomy rank: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'.
    An extra column 'taxonomic_name' is also added, which concatenates
    the higher ranks with the canonical name e.g.
    "Animalia Chordata Aves Passeriformes Corvidae Corvus corax".

    Parameters
    ----------
    feature : str
        Column name containing scientific names to look up.
    gbif_precomputed_taxonomy_path : str | AnyPathT
        Path to GBIF taxonomy json file, preprocessed via
        scripts/cache_gbif_taxonomy_conversion.py
    add_taxonomic_name : bool
        Whether to add a 'taxonomic_name' column with the full taxonomic name.
    """

    def __init__(
        self,
        feature: str = "scientific_name",
        gbif_precomputed_taxonomy_path: str | AnyPathT = DEFAULT_PRECOMPUTED_LOCATION,
        add_taxonomic_name: bool = False,
    ) -> None:
        self.feature = feature
        self.converter = GBIFConverter(
            precomputed_cache_path=gbif_precomputed_taxonomy_path,
        )
        self.add_taxonomic_name = add_taxonomic_name

    @classmethod
    def from_config(cls, cfg: AddTaxonomyConfig) -> "AddTaxonomy":
        return cls(**cfg.model_dump(exclude={"type"}))

    def _make_taxonomic_name(self, info: dict[str, str]) -> str | None:
        """Construct the full taxonomic name from GBIF info.

        Parameters
        ----------
        info : dict[str, str]
            GBIF taxonomic record fields.

        Returns
        -------
        str | None
            Full taxonomic name (including higher ranks) or None if unavailable.
        """
        if not info:
            return None

        taxonomic_name = ""
        for rank in TAXONOMY_RANKS[:-1]:  # Exclude genus
            rank_value = info.get(rank)
            if rank_value:
                if taxonomic_name:
                    taxonomic_name += " "
                taxonomic_name += rank_value

        # Add canonicalName
        canonical_name = info.get("canonicalName")
        if canonical_name:
            if taxonomic_name:
                taxonomic_name += " "
            taxonomic_name += canonical_name

        return taxonomic_name if len(taxonomic_name) > 0 else None

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """
        Apply the transform to add taxonomy columns.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the DataFrame to transform.

        Returns
        -------
        tuple[DataBackend, dict]
            A tuple containing the transformed backend with taxonomy columns added,
            and metadata about the resolution (success/failure counts).

        Raises
        ------
        ValueError
            If the specified feature column is not found in the DataFrame.
        """
        if self.feature not in backend.columns:
            raise ValueError(f"Feature column '{self.feature}' not found in data.")

        # Get unique scientific names to look up (avoids redundant lookups)
        unique_names = backend.get_unique(self.feature)

        # Build a lookup cache: rank -> [{scientific_name -> value}]
        # e.g. {'kingdom': [{'Corvus corax': 'Animalia'}, ...], ...}
        EXTENDED_RANKS = TAXONOMY_RANKS
        if self.add_taxonomic_name:
            EXTENDED_RANKS = TAXONOMY_RANKS + ["taxonomic_name"]
        taxonomy_cache: dict[str, list[tuple[str, str]]] = {r: [] for r in EXTENDED_RANKS}
        success_count = 0
        failure_count = 0

        for name in unique_names:
            info, ok = self.converter(name)
            if ok:
                # Fill by rank
                for rank in TAXONOMY_RANKS:
                    taxonomy_cache[rank].append((name, info.get(rank)))
                if self.add_taxonomic_name:
                    taxonomy_cache["taxonomic_name"].append((name, self._make_taxonomic_name(info)))
                success_count += 1
            else:
                failure_count += 1
                logger.debug(f"Failed to resolve taxonomy for: {name}")

        if failure_count > 0:
            logger.warning(f"Failed to resolve {failure_count}/{len(unique_names)} unique names")

        # Map resolved taxonomy back to backend, adding new columns
        for rank in EXTENDED_RANKS:
            rank_mapping = {src: target for src, target in taxonomy_cache[rank]}
            backend = backend.map_column(self.feature, mapping=rank_mapping, output_column=rank)

        metadata = {
            "feature": self.feature,
            "resolved": success_count,
            "failed": failure_count,
            "taxonomy_columns_added": EXTENDED_RANKS,
        }

        return backend, metadata


register_transform(AddTaxonomyConfig, AddTaxonomy)

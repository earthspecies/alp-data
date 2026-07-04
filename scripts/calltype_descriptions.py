"""Within-species call-type descriptions for the priority species set.

Unlike ``iconic_call_descriptions`` (one description per species, used for
*between*-species MCQs), this module provides, for a curated priority set of
~20 species with rich Xeno-canto ``behavior`` metadata, a distinct acoustic
description for *each* of several call types of the *same* species. These power
a *within-species* call-description MCQ: the audio is one call type of species
X; the correct option describes that call type; the distractors describe the
*other* call types of the same species X.

Call types are the canonical, acoustically-distinct categories we trust from
the Xeno-canto metadata. Note ``nocturnal flight call`` is merged into
``flight call`` (it is the same vocalization given at night), see
``CANON_CALLTYPE``.

Each description is deliberately written to be discriminable from the *other
call types of the same species* — that is the axis the MCQ tests.

Fields
------
scientific_name, common_name
    Species identity (joined against ``canonical_name`` / BirdCODE Species).
call_type
    Canonical call type (one of CANON_CALLTYPE values).
description
    Vivid acoustic description of *this* call type, free of the type name.
"""

from __future__ import annotations

from typing import TypedDict


class CallTypeDescription(TypedDict):
    scientific_name: str
    common_name: str
    call_type: str
    description: str


# Map raw Xeno-canto behavior tokens -> canonical call type. Tokens not in this
# map make a recording ineligible (ambiguous / un-describable).
CANON_CALLTYPE: dict[str, str] = {
    "song": "song",
    "call": "call",
    "flight call": "flight call",
    "nocturnal flight call": "flight call",  # same vocalization, night context
    "alarm call": "alarm call",
    "begging call": "begging call",
    "drumming": "drumming",
}


CALLTYPE_DESCRIPTIONS: list[CallTypeDescription] = [
    # ── Great Spotted Woodpecker ──────────────────────────────────────────
    {"scientific_name": "Dendrocopos major", "common_name": "Great Spotted Woodpecker",
     "call_type": "call",
     "description": "a single, sharp, explosive 'kik!' note, sometimes repeated at intervals"},
    {"scientific_name": "Dendrocopos major", "common_name": "Great Spotted Woodpecker",
     "call_type": "drumming",
     "description": "a very short, fast, accelerating mechanical drumroll rapping on wood — a "
                    "rattle of taps rather than a voice"},
    {"scientific_name": "Dendrocopos major", "common_name": "Great Spotted Woodpecker",
     "call_type": "alarm call",
     "description": "a fast, agitated series of 'kik-kik-kik-kik' notes run together in alarm"},
    {"scientific_name": "Dendrocopos major", "common_name": "Great Spotted Woodpecker",
     "call_type": "begging call",
     "description": "an insistent, grating, wheezy chittering churr from young in the nest hole"},

    # ── Black Woodpecker ──────────────────────────────────────────────────
    {"scientific_name": "Dryocopus martius", "common_name": "Black Woodpecker",
     "call_type": "song",
     "description": "a loud, ringing, slightly accelerating series of 'kwih-kwih-kwih' notes "
                    "advertised from a perch"},
    {"scientific_name": "Dryocopus martius", "common_name": "Black Woodpecker",
     "call_type": "call",
     "description": "a far-carrying, plaintive, drawn-out 'klee-aah' wail"},
    {"scientific_name": "Dryocopus martius", "common_name": "Black Woodpecker",
     "call_type": "drumming",
     "description": "an exceptionally long, loud, powerful drumroll resonating through the forest"},
    {"scientific_name": "Dryocopus martius", "common_name": "Black Woodpecker",
     "call_type": "flight call",
     "description": "a loud, rolling, chattering 'krri-krri-krri-krri' given in flight"},

    # ── Common Snipe ──────────────────────────────────────────────────────
    {"scientific_name": "Gallinago gallinago", "common_name": "Common Snipe",
     "call_type": "song",
     "description": "a persistent, rhythmic, mechanical 'chip-per, chip-per, chip-per' chanted "
                    "from a post"},
    {"scientific_name": "Gallinago gallinago", "common_name": "Common Snipe",
     "call_type": "flight call",
     "description": "a harsh, rasping, abrupt 'scaap' as the bird flushes"},
    {"scientific_name": "Gallinago gallinago", "common_name": "Common Snipe",
     "call_type": "drumming",
     "description": "a tremulous, bleating, vibrating hum produced by the outer tail feathers in a "
                    "display dive — an instrument, not a voice"},

    # ── Tawny Owl ─────────────────────────────────────────────────────────
    {"scientific_name": "Strix aluco", "common_name": "Tawny Owl",
     "call_type": "song",
     "description": "a quavering, breathy hooted phrase: 'hoooo ... ho, ho-ho-hoooooo'"},
    {"scientific_name": "Strix aluco", "common_name": "Tawny Owl",
     "call_type": "call",
     "description": "a sharp, explosive, disyllabic 'ke-wick' contact call"},
    {"scientific_name": "Strix aluco", "common_name": "Tawny Owl",
     "call_type": "begging call",
     "description": "a wheezy, rasping, insistent 'tssrip' repeated by hungry juveniles"},

    # ── Common Blackbird ──────────────────────────────────────────────────
    {"scientific_name": "Turdus merula", "common_name": "Common Blackbird",
     "call_type": "song",
     "description": "a rich, mellow, leisurely fluty warble of varied melodious phrases"},
    {"scientific_name": "Turdus merula", "common_name": "Common Blackbird",
     "call_type": "alarm call",
     "description": "a hysterical, accelerating, metallic chinking that breaks into a rattle, often "
                    "at dusk"},
    {"scientific_name": "Turdus merula", "common_name": "Common Blackbird",
     "call_type": "call",
     "description": "a thin, high, drawn-out 'srrri' / 'tsee' note"},

    # ── European Robin ────────────────────────────────────────────────────
    {"scientific_name": "Erithacus rubecula", "common_name": "European Robin",
     "call_type": "song",
     "description": "a wistful, silvery, trickling warble of high, thin musical phrases"},
    {"scientific_name": "Erithacus rubecula", "common_name": "European Robin",
     "call_type": "call",
     "description": "a sharp, hard, insistently repeated ticking 'tic-tic-tic' like a wound clock"},
    {"scientific_name": "Erithacus rubecula", "common_name": "European Robin",
     "call_type": "flight call",
     "description": "a very thin, high, sibilant 'tsiih' seep"},

    # ── Common Chaffinch ──────────────────────────────────────────────────
    {"scientific_name": "Fringilla coelebs", "common_name": "Common Chaffinch",
     "call_type": "song",
     "description": "a cheerful, accelerating, descending rattle ending in a flourishing terminal "
                    "twiddle"},
    {"scientific_name": "Fringilla coelebs", "common_name": "Common Chaffinch",
     "call_type": "call",
     "description": "a loud, ringing, metallic, well-spaced 'pink ... pink'"},
    {"scientific_name": "Fringilla coelebs", "common_name": "Common Chaffinch",
     "call_type": "flight call",
     "description": "a soft, short, quiet 'yup' or 'chup' uttered in flight"},

    # ── Great Tit ─────────────────────────────────────────────────────────
    {"scientific_name": "Parus major", "common_name": "Great Tit",
     "call_type": "song",
     "description": "a ringing, repeated two-note 'teacher-teacher-teacher' like a squeaky see-saw"},
    {"scientific_name": "Parus major", "common_name": "Great Tit",
     "call_type": "call",
     "description": "a bright, emphatic, chinking 'pink' note"},
    {"scientific_name": "Parus major", "common_name": "Great Tit",
     "call_type": "alarm call",
     "description": "a harsh, churring, scolding 'tcherr-er-er-er' rattle"},

    # ── Eurasian Wren ─────────────────────────────────────────────────────
    {"scientific_name": "Troglodytes troglodytes", "common_name": "Eurasian Wren",
     "call_type": "song",
     "description": "an astonishingly loud, fast, explosive cascade of trills ending in a hard "
                    "rattle"},
    {"scientific_name": "Troglodytes troglodytes", "common_name": "Eurasian Wren",
     "call_type": "alarm call",
     "description": "a hard, dry, machine-gun 'tic-tic-tic-tic-tic' scolding rattle"},
    {"scientific_name": "Troglodytes troglodytes", "common_name": "Eurasian Wren",
     "call_type": "call",
     "description": "a hard, single, repeated ticking 'tek' note"},

    # ── Eurasian Blackcap ─────────────────────────────────────────────────
    {"scientific_name": "Sylvia atricapilla", "common_name": "Eurasian Blackcap",
     "call_type": "song",
     "description": "a rich, fluty warble gathering into a clear, loud, ringing final flourish"},
    {"scientific_name": "Sylvia atricapilla", "common_name": "Eurasian Blackcap",
     "call_type": "call",
     "description": "a hard, repeated 'tac-tac' like two pebbles tapped together"},
    {"scientific_name": "Sylvia atricapilla", "common_name": "Eurasian Blackcap",
     "call_type": "alarm call",
     "description": "a fast, churring, scolding run of harsh 'tacc' notes"},

    # ── Song Thrush ───────────────────────────────────────────────────────
    {"scientific_name": "Turdus philomelos", "common_name": "Song Thrush",
     "call_type": "song",
     "description": "loud, ringing musical phrases, each distinct phrase repeated two or three "
                    "times"},
    {"scientific_name": "Turdus philomelos", "common_name": "Song Thrush",
     "call_type": "alarm call",
     "description": "a hard, rapid, scolding 'chook-chook-chook' rattle"},
    {"scientific_name": "Turdus philomelos", "common_name": "Song Thrush",
     "call_type": "flight call",
     "description": "a short, thin, high 'tsip' / 'seep' given in flight"},

    # ── Yellowhammer ──────────────────────────────────────────────────────
    {"scientific_name": "Emberiza citrinella", "common_name": "Yellowhammer",
     "call_type": "song",
     "description": "a rhythmic phrase of repeated notes ending in a long, drawn-out high one"},
    {"scientific_name": "Emberiza citrinella", "common_name": "Yellowhammer",
     "call_type": "call",
     "description": "a sharp, clicking, metallic 'twit' / 'zit'"},
    {"scientific_name": "Emberiza citrinella", "common_name": "Yellowhammer",
     "call_type": "flight call",
     "description": "a short, sharp, rattling 'tirrl' uttered in flight"},

    # ── Barn Owl ──────────────────────────────────────────────────────────
    {"scientific_name": "Tyto alba", "common_name": "Barn Owl",
     "call_type": "call",
     "description": "a long, harsh, drawn-out, rasping shriek"},
    {"scientific_name": "Tyto alba", "common_name": "Barn Owl",
     "call_type": "begging call",
     "description": "a prolonged, wheezy, snoring hiss from the nest"},
    {"scientific_name": "Tyto alba", "common_name": "Barn Owl",
     "call_type": "flight call",
     "description": "a sharp, repeated, twittering chirrup given in flight"},

    # ── Eurasian Curlew ───────────────────────────────────────────────────
    {"scientific_name": "Numenius arquata", "common_name": "Eurasian Curlew",
     "call_type": "song",
     "description": "a rising, bubbling, accelerating cascade of liquid trilling whistles in "
                    "display"},
    {"scientific_name": "Numenius arquata", "common_name": "Eurasian Curlew",
     "call_type": "call",
     "description": "a far-carrying, mournful, rising two-note 'cur-leee'"},
    {"scientific_name": "Numenius arquata", "common_name": "Eurasian Curlew",
     "call_type": "flight call",
     "description": "a short, plaintive, whistled 'cur-lee' repeated in flight"},

    # ── Tree Pipit ────────────────────────────────────────────────────────
    {"scientific_name": "Anthus trivialis", "common_name": "Tree Pipit",
     "call_type": "song",
     "description": "a rich, accelerating trill in parachuting song-flight, ending in a slow, "
                    "drawn-out 'see-er'"},
    {"scientific_name": "Anthus trivialis", "common_name": "Tree Pipit",
     "call_type": "flight call",
     "description": "a hoarse, buzzy, downslurred 'teez'"},
    {"scientific_name": "Anthus trivialis", "common_name": "Tree Pipit",
     "call_type": "call",
     "description": "a short, sharp, thin 'sip' note"},

    # ── European Starling ─────────────────────────────────────────────────
    {"scientific_name": "Sturnus vulgaris", "common_name": "European Starling",
     "call_type": "song",
     "description": "a rambling medley of clicks, whistles, rattles and mimicry, with long rising "
                    "whistles"},
    {"scientific_name": "Sturnus vulgaris", "common_name": "European Starling",
     "call_type": "call",
     "description": "a harsh, descending, grating 'tcheer'"},
    {"scientific_name": "Sturnus vulgaris", "common_name": "European Starling",
     "call_type": "begging call",
     "description": "an insistent, wheezy, rasping juvenile churring"},

    # ── Eurasian Blue Tit ─────────────────────────────────────────────────
    {"scientific_name": "Cyanistes caeruleus", "common_name": "Eurasian Blue Tit",
     "call_type": "song",
     "description": "a high 'tsee-tsee' followed by a fast, silvery, descending trill"},
    {"scientific_name": "Cyanistes caeruleus", "common_name": "Eurasian Blue Tit",
     "call_type": "call",
     "description": "a thin, scolding 'tsee-tsee-tsit'"},
    {"scientific_name": "Cyanistes caeruleus", "common_name": "Eurasian Blue Tit",
     "call_type": "alarm call",
     "description": "a harsh, rapid, churring, scolding rattle"},

    # ── Common Chiffchaff ─────────────────────────────────────────────────
    {"scientific_name": "Phylloscopus collybita", "common_name": "Common Chiffchaff",
     "call_type": "song",
     "description": "a monotonous, repetitive 'chiff-chaff-chiff-chaff' jumping irregularly "
                    "between two pitches"},
    {"scientific_name": "Phylloscopus collybita", "common_name": "Common Chiffchaff",
     "call_type": "call",
     "description": "a soft, plaintive, upward-inflected 'hweet'"},
    {"scientific_name": "Phylloscopus collybita", "common_name": "Common Chiffchaff",
     "call_type": "alarm call",
     "description": "a harder, repeated, churring 'tret'"},

    # ── Mistle Thrush ─────────────────────────────────────────────────────
    {"scientific_name": "Turdus viscivorus", "common_name": "Mistle Thrush",
     "call_type": "song",
     "description": "far-carrying, wild, fluty phrases sung defiantly, often in stormy weather"},
    {"scientific_name": "Turdus viscivorus", "common_name": "Mistle Thrush",
     "call_type": "call",
     "description": "a dry, harsh, rattling, machine-gun churr like a wooden football rattle"},
    {"scientific_name": "Turdus viscivorus", "common_name": "Mistle Thrush",
     "call_type": "flight call",
     "description": "a thin, high 'seep' contact note"},

    # ── Eurasian Skylark ──────────────────────────────────────────────────
    {"scientific_name": "Alauda arvensis", "common_name": "Eurasian Skylark",
     "call_type": "song",
     "description": "an unbroken, tumbling, sustained cascade of trills and warbles poured out "
                    "high overhead"},
    {"scientific_name": "Alauda arvensis", "common_name": "Eurasian Skylark",
     "call_type": "flight call",
     "description": "a liquid, bubbling, rippling 'chirrup' given in flight"},
    {"scientific_name": "Alauda arvensis", "common_name": "Eurasian Skylark",
     "call_type": "call",
     "description": "a dry, buzzy, short 'prreet'"},
]

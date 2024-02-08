# README
# Phillip Long
# November 3, 2023

# mappings for various representations of music
# copied from https://github.com/salu133445/mmt/blob/main/mmt/representation.py


# IMPORTS
##################################################
import pprint
import numpy as np
import utils
import argparse
from itertools import combinations
from read_mscz.classes import DEFAULT_VELOCITY
from utils import unique
##################################################


# CONSTANTS
##################################################

RESOLUTION = 12
MAX_BEAT = 1024
MAX_DURATION = 768  # remember to modify known durations as well!
MAX_VELOCITY = 127
DEFAULT_VALUE_CODE = -1
N_NOTES = 128
NA_VALUES = ("null", "None", None) # for loading encodings

ENCODING_DIR = "/data2/pnlong/musescore/data"
ENCODING_BASENAME = "encoding.json"
ENCODING_FILEPATH = f"{ENCODING_DIR}/{ENCODING_BASENAME}"

##################################################


# DIMENSIONS
##################################################

# (NOTE: "type" must be the first dimension!)
# (NOTE: Remember to modify N_TOKENS as well!)
DIMENSIONS = ["type", "beat", "position", "value", "duration", "instrument", "velocity", "time", "time.s"] # last 2 columns are for sorting, and will be discarded later
assert DIMENSIONS[0] == "type"

##################################################


# TYPE
##################################################

EXPRESSIVE_FEATURE_TYPE_STRING = "expressive-feature"
TYPE_CODE_MAP = {type_: code for code, type_ in enumerate(("start-of-song", "instrument", "start-of-notes", EXPRESSIVE_FEATURE_TYPE_STRING, "grace-note", "note", "end-of-song"))}
CODE_TYPE_MAP = utils.inverse_dict(TYPE_CODE_MAP)

##################################################


# BEAT
##################################################

BEAT_CODE_MAP = {i: i + 1 for i in range(MAX_BEAT + 1)}
BEAT_CODE_MAP[None] = 0
CODE_BEAT_MAP = utils.inverse_dict(BEAT_CODE_MAP)

##################################################


# POSITION
##################################################

POSITION_CODE_MAP = {i: i + 1 for i in range(RESOLUTION)}
POSITION_CODE_MAP[None] = 0
CODE_POSITION_MAP = utils.inverse_dict(POSITION_CODE_MAP)

##################################################


# VALUE
##################################################

# tempo
TEMPO_QPM_MAP = { # each value is the maximum BPM before we go up a tempo, found at https://en.wikipedia.org/wiki/Tempo
    "larghissimo": 24,
    "grave": 40, # also Adagissimo
    "largo": 54, # also Larghetto
    "adagio": 68,
    "adagietto": 80,
    "andante": 94, # or Lento
    "andantino": 108,
    "moderato": 114,
    "allegretto": 120,
    "allegro": 156,
    "vivace": 176,
    "presto": 200,
    "prestissimo": float("inf"), # some arbitrary large number
}
QPM_TEMPO_MAP = utils.inverse_dict(TEMPO_QPM_MAP)
DEFAULT_QPM = 112
def QPM_TEMPO_MAPPER(qpm: float) -> str:
    if qpm is None: # default if qpm argument is none
        qpm = DEFAULT_QPM # default bpm
    for bpm in QPM_TEMPO_MAP.keys():
        if qpm <= bpm:
            return QPM_TEMPO_MAP[bpm]

# time signature changes
time_signature_combinations = list(combinations(iterable = ["4/4", "3/4", "2/4", "5/8", "5/4", "3/8", "9/16", "9/8", "12/4"], r = 2))
time_signature_change_ratios = [f"({combination[0]})/({combination[1]})" for combination in (time_signature_combinations + [time_signature_combination[::-1] for time_signature_combination in time_signature_combinations])] # ratios between different time signatures
already_encountered_ratios = set()
TIME_SIGNATURE_CHANGE_RATIOS = []
for time_signature_change_ratio in time_signature_change_ratios: # make sure there are no duplicate ratios
    ratio = eval(time_signature_change_ratio)
    if ratio not in already_encountered_ratios:
        TIME_SIGNATURE_CHANGE_RATIOS.append(time_signature_change_ratio)
        already_encountered_ratios.add(ratio)
del time_signature_combinations, time_signature_change_ratios, time_signature_change_ratio, already_encountered_ratios, ratio # clear up some memory
TIME_SIGNATURE_CHANGE_PREFIX = "time-signature-change-"
DEFAULT_TIME_SIGNATURE_CHANGE = TIME_SIGNATURE_CHANGE_PREFIX + "other" # default if a ratio doesn't fall in the discretized list
def TIME_SIGNATURE_CHANGE_MAPPER(time_signature_change_ratio: float) -> str:
    discrete_time_signature_change_ratio_differences = [abs(time_signature_change_ratio - eval(discrete_time_signature_change_ratio)) for discrete_time_signature_change_ratio in TIME_SIGNATURE_CHANGE_RATIOS] # calculate difference with each time signature change ratio
    minimum_difference_index = min(range(len(discrete_time_signature_change_ratio_differences)), key = discrete_time_signature_change_ratio_differences.__getitem__) # get the index of the ratio that is closest
    if discrete_time_signature_change_ratio_differences[minimum_difference_index] < 1e-3: # if the minimum difference between the input ratio and one of the discrete ratios is small enough
        return TIME_SIGNATURE_CHANGE_PREFIX + str(TIME_SIGNATURE_CHANGE_RATIOS[minimum_difference_index]) # return the ratio
    else: # if not sufficiently close
        return DEFAULT_TIME_SIGNATURE_CHANGE # return the default

# dynamics
DEFAULT_DYNAMIC = "dynamic-marking"
DYNAMIC_VELOCITY_MAP = {
    "pppppp": 4, "ppppp": 8, "pppp": 12, "ppp": 16, "pp": 33, "p": 49, "mp": 64,
    "mf": 80, "f": 96, "ff": 112, "fff": 126, "ffff": 127, "fffff": 127, "ffffff": 127,
    "sfpp": 96, "sfp": 112, "sf": 112, "sff": 126, "sfz": 112, "sffz": 126, "fz": 112, "rf": 112, "rfz": 112,
    "fp": 96, "pf": 49, "s": DEFAULT_VELOCITY, "r": DEFAULT_VELOCITY, "z": DEFAULT_VELOCITY, "n": DEFAULT_VELOCITY, "m": DEFAULT_VELOCITY,
    DEFAULT_DYNAMIC: DEFAULT_VELOCITY,
}
DYNAMIC_DYNAMICS = set(tuple(DYNAMIC_VELOCITY_MAP.keys())[:tuple(DYNAMIC_VELOCITY_MAP.keys()).index("ffffff")]) # dynamics that are actually dynamic markings and not sudden dynamic hikes

# expressive features
DEFAULT_EXPRESSIVE_FEATURE_VALUES = {
    "Barline": "barline",
    "KeySignature": None,
    "TimeSignature": DEFAULT_TIME_SIGNATURE_CHANGE,
    "Fermata": "fermata",
    "SlurSpanner": "slur",
    "PedalSpanner": "pedal",
    "Tempo": QPM_TEMPO_MAPPER(qpm = None),
    "TempoSpanner": "tempo",
    "Dynamic": DEFAULT_DYNAMIC,
    "HairPinSpanner": "hair-pin",
    "Articulation": "articulation",
    "Text": "text",
    "RehearsalMark": "rehearsal-mark",
    "TextSpanner": "text-spanner",
    "TechAnnotation": "tech-annotation",
    # "Symbol": "symbol",
}
EXPRESSIVE_FEATURES = {
    # Barlines
    "Barline": ["double-barline", "end-barline", "dotted-barline", "dashed-barline", DEFAULT_EXPRESSIVE_FEATURE_VALUES["Barline"],],
    # Key Signatures
    "KeySignature": [f"key-signature-change-{distance}" for distance in range(-6, 7)],
    # Time Signatures
    "TimeSignature": [TIME_SIGNATURE_CHANGE_PREFIX + time_signature_change_ratio for time_signature_change_ratio in TIME_SIGNATURE_CHANGE_RATIOS] + [DEFAULT_EXPRESSIVE_FEATURE_VALUES["TimeSignature"],],
    # Fermata
    "Fermata": [DEFAULT_EXPRESSIVE_FEATURE_VALUES["Fermata"],],
    # SlurSpanner
    "SlurSpanner": [DEFAULT_EXPRESSIVE_FEATURE_VALUES["SlurSpanner"],],
    # PedalSpanner
    "PedalSpanner": [DEFAULT_EXPRESSIVE_FEATURE_VALUES["PedalSpanner"],],
    # Tempo
    "Tempo": list(TEMPO_QPM_MAP.keys()), # no default value, default is set above in QPM_TEMPO_MAPPER
    # TempoSpanner
    "TempoSpanner": ["lentando", "lent", "smorzando", "smorz", "sostenuto", "sosten", "accelerando", "accel", "allargando", "allarg", "rallentando", "rall", "rallent", "ritardando", "rit", DEFAULT_EXPRESSIVE_FEATURE_VALUES["TempoSpanner"],],
    # Dynamic
    "Dynamic": list(DYNAMIC_VELOCITY_MAP.keys()) + [DEFAULT_EXPRESSIVE_FEATURE_VALUES["Dynamic"],],
    # HairPinSpanner
    "HairPinSpanner": [
        "cresc", "dim", "dynamic-mezzodynamic-forte", "cresc-sempre", "molto-cresc", "dimin", "decresc",
        "cresc-poco-a-poco", "poco-a-poco-cresc", "cresc-molto", "poco-cresc", "crescendo", "più-cresc",
        "scen", "sempre-cresc", "do", "cre", "morendo", "dim-e-rit", "diminuendo", "sempre-dim", "poco-a-poco",
        "cres", "poco-a-poco-dim", "dim-sempre", "rall-e-dim", "dim-molto", "keyboard-pedal-ped", "cresc-poco",
        "dim-e-rall", "poco-a-poco-crescendo", "molto-dim", "dim-e-poco-rit", "poco", "poco-dim", "sempre-più-dim",
        "dim-poco-a-poco", "cre-scen-do", "accel-e-cresc", "ed-allarg", "poco-rit", "crescendo-molto", "crescendo-poco-a-poco",
        "string", "un-poco-cresc", "cresc-y-accel", "rit-e-dim", "ritard", "cresc-e-stringendo", "dim-poco-a-poco-e-rit",
        "molto-rit", "più-dim", "cresc-ed-accel", "crescpoco-string", "perdendo", "rall-molto", "sempre", "cresc-e-string", "e-cresc",
        "piu-cresc", "poco-rall", "poco-riten", "calando", "cresc-e-rit", "crescendo-e-rit", "dim-e-poco-riten", "dim-e-sosten", "poco-a",
        "dynamic-forte", "cre-scend-do", "dim-al-fine", "molto-crescendo", "più-piano", "smorz-e-rallent",
        "cresc-assai", "crescsempre", "descresc", "dim-poco", "dim-poco-rit", "dim-rall", "dim-rit", "dimsempre",
        "più-rinforz", "poco-rallent", "rit-et-dim", "sempre-dim-e-rit-al-fine", "a-c-c-e-l-e-r-a-n-d-o", "cresc-", "cresc-e-agitato",
        "cresc-un-piu-animato", "dim-e-molto-rall", "dimin-e-ritard", "dimin-poco-a-poco", "forzando", "poco-a-poco-cresc-molto", "poco-a-poco-dimin",
        "scendo", "un-poco-animato-e-cresc", "dynamic-mezzodynamic-piano", "dynamic-pianodynamic-piano", "cédez", "ral", "cresc-appassionato",
        "cresc-con-molto-agitazione", "cresc-e-accel", "cresc-e-accelerando", "cresc-e-animato", "cresc-e-appassionato", "cresc-e-rall-sempre",
        "cresc-e-stretto", "cresc-molto-e-stringendo", "cresc-sempre-poco-a-poco", "cresc-un-poco-animato", "di-min", "dim-e-ritardando", "dim-e-riten",
        "dim-ed-allarg", "dim-subito", "dimin-e-poco-riten", "diminuendo-subito", "e-poco-rit", "en", "gritando-shouting", "increase", "più-moto",
        "poco-a-poco-cre", "poco-crescendo", "sempre-piu-piano", "sempre-più-cresc", "sempre-più-cresc-e", "sempre-rit-e-dim-sin-al-fine",
        "string-e-cresc", "stringendo", "stringendo-e", "dynamicdynamic-forte", "alternative-solo", "i-i-c", "i-v-c", "preferred-solo", "serrez", "sw",
        "très-dim", "accelerando-e-sempre-cresc", "agitato-e-sempre-più-cresc", "al", "ancor-più-cresc", "animando", "ca",
        "calmando-dim", "cantabile-cresc", "couple-sw", "cr-es-c", "cre-scen", "cres-c", "cresc-animato", "cresc-e",
        "cresc-e-affrettando", "cresc-e-poco-rit", "cresc-e-poco-sostenuto", "cresc-e-poco-string", "cresc-e-stringendo-a-poco-a-poco", "cresc-ed",
        "cresc-ed-accel-poco-a-poco", "cresc-ed-accelerando", "cresc-ed-rit", "cresc-p-a-p", "cresc-poco-a-poco-al-mf", "cresc-sempre-al-ff",
        "cresc-sempre-al-fine", "cresc-sforzando", "cresc-un", "cresc-un-poco", "crescpoco-a-poco", "crescendo-e-rall-a-poco-a-poco", "crescendo-poco",
        "de-cres-cen-do", "di-mi-nu-en-do", "dim-colla-voce", "dim-e", "dim-e-calando", "dim-e-pochiss-rit", "dim-e-poco-rall", "dim-e-rall-molto",
        "dim-e-rallent", "dim-e-rit-poco", "dim-e-roco-rit", "dim-morendo", "dim-riten", "dim-smorz", "dime-poco-rall", "dimin-al-fine", "dimin-e-riten",
        "diminish", "diminuendo-e-leggierissimo", "diminuendo-e-ritardando", "diminuendo-un-poco", "dolce-poc-a-poco", "e", "e-cresc-molto", "e-dim",
        "e-rall", "e-rit-in-poco", "e-smorz", "e-stringendo", "en-dim", "espress-legato-poco-a-poco-cresc", "il-piu-forte-possible", "incalze-cressc-sempre",
        "lan", "mo-ren-do", "molto-cresc-ed-accelerando", "molto-rinforz", "molto-ritardando", "per", "più-cre", "più-cresc-ed-agitato", "più-rit-e-dim",
        "più-smorz-e-rit", "poco-a-poco-accelerando", "poco-a-poco-accelerando-e-crescendo", "poco-a-poco-cresc-e-risoluto", "poco-a-poco-cresc-ed-accel",
        "poco-a-poco-cresc-ed-acceler", "poco-a-poco-cresc-ed-accelerando", "poco-a-poco-cresc-ed-anim", "poco-a-poco-cresc-ed-animato", "poco-a-poco-decresc",
        "poco-a-poco-diminuendo", "poco-a-poco-rinforz", "poco-cresc-e-rit", "poco-cresc-ed-agitato", "poco-cresc-molto", "poco-ritar", "poco-stretto",
        "poco-à-poco-cresc", "pressez", "religioso", "ri", "rinforzando", "rit-e-crescendo", "ritar", "ritard-e-dimin", "riten", "riten-e-dim",
        "sempre-cresc-e-affrettando", "sempre-cresc-e-string", "sempre-cresc-ed-accel", "sempre-dim-e-legatissimo", "sempre-dim-e-più-tranquillo", "sempre-dimin",
        "sempre-più-cresc-e-string", "sempre-più-forte", "si", "slentando", "smorzn", "smorzando-e-rallent", "stretto-e-cresc",
        "stringendo-e-cresc", "stringendo-cresc", "tar", "un-peu-ralenti", "un-poco-animato-e-crescendo", "vif", "élargir",
        DEFAULT_EXPRESSIVE_FEATURE_VALUES["HairPinSpanner"],
    ],
    # Articulation (Chunks)
    "Articulation": [
        "staccato", "sforzato", "accent", "marcato", "tenuto", "trill", "portato", "staccatissimo", "up-bow", "down-bow", "tenuto-staccato", "mordent",
        "accent-staccato", "marcato-staccato", "tenuto-accent", "brass-mute-closed", "marcato-tenuto", "guitar-fade-out", "fadeout", "ouvert",
        "plusstop", "staccatissimo-wedge", "turn", "brass-mute-open", "tremblement", "harmonic", "fadein", "snappizzicato",
        "precomp-mordent-upper-prefix", "plucked-snap-pizzicato", "shortfermata", "stress", "soft-accent",
        "wiggle-vibrato-large-faster", "reverseturn", "staccatissimo-stroke", "unstress", "guitar-volume-swell", "longfermata",
        "guitar-fade-in", "volumeswell", "thumb", "schleifer", "verylongfermata", "wigglevibratolargefaster",
        "espressivo", "precomp-slide", "no-sym", "wiggle-vibrato-large-slowest", "thumb-position", "laissez-vibrer", "wiggle-sawtooth",
        "wigglevibratolargeslowest", "soft-accent-tenuto", "soft-accent-staccato", "soft-accent-tenuto-staccato", "wigglesawtoothwide", "spiccato",
        DEFAULT_EXPRESSIVE_FEATURE_VALUES["Articulation"],
    ],
    # Text
    "Text": [
        "acappella", "aria", "aria-di-sorbetto", "arietta", "arioso", "ballabile", "battaglia", "bergamasca", "burletta", "cabaletta",
        "cadenza", "cantata", "capriccio", "cavatina", "coda", "concerto", "concertino", "concerto-grosso", "da-capo-aria", "dramma-giocoso",
        "dramma-per-musica", "fantasia", "farsa", "festa-teatrale", "fioritura", "intermedio", "intermezzo", "libretto", "melodramma", "opera",
        "opera-buffa", "opera-semiseria", "opera-seria", "operetta", "oratorio", "pasticcio", "ripieno-concerto", "serenata", "soggetto-cavato", "sonata",
        "verismo", "campana", "cornetto", "fagotto", "orchestra", "piano(forte)", "piccolo", "sordun", "timpani", "tuba",
        "viola", "viola-d'amore", "viola-da-braccio", "viola-da-gamba", "violoncello", "alto", "basso", "basso-profondo", "castrato", "coloratura-soprano",
        "contralto", "falsetto", "falsettone", "leggiero-tenor", "musico", "mezzo-soprano", "passaggio", "soprano", "soprano-sfogato", "spinto",
        "spinto-soprano", "squillo", "tenore-contraltino", "tenore-di-grazia-or-leggiero-tenor", "tessitura", "accompagnato", "affrettando",
        "alla-marcia", "a-tempo", "largamente", "larghetto", "lento", "l'istesso-tempo", "mosso",
        "tardo", "(tempo)-rubato", "messa-di-voce", "stentato",
        "tremolo", "affettuoso", "agitato", "animato", "brillante", "bruscamente", "cantabile", "colossale", "comodo", "con-amore",
        "con-brio", "con-fuoco", "con-moto", "con-spirito", "dolce", "drammatico", "feroce", "festoso", "furioso",
        "giocoso", "grandioso", "grazioso", "lacrimoso", "lamentoso", "maestoso", "misterioso", "pesante", "risoluto",
        "scherzando", "solitario", "sotto-(voce)", "sonore", "semplicemente", "slancio", "tranquillo", "volante", "molto",
        "assai", "più", "ma-non-tanto", "ma-non-troppo", "meno", "subito", "lacuna", "ossia",
        "ostinato", "pensato", "ritornello", "segue", "stretto", "attacca", "cambiare", "da-capo-(al-fine)", "dal-segno", "divisi",
        "oppure", "sole", "acciaccatura", "altissimo", "appoggiatura", "arco", "arpeggio", "basso-continuo", "a-bocca-chiusa",
        "chiuso", "coloratura", "coperti", "una-corda", "due-corde", "tre-corde-or-tutte-le-corde", "glissando", "legato", "col-legno", "martellato",
        "pizzicato", "portamento", "sforzando", "scordatura", "con-sordino", "senza-sordino",
        "tutti", "vibrato", "colla-voce", "banda", "comprimario", "convenienze", "coro", "diva", "prima-donna",
        "primo-uomo", "ripieno", "bel-canto", "bravura", "bravo", "maestro", "maestro-collaboratore", "maestro-sostituto", "maestro-suggeritore", "stagione",
        DEFAULT_EXPRESSIVE_FEATURE_VALUES["Text"],
    ],
    # RehearsalMark
    "RehearsalMark": 
        list(map(chr, range(ord("a"), ord("z") + 1))) + [ # the letters
        "i", "i-i", "i-i-i", "i-v", "v", "v-i", "v-i-i", "v-i-i-i", "i-x", "x", # roman numerals
        "introduction", "intro", "bridge", "chorus", "outro", "verse", "theme", "refrain", "solo-section", "solo", "variations", "trio", # song sections
        DEFAULT_EXPRESSIVE_FEATURE_VALUES["RehearsalMark"],
    ],
    # TextSpanner
    "TextSpanner": [DEFAULT_EXPRESSIVE_FEATURE_VALUES["TextSpanner"],],
    # TechAnnotation
    "TechAnnotation": [DEFAULT_EXPRESSIVE_FEATURE_VALUES["TechAnnotation"],],
    # Symbol
    # "Symbol": [
    #     "no-sym", "keyboard-pedal-ped", "notehead-parenthesis-left", "notehead-parenthesis-right", "keyboard-pedal-up", "handbells-mallet-bell-on-table", "guitar-string-7", "strings-harmonic", "pedal ped", "strings-down-bow", "handbells-martellato", "pedalasterisk", "chant-punctum", "notehead-black", "guitar-string-8", "arrow-open-left", "artic-accent-above", "accidental-natural", "breath-mark-comma", "keyboard-pedal-p",
    #     "notehead-parenthesis", "strings-down-bow-turned", "strings-up-bow", "acc dot", "notehead-whole", "accdn-comb-dot", "ornament-trill", "dynamic-messa-di-voce", "vocal-mouth-closed", "accdn-r-h-3-ranks-full-factory", "accidental-flat", "keyboard-play-with-r-h", "keyboard-play-with-l-h", "accidental-sharp", "fermata-above", "augmentation-dot", "arrowhead-open-up", "ornament-mordent", "grace-note-slash-stem-up", "note-quarter-up",
    #     "fingering-5", "staccato", "pict-beater-hammer-plastic-up", "accidental-parens-left", "accdn-r-h-4-ranks-master", "artic-laissez-vibrer-above", "accidental-quarter-tone-sharp-arabic", "mensural-prolation-2", "brass-mute-closed", "dot", "dynamic-forte", "ornament-tremblement-couperin", "trill", "csym-parens-left-tall", "artic-staccato-above", "arrow-black-down", "accidental-quarter-tone-flat-arabic", "artic-staccato-below", "function-parens-left", "mensural-prolation-combining-three-dots",
    #     "csym-bracket-right-tall", "notehead-round-white-slashed-large", "accidental-three-quarter-tones-sharp-arabic", "fingering-4", "flat", "sforza to accent", "csym-bracket-left-tall", "left parenthesis", "ornament-pince-couperin", "plucked-left-hand-pizzicato", "repeat-left", "repeat-right", "accidental-natural-flat", "lute-italian-fret-3", "rcomma", "artic-marcato-below", "fingering-1", "mensural-signum-up", "artic-staccatissimo-above", "fingering-3",
    #     "system-divider", "accidental-bakiye-sharp", "dynamic-f-f-f-f", "ufermata", "bracket", "dynamic-piano", "figbass-4", "keyboard-pedal-s", "mensural-prolation-9", "octave-parens-left", "time-sig-fractional-slash", "accidental-sharp-arabic", "octave-parens-right", "rest-quarter", "unpitched-percussion-clef-1", "accidental-natural-sharp", "figbass-flat", "function-plus", "text-tie", "time-sig-bracket-left",
    #     "accidental-natural-arabic", "arrow-black-up-right", "figbass-3", "fingering-2", "system-divider-long", "arrow-black-up", "artic-laissez-vibrer-below", "dynamic-diminuendo-hairpin", "figbass-2", "mensural-prolation-7", "quart head", "barline-single", "dynamic-sforzato", "f-clef-change", "ornament-up-curve", "time-sig-2", "keyboard-pedal-toe-2", "lute-french-mordent-upper", "lute-italian-fret-7", "lute-italian-fret-9",
    #     "mensural-prolation-6", "ornament-left-facing-half-circle", "time-sig-common", "dynamic-p-p", "dynamic-sforzando-1", "function-parens-right", "grace-note-acciaccatura-stem-up", "lute-italian-fret-6", "lute-italian-fret-8", "mensural-proportion-proportio-dupla-2", "natural", "note-half-up", "notehead-round-black-slashed-large", "rest-16th", "right parenthesis", "staff-1-line", "text-cont-8th-beam-short-stem", "time-sig-parens-right", "accidental-double-sharp-arabic", "analytics-end-stimme",
    #     "artic-tenuto-above", "beam-accel-rit-4", "brass-lift-smooth-short", "conductor-beat-4-simple", "note -1/-4", "rest-8th", "text-cont-8th-beam-long-stem", "time-sig-parens-left", "unmeasured-tremolo", "down bow", "grace-note-slash-stem-down", "mensural-notehead-longa-white", "notehead-square-black-large", "segno", "sharp", "staff-1-line-wide", "accidental-kucuk-mucenneb-flat", "arrow-open-up-right", "arrowhead-open-right", "breath-mark-upbow",
    #     "function-one", "function-three", "g-clef-turned", "half head", "handbells-mallet-bell-suspended", "accdn-comb-r-h-3-ranks-empty", "accdn-r-h-3-ranks-bandoneon", "arrowhead-open-down", "artic-stress-below", "brass-jazz-turn", "figbass-1", "flag-16th-down", "fretboard-6-string-nut", "note-shape-triangle-up-black", "tremolo-divisi-dots-4", "tremolo-divisi-dots-6", "accdn-comb-r-h-4-ranks-empty", "accidental-bracket-left", "accidental-bracket-right", "analytics-hauptstimme",
    #     "beam-accel-rit-10", "dynamic-crescendo-hairpin", "figbass-sharp", "keyboard-play-with-l-h-end", "mensural-prolation-5", "mensural-proportion-tempus-perfectum", "mensural-rest-longa-imperfecta", "pedaldot", "strings-bow-behind-bridge", "strings-thumb-position", "tenuto", "time-sig-1", "time-sig-4", "whole head", "accidental-parens-right", "brass-mute-open", "fermata-below", "figbass-natural", "glissando-down", "keyboard-pedal-toe-1",
    #     "mensural-proportion-3", "ottava", "ottava-bassa-ba", "time-sig-bracket-right", "unicode-note-32nd-up", "accidental-comma-slash-up", "accidental-triple-flat", "dynamic-hairpin-parenthesis-left", "flag-16th-up", "g-clef-change", "guitar-string-3", "handbells-martellato-lift", "keyboard-pedal-heel-3", "lyrics-elision-wide", "mensural-proportion-2", "ornament-oblique-line-horiz-before-note", "pict-beater-hammer-plastic-down", "pict-beater-hand", "accdn-r-h-3-ranks-clarinet", "accidental-wyschnegradsky-6-twelfths-flat",
    #     "barline-heavy", "f-clef", "figbass-5", "guitar-string-4", "handbells-table-single-bell", "lute-italian-fret-4", "mensural-prolation-combining-two-dots", "notehead-round-black-slashed", "time-sig-3", "two", "accdn-r-h-3-ranks-bassoon", "accdn-r-h-3-ranks-harmonium", "accdn-r-h-3-ranks-two-choirs", "accidental-wyschnegradsky-6-twelfths-sharp", "arrowhead-black-left", "artic-marcato-above", "chant-punctum-inclinatum", "csym-parens-right-tall", "dynamic-hairpin-parenthesis-right", "figbass-combining-raising",
    #     "figbass-parens-left", "flag-32nd-up", "fretboard-filled-circle", "function-two", "keyboard-pedal-dot", "lute-italian-fret-2", "lute-italian-fret-5", "lute-italian-vibrato", "mensural-comb-stem-up", "met-note-half-up", "notehead-half-filled", "notehead-rectangular-cluster-white-top", "pict-crotales", "rest-whole", "staff-1-line-narrow", "time-sig-8", "accdn-r-h-3-ranks-organ", "artic-accent-below", "artic-staccatissimo-below", "artic-unstress-above",
    #     "barline-short", "bracket tips up", "caesura straight", "cbass clef", "dfermata", "five", "four", "g-clef", "guitar-string-1", "guitar-string-2", "guitar-volume-swell", "lute-italian-fret-0", "lute-italian-fret-1", "lute-italian-tremolo", "med-ren-natural", "mensural-notehead-longa-black", "mensural-signum-down", "notehead-round-white-slashed", "notehead-square-black", "notehead-x-black",
    #     "one", "ottava-alta", "ottava-bassa-vb", "pict-mar", "pict-timpani", "pict-vib", "prall", "prall prall", "text-cont-16th-beam-short-stem", "text-tuplet-bracket-start-short-stem", "time-sig-6", "up bow", "wind-trill-key", "accdn-r-h-3-ranks-master", "accdn-r-h-4-ranks-bass-alto", "accidental-flat-arabic", "accidental-wyschnegradsky-1-twelfths-sharp", "analytics-nebenstimme", "arrow-open-down-left", "artic-stress-above",
    #     "barline-tick", "beam-accel-rit-11", "beam-accel-rit-2", "dgrace dash", "dpedal toe", "dynamic-hairpin-bracket-left", "dynamic-m-f", "dynamic-p-f", "eight flag", "figbass-7", "flag-32nd-down", "flag-8th-up", "grace dash", "guitar-string-0", "guitar-string-5", "guitar-string-6", "keyboard-pedal-e", "keyboard-pedal-sost", "keyboard-play-with-r-h-end", "lute-barline-end-repeat",
    #     "mensural-notehead-semibrevis-black", "mensural-prolation-4", "mensural-white-longa", "met-note-quarter-up", "note -1/-8", "notehead-plus-black", "ornament-comma", "ornament-oblique-line-horiz-after-note", "pict-beater-combining-parentheses", "plus", "seven", "staff-2-lines", "staff-5-lines", "text-tuplet-3-long-stem", "text-tuplet-bracket-end-long-stem", "text-tuplet-bracket-start-long-stem", "time-sig-5", "time-sig-plus-small", "tremolo-divisi-dots-2", "tuplet-2",
    #     "unicode-note-half-up", "accdn-r-h-3-ranks-authentic-musette", "accdn-r-h-3-ranks-oboe", "accdn-r-h-3-ranks-tremolo-lower-8ve", "accdn-r-h-4-ranks-soft-tenor", "accidental-combining-raise-53-limit-comma", "analytics-hauptrhythmus", "arrow-black-down-left", "bracket-bottom", "brass-doit-short", "brass-fall-smooth-long", "breath-mark-salzedo", "caesura-short", "chant-augmentum", "coda", "dynamic-f-f", "dynamic-hairpin-bracket-right", "dynamic-niente-for-hairpin", "dynamic-p-p-p-p", "dynamic-rinforzando",
    #     "eight rest", "figbass-0", "figbass-double-flat", "fretboard-6-string", "fretboard-o", "function-bracket-left", "function-bracket-right", "guitar-wide-vibrato-stroke", "leger-line", "mensural-oblique-desc-3rd-white", "mensural-oblique-desc-5th-white", "mensural-proportion-4", "mensural-rest-brevis", "mensural-white-brevis", "met-augmentation-dot", "nine", "notehead-double-whole-square", "notehead-slash-vertical-ends-small", "ornament-down-curve", "ornament-precomp-double-cadence-upper-prefix",
    #     "ornament-turn-inverted", "ornament-turn-slash", "pict-beater-snare-sticks-up", "pict-beater-wire-brushes-up", "pict-open-rim-shot", "pict-tom-tom", "quindicesima-alta", "staff-4-lines", "text-tuplet-bracket-end-short-stem", "time-sig-0", "time-sig-7", "time-sig-9", "tuplet-8", "wiggle-random-4",
    #     DEFAULT_EXPRESSIVE_FEATURE_VALUES["Symbol"],
    # ],
}
EXPRESSIVE_FEATURE_TYPE_MAP = {expressive_feature_subtype : expressive_feature_type for expressive_feature_type in tuple(EXPRESSIVE_FEATURES.keys()) for expressive_feature_subtype in EXPRESSIVE_FEATURES[expressive_feature_type]}
VALUE_CODE_MAP = unique(l = [None,] + list(range(N_NOTES)) + sum(list(EXPRESSIVE_FEATURES.values()), [])) # ensure no duplicates
VALUE_CODE_MAP = {VALUE_CODE_MAP[i]: i for i in range(len(VALUE_CODE_MAP))}
CODE_VALUE_MAP = utils.inverse_dict(VALUE_CODE_MAP)

##################################################


# DURATION
##################################################

KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
    768,
]
DURATION_CODE_MAP = {
    i: int(np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))) + 1
    for i in range(MAX_DURATION + 1)
}
DURATION_CODE_MAP[None] = 0
CODE_DURATION_MAP = {
    i + 1: duration for i, duration in enumerate(KNOWN_DURATIONS)
}
CODE_DURATION_MAP[0] = None

##################################################


# INSTRUMENT
##################################################

PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano",
    1: "piano",
    2: "piano",
    3: "piano",
    4: "electric-piano",
    5: "electric-piano",
    6: "harpsichord",
    7: "clavinet",
    # Chromatic Percussion
    8: "celesta",
    9: "glockenspiel",
    10: "music-box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular-bells",
    15: "dulcimer",
    # Organs
    16: "organ",
    17: "organ",
    18: "organ",
    19: "church-organ",
    20: "organ",
    21: "accordion",
    22: "harmonica",
    23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar",
    25: "steel-string-guitar",
    26: "electric-guitar",
    27: "electric-guitar",
    28: "electric-guitar",
    29: "electric-guitar",
    30: "electric-guitar",
    31: "electric-guitar",
    # Basses
    32: "bass",
    33: "electric-bass",
    34: "electric-bass",
    35: "electric-bass",
    36: "slap-bass",
    37: "slap-bass",
    38: "synth-bass",
    39: "synth-bass",
    # Strings
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "strings",
    45: "strings",
    46: "harp",
    47: "timpani",
    # Ensemble
    48: "strings",
    49: "strings",
    50: "synth-strings",
    51: "synth-strings",
    52: "voices",
    53: "voices",
    54: "voices",
    55: "orchestra-hit",
    # Brass
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "trumpet",
    60: "horn",
    61: "brasses",
    62: "synth-brasses",
    63: "synth-brasses",
    # Reed
    64: "soprano-saxophone",
    65: "alto-saxophone",
    66: "tenor-saxophone",
    67: "baritone-saxophone",
    68: "oboe",
    69: "english-horn",
    70: "bassoon",
    71: "clarinet",
    # Pipe
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan-flute",
    76: None,
    77: None,
    78: None,
    79: "ocarina",
    # Synth Lead
    80: "lead",
    81: "lead",
    82: "lead",
    83: "lead",
    84: "lead",
    85: "lead",
    86: "lead",
    87: "lead",
    # Synth Pad
    88: "pad",
    89: "pad",
    90: "pad",
    91: "pad",
    92: "pad",
    93: "pad",
    94: "pad",
    95: "pad",
    # Synth Effects
    96: None,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    # Ethnic
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bag-pipe",
    110: "violin",
    111: "shehnai",
    # Percussive
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: "melodic-tom",
    118: "synth-drums",
    119: "synth-drums",
    # Sound effects
    120: None,
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
}
INSTRUMENT_PROGRAM_MAP = {
    # Pianos
    "piano": 0,
    "electric-piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    # Chromatic Percussion
    "celesta": 8,
    "glockenspiel": 9,
    "music-box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "tubular-bells": 14,
    "dulcimer": 15,
    # Organs
    "organ": 16,
    "church-organ": 19,
    "accordion": 21,
    "harmonica": 22,
    "bandoneon": 23,
    # Guitars
    "nylon-string-guitar": 24,
    "steel-string-guitar": 25,
    "electric-guitar": 26,
    # Basses
    "bass": 32,
    "electric-bass": 33,
    "slap-bass": 36,
    "synth-bass": 38,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "timpani": 47,
    # Ensemble
    "strings": 49,
    "synth-strings": 50,
    "voices": 52,
    "orchestra-hit": 55,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "horn": 60,
    "brasses": 61,
    "synth-brasses": 62,
    # Reed
    "soprano-saxophone": 64,
    "alto-saxophone": 65,
    "tenor-saxophone": 66,
    "baritone-saxophone": 67,
    "oboe": 68,
    "english-horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    # Pipe
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan-flute": 75,
    "ocarina": 79,
    # Synth Lead
    "lead": 80,
    # Synth Pad
    "pad": 88,
    # Ethnic
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bag-pipe": 109,
    "shehnai": 111,
    # Percussive
    "melodic-tom": 117,
    "synth-drums": 118,
}
KNOWN_PROGRAMS = list(k for k, v in PROGRAM_INSTRUMENT_MAP.items() if v is not None)
KNOWN_INSTRUMENTS = unique(l = INSTRUMENT_PROGRAM_MAP.keys())
INSTRUMENT_CODE_MAP = {instrument: i + 1 for i, instrument in enumerate(KNOWN_INSTRUMENTS)}
INSTRUMENT_CODE_MAP[None] = 0
CODE_INSTRUMENT_MAP = utils.inverse_dict(INSTRUMENT_CODE_MAP)

##################################################


# VELOCITY
##################################################

NON_VELOCITY = 0
VELOCITY_CODE_MAP = {i: i + 1 for i in range(MAX_VELOCITY + 1)}
VELOCITY_CODE_MAP[None] = 0
CODE_VELOCITY_MAP = utils.inverse_dict(VELOCITY_CODE_MAP)

##################################################


# TOKEN COUNTS
##################################################

N_TOKENS = [
    max(TYPE_CODE_MAP.values()) + 1,
    max(BEAT_CODE_MAP.values()) + 1,
    max(POSITION_CODE_MAP.values()) + 1,
    max(VALUE_CODE_MAP.values()) + 1,
    max(DURATION_CODE_MAP.values()) + 1,
    max(INSTRUMENT_CODE_MAP.values()) + 1,
    max(VELOCITY_CODE_MAP.values()) + 1,
]

##################################################


# ENCODING-GENERATING FUNCTIONS
##################################################

def get_encoding(include_velocity: bool = False) -> dict:
    """Return the encoding configurations."""
    encoding = {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "dimensions": DIMENSIONS[:DIMENSIONS.index("time")],
        "n_tokens": N_TOKENS,
        "type_code_map": TYPE_CODE_MAP,
        "beat_code_map": BEAT_CODE_MAP,
        "position_code_map": POSITION_CODE_MAP,
        "value_code_map": VALUE_CODE_MAP,
        "duration_code_map": DURATION_CODE_MAP,
        "instrument_code_map": INSTRUMENT_CODE_MAP,
        "code_type_map": CODE_TYPE_MAP,
        "code_beat_map": CODE_BEAT_MAP,
        "code_position_map": CODE_POSITION_MAP,
        "code_value_map": CODE_VALUE_MAP,
        "code_duration_map": CODE_DURATION_MAP,
        "code_instrument_map": CODE_INSTRUMENT_MAP,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "velocity_code_map": VELOCITY_CODE_MAP,
        "code_velocity_map": CODE_VELOCITY_MAP,
    }
    if not include_velocity:
        del encoding["velocity_code_map"], encoding["code_velocity_map"]
        encoding["dimensions"].remove("velocity") # remove velocity from dimensions
        encoding["n_tokens"] = encoding["n_tokens"][:-1] # remove token count for velocity
    return encoding


def load_encoding(filepath: str) -> dict:
    """Load encoding configurations from a JSON file. Make sure types are correct."""

    # load in encoding from json
    encoding = utils.load_json(filepath = filepath)

    # constant values
    encoding["resolution"] = int(encoding["resolution"])
    encoding["max_beat"] = int(encoding["max_beat"])
    encoding["max_duration"] = int(encoding["max_duration"])
    encoding["n_tokens"] = list(map(int, encoding["n_tokens"]))
    encoding["dimensions"] = list(map(str, encoding["dimensions"]))

    # type-code
    encoding["type_code_map"] = {str(k): int(v) for k, v in encoding["type_code_map"].items()}
    encoding["code_type_map"] = {int(k): str(v) for k, v in encoding["code_type_map"].items()}

    # instrument
    encoding["program_instrument_map"] = {int(k): str(v) if v not in NA_VALUES else None for k, v in encoding["program_instrument_map"].items()}
    encoding["instrument_program_map"] = {str(k): int(v) for k, v in encoding["instrument_program_map"].items()}
    encoding["instrument_code_map"] = {str(k) if k not in NA_VALUES else None: int(v) for k, v in encoding["instrument_code_map"].items()}
    encoding["code_instrument_map"] = {int(k): str(v) if v not in NA_VALUES else None for k, v in encoding["code_instrument_map"].items()}

    # all integer values
    for key in ("beat_code_map", "position_code_map", "duration_code_map", "code_beat_map", "code_position_map", "code_duration_map"):
        encoding[key] = {int(k) if k not in NA_VALUES else None: int(v) if v not in NA_VALUES else None for k, v in encoding[key].items()}

    # velocity
    velocity_maps = {"velocity_code_map", "code_velocity_map"}
    if all((velocity_map in encoding.keys() for velocity_map in velocity_maps)):
        for key in velocity_maps:
            encoding[key] = {int(k) if k not in NA_VALUES else None: int(v) if v not in NA_VALUES else None for k, v in encoding[key].items()}
    del velocity_maps

    # values
    encoding["value_code_map"] = {str(k) if k not in NA_VALUES else None: int(v) for k, v in encoding["value_code_map"].items()}
    encoding["code_value_map"] = {int(k): str(v) if v not in NA_VALUES else None for k, v in encoding["code_value_map"].items()}
    for i in range(128): # convert integer values
        encoding["value_code_map"][i] = int(encoding["value_code_map"][str(i)])
        del encoding["value_code_map"][str(i)]
        encoding["code_value_map"][i + 1] = int(encoding["code_value_map"][i + 1])

    return encoding

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(prog = "Representation", description = "Test Encoding/Decoding mechanisms for MuseScore data.")
    parser.add_argument("-o", "--output_dir", type = str, default = ENCODING_DIR, help = "Directory in which to store the encoding file")
    parser.add_argument("-v", "--velocity", action = "store_true", help = "Whether to add a velocity field.")
    args = parser.parse_args()

    # get the encoding
    encoding = get_encoding(include_velocity = args.velocity)

    # save the encoding
    encoding_filepath = f"{args.output_dir}/{ENCODING_BASENAME}"
    utils.save_json(filepath = encoding_filepath, data = encoding)
    print(f"Encoding saved to {encoding_filepath}")

    # load encoding
    encoding = load_encoding(filepath = encoding_filepath)

    # print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in (
            "instrument_code_map",
            "code_instrument_map",
            "program_instrument_map",
            "instrument_program_map",
        ):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent = 2)

    # print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # print the number of tokens
    print(f"{' Number of tokens ':=^40}")
    for key, value in zip(encoding["dimensions"], encoding["n_tokens"]):
        print(f"{key}: {value}")

##################################################


# SCRAPE DATA OFF THE INTERNET
##################################################

def retrieve_italian_musical_terms() -> list:

    # imports
    import requests
    from bs4 import BeautifulSoup

    # load in page
    r = requests.get("https://en.m.wikipedia.org/wiki/List_of_Italian_musical_terms_used_in_English", headers = ({"User-Agent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', "Accept-Language": "en-US, en;q=0.5"}))
    soup = BeautifulSoup(r.content)

    # find all table elements
    terms = soup.find_all("td", attrs = {"bgcolor": "#ddffdd"})
    terms = list(term.text.strip() for term in terms)

    return terms

##################################################
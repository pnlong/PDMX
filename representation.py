# README
# Phillip Long
# November 3, 2023

# mappings for various representations of music
# copied from https://github.com/salu133445/mmt/blob/main/mmt/representation.py


# IMPORTS
##################################################
import pathlib
import pprint

import muspy
import numpy as np
import pretty_midi

import utils
##################################################


# CONSTANTS
##################################################

# Configuration
RESOLUTION = 12
MAX_BEAT = 1024
MAX_DURATION = 768  # Remember to modify known durations as well!

##################################################


# DIMENSIONS
##################################################

# (NOTE: "type" must be the first dimension!)
# (NOTE: Remember to modify N_TOKENS as well!)
DIMENSIONS = ["type", "beat", "position", "value", "duration", "instrument", "time", "time.s"] # last 2 columns are for sorting, and will be discarded later
assert DIMENSIONS[0] == "type"

##################################################


# TYPE
##################################################

TYPE_CODE_MAP = {
    "start-of-song": 0,
    "instrument": 1,
    "start-of-notes": 2,
    "note": 3,
    "end-of-song": 4,
    "expressive-feature": 5,
    "grace-note": 6,
}
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

TEMPO_QPM_MAP = { # each value is the maximum BPM before we go up a tempo, found at https://en.wikipedia.org/wiki/Tempo
    "Larghissimo": 24,
    "Grave": 40, # also Adagissimo
    "Largo": 54, # also Larghetto
    "Adagio": 68,
    "Adagietto": 80,
    "Andante": 94, # or Lento
    "Andantino": 108,
    "Moderato": 114,
    "Allegretto": 120,
    "Allegro": 156,
    "Vivace": 176,
    "Presto": 200,
    "Prestissimo": 1e10, # some arbitrary large number
}
QPM_TEMPO_MAP = utils.inverse_dict(TEMPO_QPM_MAP)
def QPM_TEMPO_MAPPER(qpm: float):
    if qpm is None: # default if qpm argument is none
        qpm = 112 # default bpm
    for bpm in QPM_TEMPO_MAP.keys():
        if qpm <= bpm:
            return QPM_TEMPO_MAP[bpm]

EXPRESSIVE_FEATURES = {
    # Barlines
    "Barline": [
        "double-barline", "end-barline", "dotted-barline", "dashed-barline",
        "barline",
        ],
    # Key Signatures
    "KeySignature": ["keysig-change",],
    # Time Signatures
    "TimeSignature": ["timesig-change",],
    # Fermata
    "Fermata": ["fermata",],
    # SlurSpanner
    "SlurSpanner": ["slur",],
    # PedalSpanner
    "PedalSpanner": ["pedal",],
    # Tempo
    "Tempo": list(TEMPO_QPM_MAP.keys()) + ["tempo-marking",],
    # TempoSpanner
    "TempoSpanner": [],
    # Dynamic
    "Dynamic": [
        "pppppp", "ppppp", "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff", "fffff", "ffffff",
        "sfpp", "sfp", "sf", "sff", "sfz", "sffz", "fz", "rf", "rfz", "fp", "pf", "s", "r", "z", "n", "m",
        "dynamic-marking",
        ],
    # HairPinSpanner
    "HairPinSpanner": [],
    # TechAnnotation
    "TechAnnotation": [],
    # Articulation (Chunks)
    "Articulation": [
        "staccato", "artic-staccato-above", "artic-staccato-below", "sforzato", "artic-accent-above",
        "artic-accent-below", "marcato", "tenuto", "fermata", "ornament-trill",
        "artic-marcato-above", "portato", "artic-tenuto-above", "artic-tenuto-below", "staccatissimo",
        "trill", "artic-staccatissimo-above", "strings-up-bow", "strings-down-bow", "artic-staccatissimo-below",
        "artic-tenuto-staccato-below", "artic-tenuto-staccato-above", "downbow", "ornament-mordent", "upbow",
        "ornament-mordent-inverted", "prall", "artic-accent-staccato-above", "artic-marcato-staccato-above", "artic-tenuto-accent-below",
        "artic-tenuto-accent-above", "artic-accent-staccato-below", "brass-mute-closed", "umarcato", "artic-marcato-tenuto-above",
        "artic-marcato-below", "guitar-fade-out", "mordent", "fadeout", "ouvert",
        "plusstop", "artic-staccatissimo-wedge-above", "ornament-turn", "prallprall", "brass-mute-open",
        "ornament-tremblement", "turn", "dportato", "ustaccatissimo", "artic-staccatissimo-wedge-below",
        "dmarcato", "lutefingering-1st", "lutefingeringthumb", "strings-harmonic", "fadein",
        "dfermata", "artic-marcato-tenuto-below", "downprall", "snappizzicato", "ornament-precomp-mordent-upper-prefix",
        "artic-marcato-staccato-below", "lute-fingering-r-h-first", "plucked-snap-pizzicato-above", "shortfermata", "prallmordent",
        "artic-stress-above", "lute-fingering-r-h-second", "lute-fingering-r-h-thumb", "ornament-prall-mordent", "artic-soft-accent-below",
        "uportato", "artic-stress-below", "ornament-turn-inverted", "wiggle-vibrato-large-faster", "reverseturn",
        "artic-staccatissimo-stroke-above", "artic-unstress-above", "fermata-above", "guitar-volume-swell", "longfermata",
        "guitar-fade-in", "artic-soft-accent-above", "downmordent", "ornament-up-prall", "volumeswell",
        "thumb", "upprall", "artic-unstress-below", "dstaccatissimo", "ornament-line-prall",
        "lineprall", "schleifer", "verylongfermata", "wigglevibratolargefaster", "artic-staccatissimo-stroke-below",
        "pralldown", "ornament-prall-up", "espressivo", "upmordent", "ornament-prall-down",
        "ushortfermata", "ornament-precomp-slide", "ornament-up-mordent", "prallup", "ornament-down-mordent",
        "no-sym", "wiggle-vibrato-large-slowest", "lutefingering-2nd", "strings-thumb-position", "artic-laissez-vibrer-above",
        "artic-laissez-vibrer-below", "wiggle-sawtooth", "wigglevibratolargeslowest", "lutefingering-3rd", "ulongfermata",
        "artic-soft-accent-tenuto-below", "artic-soft-accent-staccato-above", "artic-soft-accent-tenuto-staccato-above", "lute-fingering-r-h-third", "wigglesawtoothwide",
        "articulation",
    ],
    # Text
    "Text": [],
    # TextSpanner
    "TextSpanner": [],
    # RehearsalMark
    "RehearsalMark": [],
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
    # ],
}
VALUE_CODE_MAP = [None,] + list(range(128)) + sum(list(EXPRESSIVE_FEATURES.values()), [])
VALUE_CODE_MAP = {VALUE_CODE_MAP[i]: i for i in range(len(VALUE_CODE_MAP))}
def VALUE_CODE_MAPPER(value, value_code_map: dict) -> int:
    try:
        code = value_code_map[value]
    except KeyError:
        code = -1
    return code
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
KNOWN_PROGRAMS = list(
    k for k, v in PROGRAM_INSTRUMENT_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))
INSTRUMENT_CODE_MAP = {
    instrument: i + 1 for i, instrument in enumerate(KNOWN_INSTRUMENTS)
}
INSTRUMENT_CODE_MAP[None] = 0
CODE_INSTRUMENT_MAP = utils.inverse_dict(INSTRUMENT_CODE_MAP)

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
]

##################################################


# ENCODER FUNCTIONS
##################################################

def get_encoding() -> dict:
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "dimensions": DIMENSIONS,
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
    }


def load_encoding(filename: str) -> dict:
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename = filename)
    for key in (
        "code_type_map",
        "code_beat_map",
        "code_position_map",
        "code_duration_map",
        "code_value_map",
        "code_instrument_map",
        "beat_code_map",
        "position_code_map",
        "duration_code_map",
        "value_code_map",
        "program_instrument_map",
    ):
        encoding[key] = {(int(k) if k != "null" else None): v for k, v in encoding[key].items()}
    return encoding


# def extract_notes(music: BetterMusic, resolution: int) -> np.array:
#     """Return a MusPy music object as a note sequence.
#     Each row of the output is a note specified as follows.
#         (beat, position, value, duration, program)
#     """
#     # Check resolution
#     assert music.resolution == resolution
#     # Extract notes
#     notes = []
#     for track in music:
#         if track.is_drum or track.program not in KNOWN_PROGRAMS:
#             continue
#         for note in track:
#             beat, position = divmod(note.time, resolution)
#             notes.append((beat, position, note.pitch, note.duration, track.program))
#     # Deduplicate and sort the notes
#     notes = sorted(set(notes))
#     return np.array(notes)


def encode(path: str, encoding: dict) -> np.array:
    """Encode a note sequence into a sequence of codes.
    Each row of the input is a note specified as follows.
        (beat, position, value, duration, program)
    Each row of the output is encoded as follows.
        (event_type, beat, position, value, duration, instrument)
    """

    # load in npy file from path parameter
    data = np.load(file = path)

    # get variables
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # Get maps
    type_code_map = encoding["type_code_map"]
    beat_code_map = encoding["beat_code_map"]
    position_code_map = encoding["position_code_map"]
    value_code_map = encoding["value_code_map"]
    duration_code_map = encoding["duration_code_map"]
    instrument_code_map = encoding["instrument_code_map"]
    program_instrument_map = encoding["program_instrument_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # apply encodings
    data[:, DIMENSIONS.index("type")] = list(map(lambda type_: type_code_map[str(type_)], data[:, DIMENSIONS.index("type")])) # encode type column
    data[:, DIMENSIONS.index("beat")] = list(map(lambda beat: beat_code_map[int(beat)], data[:, DIMENSIONS.index("beat")])) # encode beat
    data[:, DIMENSIONS.index("position")] = list(map(lambda position: position_code_map[int(position)], data[:, DIMENSIONS.index("position")])) # encode position
    data[:, DIMENSIONS.index("value")] = list(map(value_code_mapper, data[:, DIMENSIONS.index("value")])) # encode value column
    data[:, DIMENSIONS.index("duration")] = list(map(lambda duration: duration_code_map[int(duration)], data[:, DIMENSIONS.index("duration")])) # encode duration
    data[:, DIMENSIONS.index("instrument")] = list(map(lambda program: program_instrument_map[int(program)]), data[:, DIMENSIONS.index("instrument")]) # encode instrument column

    # Start the codes with an SOS row
    codes = [(type_code_map["start-of-song"], 0, 0, 0, 0, 0)]

    # Extract instruments
    instruments = set(program_instrument_map[note[-1]] for note in notes)

    # Encode the instruments
    instrument_codes = []
    for instrument in instruments:
        # Skip unknown instruments
        if instrument is None:
            continue
        row = [type_code_map["instrument"], 0, 0, 0, 0, 0]
        row[instrument_dim] = instrument_code_map[instrument]
        instrument_codes.append(row)

    # Sort the instruments and append them to the code sequence
    instrument_codes.sort()
    codes.extend(instrument_codes)

    # Encode the notes
    codes.append((type_code_map["start-of-notes"], 0, 0, 0, 0, 0))
    for beat, position, value, duration, program in notes:
        # Skip if max_beat has reached
        if beat > max_beat:
            continue
        # Skip unknown instruments
        instrument = program_instrument_map[program]
        if instrument is None:
            continue
        # Encode the note
        row = [type_code_map["note"], 0, 0, 0, 0, 0]
        row[beat_dim] = beat_code_map[beat]
        row[position_dim] = position_code_map[position]
        row[value_dim] = value_code_map[value]
        row[duration_dim] = duration_code_map[min(duration, max_duration)]
        row[instrument_dim] = instrument_code_map[instrument]
        codes.append(row)

    # End the codes with an EOS row
    codes.append((type_code_map["end-of-song"], 0, 0, 0, 0, 0))

    return np.array(codes)

##################################################


# DECODER FUNCTIONS
##################################################

def decode_notes(codes, encoding):
    """Decode codes into a note sequence.

    Each row of the input is encoded as follows.

        (event_type, beat, position, value, duration, instrument)

    """
    # Get variables and maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_value_map = encoding["code_value_map"]
    code_duration_map = encoding["code_duration_map"]
    code_instrument_map = encoding["code_instrument_map"]
    instrument_program_map = encoding["instrument_program_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # Decode the codes into a sequence of notes
    notes = []
    for row in codes:
        event_type = code_type_map[int(row[0])]
        if event_type in ("start-of-song", "instrument", "start-of-notes"):
            continue
        elif event_type == "end-of-song":
            break
        elif event_type == "note":
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            value = code_value_map[int(row[value_dim])]
            duration = code_duration_map[int(row[duration_dim])]
            instrument = code_instrument_map[int(row[instrument_dim])]
            program = instrument_program_map[instrument]
            notes.append((beat, position, value, duration, program))
        else:
            raise ValueError("Unknown event type.")

    return notes


def reconstruct(notes, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(resolution=resolution, tempos=[muspy.Tempo(0, 100)])

    # Append the tracks
    programs = sorted(set(note[-1] for note in notes))
    for program in programs:
        music.tracks.append(muspy.Track(program))

    # Append the notes
    for beat, position, value, duration, program in notes:
        time = beat * resolution + position
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(time, value, duration))

    return music


def decode(codes, encoding):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, value, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes = decode_notes(codes, encoding)

    # Reconstruct the music object
    music = reconstruct(notes, resolution)

    return music


def dump(data, encoding):
    """Decode the codes and dump as a string."""
    # Get maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_value_map = encoding["code_value_map"]
    code_duration_map = encoding["code_duration_map"]
    code_instrument_map = encoding["code_instrument_map"]

    # Get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # Iterate over the rows
    lines = []
    for row in data:
        event_type = code_type_map[int(row[0])]
        if event_type == "start-of-song":
            lines.append("Start of song")
        elif event_type == "end-of-song":
            lines.append("End of song")
        elif event_type == "instrument":
            instrument = code_instrument_map[int(row[instrument_dim])]
            lines.append(f"Instrument: {instrument}")
        elif event_type == "start-of-notes":
            lines.append("Start of notes")
        elif event_type == "note":
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            value = pretty_midi.note_number_to_name(
                code_value_map[int(row[value_dim])]
            )
            duration = code_duration_map[int(row[duration_dim])]
            instrument = code_instrument_map[int(row[instrument_dim])]
            lines.append(
                f"Note: beat={beat}, position={position}, value={value}, "
                f"duration={duration}, instrument={instrument}"
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    return "\n".join(lines)


def save_txt(filename, data, encoding):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, encoding))


def save_csv_notes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="beat,position,value,duration,program",
        comments="",
    )


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 6
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="type,beat,position,value,duration,instrument",
        comments="",
    )

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding.json"
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
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
            pprint.pprint(value, indent=2)

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # Print the number of tokens
    print(f"{' Number of tokens ':=^40}")
    for key, value in zip(DIMENSIONS, N_TOKENS):
        print(f"{key}: {value}")

    # Print an example
    print(f"{'Example':=^40}")
    codes = np.array(
        (
            (0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 3),
            (1, 0, 0, 0, 0, 33),
            (2, 0, 0, 0, 0, 0),
            (3, 1, 1, 49, 15, 3),
            (3, 1, 1, 61, 15, 3),
            (3, 1, 1, 65, 15, 3),
            (3, 1, 1, 68, 10, 33),
            (3, 1, 1, 68, 15, 3),
            (3, 2, 1, 68, 10, 33),
            (3, 3, 1, 68, 10, 33),
            (3, 4, 1, 61, 10, 33),
            (3, 4, 1, 61, 15, 3),
            (3, 4, 1, 65, 4, 33),
            (3, 4, 1, 65, 10, 3),
            (3, 4, 1, 68, 10, 3),
            (3, 4, 1, 73, 10, 3),
            (3, 4, 13, 63, 4, 33),
            (4, 0, 0, 0, 0, 0),
        ),
        int,
    )
    print(f"Codes:\n{codes}")

    print("-" * 40)
    print(f"Decoded:\n{dump(codes, encoding)}")

    music = decode(codes, encoding)
    print("-" * 40)
    print(f"Decoded music:\n{music}")

    encoded = encode(music, encoding)
    print("-" * 40)
    print(f"Encoded:\n{encoded}")
    assert np.all(codes == encoded)

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

    # EXPRESSIVE_FEATURES = [
    #     "A cappella",
    #     "Aria",
    #     "Aria di sorbetto",
    #     "Arietta",
    #     "Arioso",
    #     "Ballabile",
    #     "Battaglia",
    #     "Bergamasca",
    #     "Burletta",
    #     "Cabaletta",
    #     "Cadenza",
    #     "Cantata",
    #     "Capriccio",
    #     "Cavatina",
    #     "Coda",
    #     "Concerto",
    #     "Concertino",
    #     "Concerto grosso",
    #     "Da capo aria",
    #     "Dramma giocoso",
    #     "Dramma per musica",
    #     "Fantasia",
    #     "Farsa",
    #     "Festa teatrale",
    #     "Fioritura",
    #     "Intermedio",
    #     "Intermezzo",
    #     "Libretto",
    #     "Melodramma",
    #     "Opera",
    #     "Opera buffa",
    #     "Opera semiseria",
    #     "Opera seria",
    #     "Operetta",
    #     "Oratorio",
    #     "Pasticcio",
    #     "Ripieno concerto",
    #     "Serenata",
    #     "Soggetto cavato",
    #     "Sonata",
    #     "Verismo",
    #     "Campana",
    #     "Cornetto",
    #     "Fagotto",
    #     "Orchestra",
    #     "Piano(forte)",
    #     "Piccolo",
    #     "Sordun",
    #     "Timpani",
    #     "Tuba",
    #     "Viola",
    #     "Viola d'amore",
    #     "Viola da braccio",
    #     "Viola da gamba",
    #     "Violoncello",
    #     "Alto",
    #     "Basso",
    #     "Basso profondo",
    #     "Castrato",
    #     "Coloratura soprano",
    #     "Contralto",
    #     "Falsetto",
    #     "Falsettone",
    #     "Leggiero tenor",
    #     "Musico",
    #     "Mezzo-soprano",
    #     "Passaggio",
    #     "Soprano",
    #     "Soprano sfogato",
    #     "Spinto",
    #     "Spinto soprano",
    #     "Squillo",
    #     "Tenore contraltino",
    #     "Tenore di grazia or Leggiero tenor",
    #     "Tessitura",
    #     "Accelerando",
    #     "Accompagnato",
    #     "Adagio",
    #     "Adagietto",
    #     "Affrettando",
    #     "Alla marcia",
    #     "Allargando",
    #     "Allegro",
    #     "Allegretto",
    #     "Andante",
    #     "Andantino",
    #     "A tempo",
    #     "Fermata",
    #     "Grave",
    #     "Largo",
    #     "Largamente",
    #     "Larghetto",
    #     "Lento",
    #     "Lentando",
    #     "L'istesso tempo",
    #     "Moderato",
    #     "Mosso",
    #     "Presto",
    #     "Prestissimo",
    #     "Rallentando",
    #     "Ritardando",
    #     "Tardo",
    #     "Tempo",
    #     "(Tempo) rubato",
    #     "Tenuto",
    #     "Vivace",
    #     "Calando",
    #     "Crescendo",
    #     "Decrescendo",
    #     "Diminuendo",
    #     "Forte",
    #     "Fortissimo",
    #     "Mezzo forte",
    #     "Marcato",
    #     "Messa di voce",
    #     "Piano",
    #     "Pianissimo",
    #     "Mezzo piano",
    #     "Sforzando",
    #     "Stentato",
    #     "Tremolo",
    #     "Affettuoso",
    #     "Agitato",
    #     "Animato",
    #     "Brillante",
    #     "Bruscamente",
    #     "Cantabile",
    #     "Colossale",
    #     "Comodo",
    #     "Con amore",
    #     "Con brio",
    #     "Con fuoco",
    #     "Con moto",
    #     "Con spirito",
    #     "Dolce",
    #     "Drammatico",
    #     "Espressivo",
    #     "Feroce",
    #     "Festoso",
    #     "Furioso",
    #     "Giocoso",
    #     "Grandioso",
    #     "Grazioso",
    #     "Lacrimoso",
    #     "Lamentoso",
    #     "Maestoso",
    #     "Misterioso",
    #     "Morendo",
    #     "Pesante",
    #     "Risoluto",
    #     "Scherzando",
    #     "Solitario",
    #     "Sotto (voce)",
    #     "Sonore",
    #     "Semplicemente",
    #     "Slancio",
    #     "Tranquillo",
    #     "Vivace",
    #     "Volante",
    #     "Molto",
    #     "Assai",
    #     "Più",
    #     "Poco",
    #     "poco a poco",
    #     "ma non tanto",
    #     "ma non troppo",
    #     "Meno",
    #     "Subito",
    #     "Lacuna",
    #     "Ossia",
    #     "Ostinato",
    #     "Pensato",
    #     "Ritornello",
    #     "Segue",
    #     "Stretto",
    #     "Attacca",
    #     "Cambiare",
    #     "Da Capo (al fine)",
    #     "Dal Segno",
    #     "Divisi",
    #     "Oppure",
    #     "Solo",
    #     "Sole",
    #     "Acciaccatura",
    #     "Altissimo",
    #     "Appoggiatura",
    #     "Arco",
    #     "Arpeggio",
    #     "Basso continuo",
    #     "A bocca chiusa",
    #     "Chiuso",
    #     "Coloratura",
    #     "Coperti",
    #     "Una corda",
    #     "Due corde",
    #     "Tre corde or tutte le corde",
    #     "Glissando",
    #     "Legato",
    #     "Col legno",
    #     "Martellato",
    #     "Pizzicato",
    #     "Portamento",
    #     "Portato",
    #     "Sforzando",
    #     "Scordatura",
    #     "Con sordino",
    #     "Senza sordino",
    #     "Spiccato",
    #     "Staccato",
    #     "Staccatissimo",
    #     "Tutti",
    #     "Vibrato",
    #     "Colla voce",
    #     "Banda",
    #     "Comprimario",
    #     "Concertino",
    #     "Convenienze",
    #     "Coro",
    #     "Diva",
    #     "Prima donna",
    #     "Primo uomo",
    #     "Ripieno",
    #     "Bel canto",
    #     "Bravura",
    #     "Bravo",
    #     "Maestro",
    #     "Maestro collaboratore",
    #     "Maestro sostituto",
    #     "Maestro suggeritore",
    #     "Stagione"
    # ]

##################################################
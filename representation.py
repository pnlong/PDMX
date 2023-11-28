# README
# Phillip Long
# November 3, 2023

# mappings for various representations of music
# copied from https://github.com/salu133445/mmt/blob/main/mmt/representation.py


# IMPORTS
##################################################
import pprint
import numpy as np
import pandas as pd
import pretty_midi
import warnings
from typing import List
from re import sub
import utils
import argparse
from read_mscz.music import BetterMusic
from read_mscz.classes import *
##################################################


# CONSTANTS
##################################################

# configuration
RESOLUTION = 12
MAX_BEAT = 1024
MAX_DURATION = 768  # remember to modify known durations as well!

# encoding
CONDITIONINGS = ("sort", "prefix", "anticipation") # there are three options for conditioning 
DEFAULT_CONDITIONING = CONDITIONINGS[0]
ENCODING_ARRAY_TYPE = np.int64
SIGMA = 5.0 # for anticipation conditioning

##################################################





##################################################
##################################################
##################################################
#
#   __  __                _              
#  |  \/  |__ _ _ __ _ __(_)_ _  __ _ ___
#  | |\/| / _` | '_ \ '_ \ | ' \/ _` (_-<
#  |_|  |_\__,_| .__/ .__/_|_||_\__, /__/
#              |_|  |_|         |___/    
#
##################################################
##################################################
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

EXPRESSIVE_FEATURE_TYPE_STRING = "expressive-feature"
TYPE_CODE_MAP = {
    "start-of-song": 0,
    "instrument": 1,
    "start-of-notes": 2,
    "note": 3,
    "end-of-song": 4,
    EXPRESSIVE_FEATURE_TYPE_STRING: 5,
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
    "prestissimo": 1e10, # some arbitrary large number
}
QPM_TEMPO_MAP = utils.inverse_dict(TEMPO_QPM_MAP)
DEFAULT_QPM = 112
def QPM_TEMPO_MAPPER(qpm: float):
    if qpm is None: # default if qpm argument is none
        qpm = DEFAULT_QPM # default bpm
    for bpm in QPM_TEMPO_MAP.keys():
        if qpm <= bpm:
            return QPM_TEMPO_MAP[bpm]

EXPRESSIVE_FEATURES = {
    # Barlines
    "Barline": [
        "double-barline", "end-barline", "dotted-barline", "dashed-barline",
        "barline", # default value
        ],
    # Key Signatures
    "KeySignature": [
        "keysig-change",
        ],
    # Time Signatures
    "TimeSignature": [
        "timesig-change",
        ],
    # Fermata
    "Fermata": [
        "fermata",
        ],
    # SlurSpanner
    "SlurSpanner": [
        "slur",
        ],
    # PedalSpanner
    "PedalSpanner": [
        "pedal",
        ],
    # Tempo
    "Tempo": list(TEMPO_QPM_MAP.keys()), # no default value, default is set above in QPM_TEMPO_MAPPER
    # TempoSpanner
    "TempoSpanner": [
        "lentando", "lent", "smorzando", "smorz", "sostenuto", "sosten", "accelerando", "accel", "allargando", "allarg", "rallentando", "rall", "rallent", "ritardando", "rit",
        "tempo", # default value
        ],
    # Dynamic
    "Dynamic": [
        "pppppp", "ppppp", "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff", "fffff", "ffffff",
        "sfpp", "sfp", "sf", "sff", "sfz", "sffz", "fz", "rf", "rfz", "fp", "pf", "s", "r", "z", "n", "m",
        "dynamic-marking", # default value
        ],
    # HairPinSpanner
    "HairPinSpanner": [
        "cresc", "dim", "dynamic-mezzodynamic-forte", "cresc-sempre", "molto-cresc", "dimin", "decresc",
        "cresc-poco-a-poco", "poco-a-poco-cresc", "cresc-molto", "poco-cresc", "crescendo", "più-cresc",
        "scen", "sempre-cresc", "do", "cre", "morendo", "dim-e-rit", "diminuendo", "sempre-dim", "poco-a-poco",
        "smorz", "cres", "poco-a-poco-dim", "dim-sempre", "rall-e-dim", "dim-molto", "keyboard-pedal-ped", "cresc-poco",
        "dim-e-rall", "poco-a-poco-crescendo", "molto-dim", "dim-e-poco-rit", "poco", "poco-dim", "sempre-più-dim",
        "dim-poco-a-poco", "cre-scen-do", "accel-e-cresc", "ed-allarg", "poco-rit", "crescendo-molto", "crescendo-poco-a-poco",
        "smorzando", "string", "un-poco-cresc", "cresc-y-accel", "rit-e-dim", "ritard", "cresc-e-stringendo", "dim-poco-a-poco-e-rit",
        "molto-rit", "più-dim", "cresc-ed-accel", "crescpoco-string", "perdendo", "rall-molto", "sempre", "cresc-e-string", "e-cresc",
        "piu-cresc", "poco-rall", "poco-riten", "calando", "cresc-e-rit", "crescendo-e-rit", "dim-e-poco-riten", "dim-e-sosten", "poco-a",
        "dynamic-forte", "allargando", "cre-scend-do", "dim-al-fine", "molto-crescendo", "più-piano", "smorz-e-rallent", "a-tempo",
        "cresc-assai", "crescsempre", "descresc", "dim-poco", "dim-poco-rit", "dim-rall", "dim-rit", "dimsempre",
        "molto", "più-rinforz", "poco-rallent", "rit-et-dim", "sempre-dim-e-rit-al-fine", "a-c-c-e-l-e-r-a-n-d-o", "cresc-", "cresc-e-agitato",
        "cresc-un-piu-animato", "dim-e-molto-rall", "dimin-e-ritard", "dimin-poco-a-poco", "forzando", "poco-a-poco-cresc-molto", "poco-a-poco-dimin",
        "scendo", "un-poco-animato-e-cresc", "dynamic-mezzodynamic-piano", "dynamic-pianodynamic-piano", "cédez", "ral", "cresc-appassionato",
        "cresc-con-molto-agitazione", "cresc-e-accel", "cresc-e-accelerando", "cresc-e-animato", "cresc-e-appassionato", "cresc-e-rall-sempre",
        "cresc-e-stretto", "cresc-molto-e-stringendo", "cresc-sempre-poco-a-poco", "cresc-un-poco-animato", "di-min", "dim-e-ritardando", "dim-e-riten",
        "dim-ed-allarg", "dim-subito", "dimin-e-poco-riten", "diminuendo-subito", "e-poco-rit", "en", "gritando-shouting", "increase", "mf", "più-moto",
        "poco-a-poco-cre", "poco-crescendo", "rallentando", "sempre-piu-piano", "sempre-più-cresc", "sempre-più-cresc-e", "sempre-rit-e-dim-sin-al-fine",
        "string-e-cresc", "stringendo", "stringendo-e", "dynamicdynamic-forte", "alternative-solo", "i-i-c", "i-v-c", "preferred-solo", "serrez", "sw",
        "très-dim", "accelerando", "accelerando-e-sempre-cresc", "agitato-e-sempre-più-cresc", "al", "allarg", "ancor-più-cresc", "animando", "ca",
        "calmando-dim", "cantabile-cresc", "con-fuoco", "couple-sw", "cr-es-c", "cre-scen", "cres-c", "cresc-animato", "cresc-e",
        "cresc-e-affrettando", "cresc-e-poco-rit", "cresc-e-poco-sostenuto", "cresc-e-poco-string", "cresc-e-stringendo-a-poco-a-poco", "cresc-ed",
        "cresc-ed-accel-poco-a-poco", "cresc-ed-accelerando", "cresc-ed-rit", "cresc-p-a-p", "cresc-poco-a-poco-al-mf", "cresc-sempre-al-ff",
        "cresc-sempre-al-fine", "cresc-sforzando", "cresc-un", "cresc-un-poco", "crescpoco-a-poco", "crescendo-e-rall-a-poco-a-poco", "crescendo-poco",
        "de-cres-cen-do", "di-mi-nu-en-do", "dim-colla-voce", "dim-e", "dim-e-calando", "dim-e-pochiss-rit", "dim-e-poco-rall", "dim-e-rall-molto",
        "dim-e-rallent", "dim-e-rit-poco", "dim-e-roco-rit", "dim-morendo", "dim-riten", "dim-smorz", "dime-poco-rall", "dimin-al-fine", "dimin-e-riten",
        "diminish", "diminuendo-e-leggierissimo", "diminuendo-e-ritardando", "diminuendo-un-poco", "dolce-poc-a-poco", "e", "e-cresc-molto", "e-dim",
        "e-rall", "e-rit-in-poco", "e-smorz", "e-stringendo", "en-dim", "espress-legato-poco-a-poco-cresc", "il-piu-forte-possible", "incalze-cressc-sempre",
        "lan", "mo-ren-do", "molto-cresc-ed-accelerando", "molto-rinforz", "molto-ritardando", "per", "più", "più-cre", "più-cresc-ed-agitato", "più-rit-e-dim",
        "più-smorz-e-rit", "poco-a-poco-accelerando", "poco-a-poco-accelerando-e-crescendo", "poco-a-poco-cresc-e-risoluto", "poco-a-poco-cresc-ed-accel",
        "poco-a-poco-cresc-ed-acceler", "poco-a-poco-cresc-ed-accelerando", "poco-a-poco-cresc-ed-anim", "poco-a-poco-cresc-ed-animato", "poco-a-poco-decresc",
        "poco-a-poco-diminuendo", "poco-a-poco-rinforz", "poco-cresc-e-rit", "poco-cresc-ed-agitato", "poco-cresc-molto", "poco-ritar", "poco-stretto",
        "poco-à-poco-cresc", "pressez", "rallent", "religioso", "ri", "rinforzando", "rit-e-crescendo", "ritar", "ritard-e-dimin", "riten", "riten-e-dim",
        "sempre-cresc-e-affrettando", "sempre-cresc-e-string", "sempre-cresc-ed-accel", "sempre-dim-e-legatissimo", "sempre-dim-e-più-tranquillo", "sempre-dimin",
        "sempre-più-cresc-e-string", "sempre-più-forte", "sf", "si", "slentando", "smorzn", "smorzando-e-rallent", "stretto", "stretto-e-cresc",
        "stringendo-e-cresc", "stringendo-cresc", "tar", "un-peu-ralenti", "un-poco-animato-e-crescendo", "vif", "élargir",
        "hair-pin", # default value
    ],
    # Articulation (Chunks)
    "Articulation": [
        "staccato", "artic-staccato-above", "artic-staccato-below", "sforzato", "artic-accent-above",
        "artic-accent-below", "marcato", "tenuto", "ornament-trill",
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
        "artic-soft-accent-tenuto-below", "artic-soft-accent-staccato-above", "artic-soft-accent-tenuto-staccato-above", "lute-fingering-r-h-third", "wigglesawtoothwide", "spiccato",
        "articulation", # default value
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
        "tardo", "tempo", "(tempo)-rubato", "calando", "messa-di-voce", "stentato",
        "tremolo", "affettuoso", "agitato", "animato", "brillante", "bruscamente", "cantabile", "colossale", "comodo", "con-amore",
        "con-brio", "con-fuoco", "con-moto", "con-spirito", "dolce", "drammatico", "feroce", "festoso", "furioso",
        "giocoso", "grandioso", "grazioso", "lacrimoso", "lamentoso", "maestoso", "misterioso", "pesante", "risoluto",
        "scherzando", "solitario", "sotto-(voce)", "sonore", "semplicemente", "slancio", "tranquillo", "volante", "molto",
        "assai", "più", "poco", "poco-a-poco", "ma-non-tanto", "ma-non-troppo", "meno", "subito", "lacuna", "ossia",
        "ostinato", "pensato", "ritornello", "segue", "stretto", "attacca", "cambiare", "da-capo-(al-fine)", "dal-segno", "divisi",
        "oppure", "solo", "sole", "acciaccatura", "altissimo", "appoggiatura", "arco", "arpeggio", "basso-continuo", "a-bocca-chiusa",
        "chiuso", "coloratura", "coperti", "una-corda", "due-corde", "tre-corde-or-tutte-le-corde", "glissando", "legato", "col-legno", "martellato",
        "pizzicato", "portamento", "sforzando", "scordatura", "con-sordino", "senza-sordino",
        "tutti", "vibrato", "colla-voce", "banda", "comprimario", "convenienze", "coro", "diva", "prima-donna",
        "primo-uomo", "ripieno", "bel-canto", "bravura", "bravo", "maestro", "maestro-collaboratore", "maestro-sostituto", "maestro-suggeritore", "stagione",
        "text",
    ],
    # RehearsalMark
    "RehearsalMark": 
        list(map(chr, range(ord("a"), ord("z") + 1))) + [ # the letters
        "i", "i-i", "i-i-i", "i-v", "v", "v-i", "v-i-i", "v-i-i-i", "i-x", "x", # roman numerals
        "introduction", "intro", "bridge", "chorus", "outro", "verse", "theme", "refrain", "solo-section", "solo", "variations", "trio" # song sections
        "rehearsal-mark"
    ],
    # TextSpanner
    "TextSpanner": [],
    # TechAnnotation
    "TechAnnotation": [],
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
    #     "symbol",
    # ],
}
EXPRESSIVE_FEATURE_TYPE_MAP = {expressive_feature_subtype : expressive_feature_type for expressive_feature_type in tuple(EXPRESSIVE_FEATURES.keys()) for expressive_feature_subtype in EXPRESSIVE_FEATURES[expressive_feature_type]}
VALUE_CODE_MAP = [None,] + list(range(128)) + sum(list(EXPRESSIVE_FEATURES.values()), [])
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





##################################################
##################################################
##################################################
#
#   ___                 _ _           
#  | __|_ _  __ ___  __| (_)_ _  __ _ 
#  | _|| ' \/ _/ _ \/ _` | | ' \/ _` |
#  |___|_||_\__\___/\__,_|_|_||_\__, |
#                               |___/ 
#
##################################################
##################################################
##################################################
#
# ['Metadata', 'Tempo', 'KeySignature', 'TimeSignature', 'Beat', 'Barline', 'Lyric', 'Annotation', 'Note', 'Chord', 'Track', 'Text', 'Subtype', 'RehearsalMark', 'TechAnnotation', 'Dynamic', 'Fermata', 'Arpeggio', 'Tremolo', 'ChordLine', 'Ornament', 'Articulation', 'Notehead', 'Symbol', 'Point', 'Bend', 'TremoloBar', 'Spanner', 'SubtypeSpanner', 'TempoSpanner', 'TextSpanner', 'HairPinSpanner', 'SlurSpanner', 'PedalSpanner', 'TrillSpanner', 'VibratoSpanner', 'GlissandoSpanner', 'OttavaSpanner']
# Explicit objects to scrape:
#   - Notes >
#   - Grace Notes (type field) >
#   - Barlines >
#   - Time Signatures >
#   - Key Signatures >
#   - Tempo, TempoSpanner >
#   - Text, TextSpanner >
#   - RehearsalMark >
#   - Dynamic >
#   - HairPinSpanner >
#   - Fermata >
#   - TechAnnotation >
#   - Symbol >
# Implicit objects to scrape:
#   - Articulation (Total # in some time; 4 horizontally in a row) -- we are looking for articulation chunks! >
#   - SlurSpanner, higher the tempo, the longer the slur needs to be >
#   - PedalSpanner, higher the tempo, the longer the slur needs to be >
# Punting on:
#   - Notehead
#   - Arpeggio
#   - Ornament
#   - Ottava
#   - Bend
#   - TrillSpanner
#   - VibratoSpanner
#   - GlissandoSpanner
#   - Tremolo, TremoloBar
#   - Vertical Density
#   - Horizontal Density
#   - Any Drum Tracks


# TEXT CLEANUP FUNCTIONS FOR SCRAPING
##################################################

# make sure text is ok
def check_text(text: str):
    if text is not None:
        return sub(pattern = ": ", repl = ":", string = sub(pattern = ", ", repl = ",", string = " ".join(text.split()))).strip()
    return None

# clean up text objects
def clean_up_text(text: str):
    if text is not None:
        text = sub(pattern = "-", repl = " ", string = utils.split_camel_case(string = text)) # get rid of camel case, deal with long spins of dashes
        text = sub(pattern = " ", repl = "-", string = check_text(text = text)) # replace any whitespace with dashes
        text = sub(pattern = "[^\w-]", repl = "", string = text) # extract alphanumeric
        return text.lower() # convert to lower case
    return None

##################################################


# SCRAPE EXPLICIT FEATURES
##################################################

desired_expressive_feature_types = ("Text", "TextSpanner", "RehearsalMark", "Dynamic", "HairPinSpanner", "Fermata", "TempoSpanner", "TechAnnotation", "Symbol")
def scrape_annotations(annotations: List[Annotation], song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """Scrape annotations. song_length is the length of the song (in time steps). use_implied_duration is whether or not to calculate an 'implied duration' value for features without duration."""

    annotations_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    if use_implied_duration:
        encounters = dict(zip(desired_expressive_feature_types, utils.rep(x = None, times = len(desired_expressive_feature_types)))) # to track durations

    for annotation in annotations:

        # get the expressive feature type we are working with
        expressive_feature_type = annotation.annotation.__class__.__name__
        if expressive_feature_type not in desired_expressive_feature_types: # ignore expressive we are not interested in
            continue
        
        annotation_attributes = vars(annotation.annotation).keys()

        # time
        annotations_encoded["time"].append(annotation.time)

        # event type
        annotations_encoded["type"].append(EXPRESSIVE_FEATURE_TYPE_STRING)

        # duration
        if "duration" in annotation_attributes:
            duration = annotation.annotation.duration # get the duration
        elif use_implied_duration: # deal with implied duration (time until next of same type)
            if encounters[expressive_feature_type] is not None: # to deal with the first encounter
                annotations_encoded["duration"][encounters[expressive_feature_type]] = annotation.time - annotations_encoded["time"][encounters[expressive_feature_type]]
            encounters[expressive_feature_type] = len(annotations_encoded["duration"]) # update encounter index
            duration = None # append None for current duration, will be fixed later           
        else: # not use_implied_duration ; not using implied duration
            duration = 0
        annotations_encoded["duration"].append(duration) # add a duration value if there is one
        
        # deal with value field
        value = None
        if "text" in annotation_attributes:
            value = clean_up_text(text = annotation.annotation.text)
        elif "subtype" in annotation_attributes:
            value = utils.split_camel_case(string = annotation.annotation.subtype)
        if (value is None) or (value == ""): # if there is no text or subtype value, make the value the expressive feature type (e.g. "TempoSpanner")
            value = utils.split_camel_case(string = sub(pattern = "Spanner", repl = "", string = expressive_feature_type))
        # deal with special cases
        if value in ("dynamic", "other-dynamics"):
            value = "dynamic-marking"
        elif expressive_feature_type == "Fermata":
            value = "fermata"
        elif expressive_feature_type == "RehearsalMark" and value.isdigit():
            value = "rehearsal-mark"
        annotations_encoded["value"].append(check_text(text = value))
    
    # get final if using implied durationdurations
    if use_implied_duration:
        for expressive_feature_type in tuple(encounters.keys()):
            if encounters[expressive_feature_type] is not None:
                annotations_encoded["duration"][encounters[expressive_feature_type]] = song_length - annotations_encoded["time"][encounters[expressive_feature_type]]
        
    # make sure untouched columns get filled
    for dimension in filter(lambda dimension: len(annotations_encoded[dimension]) == 0, tuple(annotations_encoded.keys())):
        annotations_encoded[dimension] = utils.rep(x = None, times = len(annotations_encoded["type"]))

    # create dataframe from scraped values
    return pd.DataFrame(data = annotations_encoded, columns = DIMENSIONS)


def scrape_barlines(barlines: List[Barline], song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """Scrape barlines. song_length is the length of the song (in time steps). use_implied_duration is whether or not to calculate an 'implied duration' value for features without duration."""
    barlines = list(filter(lambda barline: not ((barline.subtype == "single") or ("repeat" in barline.subtype.lower())), barlines)) # filter out single barlines
    barlines_encoded = {key: utils.rep(x = None, times = len(barlines)) for key in DIMENSIONS} # create dictionary of lists
    barlines.append(Barline(time = song_length, measure = 0)) # for duration
    for i, barline in enumerate(barlines[:-1]):
        barlines_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        barlines_encoded["value"][i] = check_text(text = (f"{barline.subtype.lower()}-" if barline.subtype is not None else "") + "barline")
        barlines_encoded["duration"][i] = barlines[i + 1].time - barline.time if use_implied_duration else 0
        barlines_encoded["time"][i] = barline.time
    return pd.DataFrame(data = barlines_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_timesigs(timesigs: List[TimeSignature], song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """Scrape timesigs. song_length is the length of the song (in time steps). use_implied_duration is whether or not to calculate an 'implied duration' value for features without duration."""
    timesigs = timesigs[1:] # get rid of first timesig, since we are tracking changes in timesig
    timesigs_encoded = {key: utils.rep(x = None, times = len(timesigs)) for key in DIMENSIONS} # create dictionary of lists
    timesigs.append(TimeSignature(time = song_length, measure = 0, numerator = 4, denominator = 4)) # for duration
    for i, timesig in enumerate(timesigs[:-1]):
        timesigs_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        timesigs_encoded["value"][i] = check_text(text = f"timesig-change") # check_text(text = f"{timesig.numerator}/{timesig.denominator}")
        timesigs_encoded["duration"][i] = timesigs[i + 1].time - timesig.time if use_implied_duration else 0
        timesigs_encoded["time"][i] = timesig.time
    return pd.DataFrame(data = timesigs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_keysigs(keysigs: List[KeySignature], song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """Scrape keysigs. song_length is the length of the song (in time steps). use_implied_duration is whether or not to calculate an 'implied duration' value for features without duration."""
    keysigs = keysigs[1:] # get rid of first keysig, since we are tracking changes in keysig
    keysigs_encoded = {key: utils.rep(x = None, times = len(keysigs)) for key in DIMENSIONS} # create dictionary of lists
    keysigs.append(KeySignature(time = song_length, measure = 0)) # for duration
    for i, keysig in enumerate(keysigs[:-1]):
        keysigs_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        keysigs_encoded["value"][i] = check_text(text = f"keysig-change") # check_text(text = f"{keysig.root_str} {keysig.mode}") # or keysig.root or keysig.fifths
        keysigs_encoded["duration"][i] = keysigs[i + 1].time - keysig.time if use_implied_duration else 0
        keysigs_encoded["time"][i] = keysig.time
    return pd.DataFrame(data = keysigs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_tempos(tempos: List[Tempo], song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """Scrape tempos. song_length is the length of the song (in time steps). use_implied_duration is whether or not to calculate an 'implied duration' value for features without duration."""
    tempos_encoded = {key: utils.rep(x = None, times = len(tempos)) for key in DIMENSIONS} # create dictionary of lists
    tempos.append(Tempo(time = song_length, measure = 0, qpm = 0.0)) # for duration
    for i, tempo in enumerate(tempos[:-1]):
        tempos_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        tempos_encoded["value"][i] = check_text(text = QPM_TEMPO_MAPPER(qpm = tempo.qpm)) # check_text(text = tempo.text.lower() if tempo.text is not None else "tempo-marking")
        tempos_encoded["duration"][i] = tempos[i + 1].time - tempo.time if use_implied_duration else 0
        tempos_encoded["time"][i] = tempo.time
    return pd.DataFrame(data = tempos_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_notes(notes: List[Note]) -> pd.DataFrame:
    """Scrape notes (and grace notes)."""
    notes_encoded = {key: utils.rep(x = None, times = len(notes)) for key in DIMENSIONS} # create dictionary of lists
    for i, note in enumerate(notes):
        notes_encoded["type"][i] = "grace-note" if note.is_grace else "note" # get the note type (grace or normal)
        notes_encoded["value"][i] = note.pitch # or note.pitch_str
        notes_encoded["duration"][i] = note.duration
        notes_encoded["time"][i] = note.time
    return pd.DataFrame(data = notes_encoded, columns = DIMENSIONS) # create dataframe from scraped values

##################################################


# SCRAPE IMPLICIT FEATURES
##################################################

def scrape_articulations(annotations: List[Annotation], maximum_gap: int, articulation_count_threshold: int = 4) -> pd.DataFrame:
    """Scrape articulations. maximum_gap is the maximum distance (in time steps) between articulations, which, when exceeded, forces the ending of the current articulation chunk and the creation of a new one. articulation_count_threshold is the minimum number of articulations in a chunk to make it worthwhile recording."""
    articulations_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    encounters = {}
    def check_all_subtypes_for_chunk_ending(time): # helper function to check if articulation subtypes chunk ended
        for articulation_subtype in tuple(encounters.keys()):
            if (time - encounters[articulation_subtype]["end"]) > maximum_gap: # if the articulation chunk is over
                if encounters[articulation_subtype]["count"] >= articulation_count_threshold:
                    articulations_encoded["type"].append(EXPRESSIVE_FEATURE_TYPE_STRING)
                    articulations_encoded["value"].append(utils.split_camel_case(string = check_text(text = articulation_subtype if articulation_subtype is not None else "articulation")))
                    articulations_encoded["duration"].append(encounters[articulation_subtype]["end"] - encounters[articulation_subtype]["start"])
                    articulations_encoded["time"].append(encounters[articulation_subtype]["start"])
                del encounters[articulation_subtype] # erase articulation subtype, let it be recreated when it comes up again
    for annotation in annotations:
        check_all_subtypes_for_chunk_ending(time = annotation.time)
        if annotation.annotation.__class__.__name__ == "Articulation":
            articulation_subtype = annotation.annotation.subtype # get the articulation subtype
            if articulation_subtype in encounters.keys(): # if we've encountered this articulation before
                encounters[articulation_subtype]["end"] = annotation.time
                encounters[articulation_subtype]["count"] += 1
            else:
                encounters[articulation_subtype] = {"start": annotation.time, "end": annotation.time, "count": 1} # if we are yet to encounter this articulation
        else: # ignore non Articulations
            continue
    if len(annotations) > 0: # to avoid index error
        check_all_subtypes_for_chunk_ending(time = annotations[-1].time + (2 * maximum_gap)) # one final check
    for dimension in filter(lambda dimension: len(articulations_encoded[dimension]) == 0, tuple(articulations_encoded.keys())): # make sure untouched columns get filled
        articulations_encoded[dimension] = utils.rep(x = None, times = len(articulations_encoded["type"]))
    return pd.DataFrame(data = articulations_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_slurs(annotations: List[Annotation], minimum_duration: float, music: BetterMusic) -> pd.DataFrame:
    """Scrape slurs. minimum_duration is the minimum duration (in seconds) a slur needs to be to make it worthwhile recording."""
    slurs_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    for annotation in annotations:
        if annotation.annotation.__class__.__name__ == "SlurSpanner":
            if annotation.annotation.is_slur:
                start = music.metrical_time_to_absolute_time(time_steps = annotation.time)
                duration = music.metrical_time_to_absolute_time(time_steps = annotation.time + annotation.annotation.duration) - start
                if duration > minimum_duration:
                    slurs_encoded["type"].append(EXPRESSIVE_FEATURE_TYPE_STRING)
                    slurs_encoded["value"].append(check_text(text = "slur"))
                    slurs_encoded["duration"].append(annotation.annotation.duration)
                    slurs_encoded["time"].append(annotation.time)
                else: # if slur is too short
                    continue
            else: # if annotation is a tie
                continue
        else: # ignore non slurs
            continue
    for dimension in filter(lambda dimension: len(slurs_encoded[dimension]) == 0, tuple(slurs_encoded.keys())): # make sure untouched columns get filled
        slurs_encoded[dimension] = utils.rep(x = None, times = len(slurs_encoded["type"]))
    return pd.DataFrame(data = slurs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_pedals(annotations: List[Annotation], minimum_duration: float, music: BetterMusic) -> pd.DataFrame:
    """Scrape pedals. minimum_duration is the minimum duration (in seconds) a pedal needs to be to make it worthwhile recording."""
    pedals_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    for annotation in annotations:
        if annotation.annotation.__class__.__name__ == "PedalSpanner":
            start = music.metrical_time_to_absolute_time(time_steps = annotation.time)
            duration = music.metrical_time_to_absolute_time(time_steps = annotation.time + annotation.annotation.duration) - start
            if duration > minimum_duration:
                pedals_encoded["type"].append(EXPRESSIVE_FEATURE_TYPE_STRING)
                pedals_encoded["value"].append(check_text(text = "pedal"))
                pedals_encoded["duration"].append(annotation.annotation.duration)
                pedals_encoded["time"].append(annotation.time)
            else: # if pedal is too short
                continue
        else: # ignore non pedals
            continue
    for dimension in filter(lambda dimension: len(pedals_encoded[dimension]) == 0, tuple(pedals_encoded.keys())): # make sure untouched columns get filled
        pedals_encoded[dimension] = utils.rep(x = None, times = len(pedals_encoded["type"]))
    return pd.DataFrame(data = pedals_encoded, columns = DIMENSIONS) # create dataframe from scraped values

##################################################


# WRAPPER FUNCTIONS MAKE CODE EASIER TO READ
##################################################

def get_system_level_expressive_features(music: BetterMusic, use_implied_duration: bool = True) -> pd.DataFrame:
    """Wrapper function to make code more readable. Extracts system-level expressive features."""
    system_annotations = scrape_annotations(annotations = music.annotations, song_length = music.song_length, use_implied_duration = use_implied_duration)
    system_barlines = scrape_barlines(barlines = music.barlines, song_length = music.song_length, use_implied_duration = use_implied_duration)
    system_timesigs = scrape_timesigs(timesigs = music.time_signatures, song_length = music.song_length, use_implied_duration = use_implied_duration)
    system_keysigs = scrape_keysigs(keysigs = music.key_signatures, song_length = music.song_length, use_implied_duration = use_implied_duration)
    system_tempos = scrape_tempos(tempos = music.tempos, song_length = music.song_length, use_implied_duration = use_implied_duration)
    return pd.concat(objs = (system_annotations, system_barlines, system_timesigs, system_keysigs, system_tempos), axis = 0, ignore_index = True)

def get_staff_level_expressive_features(track: Track, music: BetterMusic, use_implied_duration: bool = True) -> pd.DataFrame:
    """Wrapper function to make code more readable. Extracts staff-level expressive features."""
    staff_notes = scrape_notes(notes = track.notes)
    staff_annotations = scrape_annotations(annotations = track.annotations, song_length = music.song_length, use_implied_duration = use_implied_duration)
    staff_articulations = scrape_articulations(annotations = track.annotations, maximum_gap = 2 * music.resolution) # 2 beats = 2 * music.resolution
    staff_slurs = scrape_slurs(annotations = track.annotations, minimum_duration = 1.5, music = music) # minimum duration for slurs to be recorded is 1.5 seconds
    staff_pedals = scrape_pedals(annotations = track.annotations, minimum_duration = 1.5, music = music) # minimum duration for pedals to be recorded is 1.5 seconds
    return pd.concat(objs = (staff_notes, staff_annotations, staff_articulations, staff_slurs, staff_pedals), axis = 0, ignore_index = True)

##################################################


# ENCODER FUNCTIONS
##################################################

def get_encoding() -> dict:
    """Return the encoding configurations."""
    return {
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
    }


def load_encoding(filepath: str) -> dict:
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filepath = filepath)
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


def extract_data(music: BetterMusic, use_implied_duration: bool = True) -> np.array:
    """Return a BetterMusic object as a data sequence.
    Each row of the output is a note specified as follows.
        (event_type, beat, position, value, duration, program, time, time (in seconds))
    """

    # create output dataframe
    output = np.empty(shape = (0, len(DIMENSIONS)), dtype = np.object_)

    # time column index
    time_dim = DIMENSIONS.index("time")

    # scrape system level expressive features
    system_level_expressive_features = get_system_level_expressive_features(music = music, use_implied_duration = use_implied_duration)

    for track in music.tracks:

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in KNOWN_PROGRAMS:
            continue

        # scrape staff-level features
        staff_level_expressive_features = get_staff_level_expressive_features(track = track, music = music, use_implied_duration = use_implied_duration)

        # create dataframe, do some wrangling to semi-encode values
        data = pd.concat(objs = (pd.DataFrame(columns = DIMENSIONS), system_level_expressive_features, staff_level_expressive_features), axis = 0, ignore_index = True) # combine system and staff expressive features
        data["instrument"] = utils.rep(x = track.program, times = len(data)) # add the instrument column
        data["duration"] = (RESOLUTION / music.resolution) * data["duration"] # semi-encode duration

        # convert time to seconds for certain types of sorting that might require it
        data["time.s"] = data["time"].apply(lambda time_steps: music.metrical_time_to_absolute_time(time_steps = time_steps)) # get time in seconds
        # data = data.sort_values(by = "time").reset_index(drop = True) # sort by time

        # calculate beat and position values (time signature agnostic)
        data["beat"] = data["time"].apply(lambda time_steps: int(time_steps / music.resolution)) # add beat
        data["position"] = data["time"].apply(lambda time_steps: int((RESOLUTION / music.resolution) * (time_steps % music.resolution))) # add position
        # get beats (accounting for time signature) # beats = sorted(list(set([beat.time for beat in music.beats] + [music.song_length,]))) # add song length to end of beats for calculating position
        # if len(music.time_signatures) > 0:
        #     beats = []
        #     timesigs = music.time_signatures + [TimeSignature(time = music.song_length, measure = 0, numerator = 4, denominator = 4),]
        #     for i in range(len(timesigs) - 1):
        #         beats += list(range(timesigs[i].time, timesigs[i + 1].time, int(music.resolution * (4 / timesigs[i].denominator))))
        # else: # assume 4/4
        #     beats = list(range(0, music.song_length + music.resolution, music.resolution))        
        # beat_index = 0
        # for i in data.index: # assumes data is sorted by time_step values
        #     if data.at[i, "time"] >= beats[beat_index + 1]: # if we've moved to the next beat
        #         beat_index += 1 # increment beat index
        #     data.at[i, "beat"] = beat_index  # convert base to base 0
        #     data.at[i, "position"] = int((RESOLUTION * (data.at[i, "time"] - beats[beat_index])) / (beats[beat_index + 1] - beats[beat_index]))

        # remove duplicates due to beat and position quantization
        data = data.drop_duplicates(subset = DIMENSIONS[:time_dim], keep = "first", ignore_index = True)

        # don't save low-quality data
        # if len(data) < 50:
        #     continue

        # convert to np array so that we can save as npy file, which loads faster
        data = np.array(object = data, dtype = np.object_)

        # add to output array
        output = np.concatenate((output, data), axis = 0, dtype = np.object_)

    # sort by time
    output = output[output[:, time_dim].argsort()]
    if len(output) > 0: # set start beat to 0
        output[:, time_dim] = output[:, time_dim] - output[0, time_dim]
        output[:, time_dim + 1] = output[:, time_dim + 1] - output[0, time_dim + 1]

    return output


def encode_data(data: np.array, encoding: dict, conditioning: str = DEFAULT_CONDITIONING, sigma: float = SIGMA) -> np.array:
    """Encode a note sequence into a sequence of codes.
    Each row of the input is a note specified as follows.
        (event_type, beat, position, value, duration, program, time, time (in seconds))
    Each row of the output is encoded as follows.
        (event_type, beat, position, value, duration, instrument)
    """

    # LOAD IN DATA

    # load in npy file from path parameter
    # data = np.load(file = path, allow_pickle = True)

    # get variables
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # get maps
    type_code_map = encoding["type_code_map"]
    beat_code_map = encoding["beat_code_map"]
    position_code_map = encoding["position_code_map"]
    value_code_map = encoding["value_code_map"]
    duration_code_map = encoding["duration_code_map"]
    instrument_code_map = encoding["instrument_code_map"]
    program_instrument_map = encoding["program_instrument_map"]

    # get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # make sure conditioning value is correct
    if conditioning not in CONDITIONINGS:
        conditioning = DEFAULT_CONDITIONING
    
    # ENCODE

    # start the codes with an SOS row
    codes = np.array(object = [(type_code_map["start-of-song"], 0, 0, 0, 0, 0)], dtype = ENCODING_ARRAY_TYPE)

    # extract/encode instruments
    programs = np.unique(ar = data[:, instrument_dim]) # get unique instrument values
    instrument_codes = np.zeros(shape = (len(programs), codes.shape[1]), dtype = ENCODING_ARRAY_TYPE) # create empty array
    for i, program in enumerate(programs):
        if program is None: # skip unknown programs
            continue
        instrument_codes[i, 0] = type_code_map["instrument"] # set the type to instrument
        instrument_codes[i, instrument_dim] = instrument_code_map[program_instrument_map[program]] # encode the instrument value
    instrument_codes = instrument_codes[np.argsort(a = instrument_codes[:, instrument_dim], axis = 0)] # sort the instruments
    codes = np.concatenate((codes, instrument_codes), axis = 0) # append them to the code sequence
    del instrument_codes # clear up memory

    # add start of notes row
    codes = np.append(arr = codes, values = [(type_code_map["start-of-notes"], 0, 0, 0, 0, 0)], axis = 0)

    # helper functions for mapping
    def value_code_mapper(value) -> int:
        try:
            code = value_code_map[None if value == "" else value]
        except KeyError:
            value = sub(pattern = "", repl = "", string = value) # try wrangling value a bit to get a key
            try:
                code = value_code_map[None if value == "" else value]
            except KeyError:
                code = -1
        return code
    def program_instrument_mapper(program) -> int:
        instrument = instrument_code_map[program_instrument_map[int(program)]]
        if instrument is None:
            return -1
        return instrument
    
    # encode the notes / expressive features
    core_codes = np.zeros(shape = (data.shape[0], codes.shape[1]), dtype = ENCODING_ARRAY_TYPE)
    core_codes[:, 0] = list(map(lambda type_: type_code_map[str(type_)], data[:, 0])) # encode type column
    core_codes[:, beat_dim] = list(map(lambda beat: beat_code_map[int(beat)], data[:, beat_dim])) # encode beat
    core_codes[:, position_dim] = list(map(lambda position: position_code_map[int(position)], data[:, position_dim])) # encode position
    core_codes[:, value_dim] = list(map(value_code_mapper, data[:, value_dim])) # encode value column
    core_codes[:, duration_dim] = list(map(lambda duration: duration_code_map[min(int(duration), int(max_duration))], data[:, duration_dim])) # encode duration
    core_codes[:, instrument_dim] = list(map(program_instrument_mapper, data[:, instrument_dim])) # encode instrument column
    core_codes = core_codes[core_codes[:, beat_dim] <= max_beat] # remove data if beat greater than max beat
    core_codes = core_codes[core_codes[:, instrument_dim] >= 0] # skip unknown instruments

    # apply conditioning to core_codes
    if conditioning == CONDITIONINGS[0]: # sort-order
        core_codes_with_time_steps = np.concatenate((core_codes, data[:, data.shape[1] - 2].reshape(data.shape[0], 1)), axis = 1) # add time steps column
        time_steps_column = core_codes_with_time_steps.shape[1] - 1
        core_codes_with_time_steps = core_codes_with_time_steps[core_codes_with_time_steps[:, time_steps_column].argsort()] # sort by time (time steps)
        core_codes = np.delete(arr = core_codes_with_time_steps, obj = time_steps_column, axis = 1).astype(ENCODING_ARRAY_TYPE) # remove time steps column
        del core_codes_with_time_steps, time_steps_column
    elif conditioning == CONDITIONINGS[1]: # prefix
        core_codes_with_time_steps = np.concatenate((core_codes, data[:, data.shape[1] - 2].reshape(data.shape[0], 1)), axis = 1) # add time steps column
        time_steps_column = core_codes_with_time_steps.shape[1] - 1
        expressive_feature_indicies = sorted(np.where(core_codes[:, 0] == type_code_map[EXPRESSIVE_FEATURE_TYPE_STRING])[0]) # get indicies of expressive features
        expressive_features = core_codes_with_time_steps[expressive_feature_indicies] # extract expressive features
        expressive_features = expressive_features[expressive_features[:, time_steps_column].argsort()] # sort by time
        expressive_features = np.delete(arr = expressive_features, obj = time_steps_column, axis = 1).astype(ENCODING_ARRAY_TYPE) # delete time steps column
        notes = np.delete(arr = core_codes_with_time_steps, obj = expressive_feature_indicies, axis = 0) # delete expressive features from core
        notes = notes[notes[:, time_steps_column].argsort()] # sort by time
        notes = np.delete(arr = notes, obj = time_steps_column, axis = 1).astype(ENCODING_ARRAY_TYPE) # delete time steps column
        core_codes = np.concatenate((expressive_features, codes[len(codes) - 1].reshape(1, codes.shape[1]), notes), axis = 0, dtype = ENCODING_ARRAY_TYPE) # sandwich: expressive features, start of notes row, 
        codes = np.delete(arr = codes, obj = len(codes) - 1, axis = 0) # remove start of notes row from codes
        del core_codes_with_time_steps, time_steps_column, expressive_feature_indicies, expressive_features, notes
    elif conditioning == CONDITIONINGS[2]: # anticipation
        if sigma is None: # make sure sigma is not none
            warnings.warn(f"Encountered NoneValue sigma argument for anticipation conditioning. Using sigma = {SIGMA}.", RuntimeWarning)
            sigma = SIGMA
        core_codes_with_seconds = np.concatenate((core_codes, data[:, data.shape[1] - 1].reshape(data.shape[0], 1)), axis = 1) # add seconds column
        seconds_column = core_codes_with_seconds.shape[1] - 1 # get the index of the seconds column
        for i in range(core_codes_with_seconds.shape[0]): # iterate through core_codes_with_seconds
            if core_codes_with_seconds[i, 0] == type_code_map[EXPRESSIVE_FEATURE_TYPE_STRING]: # if the type is expressive feature
                core_codes_with_seconds[i, seconds_column] -= sigma # add anticipation
        core_codes_with_seconds = core_codes_with_seconds[core_codes_with_seconds[:, seconds_column].argsort()] # sort by time (seconds)
        core_codes = np.delete(arr = core_codes_with_seconds, obj = seconds_column, axis = 1).astype(ENCODING_ARRAY_TYPE) # remove seconds column
        del core_codes_with_seconds, seconds_column

    # add core_codes to the general codes matrix
    codes = np.concatenate((codes, core_codes), axis = 0) # append them to the code sequence
    del core_codes # clear up memory

    # end the codes with an EOS row
    codes = np.append(arr = codes, values = [(type_code_map["end-of-song"], 0, 0, 0, 0, 0)], axis = 0)

    return codes


def encode(music: BetterMusic, use_implied_duration: bool = True, encoding: dict = get_encoding(), conditioning: str = DEFAULT_CONDITIONING, sigma: float = SIGMA) -> np.array:
    """Given a BetterMusic object, encode it."""

    # extract data
    data = extract_data(music = music, use_implied_duration = use_implied_duration)

    # encode data
    codes = encode_data(data = data, encoding = encoding, conditioning = conditioning, sigma = sigma)

    return codes

##################################################





##################################################
##################################################
##################################################
#
#   ___                 _ _           
#  |   \ ___ __ ___  __| (_)_ _  __ _ 
#  | |) / -_) _/ _ \/ _` | | ' \/ _` |
#  |___/\___\__\___/\__,_|_|_||_\__, |
#                               |___/ 
#
##################################################
##################################################
##################################################


# DECODER FUNCTIONS
##################################################

def decode_data(codes: np.array, encoding: dict = get_encoding()) -> List[list]:
    """Decode codes into a data sequence.
    Each row of the input is encoded as follows.
        (event_type, beat, position, value, duration, instrument)
    Each row of the output is decoded the same way.
    """

    # get variables and maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_value_map = encoding["code_value_map"]
    code_duration_map = encoding["code_duration_map"]
    code_instrument_map = encoding["code_instrument_map"]
    instrument_program_map = encoding["instrument_program_map"]

    # get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # decode the codes into a sequence of data
    data = []
    for row in codes:
        event_type = code_type_map[int(row[0])]
        if event_type in ("start-of-song", "instrument", "start-of-notes"):
            continue
        elif event_type == "end-of-song":
            break
        elif event_type in ("note", "grace-note", EXPRESSIVE_FEATURE_TYPE_STRING):
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            value = code_value_map[int(row[value_dim])]
            duration = code_duration_map[int(row[duration_dim])]
            program = instrument_program_map[code_instrument_map[int(row[instrument_dim])]]
            data.append((event_type, beat, position, value, duration, program))
        else:
            raise ValueError("Unknown event type.")

    return data


def reconstruct(data: np.array, resolution: int, encoding: dict = get_encoding()) -> BetterMusic:
    """Reconstruct a data sequence as a BetterMusic object."""

    # construct the BetterMusic object with defaults
    music = BetterMusic(resolution = resolution, tempos = [Tempo(time = 0, qpm = DEFAULT_QPM)], key_signatures = [KeySignature(time = 0)], time_signatures = [TimeSignature(time = 0)])

    # append the tracks
    programs = sorted(set(row[-1] for row in data)) # get programs
    for program in programs:
        music.tracks.append(Track(program = program, is_drum = False, name = encoding["program_instrument_map"][program])) # append to tracks

    # append the notes
    for event_type, beat, position, value, duration, program in data:
        track_idx = programs.index(program) # get track index
        time = (beat * resolution) + position # get time in time steps
        duration = (resolution / encoding["resolution"]) * duration # get duration in time steps
        if event_type in ("note", "grace-note"):
            music.tracks[track_idx].notes.append(Note(time = time, pitch = value, duration = duration, is_grace = (event_type == "grace_note")))
        elif event_type == EXPRESSIVE_FEATURE_TYPE_STRING:
            expressive_feature_type = EXPRESSIVE_FEATURE_TYPE_MAP[value]
            match expressive_feature_type:
                case "Barline":
                    music.barlines.append(Barline(time = time, subtype = value))
                case "KeySignature":
                    music.key_signatures.append(KeySignature(time = time, ))
                case "TimeSignature":
                    music.time_signatures.append(TimeSignature(time = time, numerator = 0, denominator = 0))
                case "Tempo":
                    music.tempos.append(Tempo(time = time, qpm = TEMPO_QPM_MAP[value], text = value))
                case "TempoSpanner":
                    music.annotations.append(Annotation(time = time, annotation = TempoSpanner(duration = duration, subtype = value)))
                case "RehearsalMark":
                    music.annotations.append(Annotation(time = time, annotation = RehearsalMark(text = value)))
                case "Fermata":
                    music.annotations.append(Annotation(time = time, annotation = Fermata()))
                case "Text":
                    music.annotations.append(Annotation(time = time, annotation = Text(text = value, is_system = True)))
                case "TextSpanner":
                    music.annotations.append(Annotation(time = time, annotation = TextSpanner(duration = duration, text = value, is_system = True)))
                case "SlurSpanner":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = SlurSpanner(duration = duration, is_slur = True)))
                case "PedalSpanner":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = PedalSpanner(duration = duration)))
                case "Dynamic":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = Dynamic(subtype = value)))
                case "HairPinSpanner":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = HairPinSpanner(duration = duration, subtype = value)))
                case "Articulation":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = Articulation(subtype = value)))
                case "TechAnnotation":
                    music.tracks[track_idx].annotations.append(Annotation(time = time, annotation = TechAnnotation(text = value)))
        else:
            raise ValueError("Unknown event type.")

    # return the filled BetterMusic
    return music


def decode(codes: np.array, encoding: dict = get_encoding()) -> BetterMusic:
    """Decode codes into a MusPy Music object.
    Each row of the input is encoded as follows.
        (event_type, beat, position, value, duration, instrument)
    """

    # get resolution
    resolution = encoding["resolution"]

    # decode codes into a note sequence
    data = decode_data(codes = codes, encoding = encoding)

    # reconstruct the music object
    music = reconstruct(data = data, resolution = resolution, encoding = encoding)

    return music


def dump(data: np.array, encoding: dict = get_encoding()) -> str:
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

##################################################


# SAVE DATA
##################################################

def save_txt(filepath: str, data: np.array, encoding: dict = get_encoding()):
    """Dump the codes into a TXT file."""
    with open(filepath, "w") as f:
        f.write(dump(data = data, encoding = encoding))


def save_csv_data(filepath: str, data: np.array):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(fname = filepath, X = data, fmt = "%d", delimiter = ",", header = "beat,position,value,duration,program", comments = "")


def save_csv_codes(filepath: str, data: np.array):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 6
    np.savetxt(fname = filepath, X = data, fmt = "%d", delimiter = ",", header = "type,beat,position,value,duration,instrument", comments = "")

##################################################





##################################################
##################################################
##################################################
#
#   ___     _            
#  | __|_ _| |_ _ _ __ _ 
#  | _|\ \ /  _| '_/ _` |
#  |___/_\_\\__|_| \__,_|
#
#
##################################################
##################################################
##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"

    # get arguments
    parser = argparse.ArgumentParser(prog = "Representation", description = "Test Encoding/Decoding mechanisms for MuseScore data.")
    parser.add_argument("-e", "--encoding_filepath", type = str, default = ENCODING_FILEPATH, help = "Absolute filepath to encoding file")
    args = parser.parse_args()

    # get the encoding
    encoding = get_encoding()

    # save the encoding
    utils.save_json(filepath = args.encoding_filepath, data = encoding)

    # load encoding
    encoding = load_encoding(filepath = args.encoding_filepath)

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

    # print an example
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
    print(f"Decoded:\n{dump(data = codes, encoding = encoding)}")

    music = decode(codes = codes, encoding = encoding)
    print("-" * 40)
    print(f"Decoded music:\n{music}")

    encoded = encode(music = music, encoding = encoding)
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

##################################################
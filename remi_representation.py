# README
# Phillip Long
# August 1, 2024

# Utilities for representing a REMI-Style encoding.

# python /home/pnlong/model_musescore/remi_representation.py


# IMPORTS
##################################################

import pprint
from os.path import abspath, dirname
import numpy as np
from typing import List
import utils

from read_mscz.music import MusicExpress
from read_mscz.classes import Tempo, Track, Note
from read_mscz.read_mscz import read_musescore

##################################################


# CONFIGURATION
##################################################

RESOLUTION = 12 # resolution per beat
MAX_BEAT = 1024 # max beat
MAX_DURATION = 384 # longest possible duration
N_NOTES = 128 # number of notes in midi

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
]
DURATION_MAP = {
    i: KNOWN_DURATIONS[np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))]
    for i in range(1, MAX_DURATION + 1)
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
    120: None,
    # Sound effects
    121: None,
    122: None,
    123: None,
    124: None,
    125: None,
    126: None,
    127: None,
    128: None,
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
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

##################################################


# EVENTS
##################################################

KNOWN_EVENTS = [
    "start-of-song",
    "end-of-song",
    "start-of-track",
    "end-of-track",
]
KNOWN_EVENTS.extend(f"beat_{i}" for i in range(MAX_BEAT))
KNOWN_EVENTS.extend(f"position_{i}" for i in range(RESOLUTION))
KNOWN_EVENTS.extend(f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS)
KNOWN_EVENTS.extend(f"pitch_{i}" for i in range(N_NOTES))
KNOWN_EVENTS.extend(f"duration_{i}" for i in KNOWN_DURATIONS)
EVENT_CODE_MAPS = {event: i for i, event in enumerate(KNOWN_EVENTS)}
CODE_EVENT_MAPS = utils.inverse_dict(EVENT_CODE_MAPS)

##################################################


# INDEXER CLASS
##################################################

# indexer
class Indexer:

    # initializer
    def __init__(self, data: dict = None, is_training: bool = False):
        self._dict = dict() if data is None else data
        self._is_training = is_training

    # obtain an item
    def __getitem__(self, key):
        if self._is_training and key not in self._dict:
            self._dict[key] = len(self._dict)
            return len(self._dict) - 1
        return self._dict[key]

    # get length
    def __len__(self) -> int:
        return len(self._dict)

    # check if an item is in self._dict
    def __contain__(self, item) -> bool:
        return item in self._dict

    # return the dictionary
    def get_dict(self) -> dict:
        """Return the internal dictionary."""
        return self._dict

    # set in training mode
    def train(self) -> None:
        """Set training mode."""
        self._is_training = True

    # exit training mode
    def eval(self) -> None:
        """Set evaluation mode."""
        self._is_learning = False

##################################################


# GET AND LOAD ENCODING
##################################################

# get the encoding as a dictionary
def get_encoding() -> dict:
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_beat": MAX_BEAT,
        "max_duration": MAX_DURATION,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "duration_map": DURATION_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }

# load the encoding from a file, returning a dictionary
def load_encoding(filepath: str) -> dict:
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filepath = filepath)
    for key in ("program_instrument_map", "code_event_map", "duration_map"):
        encoding[key] = {
            (int(k) if (k != "null") else None): v
            for k, v in encoding[key].items()
        }
    return encoding

##################################################


# ENCODE
##################################################

# extract notes from a music object
def extract_notes(music: MusicExpress, resolution: int = RESOLUTION) -> np.array:
    """Return a music object as a note sequence.

    Each row of the output is a note specified as follows.

        (beat, position, pitch, duration, program)

    """

    # check resolution
    resolution_scale_factor = resolution / music.resolution

    # extract notes
    notes = []
    for track in music:
        for note in track:
            beat, position = map(int, divmod(note.time * resolution_scale_factor, resolution))
            notes.append((beat, position, note.pitch, note.duration, track.program))

    # deduplicate and sort the notes
    notes = sorted(set(notes))

    # return list of events
    return np.array(notes)

# encode intermediate extraction scheme
def encode_notes(notes: np.array, encoding: dict, indexer: Indexer) -> np.array:
    """Encode the notes into a sequence of code tuples.

    Each row of the output is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """

    # get variables
    max_beat = encoding["max_beat"]
    max_duration = encoding["max_duration"]

    # get maps
    duration_map = encoding["duration_map"]
    program_instrument_map = encoding["program_instrument_map"]

    # start the codes with an SOS event
    codes = [indexer["start-of-song"]]

    # encode the notes
    last_beat = 0
    for beat, position, pitch, duration, program in notes:

        # skip if max_beat has reached
        if beat > max_beat:
            continue

        # skip unknown instruments
        instrument = program_instrument_map[program]
        if instrument is None:
            continue
        if beat > last_beat:
            codes.append(indexer[f"beat_{beat}"])
            last_beat = beat
        codes.append(indexer[f"position_{position}"])
        codes.append(indexer[f"instrument_{instrument}"])
        codes.append(indexer[f"pitch_{pitch}"])
        codes.append(indexer[f"duration_{duration_map[min(duration, max_duration)]}"])

    # end the codes with an EOS event
    codes.append(indexer["end-of-song"])

    # return codes
    return np.array(codes)

# combine extract and encode notes into a single function
def encode(music: MusicExpress, encoding: dict, indexer: Indexer) -> np.array:
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """

    # extract notes
    notes = extract_notes(music = music, resolution = encoding["resolution"])

    # encode the notes
    codes = encode_notes(notes = notes, encoding = encoding, indexer = indexer)

    # return the encoded note sequence
    return codes

##################################################


# DECODE
##################################################

# decode codes into a sequence of notes
def decode_notes(data: List[str], encoding: dict, vocabulary: dict) -> List[tuple]:
    """Decode codes into a note sequence."""

    # get variables and maps
    instrument_program_map = encoding["instrument_program_map"]

    # initialize variables
    beat = 0
    position = None
    program = None
    pitch = None
    duration = None

    # decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]

        # start of song event
        if event == "start-of-song":
            continue

        # end of song event
        elif event == "end-of-song":
            break

        # beat event
        elif event.startswith("beat"):
            beat = int(event.split("_")[1]) # reset variables
            position = None
            program = None
            pitch = None
            duration = None

        # position event
        elif event.startswith("position"):
            position = int(event.split("_")[1]) # reset variables
            program = None
            pitch = None
            duration = None
        
        # instrument event
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            program = instrument_program_map[instrument]

        # pitch event
        elif event.startswith("pitch"):
            pitch = int(event.split("_")[1])

        # duration event
        elif event.startswith("duration"):
            duration = int(event.split("_")[1])
            if (position is None) or (program is None) or (pitch is None) or (duration is None):
                continue
            notes.append((beat, position, pitch, duration, program)) # add event
        
        # unknown event type
        else:
            raise ValueError(f"Unknown event type for: {event}")

    # return list of note events
    return notes

# reconstruct note events sequence as a music object
def reconstruct(notes: List[tuple], resolution: int = RESOLUTION) -> MusicExpress:
    """Reconstruct a note sequence to a music object."""

    # construct the music object
    music = MusicExpress(resolution = resolution, tempos = [Tempo(time = 0, qpm = 100)])

    # append the tracks
    programs = sorted(set(note[-1] for note in notes))
    for program in programs:
        music.tracks.append(Track(program = program))

    # append the notes
    for beat, position, pitch, duration, program in notes:
        time = (beat * resolution) + position
        i_track = programs.index(program)
        music[i_track].notes.append(Note(time = time, pitch = pitch, duration = duration))

    # return the music object
    return music

# combine decode and reconstruct into a single function
def decode(codes: List[str], encoding: dict, vocabulary: dict) -> MusicExpress:
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """

    # get resolution
    resolution = encoding["resolution"]

    # decode codes into a note sequence
    notes = decode_notes(data = codes, encoding = encoding, vocabulary = vocabulary)

    # reconstruct the music object
    music = reconstruct(notes = notes, resolution = resolution)

    # return music object
    return music

# decode and dump as a string
def dump(data: List[str], vocabulary: dict) -> str:
    """Decode the codes and dump as a string."""

    # iterate over the rows
    lines = []
    for code in data:
        event = vocabulary[code]

        # start of song, beat, and position events
        if (event == "start-of-song") or event.startswith("beat") or event.startswith("position"):
            lines.append(event)
        
        # end of song event
        elif event == "end-of-song":
            lines.append(event)
            break
        
        # instrument, pitch, or duration events
        elif event.startswith("instrument") or event.startswith("pitch") or event.startswith("duration"):
            lines[-1] = f"{lines[-1]} {event}"
        
        # unknown event type
        else:
            raise ValueError(f"Unknown event type for: {event}")

    # join lines together
    return "\n".join(lines)

##################################################


# UTILITY FUNCTIONS
##################################################

# save codes as a text file
def save_txt(filepath: str, data: List[str], vocabulary: dict):
    """Dump the codes into a .txt file."""
    with open(filepath, "w") as file:
        file.write(dump(data = data, vocabulary = vocabulary))

# save note events as a csv file
def save_csv_notes(filepath: str, data: np.array):
    """Save the representation as a csv file."""
    assert data.shape[1] == 5
    np.savetxt(
        fname = filepath,
        X = data,
        fmt = "%d",
        delimiter = ",",
        header = "beat,position,pitch,duration,program",
        comments = "",
    )

# save encoded notes as a csv file
def save_csv_codes(filepath: str, data: np.array):
    """Save the representation as a CSV file."""
    assert data.ndim == 1
    np.savetxt(
        fname = filepath,
        X = data,
        fmt = "%d",
        delimiter = ",",
        header = "code",
        comments = "",
    )

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # get the encoding
    encoding = get_encoding()

    # save the encoding
    # filepath = f"{dirname(abspath(__file__))}/remi_encoding.json"
    # utils.save_json(filepath = filepath, encoding = encoding) # save encoding as json
    # encoding = load_encoding(filepath = filepath) # load encoding

    # print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in ("program_instrument_map", "instrument_program_map"):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(object = value, indent = 2)

    # print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_beat: {encoding['max_beat']}")
    print(f"max_duration: {encoding['max_duration']}")

    # load the example
    music = read_musescore(path = "/data2/pnlong/musescore/test_data/toploader/dancing_in_the_moonlight.mscz")

    # get the indexer
    indexer = Indexer(is_training = True)

    # encode the music
    encoded = encode(music = music, encoding = encoding, indexer = indexer)
    print(f"Codes:\n{encoded}")

    # get the learned vocabulary
    vocabulary = utils.inverse_dict(indexer.get_dict())
    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    # print decoded music
    music = decode(codes = encoded, encoding = encoding, vocabulary = vocabulary)
    print(f"Decoded musics:\n{music}")

##################################################

# README
# Phillip Long
# December 3, 2023

# Any functions related to decoding.


# IMPORTS
##################################################
import numpy as np
from pretty_midi import note_number_to_name
import argparse
from typing import List
import representation
from encode import ENCODING_ARRAY_TYPE
from read_mscz.music import BetterMusic
from read_mscz.classes import *
##################################################


# CONSTANTS
##################################################



##################################################


# DECODER FUNCTIONS
##################################################

def decode_data(codes: np.array, encoding: dict = representation.get_encoding()) -> List[list]:
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
        elif event_type in ("note", "grace-note", representation.EXPRESSIVE_FEATURE_TYPE_STRING):
            beat = code_beat_map[int(row[beat_dim])]
            position = code_position_map[int(row[position_dim])]
            value = code_value_map[int(row[value_dim])]
            duration = code_duration_map[int(row[duration_dim])]
            program = instrument_program_map[code_instrument_map[int(row[instrument_dim])]]
            data.append((event_type, beat, position, value, duration, program))
        else:
            raise ValueError("Unknown event type.")

    return data


def reconstruct(data: np.array, resolution: int, encoding: dict = representation.get_encoding()) -> BetterMusic:
    """Reconstruct a data sequence as a BetterMusic object."""

    # construct the BetterMusic object with defaults
    music = BetterMusic(resolution = resolution, tempos = [Tempo(time = 0, qpm = representation.DEFAULT_QPM)], key_signatures = [KeySignature(time = 0)], time_signatures = [TimeSignature(time = 0)])

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
        elif event_type == representation.EXPRESSIVE_FEATURE_TYPE_STRING:
            expressive_feature_type = representation.EXPRESSIVE_FEATURE_TYPE_MAP[value]
            match expressive_feature_type:
                case "Barline":
                    music.barlines.append(Barline(time = time, subtype = value))
                case "KeySignature":
                    music.key_signatures.append(KeySignature(time = time, ))
                case "TimeSignature":
                    music.time_signatures.append(TimeSignature(time = time, numerator = 0, denominator = 0))
                case "Tempo":
                    music.tempos.append(Tempo(time = time, qpm = representation.TEMPO_QPM_MAP[value], text = value))
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


def decode(codes: np.array, encoding: dict = representation.get_encoding()) -> BetterMusic:
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


def dump(codes: np.array, encoding: dict = representation.get_encoding()) -> str:
    """Decode the codes and dump as a string."""

    # get maps
    code_type_map = encoding["code_type_map"]
    code_beat_map = encoding["code_beat_map"]
    code_position_map = encoding["code_position_map"]
    code_value_map = encoding["code_value_map"]
    code_duration_map = encoding["code_duration_map"]
    code_instrument_map = encoding["code_instrument_map"]

    # get the dimension indices
    beat_dim = encoding["dimensions"].index("beat")
    position_dim = encoding["dimensions"].index("position")
    value_dim = encoding["dimensions"].index("value")
    duration_dim = encoding["dimensions"].index("duration")
    instrument_dim = encoding["dimensions"].index("instrument")

    # iterate over the rows
    lines = []
    for row in codes:
        event_type = code_type_map[int(row[0])] # get the event type

        # start of song events
        if event_type == "start-of-song":
            lines.append("Start of song")

        # end of songs events
        elif event_type == "end-of-song":
            lines.append("End of song")

        # instrument events
        elif event_type == "instrument":
            instrument = code_instrument_map[int(row[instrument_dim])]
            lines.append(f"Instrument: {instrument}")

        # start of notes events
        elif event_type == "start-of-notes":
            lines.append("Start of notes")

        # values
        elif event_type in ("note", "grace-note", representation.EXPRESSIVE_FEATURE_TYPE_STRING):
            
            # get beat
            beat = code_beat_map[int(row[beat_dim])]
            
            # get position
            position = code_position_map[int(row[position_dim])]

            # get value
            if value == representation.DEFAULT_VALUE_CODE:
                value = "TEXT"
            elif value == 0:
                value = "null"
            elif value <= 128:
                value = note_number_to_name(note_number = code_value_map[int(row[value_dim])])
            else:
                value = str(code_value_map[int(row[value_dim])])
            
            # get duration
            duration = code_duration_map[int(row[duration_dim])]
            
            # get instrument
            instrument = code_instrument_map[int(row[instrument_dim])]

            # create output
            lines.append(f"{''.join(' '.join(event_type.split('-')).title().split())}: beat={beat}, position={position}, value={value}, duration={duration}, instrument={instrument}")
        
        # unknown event type
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    # return big string
    return "\n".join(lines)

##################################################


# SAVE DATA
##################################################

def save_txt(filepath: str, codes: np.array, encoding: dict = representation.get_encoding()):
    """Dump the codes into a TXT file."""
    with open(filepath, "w") as f:
        f.write(dump(codes = codes, encoding = encoding))

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"

    # get arguments
    parser = argparse.ArgumentParser(prog = "Representation", description = "Test Encoding/Decoding mechanisms for MuseScore data.")
    parser.add_argument("-e", "--encoding", type = str, default = ENCODING_FILEPATH, help = "Absolute filepath to encoding file")
    args = parser.parse_args()

    # load encoding
    encoding = representation.load_encoding(filepath = args.encoding)

    # print an example
    print(f"{'Example':=^40}")
    codes = np.array(object = (
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
        ), dtype = ENCODING_ARRAY_TYPE)
    print(f"Codes:\n{codes}")

    music = decode(codes = codes, encoding = encoding)
    print("-" * 40)
    print(f"Decoded music:\n{music}")

    print("-" * 40)
    print(f"Decoded:\n{dump(codes = codes, encoding = encoding)}")

##################################################

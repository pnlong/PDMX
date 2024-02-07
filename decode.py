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
from encode import encode, ENCODING_ARRAY_TYPE
from read_mscz.music import BetterMusic
from read_mscz.classes import *
from read_mscz.read_mscz import read_musescore
from muspy.utils import CIRCLE_OF_FIFTHS
##################################################


# CONSTANTS
##################################################



##################################################


# DECODER FUNCTIONS
##################################################

def decode_data(codes: np.array, encoding: dict = representation.get_encoding()) -> List[list]:
    """Decode codes into a data sequence.
    Each row of the input is encoded as follows.
        (event_type, beat, position, value, duration, instrument, velocity (if included))
    Each row of the output is decoded the same way.
    """

    # whether there is a velocity field
    include_velocity = ("velocity" in encoding["dimensions"]) and (len(encoding["dimensions"]) == codes.shape[1])

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

    # if we are including velocity
    if include_velocity:
        code_velocity_map = encoding["code_velocity_map"]
        velocity_dim = encoding["dimensions"].index("velocity")

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
            value = code_value_map[max(int(row[value_dim]), 0)]
            duration = code_duration_map[int(row[duration_dim])]
            program = instrument_program_map[code_instrument_map[int(row[instrument_dim])]]
            current_row = (event_type, beat, position, value, duration, program)
            if include_velocity:
                current_row.append(code_velocity_map[int(row[velocity_dim])])
            data.append(current_row)
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
    ongoing_articulation_chunks = {track_index: {} for track_index in range(len(programs))}
    for row in data:
        event_type, beat, position, value, duration, program = row[:6]
        velocity = row[-1] if (len(row) > 6) else DEFAULT_VELOCITY # if there is a velocity value
        if any(field is None for field in (beat, position, duration, program)): # skip if invalid
            continue
        track_index = programs.index(program) # get track index
        time = (beat * resolution) + ((position / encoding["resolution"]) * resolution) # get time in time steps
        duration = (resolution / encoding["resolution"]) * duration # get duration in time steps
        if event_type in ("note", "grace-note"):
            try:
                pitch = int(value) # make sure the pitch is in fact a pitch
            except (ValueError):
                continue
            music.tracks[track_index].notes.append(Note(time = time, pitch = pitch, duration = duration, velocity = velocity, is_grace = (event_type == "grace_note")))
            for ongoing_articulation_chunk in tuple(ongoing_articulation_chunks[track_index].keys()): # add articulation if necessary
                if time <= ongoing_articulation_chunks[track_index][ongoing_articulation_chunk]: # if the duration is still on
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = Articulation(subtype = ongoing_articulation_chunk)))
                else: # duration is over, delete this chunk, since it is no longer ongoing
                    del ongoing_articulation_chunks[track_index][ongoing_articulation_chunk]
        elif event_type == representation.EXPRESSIVE_FEATURE_TYPE_STRING:
            if value in representation.EXPRESSIVE_FEATURE_TYPE_MAP.keys(): # as to not cause a KeyError
                expressive_feature_type = representation.EXPRESSIVE_FEATURE_TYPE_MAP[value]
            else:
                continue
            match expressive_feature_type:
                case "Barline":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["Barline"]:
                        music.barlines.append(Barline(time = time))
                    music.barlines.append(Barline(time = time, subtype = value.replace("-barline", "")))
                case "KeySignature":
                    fifths = music.key_signatures[-1].fifths + int(value.replace("key-signature-change-", ""))
                    if fifths <= -6:
                        fifths = fifths + 12
                    elif fifths > 6:
                        fifths = fifths - 12
                    root, root_str = CIRCLE_OF_FIFTHS[fifths + CIRCLE_OF_FIFTHS.index((0, "C"))]
                    key_signature_obj = KeySignature(time = time, root = root, fifths = fifths, root_str = root_str)
                    if time == 0:
                        music.key_signatures[0] = key_signature_obj
                    else:
                        music.key_signatures.append(key_signature_obj)
                case "TimeSignature":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["TimeSignature"]: # skip if unknown time signature change
                        continue
                    numerator, denominator = 4 * eval(value.replace(representation.TIME_SIGNATURE_CHANGE_PREFIX, "")) * (music.time_signatures[-1].numerator / music.time_signatures[-1].denominator), 4 # default is 4/4
                    while denominator < 16:
                        if min(numerator % 1, -(numerator % 1) + 1) < 0.25:
                            break
                        else:
                            numerator, denominator = numerator * 2, denominator * 2
                    time_signature_obj = TimeSignature(time = time, numerator = round(numerator), denominator = denominator)
                    if time == 0:
                        music.time_signatures[0] = time_signature_obj
                    else:
                        music.time_signatures.append(time_signature_obj)
                case "Fermata":
                    music.annotations.append(Annotation(time = time, annotation = Fermata()))
                case "SlurSpanner":
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = SlurSpanner(duration = duration, is_slur = True)))
                case "PedalSpanner":
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = PedalSpanner(duration = duration)))
                case "Tempo":
                    tempo_obj = Tempo(time = time, qpm = representation.TEMPO_QPM_MAP[value], text = value)
                    if time == 0:
                        music.tempos[0] = tempo_obj
                    else:
                        music.tempos.append(tempo_obj)
                case "TempoSpanner":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["TempoSpanner"]: # skip if unknown TempoSpanner
                        continue
                    music.annotations.append(Annotation(time = time, annotation = TempoSpanner(duration = duration, subtype = value)))
                case "Dynamic":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["Dynamic"]: # skip if unknown Dynamic
                        continue
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = Dynamic(subtype = value, velocity = representation.DYNAMIC_VELOCITY_MAP[value])))
                case "HairPinSpanner":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["HairPinSpanner"]: # skip if unknown HairPinSpanner
                        continue
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = HairPinSpanner(duration = duration, subtype = value)))
                case "Articulation":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["Articulation"]: # skip if unknown articulation
                        continue
                    ongoing_articulation_chunks[track_index][value] = time + duration # add stop time to ongoing articulation chunks
                case "Text":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["Text"]: # skip if unknown text
                        continue
                    music.annotations.append(Annotation(time = time, annotation = Text(text = value, is_system = True)))
                case "RehearsalMark":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["RehearsalMark"]: # skip if unknown RehearsalMark
                        continue
                    music.annotations.append(Annotation(time = time, annotation = RehearsalMark(text = value)))
                case "TextSpanner":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["TextSpanner"]: # skip if unknown TextSpanner
                        continue
                    music.annotations.append(Annotation(time = time, annotation = TextSpanner(duration = duration, text = value, is_system = True)))
                case "TechAnnotation":
                    if value == representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["TechAnnotation"]: # skip if unknown TechAnnotation
                        continue
                    music.tracks[track_index].annotations.append(Annotation(time = time, annotation = TechAnnotation(text = value)))
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

    # whether there is a velocity field
    include_velocity = ("velocity" in encoding["dimensions"]) and (len(encoding["dimensions"]) == codes.shape[1])

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

    # if we are including velocity
    if include_velocity:
        code_velocity_map = encoding["code_velocity_map"]
        velocity_dim = encoding["dimensions"].index("velocity")
    
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
            value = int(row[value_dim])
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

            if include_velocity:
                velocity = code_velocity_map[int(row[velocity_dim])]

            # create output
            lines.append(f"{''.join(' '.join(event_type.split('-')).title().split())}: beat={beat}, position={position}, value={value}, duration={duration}, instrument={instrument}" + (f", velocity={velocity}" if include_velocity else ""))

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

    # get arguments
    parser = argparse.ArgumentParser(prog = "Representation", description = "Test Encoding/Decoding mechanisms for MuseScore data.")
    parser.add_argument("-e", "--encoding", type = str, default = representation.ENCODING_FILEPATH, help = "Absolute filepath to encoding file")
    args = parser.parse_args()

    # load encoding
    encoding = representation.load_encoding(filepath = args.encoding)

    # print an example
    print(f"{'Example':=^40}")
    # codes = np.array(object = (
    #         (0, 0, 0, 0, 0, 0),
    #         (1, 0, 0, 0, 0, 3),
    #         (1, 0, 0, 0, 0, 33),
    #         (2, 0, 0, 0, 0, 0),
    #         (5, 1, 1, 49, 15, 3),
    #         (5, 1, 1, 61, 15, 3),
    #         (5, 1, 1, 65, 15, 3),
    #         (5, 1, 1, 68, 10, 33),
    #         (5, 1, 1, 68, 15, 3),
    #         (5, 2, 1, 68, 10, 33),
    #         (5, 3, 1, 68, 10, 33),
    #         (5, 4, 1, 61, 10, 33),
    #         (5, 4, 1, 61, 15, 3),
    #         (5, 4, 1, 65, 4, 33),
    #         (5, 4, 1, 65, 10, 3),
    #         (5, 4, 1, 68, 10, 3),
    #         (5, 4, 1, 73, 10, 3),
    #         (5, 4, 12, 63, 4, 33),
    #         (6, 0, 0, 0, 0, 0),
    #     ), dtype = ENCODING_ARRAY_TYPE)
    music = read_musescore(path = "/data2/pnlong/musescore/test_data/laufey/from_the_start.mscz")
    codes = encode(music = music, encoding = encoding)
    print(f"Codes:\n{codes}")

    music = decode(codes = codes, encoding = encoding)
    print("-" * 40)
    print(f"Decoded music:\n{music}")

    print("-" * 40)
    print(f"Decoded:\n{dump(codes = codes, encoding = encoding)}")

##################################################

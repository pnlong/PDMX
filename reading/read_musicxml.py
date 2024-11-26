# README
# Phillip Long
# October 23, 2024

# Create an object that can read MusicXML files (.musicxml) into a prettier, pythonic format.

# python /home/pnlong/model_musescore/reading/read_musicXML.py

# IMPORTS / CONSTANTS
##################################################

import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from fractions import Fraction
from functools import reduce
from operator import attrgetter
from os.path import dirname, join, basename
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from xml.etree.ElementTree import Element
from zipfile import ZipFile
from re import sub
import numpy as np
from muspy.utils import CIRCLE_OF_FIFTHS, MODE_CENTERS, NOTE_TYPE_MAP, TONAL_PITCH_CLASSES

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from classes import *
from music import MusicRender
from read_musescore import OTTAVA_OCTAVE_SHIFT_FACTORS, TRANSPOSE_CHROMATIC_TO_CIRCLE_OF_FIFTHS_STEPS, _get_text, _get_required_text, _get_required_attr, get_nice_measure_number, get_beats, print_measure_indicies
from output import DYNAMIC_VELOCITY_MAP, DEFAULT_VELOCITY

T = TypeVar("T")

##################################################


# CONSTANTS
##################################################

# convert pitch strings to indicies
TONAL_PITCH_CLASSES_INVERSE = {value: key for key, value in TONAL_PITCH_CLASSES.items()}

# pitches as index
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PITCH_INDEX_MAP = {note: i for i, note in enumerate(NOTES)}

##################################################


# EXCEPTIONS
##################################################

class MusicXMLError(Exception):
    """A class for MusicXML related errors."""

class MusicXMLWarning(Warning):
    """A class for MusicXML related warnings."""

##################################################


# FOR GETTING THE XML, BASIC INFO
##################################################

def _get_root(path: str, compressed: bool = None):
    """Return root of the element tree."""
    if compressed is None:
        compressed = path.endswith(".mxl")

    if not compressed:
        tree = ET.parse(path)
        return tree.getroot()

    # Find out the main MusicXML file in the compressed ZIP archive
    try:
        zip_file = ZipFile(file = path)
    except: # zipfile.BadZipFile: File is not a zip file
        raise MusicXMLError(f"{path} is not a zip file")
    if "META-INF/container.xml" not in zip_file.namelist():
        raise MusicXMLError("Container file ('container.xml') not found.")
    container = ET.fromstring(zip_file.read("META-INF/container.xml")) # read the container file as xml elementtree
    rootfile = container.findall(path = "rootfiles/rootfile") # find all branches in the container file, look for .xml
    rootfile = [file for file in rootfile if "xml" in file.get("full-path")]
    if len(rootfile) == 0:
        raise MusicXMLError("Element 'rootfile' tag not found in the container file ('container.xml').")
    filename = _get_required_attr(element = rootfile[0], attr = "full-path")
    if filename in zip_file.namelist():
        root = ET.fromstring(zip_file.read(filename))
    else:
        try:
            root = ET.fromstring(zip_file.read(tuple(path for path in zip_file.namelist() if path.endswith(".xml"))[0]))
        except (IndexError):
            raise MusicXMLError("No .xml file could be found in .mxl.")
    return root


def _get_divisions(root: Element) -> List[int]:
    """Return a list of divisions."""
    divisions = []
    for division in root.findall("part/measure/attributes/divisions"):
        if division.text is None:
            continue
        if not float(division.text).is_integer():
            raise MusicXMLError(
                "Noninteger 'division' values are not supported."
            )
        divisions.append(int(division.text))
    return divisions

##################################################


# FOR FIGURING OUT MEASURE SEQUENCE
##################################################

def parse_repeats(elem: Element) -> Tuple[list, list]:
    """Read repeats."""
    # Initialize with a start marker
    start_repeats = [0]
    end_repeats = [[],]

    # Find all repeats in all measures
    for i, measure in enumerate(elem.findall(path = "measure")):
        repeat = measure.find(path = "barline/repeat")
        if repeat is not None:
            repeat_direction = repeat.get("direction")
            # start repeat
            if repeat_direction == "forward":
                start_repeats.append(i)
                end_repeats.append([])
            # end repeat
            elif repeat_direction == "backward":
                end_repeats[len(start_repeats) - 1].append(i)

    # if there is an implied repeat at the end
    if len(end_repeats[-1]) == 0 and len(start_repeats) >= 2:
        end_repeats[-1] = [i]

    # remove faulty repeats (start repeats with no corresponding end repeats)
    start_repeats_filtered, end_repeats_filtered = [], []
    for i in range(len(end_repeats)):
        if len(end_repeats[i]) > 0:
            start_repeats_filtered.append(start_repeats[i])
            end_repeats_filtered.append(end_repeats[i])
    
    return start_repeats_filtered, end_repeats_filtered


def parse_markers(elem: Element) -> Dict[str, int]:
    """Return a marker-measure map parsed from a staff element."""
    # Initialize with a start marker
    markers: Dict[str, int] = {"start": 0}

    # Find all markers in all measures
    for i, measure in enumerate(elem.findall(path = "measure")):
        directions = measure.findall(path = "direction/sound")
        for direction in directions:
            if direction.get("segno") is not None: # segno
                markers["segno"] = i
            elif direction.get("tocoda") is not None: # to coda
                markers["tocoda"] = i
            elif direction.get("coda") is not None: # coda
                markers["coda"] = i
            elif direction.get("fine") is not None: # fine
                markers["fine"] = i

    return markers


def get_measure_ordering(elem: Element, timeout: int = None) -> List[int]:
    """Return a list of measure indices parsed from a staff element.

    This function returns the ordering of measures, considering all
    repeats and jumps.

    """
    # create measure indices list
    measure_indicies = []

    # Get the markers and repeats
    markers = parse_markers(elem = elem)
    start_repeats, end_repeats = parse_repeats(elem = elem)
    voltas_encountered = [] # voltas that have been past already
    current_repeat_idx = -1
    current_end_repeat_idx = 0

    # Record start time to check for timeout
    if timeout is not None:
        start_time = time.time()

    # boolean flags
    before_jump = True
    add_current_measure_idx = True

    # jump related indicies
    jump_to_idx = None
    play_until_idx = None
    continue_at_idx = None

    # Iterate over all measures
    measures = elem.findall(path = "measure")
    measure_idx = 0
    while measure_idx < len(measures):

        # Check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # Get the measure element
        measure = measures[measure_idx]

        #                                jump
        #                                v
        #       ║----|----|----|----|----|----|----║
        #            ^         ^              ^
        #            jump-to   play-until     continue at

        # set default next measure id
        next_measure_idx = measure_idx + 1

        # look for start repeat
        if measure_idx in start_repeats: # if this measure is the start of a repeat section
            previous_repeat_idx = current_repeat_idx
            current_repeat_idx = start_repeats.index(measure_idx)
            if previous_repeat_idx != current_repeat_idx:
                current_end_repeat_idx = 0

        # look for jump
        jump = measure.find(path = "direction/sound") # search for jump element
        if jump is not None and ((jump.get("dalsegno") is not None) or (jump.get("dacapo") is not None)) and before_jump: # if jump is found
            is_dalsegno = jump.get("dalsegno") is not None
            # set jump related indicies
            jump_to_idx = markers.get("segno" if is_dalsegno else "start")
            play_until_idx = markers.get("tocoda") if markers.get("tocoda") is not None else markers.get("fine")
            continue_at_idx = markers.get("coda") # will be None if there is no coda
            # set boolean flags
            before_jump = False
            # reset some variables
            voltas_encountered = []
            # go to the jump to measure
            next_measure_idx = jump_to_idx
        
        # look for end repeat (if there are any)
        if len(end_repeats) > 0:
            if len(end_repeats[current_repeat_idx][current_end_repeat_idx:]) > 0 and measure_idx == end_repeats[current_repeat_idx][current_end_repeat_idx]: # if there is an end repeat to look out for (still an outstanding end_repeat) and we are at the end repeat
                next_measure_idx = start_repeats[current_repeat_idx]
                current_end_repeat_idx += 1

        # look for first, second, etc. endings
        volta = measure.find(path = "barline/ending")
        if volta is not None and volta.get("type") == "start":
            if measure_idx not in (encounter[0] for encounter in voltas_encountered): # if we have not yet seen this volta
                volta_duration = 0
                for volta_exploration_measure in measures[measure_idx:]: # determine the volta duration
                    potential_volta_ends = volta_exploration_measure.findall(path = "barline/ending")
                    volta_duration += 1
                    if (len(potential_volta_ends) > 0) and (potential_volta_ends[-1].get("type") in ("stop", "discontinue")): # end of a volta will always be the last element in potential_volta_ends
                        break
                voltas_encountered.append((measure_idx, volta_duration))
            else: # if we have already seen this volta, skip volta_duration measures ahead
                for volta_duration in (encounter[1] for encounter in voltas_encountered if encounter[0] >= measure_idx):
                    measure_idx += volta_duration
                next_measure_idx = measure_idx
                add_current_measure_idx = False

        # look for Coda / Fine markings
        if not before_jump and measure_idx == play_until_idx:
            if continue_at_idx: # Coda
                next_measure_idx = continue_at_idx
            else: # Fine
                next_measure_idx = len(measures) # end the while loop at the end of this iteration

        # add the current measure index to measure indicies if I am allowed to
        if add_current_measure_idx:
            measure_indicies.append(measure_idx)
            # for debugging
            # print(measure_idx + 1, end = " " if measure_idx + 1 == next_measure_idx else "\n")
        else: # flick off the switch
            add_current_measure_idx = True
        
        # Proceed to the next measure
        measure_idx = next_measure_idx

    return measure_indicies


def get_improved_measure_ordering(measures: List[Element], measure_indicies: List[int], resolution: int) -> List[int]:
    """Return a list of measure indices parsed from a staff element.

    This function returns the ordering of measures, considering all
    repeats and jumps, as well as part-specific repeats.
    
    """

    # scrape measures for repeat counts
    measure_len = round(resolution * 4) # for keeping track of measure length
    measure_repeat_counts = [0] * len(measure_indicies)
    i = 0
    while i < len(measure_indicies):

        # get the measure element
        measure = measures[measure_indicies[i]] # get the measure element

        # check to update measure length
        time_signature = measure.find(path = "attributes/time") # check for time signature change
        if time_signature is not None:
            numerator, denominator = parse_time(elem = time_signature)
            measure_len = round((resolution / (denominator / 4)) * numerator)

        # check if the measure is a repeat
        measure_repeat_count = _get_text(element = measure, path = "forward/duration") # check if this measure is a repeat
        if measure_repeat_count is not None and len(list(filter(lambda elem: elem.tag == "note", measure))) == 0:
            duration = int(measure_repeat_count) // measure_len # duration of measure repeat (in measures)
            measure_repeat_counts[i:(i + duration)] = [duration] * duration
            i += duration
        else:
            i += 1

    # get improved measure indicies
    measure_indicies_improved = [0] * len(measure_indicies)
    for i in range(len(measure_indicies)):
        j = i
        while measure_repeat_counts[j] > 0:
            j -= measure_repeat_counts[j]
            if j < 0: # if j becomes an invalid index
                j = 0 # set j to 0
                break # break out of while loop
        measure_indicies_improved[i] = measure_indicies[j]

    # return improved measure indicies
    return measure_indicies_improved

##################################################


# PARSE METADATA
##################################################

def parse_metadata(root: Element) -> Metadata:
    """Return a Metadata object parsed from a MusicXML file."""

    # get title
    title = root.find(path = "work/movement-title")
    if title is not None and title.text is not None:
        title = title.text
    if title is None: # only use work title if movement title is not found
        title = root.find(path = "work/work-title")
        if title is not None and title.text is not None:
            title = title.text

    # get subtitle
    subtitle = root.find(path = "work/work-subtitle")
    if subtitle is not None and subtitle.text is not None:
        subtitle = subtitle.text

    # creators
    creators = []
    for creator in root.findall("identification/creator"):
        name = _get_required_attr(element = creator, attr = "type")
        if name in ("arranger", "composer", "lyricist") and creator.text is not None:
            creators.append(creator.text)

    # copyrights
    copyrights = []
    for copyright in root.findall("identification/rights"):
        if copyright.text is not None:
            copyrights.append(copyright.text)

    # iterate over meta tags
    for meta_tag in root.findall(path = "Score/metaTag"):
        name = _get_required_attr(element = meta_tag, attr = "name")
        if name == "movementTitle":
            title = meta_tag.text
        if name == "subtitle":
            subtitle = meta_tag.text
        # Only use 'workTitle' when movementTitle is not found
        if title is None and name == "workTitle":
            title = meta_tag.text
        if name in ("arranger", "composer", "lyricist"):
            if meta_tag.text is not None:
                creators.append(meta_tag.text)
        if name == "copyright":
            if meta_tag.text is not None:
                copyrights.append(meta_tag.text)

    return Metadata(title = title, subtitle = subtitle, creators = creators, copyright = " ".join(copyrights) if copyrights else None, source_format = "musicxml")


def parse_part_info(part_header: Element, part: Element) -> OrderedDict:
    """Return part information parsed from a score part element."""
    part_info: OrderedDict = OrderedDict()

    # Instrument
    part_info["id"] = part_header.get("id")
    part_info["name"] = _get_text(element = part_header, path = "part-name", remove_newlines = True)

    # transposes
    transpose_chromatic = _get_text(element = part, path = "measure/attributes/transpose/chromatic", remove_newlines = True)
    transpose_chromatic = int(transpose_chromatic) if transpose_chromatic is not None else 0
    part_info["transposeChromatic"] = transpose_chromatic
    transpose_diatonic = _get_text(element = part, path = "measure/attributes/transpose/diatonic", remove_newlines = True)
    transpose_diatonic = int(transpose_diatonic) if transpose_diatonic is not None else 0
    part_info["transposeDiatonic"] = transpose_diatonic
    transpose_octave = _get_text(element = part, path = "measure/attributes/transpose/octave-change", remove_newlines = True)
    transpose_octave = int(transpose_octave) if transpose_octave is not None else 0
    part_info["transposeOctave"] = transpose_octave

    # MIDI program and channel
    program = part_header.find(path = "midi-instrument/midi-program")
    if program is not None:
        part_info["program"] = (int(program.text) - 1) if program.text is not None else 0
    else:
        part_info["program"] = 0
    is_drum = (((int(_get_text(element = part_header, path = "midi-instrument/midi-channel", default = 0)) - 1) == 9) or
               (_get_text(element = part, path = "measure/attributes/clef/sign", default = "") == "percussion") or
               ("drum" in str(part_info["name"]).lower()))
    part_info["is_drum"] = is_drum

    # extra information if drums
    if is_drum:
        instrument_id_to_pitch = dict()
        for midi_instrument in part_header.findall(path = "midi-instrument"):
            midi_instrument_id = midi_instrument.get("id")
            if midi_instrument_id is not None:
                instrument_id_to_pitch[midi_instrument_id] = int(_get_required_text(element = midi_instrument, path = "midi-unpitched")) - 1 # subtract 1 because of discrepancy between 1-based and 0-based indexing
        part_info["instrument_id_to_pitch"] = instrument_id_to_pitch

    return part_info


def get_parts_info(part_list: Element, parts: Element) -> List[OrderedDict]:
    """Return part information from a list of all the parts elements in the part-list section."""

    # initialize return collections
    parts_info: List[OrderedDict] = [] # for storing info on each part

    # iterate through the parts
    for part_header, part in zip(part_list, parts):
        current_part_info = parse_part_info(part_header = part_header, part = part) # get the part info
        parts_info.append(current_part_info) # add the current part info to parts_info
    
    # return a value or raise an error
    if len(parts_info) == 0: # Raise an error if there is no Part information
        raise MusicXMLError("Part information is missing (i.e. there are no parts).")
    else:
        return parts_info


def get_musicxml_version(path: str, compressed: bool = None) -> str:
    """Determine the version of a MusicXML file.

    Parameters
    ----------
    path : str
        Path to the MusicXML file to read.
    compressed : bool, optional
        Whether it is a compressed MusicXML file. Defaults to infer from the filename.

    Returns
    -------
    :str:
        Version of MusicXML
    """

    # get element tree root
    root = _get_root(path = path, compressed = compressed)

    # detect MusicXML version
    musicxml_version = root.get("version")

    return musicxml_version

##################################################


# PARSING VARIOUS ELEMENT TYPES IN XML
##################################################

def parse_metronome(elem: Element) -> Optional[float]:
    """Return a qpm value parsed from a metronome element."""
    beat_unit = _get_text(element = elem, path = "beat-unit")
    if beat_unit is not None:
        per_minute = _get_text(element = elem, path = "per-minute")
        if per_minute is not None and beat_unit in NOTE_TYPE_MAP:
            qpm = NOTE_TYPE_MAP[beat_unit] * float(per_minute)
            if elem.find(path = "beat-unit-dot") is not None:
                qpm *= 1.5
            return qpm
    return None


def parse_time(elem: Element) -> Tuple[int, int]:
    """Return the numerator and denominator of a time element."""
    # Numerator
    beats = _get_text(element = elem, path = "beats", default = "4")
    if "+" in beats:
        numerator = sum(int(beat) for beat in beats.split("+"))
    else:
        numerator = int(beats)

    # Denominator
    beat_type = _get_text(element = elem, path = "beat-type", default = "4")
    if "+" in beat_type:
        raise RuntimeError("Compound time signatures with separate fractions are not supported.")
    denominator = int(beat_type)

    return numerator, denominator


def parse_key(elem: Element, transpose_chromatic: int = 0) -> Tuple[int, str, int, str]:
    """Return the key parsed from a key element."""

    fifths_text = _get_text(element = elem, path = "fifths")
    if fifths_text is None:
        return None, None, None, None
        # raise MusicXMLError("'accidental', 'subtype', or 'concertKey' subelement not found for KeySig element.")
    fifths = int(fifths_text) + transpose_chromatic # adjust for tuning
    mode = "major" # MusicXML doesn't provide any information on mode, so we assume major
    idx = MODE_CENTERS[mode] + fifths
    if idx < 0 or idx > 20:
        return None, mode, fifths, None  # type: ignore
    root, root_str = CIRCLE_OF_FIFTHS[MODE_CENTERS[mode] + fifths]
    return root, mode, fifths, root_str


def parse_lyric(elem: Element) -> str:
    """Return the lyric text parsed from a lyric element."""
    text = _get_required_text(element = elem, path = "text")
    syllabic = elem.find(path = "syllabic")
    if syllabic is not None:
        if syllabic.text == "begin":
            text = f"{text} -"
        elif syllabic.text == "middle":
            text = f"- {text} -"
        elif syllabic.text == "end":
            text = f"- {text}"
    return text


def parse_pitch(elem: Element, transpose_chromatic: int = 0) -> Tuple[int, str]:
    """Return the pitch and pitch_str of a pitch element."""

    # get basic info (string and octave)
    if elem.tag == "pitch":
        pitch_str = _get_required_text(element = elem, path = "step")
        pitch_octave = int(_get_required_text(element = elem, path = "octave"))
    elif elem.tag == "unpitched":
        pitch_str = _get_required_text(element = elem, path = "display-step")
        pitch_octave = int(_get_required_text(element = elem, path = "display-octave"))

    # account for tuning
    pitch_index = NOTES.index(pitch_str) + transpose_chromatic
    if pitch_index < 0:
        pitch_octave -= 1
    elif pitch_index >= len(NOTES):
        pitch_octave += 1
    pitch_str = NOTES[pitch_index % len(NOTES)]

    # get the base pitch
    base_pitch = NOTE_TO_PITCH_INDEX_MAP[pitch_str] + (12 * (pitch_octave + 1))
    pitch_alter = int(_get_text(element = elem, path = "alter", default = "0"))
    pitch = base_pitch + pitch_alter

    # alter pitch_str according to pitch_alter
    if pitch_alter > 0: # sharps
        pitch_str += "#" * pitch_alter
    elif pitch_alter < 0: # flats
        pitch_str += "b" * -pitch_alter

    # return pitch and pitch str
    return pitch, pitch_str


def parse_unpitched(elem: Element, elem_parent: Element, part_info: OrderedDict) -> Tuple[int, str]:
    """Return the pitch and pitch_str of an unpitched element."""

    # basic info
    pitch_str = _get_required_text(element = elem, path = "display-step")

    # instrument_id_to_pitch map
    instrument_id_to_pitch = part_info.get("instrument_id_to_pitch", dict())

    # determine pitch
    instrument = elem_parent.find(path = "instrument")
    if instrument is not None and instrument.get("id") is not None:
        instrument_id = instrument.get("id")
        pitch = instrument_id_to_pitch.get(instrument_id, 0)
        return pitch, pitch_str
    
    # if nothing is found
    else:
        pitch_estimate, _ = parse_pitch(elem = elem)
        return pitch_estimate, pitch_str    

##################################################


# PARSE A PART FOR MUSICAL INFORMATION CONSTANT ACROSS ALL PARTS
##################################################

def parse_constant_features(
        part: Element,
        resolution: int,
        measure_indicies: List[int],
        timeout: int = None,
        part_info: OrderedDict = OrderedDict([("transposeChromatic", 0), ("transposeDiatonic", 0), ("transposeOctave", 0), ("instrument_id_to_pitch", dict())])
    ) -> Tuple[List[Tempo], List[KeySignature], List[TimeSignature], List[Barline], List[Beat], List[Annotation]]:
    """Return data parsed from a meta part element.

    This function only parses the tempos, key and time signatures. Use `parse_part` to parse the notes and lyrics.

    """

    # initialize lists
    tempos: List[Tempo] = []
    key_signatures: List[KeySignature] = []
    time_signatures: List[TimeSignature] = []
    barlines: List[Barline] = []
    annotations: List[Annotation] = []

    # Initialize variables
    time_ = 0
    measure_len = round(resolution * 4)
    is_tuple = False
    notes_left_in_tuple = 0
    downbeat_times: List[int] = []
    transpose_circle_of_fifths = TRANSPOSE_CHROMATIC_TO_CIRCLE_OF_FIFTHS_STEPS[part_info["transposeChromatic"] % -12] # number of semitones to transpose so that chord symbols are concert pitch
    previous_note_duration = 0

    # record start time to check for timeout
    if timeout is not None:
        start_time = time.time()

    # create a dictionary to handle spanners
    spanners: Dict[str, Tuple[int, int]] = {}

    # Iterate over all elements
    measures = part.findall(path = "measure")
    measure_indicies_improved = get_improved_measure_ordering(measures = measures, measure_indicies = measure_indicies, resolution = resolution)
    for measure_idx, measure_idx_to_actually_read in zip(measure_indicies, measure_indicies_improved):

        # Check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # Get the measure element
        measure = measures[measure_idx_to_actually_read]
        is_measure_written_out = (measure_idx == measure_idx_to_actually_read)

        # initialize position
        position = 0
        max_position = position

        # Barlines, check for special types
        barlines_in_measure = list(map(lambda barline: barline.text, measure.findall("barline/bar-style")))
        barline_subtype = "double" if "light-light" in barlines_in_measure else "single"
        barline = Barline(time = time_, measure = get_nice_measure_number(i = measure_idx), subtype = barline_subtype)
        barlines.append(barline)

        # Collect the measure start times
        downbeat_times.append(time_)

        # Iterate over voice elements
        for elem in measure:

            # key and time signatures
            if is_measure_written_out and elem.tag == "attributes":
                for attribute in elem:
                    
                    # key signatures
                    if attribute.tag == "key":
                        root, mode, fifths, root_str = parse_key(elem = attribute, transpose_chromatic = transpose_circle_of_fifths)
                        if fifths is not None:
                            key_signatures.append(KeySignature(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), root = root, mode = mode, fifths = fifths, root_str = root_str))
                    
                    # time signatures
                    elif attribute.tag == "time":
                        numerator, denominator = parse_time(elem = elem)
                        measure_len = round((resolution / (denominator / 4)) * numerator)
                        time_signatures.append(TimeSignature(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), numerator = numerator, denominator = denominator))
                        del numerator, denominator

            # tempos, tempo spanners, text, text spanners, rehearsal marks
            elif is_measure_written_out and elem.tag == "direction":

                # different elements
                sound = elem.find(path = "sound")
                words = elem.find(path = "direction-type/words")
                rehearsal = elem.find(path = "direction-type/rehearsal")
                direction_types = elem.findall(path = "direction-type")

                # helper boolean flags
                start_of_tempo = (sound is not None and sound.get("tempo") is not None)
                start_of_words = (words is not None)
                start_of_rehearsal = (rehearsal is not None)
                start_of_spanner = (words is not None and len(direction_types) == 2 and direction_types[1][0].get("type") == "start" and direction_types[0][0].tag == "words" and (words.text is not None and words.text != ""))
                end_of_spanner = (words is None and len(direction_types) == 1 and direction_types[0][0].get("type") == "stop" and f"{direction_types[0][0].get('number')}-{direction_types[0][0].tag}" in spanners.keys())

                # tempo elements
                if start_of_tempo:
                    tempo_text = words.text if words is not None else None
                    metronome = elem.find(path = "direction-type/metronome")
                    tempo_qpm = None
                    if metronome is not None: # no relevant text with metronome
                        tempo_qpm = parse_metronome(elem = metronome)
                    if tempo_qpm is None:
                        tempo_qpm = float(sound.get("tempo"))
                    tempos.append(Tempo(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), qpm = tempo_qpm, text = tempo_text))
                    del metronome
                
                # start of text/tempo spanners
                elif start_of_words and start_of_spanner:
                    is_tempo = (len(direction_types[1][0].attrib) == 2) # tempo spanner spanners only have the "type" and "number" keys, nothing else
                    spanners[f"{direction_types[1][0].get('number')}-{direction_types[1][0].tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                    if is_tempo:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TempoSpanner(duration = 0, subtype = words.text)))
                    else:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TextSpanner(duration = 0, text = words.text, is_system = True)))
                    del is_tempo

                # end of tempo/text spanner elements
                elif end_of_spanner:
                    number = f"{direction_types[0][0].get('number')}-{direction_types[0][0].tag}"
                    spanner_duration = time_ + position - spanners[number][0] # calculate duration of spanner
                    annotations[spanners[number][1]].annotation.duration = spanner_duration
                    del number, spanner_duration
                    
                # system text (non-spanning)
                elif start_of_words and not start_of_spanner:
                    style = "tempo" if str(words.text).lower() in ("swing", "straight") else None
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Text(text = words.text, is_system = True, style = style)))
                    del style

                # rehearsal mark
                elif start_of_rehearsal:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = RehearsalMark(text = rehearsal.text)))

            # forward elements
            elif elem.tag == "forward":
                duration = int(_get_required_text(element = elem, path = "duration"))
                position += duration
            
            # backup elements
            elif elem.tag == "backup":
                duration = int(_get_required_text(element = elem, path = "duration"))
                position -= duration
            
            # fermatas, note-based stuff
            elif elem.tag == "note":

                # boolean flags
                is_grace = (elem.find("grace") is not None)
                is_chord = (elem.find(path = "chord") is not None)

                # deal with duration
                duration = _get_text(element = elem, path = "duration")
                if duration is None:
                    duration_type = _get_text(element = elem, path = "type", default = "quarter")
                    duration = NOTE_TYPE_MAP[duration_type] * resolution
                    dots = elem.findall(path = "dot")
                    if len(dots) > 0: # already accounted for by MusicXML
                        duration *= 2 - 0.5 ** len(dots)
                    if is_tuple: # already accounted for by MusicXML
                        duration *= tuple_ratio
                else:
                    duration = int(duration)
                duration = round(duration)

                # if this is a chord, update position to reflect that
                if is_chord:
                    position -= previous_note_duration
                elif not is_chord and is_tuple: # adjust number of unique notes are left in tuple
                    notes_left_in_tuple -= 1

                # notation-based stuff
                notations = elem.find(path = "notations")
                if notations is not None:

                    # fermata
                    fermata = notations.find(path = "fermata")
                    if fermata is not None:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Fermata(is_fermata_above = (fermata.get("type") == "upright"))))

                    # tuplets, we don't need, since MusicXML already adjusts duration for us
                    tuplet = notations.find("tuplet")
                    if tuplet is not None:
                        if tuplet.get("type") == "start": # start of tuple
                            is_tuple = True
                            normal_notes = _get_text(element = elem, path = "time-modification/normal-notes")
                            actual_notes = _get_text(element = elem, path = "time-modification/actual-notes")
                            if normal_notes is None and actual_notes is None:
                                normal_notes = _get_text(element = tuplet, path = "tuplet-normal/tuplet-number", default = "2")
                                actual_notes = _get_text(element = tuplet, path = "tuplet-actual/tuplet-number", default = "3")
                            normal_notes, actual_notes = int(normal_notes), int(actual_notes)
                            tuple_ratio = normal_notes / actual_notes
                            notes_left_in_tuple = actual_notes
                        elif tuplet.get("type") == "stop" or (is_tuple and notes_left_in_tuple == 0): # handle last tuplet note
                            is_tuple = False

                # update position
                if not is_grace:
                    position += duration

                # make note of note duration
                previous_note_duration = duration

            # update positions
            if position > max_position:
                max_position = position
        
        # update time
        if measure_idx == 0: # deal with potential pickups
            time_ += max_position # get the maximum position (don't want some unfinished voice)
        else:
            time_ += measure_len

    # Sort tempos, key and time signatures
    tempos.sort(key = attrgetter("time"))
    key_signatures.sort(key = attrgetter("time"))
    time_signatures.sort(key = attrgetter("time"))
    annotations.sort(key = attrgetter("time"))
    annotations = list(filter(lambda annotation: (annotation.annotation.duration > 0) if hasattr(annotation.annotation, "duration") else True, annotations)) # filter out 0-duration annotations

    # Get the beats
    beats = get_beats(downbeat_times = downbeat_times, measure_indicies = measure_indicies, time_signatures = time_signatures, resolution = resolution, is_sorted = True)

    return tempos, key_signatures, time_signatures, barlines, beats, annotations

##################################################


# PARSE A STAFF FOR GENERAL MUSICAL INFORMATION
##################################################

def parse_part(
        part: Element,
        resolution: int,
        measure_indicies: List[int],
        timeout: int = None,
        part_info: OrderedDict = OrderedDict([("transposeChromatic", 0), ("transposeDiatonic", 0), ("transposeOctave", 0), ("instrument_id_to_pitch", dict())])
    ) -> Tuple[List[Note], List[Chord], List[Lyric], List[Annotation]]:
    """Return notes and lyrics parsed from a part element.

    This function only parses the notes and lyrics. Use
    `parse_constant_features` to parse the tempos, key and time
    signatures.

    """

    # Initialize lists
    notes: List[Note] = []
    chords: List[Chord] = []
    lyrics: List[Lyric] = []
    annotations: List[Annotation] = []

    # Initialize variables
    time_ = 0
    velocity = 64
    measure_len = round(resolution * 4)
    is_tuple = False
    notes_left_in_tuple = 0
    ottava_shift = 0
    base_ottava_shift = part_info["transposeOctave"] * 12
    transpose_chromatic = part_info["transposeChromatic"]
    transpose_circle_of_fifths = TRANSPOSE_CHROMATIC_TO_CIRCLE_OF_FIFTHS_STEPS[transpose_chromatic % -12] # number of semitones to transpose so that chord symbols are concert pitch
    previous_note_duration = 0

    # Record start time to check for timeout
    if timeout is not None:
        start_time = time.time()

    # Create a dictionary to handle ties
    ties: Dict[int, int] = {}

    # create a dictionary to handle spanners
    spanners: Dict[str, Tuple[int, int]] = {}

    # create a set to handle arpeggio times
    arpeggio_times: set = set()

    # Iterate over all elements
    measures = part.findall(path = "measure")
    measure_indicies_improved = get_improved_measure_ordering(measures = measures, measure_indicies = measure_indicies, resolution = resolution)
    for measure_idx, measure_idx_to_actually_read in zip(measure_indicies, measure_indicies_improved):

        # Check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # Get the measure element
        measure = measures[measure_idx_to_actually_read]
        is_measure_written_out = (measure_idx == measure_idx_to_actually_read)

        # initialize position
        position = 0
        max_position = position

        # Iterate over voice elements
        for elem in measure:                

            # keep track of measure length
            if is_measure_written_out and elem.tag == "attributes":
                time_signature = elem.find(path = "time")
                if time_signature is not None:
                    numerator, denominator = parse_time(elem = time_signature)
                    measure_len = round((resolution / (denominator / 4)) * numerator)
                    del numerator, denominator

            # tempos, tempo spanners, text, text spanners, rehearsal marks
            elif is_measure_written_out and elem.tag == "direction":

                # different elements
                dynamics = elem.find(path = "direction-type/dynamics")
                wedge = elem.find(path = "direction-type/wedge")
                pedal = elem.find(path = "direction-type/pedal")
                ottava = elem.find(path = "direction-type/octave_shift")
                words = elem.find(path = "direction-type/words")
                direction_types = elem.findall(path = "direction-type")

                # helper boolean flags
                start_of_dynamics = (dynamics is not None and len(dynamics) == 1)
                start_of_wedge = (wedge is not None)
                start_of_pedal = (pedal is not None)
                start_of_ottava = (ottava is not None and ottava.get("type") != "stop")
                start_of_words = (words is not None)
                start_of_spanner = (words is not None and len(direction_types) == 2 and direction_types[1][0].get("type") == "start" and direction_types[0][0].tag == "words" and (words.text is not None and words.text != ""))
                end_of_spanner = (words is None and len(direction_types) == 1 and direction_types[0][0].get("type") == "stop" and f"{direction_types[0][0].get('number')}-{direction_types[0][0].tag}" in spanners.keys())

                # dynamics
                if start_of_dynamics:
                    subtype = dynamics[0].tag
                    velocity = DYNAMIC_VELOCITY_MAP.get(subtype, DEFAULT_VELOCITY)
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Dynamic(subtype = subtype, velocity = velocity)))
                    del subtype # don't delete velocity, as we will use it for notes
                
                # start of text/tempo spanners
                elif start_of_words and start_of_spanner:
                    is_tempo = (len(direction_types[1][0].attrib) == 2) # tempo spanner spanners only have the "type" and "number" keys, nothing else
                    if not is_tempo: # we don't care about tempo markings on the part level
                        spanners[f"{direction_types[1][0].get('number')}-{direction_types[1][0].tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TextSpanner(duration = 0, text = words.text, is_system = False)))
                    del is_tempo

                # start of wedge (hairpin) spanner
                elif start_of_wedge:
                    spanners[f"{wedge.get('number')}-{wedge.tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = HairPinSpanner(duration = 0, subtype = wedge.get("type"))))

                # start of pedal spanner
                elif start_of_pedal:
                    spanners[f"{pedal.get('number')}-{pedal.tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = PedalSpanner(duration = 0)))

                # start of ottava
                elif start_of_ottava:
                    ottava_subtype = str(ottava.get("size"))
                    if ottava_subtype != "8":
                        ottava_subtype += "ma"
                    else:
                        ottava_subtype += "va" if ottava.get("type") == "up" else "ba"
                    ottava_shift = 12 * OTTAVA_OCTAVE_SHIFT_FACTORS[int(ottava.get("size"))] * (-1 if ottava.get("type") == "down" else 1)
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = OttavaSpanner(duration = 0, subtype = ottava_subtype)))
                    del ottava_subtype

                # end of tempo/text/wedge spanner elements
                elif end_of_spanner:
                    number = f"{direction_types[0][0].get('number')}-{direction_types[0][0].tag}"
                    spanner_duration = time_ + position - spanners[number][0] # calculate duration of spanner
                    annotations[spanners[number][1]].annotation.duration = spanner_duration
                    del number, spanner_duration
                    if direction_types[0][0].tag == "octave-shift": # reset octave shift if ottava is over
                        ottava_shift = base_ottava_shift

                # system text (non-spanning)
                elif start_of_words and not start_of_spanner:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Text(text = words.text, is_system = False, style = None)))

            # forward elements
            elif elem.tag == "forward":
                duration = int(_get_required_text(element = elem, path = "duration"))
                position += duration

            # backup elements
            elif elem.tag == "backup":
                duration = int(_get_required_text(element = elem, path = "duration"))
                position -= duration

            # chord symbol
            elif elem.tag == "harmony":
                base_root = _get_text(element = elem, path = "root/root-step")
                if base_root is not None:
                    root_alter = int(_get_text(element = elem, path = "root/root-alter", default = "0"))
                    if root_alter > 0: # sharps
                        base_root += "#" * root_alter
                    elif root_alter < 0: # flats
                        base_root += "b" * -root_alter
                    root = TONAL_PITCH_CLASSES_INVERSE[base_root] + transpose_circle_of_fifths # account for tuning
                    root_str = TONAL_PITCH_CLASSES[root]
                    name = elem.find("kind")
                    if name is not None:
                        name = name.get("text")
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = ChordSymbol(root_str = root_str, name = name)))
                    del root_alter, root, root_str, name
                del base_root
            
            # fermatas, note-based stuff
            elif elem.tag == "note":

                # boolean flags
                is_grace = (elem.find(path = "grace") is not None)
                is_rest = (elem.find(path = "rest") is not None)
                is_chord = (elem.find(path = "chord") is not None)
                tie = elem.find(path = "tie")
                is_outgoing_tie = (tie is not None and tie.get("type") == "start")
                del tie

                # pitch
                pitch = elem.find(path = "pitch")
                unpitched = elem.find(path = "unpitched")
                if pitch is not None:
                    pitch, pitch_str = parse_pitch(elem = pitch, transpose_chromatic = transpose_chromatic)
                    pitch += ottava_shift
                elif unpitched is not None:
                    pitch, pitch_str = parse_unpitched(elem = unpitched, elem_parent = elem, part_info = part_info)
                else:
                    is_rest = True

                # deal with duration
                duration = _get_text(element = elem, path = "duration")
                if duration is None:
                    duration_type = _get_text(element = elem, path = "type", default = "quarter")
                    duration = NOTE_TYPE_MAP[duration_type] * resolution
                    dots = elem.findall(path = "dot")
                    if len(dots) > 0: # already accounted for by MusicXML
                        duration *= 2 - 0.5 ** len(dots)
                    if is_tuple: # already accounted for by MusicXML
                        duration *= tuple_ratio
                else:
                    duration = int(duration)
                duration = round(duration)

                # if this is a chord, update position to reflect that
                if is_chord:
                    position -= previous_note_duration
                elif not is_chord and is_tuple: # adjust number of unique notes are left in tuple
                    notes_left_in_tuple -= 1

                # notation-based stuff
                notations = elem.find(path = "notations")
                if notations is not None:

                    # stuff to find
                    ornaments = notations.find(path = "ornaments")
                    articulations = notations.find(path = "articulations")
                    fermata = notations.find(path = "fermata")
                    glissando = notations.find(path = "glissando")
                    slur = notations.find(path = "slur")
                    arpeggiate = notations.find(path = "arpeggiate")
                    non_arpeggiate = notations.find(path = "non-arpeggiate")
                    tuplet = notations.find("tuplet")

                    # ornaments
                    if ornaments is not None:
                        analyze_current_ornament = True
                        for i, ornament in enumerate(ornaments): # iterate through ornaments
                            
                            # if we need to skip current iteration
                            if not analyze_current_ornament:
                                analyze_current_ornament = True
                                continue

                            # start of trill spanner
                            if ornament.tag == "trill-mark":
                                annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TrillSpanner(duration = 0, subtype = "trill")))
                                if len(ornaments) > 1 and ornaments[i + 1].get("type") == "start":
                                    spanners[f"{ornaments[i + 1].get('number')}-{ornaments[i + 1].tag}"] = (time_ + position, len(annotations) - 1) # add (start_time, index) associated with this spanner number
                                    analyze_current_ornament = False # skip the next ornament, as it's the start of a trill spanner
                            
                            # end of trill spanner
                            elif ornament.get("number") is not None and f"{ornament.get('number')}-{ornament.tag}" in spanners.keys(): # end of glissando
                                number = f"{ornament.get('number')}-{ornament.tag}"
                                spanner_duration = time_ + position + duration - spanners[number][0] # calculate duration of spanner
                                annotations[spanners[number][1]].annotation.duration = spanner_duration
                                del number, spanner_duration

                            # tremolo
                            elif ornament.tag == "tremolo":
                                annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Tremolo(subtype = ornament.get("type"))))

                            # ornaments
                            else:
                                annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Ornament(subtype = ornament.tag)))

                    # articulations
                    if articulations is not None:
                        for articulation in articulations:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Articulation(subtype = articulation.tag)))

                    # fermata
                    if fermata is not None:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Fermata(is_fermata_above = (fermata.get("type") == "upright"))))

                    # glissando spanners
                    if glissando is not None:
                        if glissando.get("type") == "start": # start of glissando
                            spanners[f"{glissando.get('number')}-{glissando.tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = GlissandoSpanner(duration = 0, is_wavy = (glissando.get("line-type") == "wavy"))))
                        elif glissando.get("type") == "stop" and f"{glissando.get('number')}-{glissando.tag}" in spanners.keys(): # end of glissando
                            number = f"{glissando.get('number')}-{glissando.tag}"
                            spanner_duration = time_ + position - spanners[number][0] # calculate duration of spanner
                            annotations[spanners[number][1]].annotation.duration = spanner_duration
                            del number, spanner_duration

                    # slur spanners
                    if slur is not None:
                        if slur.get("type") == "start": # start of slur
                            spanners[f"{slur.get('number')}-{slur.tag}"] = (time_ + position, len(annotations)) # add (start_time, index) associated with this spanner number
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = SlurSpanner(duration = 0, is_slur = True)))
                        elif slur.get("type") == "stop" and f"{slur.get('number')}-{slur.tag}" in spanners.keys(): # end of slur
                            number = f"{slur.get('number')}-{slur.tag}"
                            spanner_duration = time_ + position - spanners[number][0] # calculate duration of spanner
                            annotations[spanners[number][1]].annotation.duration = spanner_duration
                            del number, spanner_duration

                    # arpeggios
                    if arpeggiate is not None or non_arpeggiate is not None:
                        subtype = 0
                        if arpeggiate is not None and arpeggiate.get("direction") is not None:
                            subtype = 1 + int(arpeggiate.get("direction") == "down")
                        elif non_arpeggiate is not None:
                            subtype = 3
                        if (time_ + position) not in arpeggio_times:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Arpeggio(subtype = subtype)))
                            arpeggio_times.add(time_ + position)
                        del subtype

                    # tuplets, we don't need, since MusicXML already adjusts duration for us
                    if tuplet is not None:
                        if tuplet.get("type") == "start": # start of tuple
                            is_tuple = True
                            normal_notes = _get_text(element = elem, path = "time-modification/normal-notes")
                            actual_notes = _get_text(element = elem, path = "time-modification/actual-notes")
                            if normal_notes is None and actual_notes is None:
                                normal_notes = _get_text(element = tuplet, path = "tuplet-normal/tuplet-number", default = "2")
                                actual_notes = _get_text(element = tuplet, path = "tuplet-actual/tuplet-number", default = "3")
                            normal_notes, actual_notes = int(normal_notes), int(actual_notes)
                            tuple_ratio = normal_notes / actual_notes
                            notes_left_in_tuple = actual_notes
                        elif tuplet.get("type") == "stop" or (is_tuple and notes_left_in_tuple == 0): # handle last tuplet note
                            is_tuple = False

                # notehead/symbols
                notehead = elem.find(path = "notehead")
                if notehead is not None and notehead.text != "normal":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Notehead(subtype = notehead.text)))
                    if notehead.text == "slash": # if the notehead is a slash, the note shouldn't be played
                        is_rest = True
                if notehead is not None and notehead.get("parentheses") == "yes":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Symbol(subtype = "noteheadParenthesisLeft")))
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Symbol(subtype = "noteheadParenthesisRight")))

                # lyrics
                for lyric in elem.findall(path = "lyric"):
                    lyric_text = parse_lyric(elem = lyric)
                    lyrics.append(Lyric(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), lyric = lyric_text))
                
                # Handle grace note
                if is_grace:
                    notes.append(Note(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), pitch = pitch, duration = duration, velocity = velocity, pitch_str = pitch_str, is_grace = True))
                    continue

                # Check if it is an incoming tied note
                if not is_rest:
                    if pitch in ties.keys():
                        note_idx = ties[pitch]
                        notes[note_idx].duration += duration
                        if is_outgoing_tie: # if the tie continues
                            ties[pitch] = note_idx
                        else: # if the tie ended
                            del ties[pitch]

                    # Append a new note to the note list
                    else:
                        notes.append(Note(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), pitch = pitch, duration = duration, velocity = velocity, pitch_str = pitch_str))
                        if is_outgoing_tie: # start of a tie, make note of it
                            ties[pitch] = len(notes) - 1
                        if is_chord: # append to current chord
                            chords[-1].pitches.append(pitch)
                            chords[-1].pitches_str.append(pitch_str)
                        else: # create new cord
                            chords.append(Chord(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), pitches = [pitch,], duration = duration, velocity = velocity, pitches_str = [pitch_str,]))

                # update position
                if not is_grace:
                    position += duration

                # make note of note duration
                previous_note_duration = duration

            # update positions
            if position > max_position:
                max_position = position
        
        # update time
        if measure_idx == 0: # deal with potential pickups
            time_ += max_position # get the maximum position (don't want some unfinished voice)
        else:
            time_ += measure_len

    # Sort notes
    notes.sort(key = attrgetter("time", "pitch", "duration", "velocity"))

    # Sort chords
    chords.sort(key = attrgetter("time", "duration", "velocity"))

    # Sort lyrics
    lyrics.sort(key = attrgetter("time"))

    # Sort annotations
    annotations.sort(key = attrgetter("time"))
    annotations = list(filter(lambda annotation: (annotation.annotation.duration > 0) if hasattr(annotation.annotation, "duration") else True, annotations)) # filter out 0-duration annotations

    return notes, chords, lyrics, annotations

##################################################


# MY BETTER READ MUSESCORE FUNCTION, EXTRACTS EXPRESSIVE FEATURES
##################################################

def read_musicxml(path: str, resolution: int = None, compressed: bool = None, timeout: int = None) -> MusicRender:
    """Read a MusicXML file into a MusicRender object, paying close attention to details such as articulation and expressive features.

    Parameters
    ----------
    path : str
        Path to the MusicXML file to read.
    resolution : int, optional
        Time steps per quarter note. Defaults to the least common
        multiple of all divisions.
    compressed : bool, optional
        Whether it is a compressed MusicXML file. Defaults to infer
        from the filename.

    Returns
    -------
    :class:`MusicRender`
        Converted MusicRender object.

    """

    # get element tree for MusicXML file
    root = _get_root(path = path, compressed = compressed)
    # ET.ElementTree(root).write(f"{dirname(path)}/xml.xml")

    # metadata
    metadata = parse_metadata(root = root)
    metadata.source_filename = basename(path)

    # find the resolution
    if resolution is None:
        divisions = _get_divisions(root = root)
        resolution = max(divisions, key = divisions.count) if len(divisions) > 0 else muspy.DEFAULT_RESOLUTION

    # part information
    parts = root.findall(path = "part")
    parts_info = get_parts_info(part_list = root.findall(path = "part-list/score-part"), parts = parts)
    if len(parts) == 0: # Return empty music object with metadata if no parts are found
        return MusicRender(metadata = metadata, resolution = resolution)

    # parse measure ordering from the meta staff, expanding all repeats and jumps
    measure_indicies = get_measure_ordering(elem = parts[0], timeout = timeout) # feed in the first staff, since measure ordering are constant across all staffs

    # parse the  part element
    (tempos, key_signatures, time_signatures, barlines, beats, annotations) = parse_constant_features(part = parts[0], resolution = resolution, measure_indicies = measure_indicies, timeout = timeout, part_info = parts_info[0]) # feed in the first staff to extract features constant across all parts

    # initialize lists
    tracks: List[Track] = []

    # record start time to check for timeout
    start_time = time.time()

    # iterate over all parts
    for part, part_info in zip(parts, parts_info):
        if timeout is not None and time.time() - start_time > timeout: # check for timeout
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")
        (notes, chords, lyrics, annotations_part) = parse_part(part = part, resolution = resolution, measure_indicies = measure_indicies, timeout = timeout, part_info = part_info) # parse the part
        tracks.append(Track(program = part_info["program"], is_drum = part_info["is_drum"], name = part_info["name"], notes = notes, chords = chords, lyrics = lyrics, annotations = annotations_part)) # append to tracks

    # make sure everything is sorted
    tempos.sort(key = attrgetter("time"))
    key_signatures.sort(key = attrgetter("time"))
    time_signatures.sort(key = attrgetter("time"))
    annotations.sort(key = attrgetter("time"))
    for track in tracks:
        track.notes.sort(key = attrgetter("time", "pitch", "duration", "velocity"))
        track.chords.sort(key = attrgetter("time", "duration", "velocity"))
        track.lyrics.sort(key = attrgetter("time"))
        track.annotations.sort(key = attrgetter("time"))

    return MusicRender(metadata = metadata, resolution = resolution, tempos = tempos, key_signatures = key_signatures, time_signatures = time_signatures, barlines = barlines, beats = beats, tracks = tracks, annotations = annotations)

##################################################


# TEST
##################################################

if __name__ == "__main__":

    # imports
    from tqdm import tqdm
    import multiprocessing

    # paths to load
    paths = [
        "/data2/pnlong/musescore/test_data/marley/jamming.mxl",
        "/data2/pnlong/musescore/test_data/debussy/clair_de_lune.mxl",
        "/data2/pnlong/musescore/test_data/carey/all_i_want_for_christmas_is_you.mxl",
        "/data2/pnlong/musescore/test_data/jackson/i_want_you_back.mxl",
        "/data2/pnlong/musescore/test_data/newman/youve_got_a_friend_in_me.mxl",
    ]

    # helper function
    def make_example(path: str):
        """Make example."""
        stem = ".".join(path.split(".")[:-1])
        music = read_musicxml(path = path)
        music.print(output_filepath = join(dirname(path), "xml.yml"))
        music.write(f"{stem}.mid") # write as symbolic
        music.write(f"{stem}.wav") # write as audio

    # multiprocessing
    with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
        _ = list(pool.map(func = make_example, iterable = tqdm(iterable = paths, desc = f"Reading MusicXML Files", total = len(paths)), chunksize = 1))

##################################################
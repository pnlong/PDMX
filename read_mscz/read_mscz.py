# README
# Phillip Long
# September 26, 2023

# Create an object that can read musescore files (.mscz) into a prettier, pythonic format.
# Copied from https://github.com/salu133445/muspy/blob/main/muspy/inputs/musescore.py

# python /home/pnlong/parse_musescore/read_mscz/read_mscz.py

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

# muspy imports
from .classes import *
from .music import BetterMusic
from muspy.utils import CIRCLE_OF_FIFTHS, MODE_CENTERS, NOTE_TYPE_MAP, TONAL_PITCH_CLASSES

# create type variable
T = TypeVar("T")

##################################################


# EXCEPTIONS
##################################################

class MuseScoreError(Exception):
    """A class for MuseScore related errors."""

class MuseScoreWarning(Warning):
    """A class for MuseScore related warnings."""

##################################################


# HELPER FUNCTIONS
##################################################

def _gcd(a: int, b: int) -> int:
    """Return greatest common divisor using Euclid's Algorithm.

    Code copied from https://stackoverflow.com/a/147539.

    """
    while b:
        a, b = b, a % b
    return a


def _lcm_two_args(a: int, b: int) -> int:
    """Return least common multiple.

    Code copied from https://stackoverflow.com/a/147539.

    """
    return a * b // _gcd(a, b)


def _lcm(*args: int) -> int:
    """Return lcm of args.

    Code copied from https://stackoverflow.com/a/147539.

    """
    return reduce(_lcm_two_args, args)  # type: ignore


def prettify_ET(root: ET, output_filepath: str = None): # prettify an xml element tree
    """Return a pretty-printed (or outputted) XML string for the Element `root`.
    """
    rough_string = ET.tostring(element = root, encoding = "utf-8") # get string for xml
    rough_string = str(rough_string, "UTF-8") # convert from type bytes to str
    if output_filepath:
        with open(output_filepath, "w") as file:
            file.write(rough_string)
    else:
        print(rough_string)

##################################################


# GET ATTRIBUTES FROM XML
##################################################

def _get_required(element: Element, path: str) -> Element:
    """Return a required child element of an element.

    Raise a MuseScoreError if not found.

    """
    elem = element.find(path = path)
    if elem is None:
        raise MuseScoreError(f"Element `{path}` is required for an '{element.tag}' element.")
    return elem


def _get_required_attr(element: Element, attr: str) -> str:
    """Return a required attribute of an element.

    Raise a MuseScoreError if not found.

    """
    attribute = element.get(attr)
    if attribute is None:
        raise MuseScoreError(f"Attribute '{attr}' is required for an '{element.tag}' element.")
    return attribute


def _get_required_text(element: Element, path: str, remove_newlines: bool = False) -> str:
    """Return a required text from a child element of an element.

    Raise a MuseScoreError otherwise.

    """
    elem = _get_required(element = element, path = path)
    if elem.text is None:
        return ""
        # raise MuseScoreError(f"Text content '{path}' of an element '{element.tag}' must not be empty.")
    if remove_newlines:
        return " ".join(elem.text.splitlines())
    return elem.text


def _get_raw_text(element: Element, path: str = "text") -> str:
    """Returns the raw XML-level text in an element"""

    # find the element
    elem = element.find(path = path)

    # if that element is not found
    if elem is None:
        return None
    
    # wrangle the raw string
    text = str(ET.tostring(element = elem, encoding = "utf-8"), encoding = "UTF-8").strip()
    path = basename(path) if "/" in path else path
    text = sub(pattern = f"<{path}>|</{path}>|<font face=\"Edwin\" />", repl = "", string = text)
    return text

##################################################


# FOR GETTING THE XML, BASIC INFO
##################################################

def _get_root(path: str, compressed: bool = None):
    """Return root of the element tree."""
    if compressed is None:
        compressed = path.endswith(".mscz")

    if not compressed:
        tree = ET.parse(path)
        return tree.getroot()

    # Find out the main MSCX file in the compressed ZIP archive
    try:
        zip_file = ZipFile(file = path)
    except: # zipfile.BadZipFile: File is not a zip file
        raise MuseScoreError(f"{path} is not a zip file")
    if "META-INF/container.xml" not in zip_file.namelist():
        raise MuseScoreError("Container file ('container.xml') not found.")
    container = ET.fromstring(zip_file.read("META-INF/container.xml")) # read the container file as xml elementtree
    rootfile = container.findall(path = "rootfiles/rootfile") # find all branches in the container file, look for .mscx
    rootfile = tuple(file for file in rootfile if "mscx" in file.get("full-path"))
    if rootfile is None:
        raise MuseScoreError("Element 'rootfile' tag not found in the container file ('container.xml').")
    filename = _get_required_attr(element = rootfile[0], attr = "full-path")
    if filename in zip_file.namelist():
        root = ET.fromstring(zip_file.read(filename))
    else:
        try:
            root = ET.fromstring(zip_file.read(tuple(path for path in zip_file.namelist() if path.endswith(".mscx"))[0]))
        except (IndexError):
            raise MuseScoreError("No .mscx file could be found in .mscz.")
    return root


def _get_divisions(root: Element):
    """Return a list of divisions."""
    divisions = []
    for division in root.findall(path = "Division"):
        if division.text is None:
            continue
        if not float(division.text).is_integer():
            raise MuseScoreError(
                "Noninteger 'division' values are not supported."
            )
        divisions.append(int(division.text))
    return divisions

##################################################


# FOR FIGURING OUT MEASURE SEQUENCE
##################################################

def parse_repeats(elem: Element) -> Tuple[list, list]:
    """Return a list of dictionaries where the keys are startRepeats and the value are endRepeats parsed from a staff element."""
    # Initialize with a start marker
    start_repeats = [0]
    end_repeats = [[],]

    # Find all repeats in all measures
    for i, measure in enumerate(elem.findall(path = "Measure")):
        # check for startRepeat
        if measure.find("startRepeat") is not None:
            start_repeats.append(i)
            end_repeats.append([])
        # check for endRepeat
        if measure.find("endRepeat") is not None:
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
    for i, measure in enumerate(elem.findall(path = "Measure")):
        for marker in measure.findall(path = "Marker"):
            label = _get_text(element = marker, path = "label")
            if label is not None:
                markers[label] = i

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
    measures = elem.findall(path = "Measure")
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
        jump = measure.find(path = "Jump") # search for jump element
        if jump is not None and before_jump: # if jump is found
            # set jump related indicies
            jump_to_idx = markers.get(_get_text(element = jump, path = "jumpTo"))
            play_until_idx = markers.get(_get_text(element = jump, path = "playUntil"))
            continue_at_idx = markers.get(_get_text(element = jump, path = "continueAt"))
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
        volta = measure.find(path = "voice/Spanner/Volta/..") # select the parent element
        if volta is not None:
            if measure_idx not in (encounter[0] for encounter in voltas_encountered):
                volta_duration = int(_get_required_text(element = volta, path = "next/location/measures"))
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


def print_measure_indicies(measure_indicies: List[int]):
    """Print the measure indicies in a readable way (new line for every jump)"""

    # convert into measure numbers as opposed to indicies
    measure_indicies = [measure_idx + 1 for measure_idx in measure_indicies]
    
    # create empty output
    measure_indicies_formatted = ["",] * len(measure_indicies)
    for i in range(len(measure_indicies) - 1):
        if measure_indicies[i] + 1 == measure_indicies[i + 1]:
            measure_indicies_formatted[i] = f"{measure_indicies[i]} "
        else:
            measure_indicies_formatted[i] = f"{measure_indicies[i]}\n"
    measure_indicies_formatted[-1] = str(measure_indicies[-1])

    # print output
    print(*measure_indicies_formatted, sep = "", end = "\n")



def get_nice_measure_number(i: int) -> int:
    """Transform measure number into indicies not starting at 0."""
    return i + 1


def get_beats(downbeat_times: List[int], measure_indicies: List[int], time_signatures: List[TimeSignature], resolution: int = muspy.DEFAULT_RESOLUTION, is_sorted: bool = False) -> List[Beat]:
    """Return beats given downbeat positions and time signatures.

    Parameters
    ----------
    downbeat_times : sequence of int
        Positions of the downbeats.
    measure_indicies : sequence of int
        Measure indicies
    time_signatures : sequence of :class:`muspy.TimeSignature`
        Time signature objects.
    resolution : int, default: `muspy.DEFAULT_RESOLUTION` (24)
        Time steps per quarter note.
    is_sorted : bool, default: False
        Whether the downbeat times and time signatures are sorted.

    Returns
    -------
    list of :class:`read_mscz.Beat`
        Computed beats.

    """
    # convert measure indicies into a better format
    measure_indicies = list(map(get_nice_measure_number, measure_indicies))

    # Return a list of downbeats if no time signatures is given
    if not time_signatures:
        return [Beat(time = int(round(time)), measure = measure_idx, is_downbeat = True) for time, measure_idx in zip(downbeat_times, measure_indicies)]

    # Sort the downbeats and time signatures if necessary
    if not is_sorted:
        downbeat_times = sorted(downbeat_times)
        time_signatures = sorted(time_signatures, key = attrgetter("time"))

    # Compute the beats
    beats: List[Beat] = []
    sign_idx = 0
    downbeat_idx = 0
    while downbeat_idx < len(downbeat_times):
        # Select the correct time signatures
        if sign_idx + 1 < len(time_signatures) and downbeat_times[downbeat_idx] < time_signatures[sign_idx + 1].time:
            sign_idx += 1
            continue

        # Set time signature
        time_sign = time_signatures[sign_idx]
        beat_resolution = resolution / (time_sign.denominator / 4)

        # Get the next downbeat
        if downbeat_idx < len(downbeat_times) - 1:
            end: float = downbeat_times[downbeat_idx + 1]
        else:
            end = downbeat_times[downbeat_idx] + (beat_resolution * time_sign.numerator)

        # Append the downbeat
        start = int(round(downbeat_times[downbeat_idx]))
        beats.append(Beat(time = start, measure = measure_indicies[downbeat_idx], is_downbeat = True))

        # Append beats
        beat_times = np.arange(start = start + beat_resolution, stop = end, step = beat_resolution)
        for time in beat_times:
            beats.append(Beat(time = int(round(time)), measure = measure_indicies[downbeat_idx], is_downbeat = False))

        downbeat_idx += 1

    return beats

##################################################


# PARSE METADATA
##################################################

def parse_metadata(root: Element) -> Metadata:
    """Return a Metadata object parsed from a MuseScore file."""
    # creators and copyrights
    title, subtitle = None, None
    creators = []
    copyrights = []

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

    return Metadata(title = title, subtitle = subtitle, creators = creators, copyright = " ".join(copyrights) if copyrights else None, source_format = "musescore")


def parse_part_info(elem: Element, musescore_version: int) -> Tuple[Optional[List[str]], OrderedDict]:
    """Return part information parsed from a score part element."""
    part_info: OrderedDict = OrderedDict()

    # Staff IDs
    staffs = elem.findall(path = "Staff")
    if musescore_version >= 2:
        staff_ids = [_get_required_attr(element = staff, attr = "id") for staff in staffs] 
    else: # MuseScore 1
        staff_ids = list(range(1, len(staffs) + 1))  # MuseScore 1.x

    # Instrument
    instrument = _get_required(element = elem, path = "Instrument")
    part_info["id"] = _get_text(element = instrument, path = "instrumentId", remove_newlines = True)
    part_info["name"] = _get_text(element = instrument, path = "trackName", remove_newlines = True)

    # MIDI program and channel
    program = instrument.find(path = "Channel/program")
    if program is not None:
        program = program.get("value")
        part_info["program"] = int(program) if program is not None else 0
    else:
        part_info["program"] = 0
    part_info["is_drum"] = (int(_get_text(instrument, "Channel/midiChannel", 0)) == 10)

    return staff_ids, part_info


def get_part_staff_info(elem: Element, musescore_version: int) -> Tuple[List[OrderedDict], OrderedDict]:
    """Return part information and the mappings between staff and parts from a list of all the parts elements."""

    # initialize return collections
    parts_info: List[OrderedDict] = [] # for storing info on each part
    staff_part_map: OrderedDict = OrderedDict() # for connecting staff ids (keys) to the part they belong to (values)

    # iterate through the parts
    for part_id, part in enumerate(elem):

        # get part info
        staff_ids, current_part_info = parse_part_info(elem = part, musescore_version = musescore_version) # get the part info
        parts_info.append(current_part_info) # add the current part info to parts_info

        # Deal with quirks of MuseScore 1
        if musescore_version < 2:
            current_max_staff_id = max((int(staff_id) for staff_id in staff_part_map.keys())) if len(staff_part_map.keys()) > 0 else 0 # get the current largest staff id
            staff_ids = tuple(str(staff_id + current_max_staff_id) for staff_id in staff_ids) # adjust staff ids to total scale, not just within each part (essentially, convert MuseScore 1's lack of staff ids within parts to Musescore>1)
        
        # assign each staff id to a part
        for staff_id in staff_ids:
            staff_part_map[staff_id] = part_id
    
    # return a value or raise an error
    if not parts_info: # Raise an error if there is no Part information
        raise MuseScoreError("Part information is missing (i.e. there are no parts).")
    else:
        return parts_info, staff_part_map


def get_musescore_version(path: str, compressed: bool = None) -> str:
    """Determine the version of a MuseScore file.

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    compressed : bool, optional
        Whether it is a compressed MuseScore file. Defaults to infer from the filename.

    Returns
    -------
    :str:
        Version of Musescore
    """

    # get element tree root
    root = _get_root(path = path, compressed = compressed)

    # detect MuseScore version
    musescore_version = root.get("version")

    return musescore_version

##################################################


# PARSING VARIOUS ELEMENT TYPES IN XML
##################################################

def _get_text(element: Element, path: str, default: T = None, remove_newlines: bool = False) -> Union[str, T]:
    """Return the text of the first matching element."""
    elem = element.find(path = path)
    if elem is not None and elem.text is not None:
        if remove_newlines:
            return " ".join(elem.text.splitlines())
        return elem.text
    return default  # type: ignore


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
    beats = _get_text(elem, "sigN")
    if beats is None:
        beats = _get_text(elem, "nom1")
    if beats is None:
        raise MuseScoreError("Neither 'sigN' nor 'nom1' element is found for a TimeSig element.")
    if "+" in beats:
        numerator = sum(int(beat) for beat in beats.split("+"))
    else:
        numerator = int(beats)

    # Denominator
    beat_type = _get_text(elem, "sigD")
    if beat_type is None:
        beat_type = _get_text(elem, "den")
    if beat_type is None:
        raise MuseScoreError("Neither 'sigD' nor 'den' element is found for a TimeSig element.")
    if "+" in beat_type:
        raise RuntimeError("Compound time signatures with separate fractions are not supported.")
    denominator = int(beat_type)

    return numerator, denominator


def parse_key(elem: Element) -> Tuple[int, str, int, str]:
    """Return the key parsed from a key element."""

    mode = _get_text(element = elem, path = "mode")
    fifths_text = _get_text(element = elem, path = "accidental")  # MuseScore 2.x and 3.x
    if fifths_text is None:
        fifths_text = _get_text(element = elem, path = "subtype")  # MuseScore 1.x
        if fifths_text is None:
            fifths_text = _get_text(element = elem, path = "concertKey") # last 
    if fifths_text is None:
        raise MuseScoreError("'accidental', 'subtype', or 'concertKey' subelement not found for KeySig element.")
    fifths = int(fifths_text)
    if mode is None:
        return None, None, fifths, None
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

##################################################


# HELPER FUNCTIONS FOR PARSING
##################################################

def get_spanner_duration(spanner: Element, resolution: int, default_measure_len: int) -> int:
    """Returns the duration (in universal time) of a spanner element."""

    # get the duration (in measures) of the spanner
    gradual_tempo_change_duration = float(_get_required_text(element = spanner, path = "next/location/measures")) # the duration of the spanner
    fractions = _get_text(element = spanner, path = "next/location/fractions")
    if fractions is not None:
        gradual_tempo_change_duration += float(Fraction(fractions))

    # convert spanner to int
    gradual_tempo_change_duration = float(gradual_tempo_change_duration) if gradual_tempo_change_duration else 0.0

    if gradual_tempo_change_duration > 0 :
        gradual_tempo_change_duration = int(resolution * 4 * gradual_tempo_change_duration)
    else:
        gradual_tempo_change_duration = default_measure_len

    return gradual_tempo_change_duration

##################################################


# PARSE A STAFF FOR MUSICAL INFORMATION CONSTANT ACROSS ALL PARTS
##################################################

def parse_constant_features(staff: Element, resolution: int, measure_indicies: List[int], timeout: int = None) -> Tuple[List[Tempo], List[KeySignature], List[TimeSignature], List[Barline], List[Beat], List[Annotation]]:
    """Return data parsed from a meta staff element.

    This function only parses the tempos, key and time signatures. Use
    `parse_staff` to parse the notes and lyrics.

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
    downbeat_times: List[int] = []

    # record start time to check for timeout
    if timeout is not None:
        start_time = time.time()

    # Iterate over all elements
    measures = staff.findall(path = "Measure")
    for measure_idx in measure_indicies:

        # Check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # Get the measure element
        measure = measures[measure_idx]

        # Barlines, check for special types
        barline_subtype = _get_text(element = measure, path = "voice/BarLine/subtype")
        barline = Barline(time = time_, measure = get_nice_measure_number(i = measure_idx), subtype = barline_subtype)
        barlines.append(barline)

        # Collect the measure start times
        downbeat_times.append(time_)

        # Get measure duration
        measure_len_text = measure.get("len")
        if measure_len_text is not None:
            measure_len = round(resolution * 4 * Fraction(measure_len_text))

        # initialize position
        position = 0

        # get voice elements
        voices = list(measure.findall(path = "voice"))  # MuseScore 3.x, 4.x
        if not voices:
            voices = [measure]  # MuseScore 1.x and 2.x

        # Iterate over voice elements
        for voice in voices:

            # Reset position
            position = 0

            # Iterate over child elements
            for elem in voice:

                # Key signatures
                if elem.tag == "KeySig":
                    root, mode, fifths, root_str = parse_key(elem = elem)
                    key_signatures.append(KeySignature(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), root = root, mode = mode, fifths = fifths, root_str = root_str))

                # Time signatures
                if elem.tag == "TimeSig":
                    numerator, denominator = parse_time(elem = elem)
                    time_signatures.append(TimeSignature(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), numerator = numerator, denominator = denominator))

                # Tempo elements
                if elem.tag == "Tempo":
                    tempo_qpm = 60 * float(_get_required_text(element = elem, path = "tempo"))
                    tempo_text = _get_raw_text(element = elem)
                    tempos.append(Tempo(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), qpm = tempo_qpm, text = tempo_text))

                # Tempo spanner elements
                if elem.tag == "Spanner" and elem.get("type") == "GradualTempoChange" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TempoSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), subtype = _get_raw_text(element = elem, path = "GradualTempoChange/tempoChangeType"))))
                    
                # Text spanner elements (system)
                if elem.tag == "Spanner" and elem.get("type") == "TextLine" and elem.find(path = "next/location/measures") is not None:
                    text_line_is_system = "system" in elem.find(path = "TextLine").attrib.keys()
                    if text_line_is_system: # only append the text spanner if it is system text
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TextSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), text = _get_required_text(element = elem, path = "TextLine/beginText"), is_system = text_line_is_system)))
                
                # System Text
                if elem.tag == "SystemText": # save staff text for other function
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Text(text = _get_raw_text(element = elem, path = "text"), is_system = True, style = _get_raw_text(element = elem, path = "style"))))

                # Rehearsal Mark
                if elem.tag == "RehearsalMark":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = RehearsalMark(text = _get_raw_text(element = elem))))

                # Tuplet elements
                if elem.tag == "Tuplet":
                    is_tuple = True
                    normal_notes = int(_get_required_text(element = elem, path = "normalNotes"))
                    actual_notes = int(_get_required_text(element = elem, path = "actualNotes"))
                    tuple_ratio = normal_notes / actual_notes

                # Rest elements
                if elem.tag == "Rest":
                    # Move time position forward if it is a rest
                    duration_type = _get_required_text(element = elem, path = "durationType")
                    if duration_type == "measure":
                        duration_text = _get_text(element = elem, path = "duration")
                        if duration_text is not None:
                            duration = (resolution * 4 * float(Fraction(duration_text)))
                        else:
                            duration = measure_len
                        position += round(duration)
                        continue
                    duration = NOTE_TYPE_MAP[duration_type] * resolution
                    position += round(duration)
                    continue

                # Chord elements
                if elem.tag == "Chord":
                    # Compute duration
                    duration_type = _get_required_text(element = elem, path = "durationType")
                    duration = NOTE_TYPE_MAP[duration_type] * resolution

                    # Handle tuplets
                    if is_tuple:
                        duration *= tuple_ratio

                    # Handle dots
                    dots = elem.find(path = "dots")
                    if dots is not None and dots.text:
                        duration *= 2 - 0.5 ** int(dots.text)

                    # Round the duration
                    duration = round(duration)

                    # Grace notes
                    is_grace = False
                    for child in elem:
                        if "grace" in child.tag or child.tag in ("appoggiatura", "acciaccatura"):
                            is_grace = True

                    if not is_grace:
                        position += duration

                # Handle last tuplet note
                if elem.tag == "endTuplet":
                    old_duration = round(NOTE_TYPE_MAP[duration_type] * resolution)
                    new_duration = normal_notes * old_duration - (actual_notes - 1) * round(old_duration * tuple_ratio)
                    if duration != new_duration:
                        position += int(new_duration - duration)
                    is_tuple = False

        time_ += position

    # Sort tempos, key and time signatures
    tempos.sort(key = attrgetter("time"))
    key_signatures.sort(key = attrgetter("time"))
    time_signatures.sort(key = attrgetter("time"))
    annotations.sort(key = attrgetter("time"))

    # Get the beats
    beats = get_beats(downbeat_times = downbeat_times, measure_indicies = measure_indicies, time_signatures = time_signatures, resolution = resolution, is_sorted = True)

    return tempos, key_signatures, time_signatures, barlines, beats, annotations

##################################################


# PARSE A STAFF FOR GENERAL MUSICAL INFORMATION
##################################################

def parse_staff(staff: Element, resolution: int, measure_indicies: List[int], timeout: int = None) -> Tuple[List[Note], List[Lyric], List[Annotation]]:
    """Return notes and lyrics parsed from a staff element.

    This function only parses the notes and lyrics. Use
    `parse_constant_features` to parse the tempos, key and time
    signatures.

    """
    # Initialize lists
    notes: List[Note] = []
    lyrics: List[Lyric] = []
    annotations: List[Annotation] = []

    # Initialize variables
    time_ = 0
    velocity = 64
    measure_len = round(resolution * 4)
    is_tuple = False

    # Record start time to check for timeout
    if timeout is not None:
        start_time = time.time()

    # Create a dictionary to handle ties
    ties: Dict[int, int] = {}

    # Iterate over all elements
    measures = staff.findall(path = "Measure")
    for measure_idx in measure_indicies:

        # Check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # Get the measure element
        measure = measures[measure_idx]

        # Get measure duration
        measure_len_text = measure.get("len")
        if measure_len_text is not None:
            measure_len = round(resolution * 4 * Fraction(measure_len_text))

        # Initialize position
        position = 0

        # Get voice elements
        voices = list(measure.findall(path = "voice"))  # MuseScore 3.x
        if not voices:
            voices = [measure]  # MuseScore 1.x and 2.x

        # Iterate over voice elements
        for voice in voices:
            # Initialize position
            position = 0

            # Iterate over child elements
            for elem in voice:

                # Dynamic elements
                if elem.tag == "Dynamic":
                    velocity = int(round(float(_get_text(element = elem, path = "velocity", default = velocity))))
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Dynamic(subtype = _get_required_text(element = elem, path = "subtype"), velocity = velocity)))

                # Hairpin elements
                if elem.tag == "Spanner" and elem.get("type") == "HairPin" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = HairPinSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), subtype = _get_text(element = elem, path = "HairPin/beginText"), hairpin_type = int(_get_required_text(element = elem, path = "HairPin/subtype")))))

                # Text spanner elements (staff)
                if elem.tag == "Spanner" and elem.get("type") == "TextLine" and elem.find(path = "next/location/measures") is not None:
                    text_line_is_system = "system" in elem.find(path = "TextLine").attrib.keys()
                    if not text_line_is_system: # only append the text spanner if it is staff text
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TextSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), text = _get_raw_text(element = elem, path = "TextLine/beginText"), is_system = text_line_is_system)))
                
                # Staff Text
                if elem.tag == "StaffText":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Text(text = _get_raw_text(element = elem), is_system = False, style = _get_raw_text(element = elem, path = "style"))))

                # Pedals
                if elem.tag == "Spanner" and elem.get("type") == "Pedal" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = PedalSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len))))

                # Trill Spanners
                if elem.tag == "Spanner" and elem.get("type") == "Trill" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TrillSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), subtype = _get_text(element = elem, path = "Trill/subtype"), ornament = _get_text(element = elem, path = "Trill/Ornament/subtype"))))

                # Vibrato Spanners
                if elem.tag == "Spanner" and elem.get("type") == "Vibrato" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = VibratoSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), subtype = _get_text(element = elem, path = "Vibrato/subtype"))))

                # Glissando Spanners
                if elem.tag == "Spanner" and elem.get("type") == "Glissando" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = GlissandoSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), is_wavy = bool(_get_text(element = elem, path = "Trill/subtype")))))

                # Ottava Spanners
                if elem.tag == "Spanner" and elem.get("type") == "Ottava" and elem.find(path = "next/location/measures") is not None:
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = OttavaSpanner(duration = get_spanner_duration(spanner = elem, resolution = resolution, default_measure_len = measure_len), subtype = _get_text(element = elem, path = "Ottava/subtype"))))

                # Fermatas
                if elem.tag == "Fermata":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Fermata(is_fermata_above = (_get_text(element = elem, path = "subtype") == "fermataAbove"))))

                # Technical Annotation
                if elem.tag == "PlayTechAnnotation":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TechAnnotation(text = _get_raw_text(element = elem), tech_type = _get_raw_text(element = elem, path = "playTechType"), is_system = False)))
                
                # Tremolo Bar
                if elem.tag == "TremoloBar":
                    annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = TremoloBar(points = [Point(time = int(point.get("time")), pitch = int(point.get("pitch")), vibrato = int(point.get("vibrato"))) for point in elem.findall("point")])))

                # Tuplet elements
                if elem.tag == "Tuplet":
                    is_tuple = True
                    normal_notes = int(_get_required_text(element = elem, path = "normalNotes"))
                    actual_notes = int(_get_required_text(element = elem, path = "actualNotes"))
                    tuple_ratio = normal_notes / actual_notes

                # Rest elements
                if elem.tag == "Rest":
                    # Move time position forward if it is a rest
                    duration_type = _get_required_text(element = elem, path = "durationType")
                    if duration_type == "measure":
                        duration_text = _get_text(element = elem, path = "duration")
                        if duration_text is not None:
                            duration = (resolution * 4 * float(Fraction(duration_text)))
                        else:
                            duration = measure_len
                        position += round(duration)
                        continue
                    duration = NOTE_TYPE_MAP[duration_type] * resolution
                    position += round(duration)
                    continue

                # Chord elements
                if elem.tag == "Chord":
                    # Compute duration
                    duration_type = _get_required_text(element = elem, path = "durationType")
                    duration = NOTE_TYPE_MAP[duration_type] * resolution

                    # Handle tuplets
                    if is_tuple:
                        duration *= tuple_ratio

                    # Handle dots
                    dots = elem.find(path = "dots")
                    if dots is not None and dots.text:
                        duration *= 2 - 0.5 ** int(dots.text)

                    # Round the duration
                    duration = round(duration)

                    # Grace notes
                    is_grace = False
                    for child in elem:
                        if "grace" in child.tag or child.tag in ("appoggiatura", "acciaccatura"):
                            is_grace = True

                    # check for slurs and ties
                    is_outgoing_tie = False
                    for spanner in elem.findall(path = "Spanner"):
                        # Check if it is a tied chord
                        if spanner.get("type") == "Tie" and spanner.find(path = "next/location/measures") is not None:
                            is_outgoing_tie = True
                    
                        # Check for any slurs
                        if spanner.get("type") == "Slur" and spanner.find(path = "next/location/measures") is not None:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = SlurSpanner(duration = get_spanner_duration(spanner = spanner, resolution = resolution, default_measure_len = measure_len), is_slur = True)))

                    # Check for ornament
                    if elem.find(path = "Ornament") is not None:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Ornament(subtype = _get_text(element = elem, path = "Ornament/subtype"))))

                    # Check for arpeggio
                    if elem.find(path = "Arpeggio") is not None:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Arpeggio(subtype = int(_get_text(element = elem, path = "Arpeggio/subtype")))))

                    # Check for tremolo
                    if elem.find(path = "Tremolo") is not None:
                        annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Tremolo(subtype = _get_text(element = elem, path = "Tremolo/subtype"))))

                    # Check for articulation
                    if elem.find(path = "Articulation") is not None:
                        for articulation_subtype in elem.findall(path = "Articulation/subtype"): # in the case of multiple articulations
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Articulation(subtype = articulation_subtype.text)))

                    # Lyrics
                    lyric = elem.find(path = "Lyrics")
                    if lyric is not None:
                        lyric_text = parse_lyric(elem = lyric)
                        lyrics.append(Lyric(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), lyric = lyric_text))

                    # Collect notes
                    for note in elem.findall(path = "Note"):

                        # Get pitch
                        pitch = int(_get_required_text(element = note, path = "pitch"))
                        pitch_str = TONAL_PITCH_CLASSES[int(_get_required_text(element = note, path = "tpc"))]

                        # Check for bend
                        if note.find(path = "Bend") is not None:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Bend(points = [Point(time = int(point.get("time")), pitch = int(point.get("pitch")), vibrato = int(point.get("vibrato"))) for point in note.findall("Bend/point")])))

                        # Check for ChordLines (falls, doits, scoops, etc.)
                        if note.find(path = "ChordLine") is not None:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = ChordLine(subtype = _get_required_text(element = note, path = "ChordLine/subtype"), is_straight = bool(_get_text(element = note, path = "ChordLine/straight")))))

                        # get notehead
                        if note.find(path = "head") is not None:
                            annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Notehead(subtype = _get_required_text(element = note, path = "head"))))
                        
                        # get symbol(s)
                        if note.find(path = "Symbol/name") is not None:
                            for symbol_name in note.findall(path = "Symbol/name"): # in the case of multiple articulations
                                annotations.append(Annotation(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), annotation = Symbol(subtype = symbol_name.text)))

                        # Handle grace note
                        if is_grace:
                            # Append a new note to the note list
                            notes.append(Note(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), pitch = pitch, duration = duration, velocity = velocity, pitch_str = pitch_str, is_grace = True))
                            continue
                        
                        # Check if it is a tied note
                        for spanner in note.findall(path = "Spanner"):
                            if spanner.get("type") == "Tie" and spanner.find(path = "next/location/measures") is not None:
                                is_outgoing_tie = True

                        # Check if it is an incoming tied note
                        if pitch in ties:
                            note_idx = ties[pitch]
                            notes[note_idx].duration += duration

                            if is_outgoing_tie:
                                ties[pitch] = note_idx
                            else:
                                del ties[pitch]

                        else:
                            # Append a new note to the note list
                            notes.append(Note(time = time_ + position, measure = get_nice_measure_number(i = measure_idx), pitch = pitch, duration = duration, velocity = velocity, pitch_str = pitch_str))

                        if is_outgoing_tie:
                            ties[pitch] = len(notes) - 1

                    if not is_grace:
                        position += duration

                # Handle last tuplet note
                if elem.tag == "endTuplet":
                    old_duration = round(NOTE_TYPE_MAP[duration_type] * resolution)
                    new_duration = normal_notes * old_duration - (actual_notes - 1) * round(old_duration * tuple_ratio)
                    if notes[-1].duration != new_duration:
                        notes[-1].duration = new_duration
                        position += int(new_duration - duration)
                    is_tuple = False

        time_ += position

    # Sort notes
    notes.sort(key = attrgetter("time", "pitch", "duration", "velocity"))

    # Sort lyrics
    lyrics.sort(key = attrgetter("time"))

    # Sort annotations
    annotations.sort(key = attrgetter("time"))

    return notes, lyrics, annotations

##################################################


# MY BETTER READ MUSESCORE FUNCTION, EXTRACTS EXPRESSIVE FEATURES
##################################################

def read_musescore(path: str, resolution: int = None, compressed: bool = None, timeout: int = None) -> BetterMusic:
    """Read the a MuseScore file into a BetterMusic object, paying close attention to details such as articulation and expressive features.

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    resolution : int, optional
        Time steps per quarter note. Defaults to the least common
        multiple of all divisions.
    compressed : bool, optional
        Whether it is a compressed MuseScore file. Defaults to infer
        from the filename.

    Returns
    -------
    :class:`BetterMusic`
        Converted BetterMusic object.

    """

    # get element tree for MuseScore file
    root = _get_root(path = path, compressed = compressed)

    # detect MuseScore version
    musescore_version = int(root.get("version")[0]) # the file format differs slightly between musescore 1 and the rest

    # get the score element
    if musescore_version >= 2:
        score = root.find(path = "Score")
    else: # MuseScore 1
        score = root # No "Score" tree

    # metadata
    metadata = parse_metadata(root = root)
    metadata.source_filename = basename(path)

    # find the resolution
    if resolution is None:
        divisions = _get_divisions(root = score)
        resolution = max(divisions, key = divisions.count) # _lcm(*divisions) if divisions else 1

    # staff/part information
    parts = score.findall(path = "Part") # get all the parts
    parts_info, staff_part_map = get_part_staff_info(elem = parts, musescore_version = musescore_version)

    # get all the staff elements
    staffs = score.findall(path = "Staff")
    if len(staffs) == 0: # Return empty music object with metadata if no staff is found
        return BetterMusic(metadata = metadata, resolution = resolution)

    # parse measure ordering from the meta staff, expanding all repeats and jumps
    measure_indicies = get_measure_ordering(elem = staffs[0], timeout = timeout) # feed in the first staff, since measure ordering are constant across all staffs

    # parse the  part element
    (tempos, key_signatures, time_signatures, barlines, beats, annotations) = parse_constant_features(staff = staffs[0], resolution = resolution, measure_indicies = measure_indicies, timeout = timeout) # feed in the first staff to extract features constant across all parts

    # initialize lists
    tracks: List[Track] = []

    # record start time to check for timeout
    start_time = time.time()

    # iterate over all staffs
    part_track_map: Dict[int, int] = {} # keeps track of parts we have already looked at
    for staff in staffs:

        # check for timeout
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Abort the process as it runned over {timeout} seconds.")

        # get the staff ID
        staff_id = staff.get("id")
        if staff_id is None:
            if len(score.findall(path = "Staff")) > 1:
                continue
            staff_id = next(iter(staff_part_map))
        if staff_id not in staff_part_map:
            continue

        # parse the staff
        notes, lyrics, annotations_staff = parse_staff(staff = staff, resolution = resolution, measure_indicies = measure_indicies, timeout = timeout)

        # extend lists
        part_id = staff_part_map[staff_id]
        if part_id in part_track_map:
            track_id = part_track_map[part_id]
            tracks[track_id].notes.extend(notes)
            tracks[track_id].lyrics.extend(lyrics)
            tracks[track_id].annotations.extend(annotations_staff)
        else:
            part_track_map[part_id] = len(tracks)
            tracks.append(Track(program = parts_info[part_id]["program"], is_drum = parts_info[part_id]["is_drum"], name = parts_info[part_id]["name"], notes = notes, lyrics = lyrics, annotations = annotations_staff))

    # make sure everything is sorted
    tempos.sort(key = attrgetter("time"))
    key_signatures.sort(key = attrgetter("time"))
    time_signatures.sort(key = attrgetter("time"))
    annotations.sort(key = attrgetter("time"))
    for track in tracks:
        track.notes.sort(key = attrgetter("time", "pitch", "duration", "velocity"))
        track.lyrics.sort(key = attrgetter("time"))
        track.annotations.sort(key = attrgetter("time"))

    return BetterMusic(metadata = metadata, resolution = resolution, tempos = tempos, key_signatures = key_signatures, time_signatures = time_signatures, barlines = barlines, beats = beats, tracks = tracks, annotations = annotations)

##################################################


# TEST
##################################################

if __name__ == "__main__":

    paths = ["/data2/pnlong/musescore/data/chopin/Chopin_Trois_Valses_Op64.mscz", "/data2/pnlong/musescore/data/goodman/in_the_mood.mscz", "/data2/pnlong/musescore/data/laufey/from_the_start.mscz", "/data2/pnlong/musescore/data/maroon/this_love.mscz", "/data2/pnlong/musescore/data/test1/QmbboPyM7KRorFpbmqnoCYDjbN2Up2mY969kggS3JLqRCF.mscz", "/data2/pnlong/musescore/data/test2/QmbbxbpgJHyNRzjkbyxdoV5saQ9HY38MauKMd5CijTPFiF.mscz", "/data2/pnlong/musescore/data/test3/QmbbjJJAMixkH5vqVeffBS1h2tJHQG1DXpTJHJonpxGmSN.mscz", "/data2/pnlong/musescore/data/toploader/dancing_in_the_moonlight.mscz"]

    for path in paths:
        mscz = read_musescore(path = path)
        mscz.print(output_filepath = join(dirname(path), "mscx.yml"))

##################################################
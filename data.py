# README
# Phillip Long
# November 1, 2023

# Make dataset of musescore files w/ expressive features.

# python /home/pnlong/model_musescore/data.py


# IMPORTS
##################################################

import argparse
import pandas as pd
import numpy as np
from os import makedirs
from os.path import exists
from tqdm import tqdm
from typing import List
import logging
from time import perf_counter, strftime, gmtime
import multiprocessing
import random
from re import sub

import representation
from representation import DIMENSIONS
from read_mscz.read_mscz import read_musescore, get_musescore_version
from read_mscz.classes import *
from read_mscz.music import BetterMusic

##################################################


# CONSTANTS
##################################################

OUTPUT_DIR = "/data2/pnlong/musescore"
MSCZ_FILEPATHS = f"{OUTPUT_DIR}/relevant_mscz_files.txt"
OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n_expressive_features")
NA_STRING = "NA"
EXPRESSIVE_FEATURE_TYPE_STRING = "expressive-feature"

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Data", description = "Extract Notes and Expressive Features from MuseScore Data.")
    parser.add_argument("-p", "--paths", type = str, default = MSCZ_FILEPATHS, help = "List of (absolute) filepaths to MuseScore files whose data will be extracted")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# HELPER FUNCTIONS
##################################################

# implementation of R's rep function
def rep(x: object, times: int, flatten: bool = False):
    l = [x] * times
    if flatten:
        l = sum(l, [])
    return l

# make sure text is ok
def check_text(text: str):
    if text is not None:
        return sub(pattern = ": ", repl = ":", string = sub(pattern = ", ", repl = ",", string = " ".join(text.split()))).strip()
    return None

# convert camel case to words
def split_camel_case(string: str, sep: str = "-"):
    splitter = "_"
    if string is not None:
        string = [*string] # convert string to list of characters
        currently_in_digit = False # boolean flag for dealing with numbers
        for i, character in enumerate(string):
            if not character.isdigit() and currently_in_digit: # update whether we are inside of digit
                currently_in_digit = False
            if character.isupper():
                string[i] = splitter + character
            elif character.isdigit() and not currently_in_digit:
                string[i] = splitter + character
                currently_in_digit = True
        words = "".join(string).split(splitter) # convert to list of words
        words = filter(lambda word: word != "", words) # filter out empty words
        return sep.join(words).lower() # join into one string
    return None

# clean up text objects
def clean_up_text(text: str):
    if text is not None:
        text = sub(pattern = "-", repl = " ", string = split_camel_case(string = text)) # get rid of camel case, deal with long spins of dashes
        text = sub(pattern = " ", repl = "-", string = check_text(text = text)) # replace any whitespace with dashes
        text = sub(pattern = "[^\w-]", repl = "", string = text) # extract alphanumeric
        return text.lower() # convert to lower case
    return None

# create a csv row
def create_csv_row(info: list, sep: str = ",") -> str:
    return sep.join((str(item) if item != None else NA_STRING for item in info)) + "\n"

# write a list to a file
def write_to_file(info: dict, output_filepath: str, columns: list = None):
        
        # if there are provided columns
        if columns is not None:
            info = {column: info[column] for column in columns} # reorder columns if possible
            if not exists(output_filepath): # write columns if they are not there yet
                with open(output_filepath, "w") as output:
                    output.write(create_csv_row(info = columns))

        # open connection to file
        with open(output_filepath, "a") as output:
            # write info
            output.write(create_csv_row(info = list(info.values())))

##################################################


# OBJECTS TO SCRAPE
##################################################

# ['Metadata', 'Tempo', 'KeySignature', 'TimeSignature', 'Beat', 'Barline', 'Lyric', 'Annotation', 'Note', 'Chord', 'Track', 'Text', 'Subtype', 'RehearsalMark', 'TechAnnotation', 'Dynamic', 'Fermata', 'Arpeggio', 'Tremolo', 'ChordLine', 'Ornament', 'Articulation', 'Notehead', 'Symbol', 'Point', 'Bend', 'TremoloBar', 'Spanner', 'SubtypeSpanner', 'TempoSpanner', 'TextSpanner', 'HairPinSpanner', 'SlurSpanner', 'PedalSpanner', 'TrillSpanner', 'VibratoSpanner', 'GlissandoSpanner', 'OttavaSpanner']

# Explicit objects to scrape:
# - Notes >
# - Grace Notes (type field) >
# - Barlines >
# - Time Signatures >
# - Key Signatures >
# - Tempo, TempoSpanner >
# - Text, TextSpanner >
# - RehearsalMark >
# - Dynamic >
# - HairPinSpanner >
# - Fermata >
# - TechAnnotation >
# - Symbol >

# Implicit objects to scrape:
# - Articulation (Total # in some time; 4 horizontally in a row) -- we are looking for articulation chunks! >
# - SlurSpanner, higher the tempo, the longer the slur needs to be >
# - PedalSpanner, higher the tempo, the longer the slur needs to be >

# Punting on:
# - Notehead
# - Arpeggio
# - Ornament
# - Ottava
# - Bend
# - TrillSpanner
# - VibratoSpanner
# - GlissandoSpanner
# - Tremolo, TremoloBar
# - Vertical Density
# - Horizontal Density
# - Any Drum Tracks

##################################################


# SCRAPE EXPLICIT FEATURES
##################################################

desired_expressive_feature_types = ("Text", "TextSpanner", "RehearsalMark", "Dynamic", "HairPinSpanner", "Fermata", "TempoSpanner", "TechAnnotation", "Symbol")
def scrape_annotations(annotations: List[Annotation], song_length: int) -> pd.DataFrame:
    """Scrape annotations. song_length is the length of the song (in time steps)."""

    annotations_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    encounters = dict(zip(desired_expressive_feature_types, rep(x = None, times = len(desired_expressive_feature_types)))) # to track durations

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
        else: # deal with implied duration (time until next of same type)
            if encounters[expressive_feature_type] is not None: # to deal with the first encounter
                annotations_encoded["duration"][encounters[expressive_feature_type]] = annotation.time - annotations_encoded["time"][encounters[expressive_feature_type]]
            encounters[expressive_feature_type] = len(annotations_encoded["duration"]) # update encounter index
            duration = None # append None for current duration, will be fixed later           
        annotations_encoded["duration"].append(duration) # add a duration value if there is one
        
        # deal with value field
        value = None
        if "text" in annotation_attributes:
            value = clean_up_text(text = annotation.annotation.text)
        elif "subtype" in annotation_attributes:
            value = split_camel_case(string = annotation.annotation.subtype)
        if (value is None) or (value == ""): # if there is no text or subtype value, make the value the expressive feature type (e.g. "TempoSpanner")
            value = split_camel_case(string = sub(pattern = "Spanner", repl = "", string = expressive_feature_type))
        # deal with special cases
        if value in ("dynamic", "other-dynamics"):
            value = "dynamic-marking"
        elif expressive_feature_type == "Fermata":
            value = "fermata"
        elif expressive_feature_type == "RehearsalMark" and value.isdigit():
            value = "rehearsal-mark"
        annotations_encoded["value"].append(check_text(text = value))
    
    # get final durations
    for expressive_feature_type in tuple(encounters.keys()):
        if encounters[expressive_feature_type] is not None:
            annotations_encoded["duration"][encounters[expressive_feature_type]] = song_length - annotations_encoded["time"][encounters[expressive_feature_type]]
    
    # make sure untouched columns get filled
    for dimension in filter(lambda dimension: len(annotations_encoded[dimension]) == 0, tuple(annotations_encoded.keys())):
        annotations_encoded[dimension] = rep(x = None, times = len(annotations_encoded["type"]))

    # create dataframe from scraped values
    return pd.DataFrame(data = annotations_encoded, columns = DIMENSIONS)


def scrape_barlines(barlines: List[Barline], song_length: int) -> pd.DataFrame:
    """Scrape barlines. song_length is the length of the song (in time steps)."""
    barlines = list(filter(lambda barline: not ((barline.subtype == "single") or ("repeat" in barline.subtype.lower())), barlines)) # filter out single barlines
    barlines_encoded = {key: rep(x = None, times = len(barlines)) for key in DIMENSIONS} # create dictionary of lists
    barlines.append(Barline(time = song_length, measure = 0)) # for duration
    for i, barline in enumerate(barlines[:-1]):
        barlines_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        barlines_encoded["value"][i] = check_text(text = (f"{barline.subtype.lower()}-" if barline.subtype is not None else "") + "barline")
        barlines_encoded["duration"][i] = barlines[i + 1].time - barline.time
        barlines_encoded["time"][i] = barline.time
    return pd.DataFrame(data = barlines_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_timesigs(timesigs: List[TimeSignature], song_length: int) -> pd.DataFrame:
    """Scrape timesigs. song_length is the length of the song (in time steps)."""
    timesigs = timesigs[1:] # get rid of first timesig, since we are tracking changes in timesig
    timesigs_encoded = {key: rep(x = None, times = len(timesigs)) for key in DIMENSIONS} # create dictionary of lists
    timesigs.append(TimeSignature(time = song_length, measure = 0, numerator = 4, denominator = 4)) # for duration
    for i, timesig in enumerate(timesigs[:-1]):
        timesigs_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        timesigs_encoded["value"][i] = check_text(text = f"timesig-change") # check_text(text = f"{timesig.numerator}/{timesig.denominator}")
        timesigs_encoded["duration"][i] = timesigs[i + 1].time - timesig.time
        timesigs_encoded["time"][i] = timesig.time
    return pd.DataFrame(data = timesigs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_keysigs(keysigs: List[KeySignature], song_length: int) -> pd.DataFrame:
    """Scrape keysigs. song_length is the length of the song (in time steps)."""
    keysigs = keysigs[1:] # get rid of first keysig, since we are tracking changes in keysig
    keysigs_encoded = {key: rep(x = None, times = len(keysigs)) for key in DIMENSIONS} # create dictionary of lists
    keysigs.append(KeySignature(time = song_length, measure = 0)) # for duration
    for i, keysig in enumerate(keysigs[:-1]):
        keysigs_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        keysigs_encoded["value"][i] = check_text(text = f"keysig-change") # check_text(text = f"{keysig.root_str} {keysig.mode}") # or keysig.root or keysig.fifths
        keysigs_encoded["duration"][i] = keysigs[i + 1].time - keysig.time
        keysigs_encoded["time"][i] = keysig.time
    return pd.DataFrame(data = keysigs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_tempos(tempos: List[Tempo], song_length: int) -> pd.DataFrame:
    """Scrape tempos. song_length is the length of the song (in time steps)."""
    tempos_encoded = {key: rep(x = None, times = len(tempos)) for key in DIMENSIONS} # create dictionary of lists
    tempos.append(Tempo(time = song_length, measure = 0, qpm = 0.0)) # for duration
    for i, tempo in enumerate(tempos[:-1]):
        tempos_encoded["type"][i] = EXPRESSIVE_FEATURE_TYPE_STRING
        tempos_encoded["value"][i] = check_text(text = representation.QPM_TEMPO_MAPPER(qpm = tempo.qpm)) # check_text(text = tempo.text.lower() if tempo.text is not None else "tempo-marking")
        tempos_encoded["duration"][i] = tempos[i + 1].time - tempo.time
        tempos_encoded["time"][i] = tempo.time
    return pd.DataFrame(data = tempos_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_notes(notes: List[Note]) -> pd.DataFrame:
    """Scrape notes (and grace notes)."""
    notes_encoded = {key: rep(x = None, times = len(notes)) for key in DIMENSIONS} # create dictionary of lists
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
                    articulations_encoded["value"].append(split_camel_case(string = check_text(text = articulation_subtype if articulation_subtype is not None else "articulation")))
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
    check_all_subtypes_for_chunk_ending(time = annotations[-1].time + (2 * maximum_gap)) # one final check
    for dimension in filter(lambda dimension: len(articulations_encoded[dimension]) == 0, tuple(articulations_encoded.keys())): # make sure untouched columns get filled
        articulations_encoded[dimension] = rep(x = None, times = len(articulations_encoded["type"]))
    return pd.DataFrame(data = articulations_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_slurs(annotations: List[Annotation], minimum_duration: float, mscz: BetterMusic) -> pd.DataFrame:
    """Scrape slurs. minimum_duration is the minimum duration (in seconds) a slur needs to be to make it worthwhile recording."""
    slurs_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    for annotation in annotations:
        if annotation.annotation.__class__.__name__ == "SlurSpanner":
            if annotation.annotation.is_slur:
                start = mscz.metrical_time_to_absolute_time(time_steps = annotation.time)
                duration = mscz.metrical_time_to_absolute_time(time_steps = annotation.time + annotation.annotation.duration) - start
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
        slurs_encoded[dimension] = rep(x = None, times = len(slurs_encoded["type"]))
    return pd.DataFrame(data = slurs_encoded, columns = DIMENSIONS) # create dataframe from scraped values


def scrape_pedals(annotations: List[Annotation], minimum_duration: float, mscz: BetterMusic) -> pd.DataFrame:
    """Scrape pedals. minimum_duration is the minimum duration (in seconds) a pedal needs to be to make it worthwhile recording."""
    pedals_encoded = {key: [] for key in DIMENSIONS} # create dictionary of lists
    for annotation in annotations:
        if annotation.annotation.__class__.__name__ == "PedalSpanner":
            start = mscz.metrical_time_to_absolute_time(time_steps = annotation.time)
            duration = mscz.metrical_time_to_absolute_time(time_steps = annotation.time + annotation.annotation.duration) - start
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
        pedals_encoded[dimension] = rep(x = None, times = len(pedals_encoded["type"]))
    return pd.DataFrame(data = pedals_encoded, columns = DIMENSIONS) # create dataframe from scraped values

##################################################


# WRAPPER FUNCTIONS MAKE CODE EASIER TO READ
##################################################

def get_system_level_expressive_features(mscz: BetterMusic) -> pd.DataFrame:
    """Wrapper function to make code more readable. Extracts system-level expressive features."""
    song_length = mscz.get_song_length() # get the song length
    system_annotations = scrape_annotations(annotations = mscz.annotations, song_length = song_length)
    system_barlines = scrape_barlines(barlines = mscz.barlines, song_length = song_length)
    system_timesigs = scrape_timesigs(timesigs = mscz.time_signatures, song_length = song_length)
    system_keysigs = scrape_keysigs(keysigs = mscz.key_signatures, song_length = song_length)
    system_tempos = scrape_tempos(tempos = mscz.tempos, song_length = song_length)
    return pd.concat(objs = (system_annotations, system_barlines, system_timesigs, system_keysigs, system_tempos), axis = 0, ignore_index = True)

def get_staff_level_expressive_features(mscz: BetterMusic, track: Track) -> pd.DataFrame:
    """Wrapper function to make code more readable. Extracts staff-level expressive features."""
    staff_notes = scrape_notes(notes = track.notes)
    staff_annotations = scrape_annotations(annotations = track.annotations, song_length = mscz.get_song_length())
    staff_articulations = scrape_articulations(annotations = track.annotations, maximum_gap = 2 * mscz.resolution) # 2 beats = 2 * mscz.resolution
    staff_slurs = scrape_slurs(annotations = track.annotations, minimum_duration = 1.5, mscz = mscz) # minimum duration for slurs to be recorded is 1.5 seconds
    staff_pedals = scrape_pedals(annotations = track.annotations, minimum_duration = 1.5, mscz = mscz) # minimum duration for pedals to be recorded is 1.5 seconds
    return pd.concat(objs = (staff_notes, staff_annotations, staff_articulations, staff_slurs, staff_pedals), axis = 0, ignore_index = True)

##################################################


# EXTRACTION FUNCTION (EXTRACT RELEVANT DATA FROM A GIVEN MUSESCORE FILE
##################################################

def extract(path: str, path_output_prefix: str):
    """Extract relevant information from a .mscz file, output as tokens

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    path_output_prefix : str
        Prefix to path where tokenized information will be outputted

    Returns
    -------
    :void:
    """
    
    # LOAD IN MSCZ FILE, CONSTANTS
    ##################################################

    # finish output dictionary
    try:
        metadata_path = METADATA[path]
    except KeyError:
        metadata_path = None
    try:
        version = get_musescore_version(path = path)
    except:
        version = None

    # try to read musescore
    try:
        mscz = read_musescore(path = path, timeout = 10)
    except: # if that fails
        return # exit here

    # start timer
    start_time = perf_counter()

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################
        
    # scrape system level expressive features
    system_level_expressive_features = get_system_level_expressive_features(mscz = mscz)

    for i, track in enumerate(mscz.tracks):

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in representation.KNOWN_PROGRAMS:
            continue

        # scrape staff-level features
        staff_level_expressive_features = get_staff_level_expressive_features(mscz = mscz, track = track)

        # create dataframe, do some wrangling to semi-encode values
        data = pd.concat(objs = (pd.DataFrame(columns = DIMENSIONS), system_level_expressive_features, staff_level_expressive_features), axis = 0, ignore_index = True) # combine system and staff expressive features
        data["instrument"] = rep(x = track.program, times = len(data)) # add the instrument column
        # timings
        data["time.s"] = data["time"].apply(lambda time_steps: mscz.metrical_time_to_absolute_time(time_steps = time_steps)) # get time in seconds
        data = data.sort_values(by = "time").reset_index(drop = True) # sort by time
        data["time"] = data["time"] - data.at[0, "time"] # set start beat to 0
        data["time.s"] = data["time.s"] - data.at[0, "time.s"] # set start beat to 0
        data["beat"] = data["time"].apply(lambda time_steps: time_steps // mscz.resolution) # add beat
        data["position"] = data["time"].apply(lambda time_steps: time_steps % mscz.resolution) # add position

        # don't save low-quality data
        if len(data) < 50:
            continue

        # convert to np array so that we can save as npy file, which loads faster
        data = np.array(object = data)

        # create output path from path_output_prefix
        path_output = f"{path_output_prefix}.{i}.npy"

        # save encoded data
        np.save(file = path_output, arr = data)

        # create current output dictionary; OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n_expressive_features")
        current_output = {
            "path" : path_output,
            "musescore" : path,
            "track" : i,
            "metadata" : metadata_path,
            "version" : version,
            "n_expressive_features" : len(data)
        }

        # write mapping
        write_to_file(info = current_output, output_filepath = MAPPING_OUTPUT_FILEPATH, columns = OUTPUT_COLUMNS)

        ##################################################

    
    # END STATS
    ##################################################

    end_time = perf_counter()
    total_time = end_time - start_time

    write_to_file(info = {"time": total_time}, output_filepath = TIMING_OUTPUT_FILEPATH)

    ##################################################


##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse arguments
    args = parse_args()
    if not exists(args.output_dir): # make output_dir if it doesn't yet exist
        makedirs(args.output_dir)

    # some constants
    METADATA_MAPPING_FILEPATH = f"{args.output_dir}/metadata_to_data.csv"
    TIMING_OUTPUT_FILEPATH = f"{args.output_dir}/data_timing.txt"
    MAPPING_OUTPUT_FILEPATH = f"{args.output_dir}/data.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = METADATA_MAPPING_FILEPATH, sep = ",", header = 0, index_col = False)
    METADATA = {data : (metadata if not pd.isna(metadata) else None) for data, metadata in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # set up logging
    logging.basicConfig(level = logging.INFO)

    ##################################################


    # GET PATHS
    ##################################################

    # create list of paths if does not exist
    if not exists(args.paths):
        from extract_expressive_features import MAPPING_OUTPUT_FILEPATH # import path to data frame
        data = pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False) # load in data frame
        data = data[data["is_valid"] & data["is_public_domain"] & (data["n_expressive_features"] > 0) & (data["expressive_features"].apply(lambda expressive_features_path: exists(str(expressive_features_path))))] # filter
        paths = pd.unique(values = data["path"]).to_list()
        with open(args.paths, "w") as file:
            file.write("\n".join(paths))
        del data, paths

    # load in paths
    with open(args.paths) as file:
        paths = [path.strip() for path in file.readlines()]

    # see if I've already completed some path
    if exists(MAPPING_OUTPUT_FILEPATH):
        completed_paths = set(pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["path"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining Already-Completed Paths") if path not in completed_paths)
        paths = tuple(random.sample(paths, len(paths)))

    # get prefix for output pickle files
    def get_path_output_prefixes(path: str) -> str:
        path = "/".join(path.split("/")[-3:]) # get the base name
        return f"{args.output_dir}/{path.split('.')[0]}"
    path_output_prefixes = tuple(map(get_path_output_prefixes, paths))

    ##################################################

    # USE MULTIPROCESSING
    ##################################################

    chunk_size = 1
    start_time = perf_counter() # start the timer
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = pool.starmap(func = extract,
                                iterable = tqdm(iterable = zip(paths, path_output_prefixes), desc = "Extracting Data from MuseScore Files", total = len(paths)),
                                chunksize = chunk_size)
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    logging.info(f"Total time: {total_time}")

    ##################################################


##################################################
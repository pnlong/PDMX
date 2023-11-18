# README
# Phillip Long
# September 24, 2023

# parse through all musescore (.mscz) files and extract expressive features to determine if we have quality data

# python /home/pnlong/model_musescore/parse_mscz.py


# IMPORTS
##################################################
import glob
from os.path import isfile, exists
import random
import pandas as pd
from tqdm import tqdm
from time import perf_counter, strftime, gmtime
import multiprocessing
import argparse
import logging
from typing import List
import json
import pickle
from read_mscz.read_mscz import read_musescore, get_musescore_version
from read_mscz.classes import *
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore/expressive_features"
FILE_OUTPUT_DIR = "/data2/pnlong/musescore"
EXPRESSIVE_FEATURE_COLUMNS = ("time", "measure", "duration", "type", "feature", "comment")
OUTPUT_COLUMNS = ("path", "track", "expressive_features", "metadata", "version", "is_public_domain", "is_valid", "n_expressive_features")
NA_STRING = "NA"
N_EXPRESSIVE_FEATURES_TO_STORE_THRESHOLD = 2
##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description = "Extract expressive features from MuseScore files")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-o", "--file_output_dir", type = str, default = FILE_OUTPUT_DIR, help = "Directory to output any data tables")
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
    if text:
        return " ".join(text.split()).replace(", ", ",").replace(": ", ":").strip()
    else:
        return None

# initialize empty lists
def initialize_empty_lists(length: int) -> tuple:
    # create empty lists
    times = rep(x = None, times = length)
    measures = rep(x = None, times = length)
    types = rep(x = None, times = length)
    durations = rep(x = None, times = length)
    features = rep(x = None, times = length)
    comments = rep(x = None, times = length)
    return (times, measures, types, durations, features, comments)

# create a csv row
def create_csv_row(info: list, sep: str = ",") -> str:
    return sep.join((str(item) if item != None else NA_STRING for item in info)) + "\n"

# write a list to a file
def write_to_file(info: list, output_filepath: str, columns: list = None):
        
        # write columns if the file is new
        if not exists(output_filepath) and columns is not None: # write columns if they are not there yet
            with open(output_filepath, "w") as output:
                output.write(create_csv_row(info = columns))

        # open connection to file
        with open(output_filepath, "a") as output:
            # write info
            output.write(create_csv_row(info = info))

##################################################


# SCRAPE OBJECTS FROM A LIST OF OBJECTS
##################################################

def scrape_annotations(annotations: List[Annotation], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of annotations, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(annotations))

    # loop
    for i, annotation in enumerate(annotations):

        # get timings
        times[i] = int(annotation.time)
        measures[i] = int(annotation.measure)

        # get the type of annotation
        types[i] = annotation.annotation.__class__.__name__

        # get annotations attributes
        annotation_attributes = vars(annotation.annotation).copy()

        # check for duration
        if "Spanner" in types[i]: # if spanner, add duration
            durations[i] = annotation_attributes["duration"]
            del annotation_attributes["duration"]

        # get what type of feature it is
        feature = None
        if "text" in annotation_attributes.keys(): # if a text element
            feature = "text"
        elif "subtype" in annotation_attributes.keys(): # if a subtype element
            feature = "subtype"
        if feature: # if there is a feature to add
            features[i] = check_text(text = str(annotation_attributes[feature]))
            del annotation_attributes[feature]
        else: # so there is no text or subtype
            features[i] = check_text(text = types[i].lower())

        # any extra info add to comments
        if len(annotation_attributes.keys()) > 0:
            for key in (key for key in annotation_attributes.keys() if hasattr(annotation_attributes[key], "__dict__")): # deal with nested objects
                annotation_attributes[key] = json.dumps(obj = vars(annotation_attributes[key]))
            for key in (key for key in annotation_attributes.keys() if type(annotation_attributes[key]) is list or type(annotation_attributes[key]) is tuple): # deal with list of nested objects
                for k in range(len(annotation_attributes[key])):
                    annotation_attributes[key][k] = json.dumps(obj = vars(annotation_attributes[key][k]))
            comments[i] = check_text(text = json.dumps(obj = annotation_attributes))
        
    # create dataframe from scraped values
    annotations = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return annotations


def scrape_barlines(barlines: List[Barline], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of barlines, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    barlines = list(filter(lambda barline: not ((barline.subtype == "single") or ("repeat" in barline.subtype.lower())), barlines)) # filter out single barlines
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(barlines))

    # loop
    for i, barline in enumerate(barlines):

        # get timings
        times[i] = int(barline.time)
        measures[i] = int(barline.measure)
        # barlines have no duration

        # update feature and subtype
        types[i] = "Barline"
        features[i] = barline.subtype
        
    # create dataframe from scraped values
    barlines = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return barlines


def scrape_timesigs(timesigs: List[TimeSignature], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of timesigs, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(timesigs))

    # loop
    for i, timesig in enumerate(timesigs):

        # get timings
        times[i] = int(timesig.time)
        measures[i] = int(timesig.measure)
        # timesigs have no duration

        # update feature and subtype
        types[i] = "TimeSignature"
        features[i] = f"{timesig.numerator}/{timesig.denominator}"
        
    # create dataframe from scraped values
    timesigs = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return timesigs


def scrape_keysigs(keysigs: List[KeySignature], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of keysigs, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(keysigs))

    # loop
    for i, keysig in enumerate(keysigs):

        # get timings
        times[i] = int(keysig.time)
        measures[i] = int(keysig.measure)
        # keysigs have no duration

        # (root, mode, fifths, root_str)
        # update feature and subtype
        types[i] = "KeySignature"
        features[i] = check_text(text = f"{keysig.root_str} {keysig.mode}")

        # comment with extra info
        comments[i] = check_text(text = json.dumps(obj = {key: vars(keysig)[key] for key in ("root", "fifths")}))
        
    # create dataframe from scraped values
    keysigs = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return keysigs


def scrape_tempos(tempos: List[Tempo], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of tempos, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(tempos))

    # loop
    for i, tempo in enumerate(tempos):

        # get timings
        times[i] = int(tempo.time)
        measures[i] = int(tempo.measure)
        # tempos have no duration

        # (root, mode, fifths, root_str)
        # update feature and subtype
        types[i] = "Tempo"
        features[i] = check_text(text = tempo.text)

        # comment with extra info
        comments[i] = check_text(text = json.dumps(obj = {key: vars(tempo)[key] for key in ("qpm",)}))
        
    # create dataframe from scraped values
    tempos = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return tempos


def scrape_notes(notes: List[Note], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of notes, create a dataframe with the scraped expressive tokens."""

    # get horizontal density
    ## IMPLEMENT ON A LATER DATE
    
    # create empty lists
    notes = list(filter(lambda note: note.is_grace, notes)) # filter out normal notes
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(notes))

    # loop
    for i, note in enumerate(notes):

        # set the note type and features
        types[i] = "GraceNote"

        # get timings
        times[i] = int(note.time)
        measures[i] = int(note.measure)
        durations[i] = int(note.duration)
        
    # create dataframe from scraped values
    notes = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return notes


def scrape_lyrics(lyrics: List[Lyric], out_columns: List[str] = EXPRESSIVE_FEATURE_COLUMNS) -> pd.DataFrame:
    """Given a list of lyrics, create a dataframe with the scraped expressive tokens."""
    
    # create empty lists
    lyrics = list(filter(lambda lyric: len("".join(lyric.lyric.split() if lyric.lyric is not None else "")) > 0, lyrics)) # filter out whitespace lyrics
    times, measures, types, durations, features, comments = initialize_empty_lists(length = len(lyrics))

    # loop
    for i, lyric in enumerate(lyrics):

        # check if this is a relevant lyric
        comments[i] = check_text(text = lyric.lyric)

        # set the lyric type and features
        types[i] = "Lyric"
        # I stored the lyric text inside of comment, not features

        # get timings
        times[i] = int(lyric.time)
        measures[i] = int(lyric.measure)
        # there is no duration for a lyric
        
    # create dataframe from scraped values
    lyrics = pd.DataFrame(data = dict(zip(out_columns, (times, measures, durations, types, features, comments))), columns = out_columns)
    return lyrics


##################################################


# FUNCTION TO EXTRACT EXPRESSIVE FEATURES
##################################################

def extract_expressive_features(path: str, path_output_prefix: str):
    """Extract expressive features from a .mscz file, output as tokens

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    path_output_prefix : str
        Prefix to path where tokenized expressive features will be outputted

    Returns
    -------
    :void:
    """

    # NOTES
    # when there is a subending in the song, an extra hidden 2 measures are added

    # NEED TO EXTRACT                                                               |   COMPLETED?
    # ------------------------------------------------------------------------------+----------------------
    # - all text (including dynamic markings)                                       |   Done
    # - long form dynamics                                                          |   Done
    # - slurs                                                                       |   Done
    # - extended periods of similar articulations / note level attributes           |   Done
    # - all fancy barlines                                                          |   Done
    # - abrupt changes in relative horizontal density                               |   Not Done
    # - abrupt changes in relative vertical/harmonic density                        |   Not Done
    #     - shelve this for now, as its really only for multitrack music


    # LOAD IN MSCZ FILE, CONSTANTS
    ##################################################

    # for debugging
    # path = "/data2/pnlong/musescore/data/chopin/Chopin_Trois_Valses_Op64.mscz"
    # path2 = "/data2/zachary/musescore/data/b/b/QmbbxbpgJHyNRzjkbyxdoV5saQ9HY38MauKMd5CijTPFiF.mscz"

    # finish output dictionary
    try:
        metadata_path = METADATA[path]
    except KeyError:
        metadata_path = None
    is_public_domain = False
    if metadata_path:
        try:
            with open(metadata_path, "r") as metadata_file:
                metadata_for_path = json.load(fp = metadata_file)
                is_public_domain = bool(metadata_for_path["data"]["is_public_domain"])
        except OSError:
            metadata_path = None
    try:
        version = get_musescore_version(path = path)
    except:
        version = None

    # try to read musescore
    try:
        mscz = read_musescore(path = path, timeout = 10)
    # if that fails
    except Exception as exc:
        # print most recent error, determine error type
        error_message = str(exc)
        if "zip" in error_message:
            error_type = "zip"
        elif "KeySig" in error_message:
            error_type = "keysig"
        elif "required for" in error_message:
            error_type = "required"
        elif "-2" in error_message:
            error_type = "-2"
        elif "int()" in error_message:
            error_type = "int"
        elif "max()" in error_message:
            error_type = "max"
        else:
            error_type = "other"
        # output error message to file
        write_to_file(info = (path, error_type, error_message.replace(",", "")), output_filepath = ERROR_MESSAGE_OUTPUT_FILEPATH, columns = ("path", "error_type", "error_message"))
        # write mapping
        write_to_file(info = (path, None, None, metadata_path, version, is_public_domain, False, None), output_filepath = MAPPING_OUTPUT_FILEPATH, columns = OUTPUT_COLUMNS)

        return None # exit here

    # start timer
    start_time = perf_counter()

    ##################################################


    # SCRAPE SYSTEM-LEVEL EXPRESSIVE FEATURES
    ##################################################

    system_annotations = scrape_annotations(annotations = mscz.annotations, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
    system_barlines = scrape_barlines(barlines = mscz.barlines, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
    system_timesigs = scrape_timesigs(timesigs = mscz.time_signatures, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
    system_keysigs = scrape_keysigs(keysigs = mscz.key_signatures, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
    system_tempos = scrape_tempos(tempos = mscz.tempos, out_columns = EXPRESSIVE_FEATURE_COLUMNS)

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################
    for i, track in enumerate(mscz.tracks):

        # scrape various annotations
        expressive_features = pd.DataFrame(columns = EXPRESSIVE_FEATURE_COLUMNS) # create dataframe
        staff_annotations = scrape_annotations(annotations = track.annotations, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
        staff_notes = scrape_notes(notes = track.notes, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
        staff_lyrics = scrape_lyrics(lyrics = track.lyrics, out_columns = EXPRESSIVE_FEATURE_COLUMNS)
        expressive_features = pd.concat(
            objs = (expressive_features, system_annotations, system_barlines, system_timesigs, system_keysigs, system_tempos, staff_annotations, staff_notes, staff_lyrics),
            axis = 0, ignore_index = True) # update current output
        expressive_features = expressive_features.sort_values("time") # sort by time
        seconds_column_name = "time_seconds"
        expressive_features[seconds_column_name] = expressive_features["time"].apply(lambda time_steps: mscz.metrical_time_to_absolute_time(time_steps = time_steps))
        expressive_feature_columns = list(EXPRESSIVE_FEATURE_COLUMNS)
        expressive_feature_columns.insert(expressive_feature_columns.index("time") + 1, seconds_column_name)
        expressive_features = expressive_features[expressive_feature_columns].reset_index(drop = True) # reorder columns, reset indicies

        current_output = {
            "path" : path,
            "metadata" : metadata_path,
            "version" : version,
            "is_public_domain" : is_public_domain,
            "track" : i,
            "program" : track.program,
            "is_drum" : bool(track.is_drum),
            "resolution" : mscz.resolution,
            "track_length" : {
                "time_steps": mscz.get_song_length(),
                "seconds": mscz.metrical_time_to_absolute_time(time_steps = mscz.get_song_length()),
                "bars": len(mscz.barlines),
                "beats": len(mscz.beats)},
            "n_annotations" : {
                "system": len(system_annotations) + len(system_barlines) + len(system_timesigs) + len(system_keysigs) + len(system_tempos),
                "staff": len(staff_annotations) + len(staff_notes) + len(staff_lyrics)},
            "expressive_features" : expressive_features
        }

        ##################################################

    
        # OUTPUT
        ##################################################

        if len(expressive_features) >= N_EXPRESSIVE_FEATURES_TO_STORE_THRESHOLD:

            # create output path from path_output_prefix
            path_output = f"{path_output_prefix}.{i}.pickle"

            # write output dictionary as pickle
            with open(path_output, "wb") as pickle_file:
                pickle.dump(obj = current_output, file = pickle_file, protocol = pickle.HIGHEST_PROTOCOL)
            
            # to reload
            # with open(path_output, "rb") as pickle_file:
            #     pickled = pickle.load(file = pickle_file)

        else: # if not enough expressive features to bother storing
            path_output = None # because we didn't write this file

        # write mapping
        write_to_file(info = (path, i, path_output, metadata_path, version, is_public_domain, True, len(expressive_features)), output_filepath = MAPPING_OUTPUT_FILEPATH, columns = OUTPUT_COLUMNS)

        ##################################################

    
    # END STATS
    ##################################################

    end_time = perf_counter()
    total_time = end_time - start_time

    write_to_file(info = (total_time,), output_filepath = TIMING_OUTPUT_FILEPATH)

    ##################################################


##################################################


# for multiprocessing, must use a main class
if __name__ == "__main__":

    # ARGS AND CONSTANTS
    ##################################################

    args = parse_args()

    # constant filepaths
    METADATA_MAPPING_FILEPATH = f"{args.file_output_dir}/metadata_to_data.csv"
    ERROR_MESSAGE_OUTPUT_FILEPATH = f"{args.file_output_dir}/read_mscz_errors.csv"
    TIMING_OUTPUT_FILEPATH = f"{args.file_output_dir}/read_mscz_timing.txt"
    MAPPING_OUTPUT_FILEPATH = f"{args.file_output_dir}/expressive_features.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = METADATA_MAPPING_FILEPATH, sep = ",", header = 0, index_col = False)
    METADATA = {data : (metadata if not pd.isna(metadata) else None) for data, metadata in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # set logging level
    logging.basicConfig(level = logging.INFO)

    ##################################################

    # GET LIST OF FILES
    ##################################################

    # use glob to get all mscz files
    paths = glob.iglob(pathname = f"/data2/zachary/musescore/data/**", recursive = True) # glob filepaths recursively, generating an iterator object
    paths = tuple(path for path in paths if isfile(path) and path.endswith("mscz")) # filter out non-file elements that were globbed
    if exists(MAPPING_OUTPUT_FILEPATH):
        completed_paths = set(pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["path"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining already-completed paths") if path not in completed_paths)
        paths = tuple(random.sample(paths, len(paths)))

    # get prefix for output pickle files
    def get_path_output_prefixes(path: str) -> str:
        path = "/".join(path.split("/")[-3:]) # get the base name
        return f"{args.output_dir}/{path.split('.')[0]}"
    path_output_prefixes = tuple(map(get_path_output_prefixes, paths))

    ##################################################


    # HELPER FUNCTION TO ALLOW PROGRESS BAR USAGE
    ##################################################

    # def extract_expressive_features_helper(arguments: list):
    #     extract_expressive_features(path = arguments[0], path_output_prefix = arguments[1])

    ##################################################


    # SCRAPE EXPRESSIVE FEATURES
    ##################################################

    # use multiprocessing
    logging.info(f"N_PATHS = {len(paths)}") # print number of paths to process
    chunk_size = 1
    start_time = perf_counter() # start the timer
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = pool.starmap(func = extract_expressive_features, iterable = tqdm(iterable = zip(paths, path_output_prefixes), desc = "Extracting Expressive Features from MuseScore Data", total = len(paths)), chunksize = chunk_size)
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    logging.info(f"Total time: {total_time}")

    ##################################################

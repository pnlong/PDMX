# README
# Phillip Long
# September 24, 2023

# parse through all musescore (.mscz) files and extract expressive features to determine if we have quality data

# python /home/pnlong/model_musescore/parse_mscz.py


# IMPORTS
##################################################

import glob
from os.path import isfile, exists, basename, dirname
import random
import subprocess
import pandas as pd
from typing import List
from tqdm import tqdm
from time import perf_counter, strftime, gmtime
import multiprocessing
import argparse
import logging
from copy import deepcopy
from re import sub
import json
import pickle
from read_mscz.read_mscz import read_musescore, get_musescore_version
from read_mscz.classes import Lyric
import representation
from encode import extract_data, get_system_level_expressive_features
from utils import write_to_file

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore"
METADATA_MAPPING = f"{INPUT_DIR}/metadata_to_data.csv"
OUTPUT_DIR = f"{INPUT_DIR}/expressive_features"
TIME_IN_SECONDS_COLUMN_NAME = "time.s"
EXPRESSIVE_FEATURE_COLUMNS = ["time", TIME_IN_SECONDS_COLUMN_NAME, "type", "value"]
OUTPUT_COLUMNS = ["path", "track", "expressive_features", "metadata", "version", "is_public_domain", "is_valid", "is_user_pro", "program", "complexity", "genres", "tags", "n_expressive_features", "n_expressive_features_with_lyrics", "n_tokens", "n_tokens_with_lyrics", "in_dataset"]
OUTPUT_COLUMNS_BY_PATH = ["path", "metadata", "version", "is_public_domain", "is_valid", "is_user_pro", "complexity", "genres", "tags", "n_tracks", "n_expressive_features", "n_expressive_features_with_lyrics", "n_tokens", "n_tokens_with_lyrics", "in_dataset"]
ERROR_COLUMNS = ["path", "error_type", "error_message"]
N_EXPRESSIVE_FEATURES_TO_STORE_THRESHOLD = 2
LIST_FEATURE_JOIN_STRING = "-"

PUBLIC_LICENSES = ("https://creativecommons.org/publicdomain/mark/1.0/", "https://creativecommons.org/publicdomain/zero/1.0/", "https://creativecommons.org/licenses/by/4.0/")

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore", description = "Extract expressive features from MuseScore files.")
    parser.add_argument("-m", "--metadata_mapping", type = str, default = METADATA_MAPPING, help = "Absolute filepath to metadata-to-data table")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# HELPER FUNCTION(S) FOR EXTRACTING EXPRESSIVE FEATURES NOT EXTRACTING IN encode.py
##################################################

def scrape_lyrics(lyrics: List[Lyric]) -> pd.DataFrame:
    lyrics_encoded = [(lyric.time, "Lyric", lyric.lyric) for lyric in lyrics]
    return pd.DataFrame(data = lyrics_encoded, columns = EXPRESSIVE_FEATURE_COLUMNS[:1] + EXPRESSIVE_FEATURE_COLUMNS[2:])

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
    # path = "/data2/pnlong/musescore/test_data/chopin/Chopin_Trois_Valses_Op64.mscz"
    # path2 = "/data2/zachary/musescore/test_data/b/b/QmbbxbpgJHyNRzjkbyxdoV5saQ9HY38MauKMd5CijTPFiF.mscz"

    # finish output dictionary
    try:
        metadata_path = METADATA[path]
    except KeyError:
        metadata_path = None
    is_public_domain, genres, complexity, tags, is_user_pro = False, [], None, [], False # set defaults
    if metadata_path:
        try:
            with open(metadata_path, "r") as metadata_file:
                metadata_for_path = json.load(fp = metadata_file)
                license_string = metadata_for_path["data"].get("license_string", "")
                is_public_domain = any(((license in license_string) for license in PUBLIC_LICENSES))
                # is_public_domain = bool(metadata_for_path["data"].get("is_public_domain", False))
                genres = list(map(lambda genre: str(genre["name"]), metadata_for_path["data"].get("genres", [])))
                complexity = int(metadata_for_path["data"].get("complexity", None))
                if "score" in metadata_for_path["data"].keys():
                    tags = list(map(str, metadata_for_path["data"]["score"].get("tags", [])))
                    if "user" in metadata_for_path["data"]["score"].keys():
                        is_user_pro = bool(metadata_for_path["data"]["score"]["user"].get("is_pro", False))
        except (OSError):
            metadata_path = None
    genres_string = LIST_FEATURE_JOIN_STRING.join(filter(lambda genre: len(genre) > 0, map(lambda genre: sub(pattern = "[^a-z]", repl = "", string = genre.lower()), genres))).strip() # store genres as a single string
    if len(genres_string) == 0:
        genres_string = None
    tags_string = LIST_FEATURE_JOIN_STRING.join(filter(lambda tag: len(tag) > 0, map(lambda tag: sub(pattern = "[^a-z]", repl = "", string = tag.lower()), tags))).strip() # store tags as a single string
    if (len(tags_string) == 0):
        tags_string = None
    try:
        version = get_musescore_version(path = path)
    except:
        version = None

    # try to read musescore
    try:
        music = read_musescore(path = path, timeout = 10)
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
        write_to_file(info = dict(zip(ERROR_COLUMNS, (path, error_type, error_message.replace(",", "")))), columns = ERROR_COLUMNS, output_filepath = ERROR_MESSAGE_OUTPUT_FILEPATH)
        # write mapping by track
        write_to_file(info = dict(zip(OUTPUT_COLUMNS, (path, None, None, metadata_path, version, is_public_domain, False, is_user_pro, None, complexity, genres_string, tags_string, None, None, None, None, False))), columns = OUTPUT_COLUMNS, output_filepath = OUTPUT_FILEPATH)
        # write mapping by path
        write_to_file(info = dict(zip(OUTPUT_COLUMNS_BY_PATH, (path, metadata_path, version, is_public_domain, False, is_user_pro, complexity, genres_string, tags_string, None, None, None, None, None, False))), columns = OUTPUT_COLUMNS_BY_PATH, output_filepath = OUTPUT_FILEPATH_BY_PATH)
        return None # exit here

    # start timer
    start_time = perf_counter()

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################
    
    system_lyrics = scrape_lyrics(lyrics = music.lyrics) # get system-level lyrics
    system_lyrics[EXPRESSIVE_FEATURE_COLUMNS[1]] = system_lyrics[EXPRESSIVE_FEATURE_COLUMNS[0]].apply(lambda time: music.metrical_time_to_absolute_time(time_steps = time)) # add time in seconds column
    system_lyrics = system_lyrics[EXPRESSIVE_FEATURE_COLUMNS] # reorder columns
    n_system_expressive_features = len(get_system_level_expressive_features(music = music, use_implied_duration = False, include_annotation_class_name = False)) # start with count of system-level expressive features
    n_system_expressive_features_with_lyrics = n_system_expressive_features + len(system_lyrics) # add lyrics to the count
    n_expressive_features_by_path, n_expressive_features_by_path_with_lyrics = n_system_expressive_features, n_system_expressive_features_with_lyrics
    n_tracks, n_notes_by_path = 0, 0
    for i, track in enumerate(music.tracks):

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in representation.KNOWN_PROGRAMS:
            continue
        else:
            n_tracks += 1
        
        # create MusicExpress object with just one track (we are not doing multitrack)
        track_music = deepcopy(x = music)
        track_music.tracks = [track,]

        # extract data
        data = extract_data(music = track_music, use_implied_duration = False, include_velocity = True, include_annotation_class_name = True) # duration doesn't matter for our purposes here, include velocity as to not mess up indexing of dimensions
        expressive_features = (data[:, 0] == representation.EXPRESSIVE_FEATURE_TYPE_STRING)
        n_notes = sum(~expressive_features)
        n_notes_by_path += n_notes
        data = data[expressive_features] # filter down to just expressive features
        data = pd.DataFrame(data = dict(zip(EXPRESSIVE_FEATURE_COLUMNS, [data[:, representation.DIMENSIONS.index("time")], data[:, representation.DIMENSIONS.index("time.s")], data[:, data.shape[1] - 1], data[:, representation.DIMENSIONS.index("value")]]))) # create pandas data frame
        n_expressive_features = len(data)
        n_expressive_features_by_path += (n_expressive_features - n_system_expressive_features) # update n_expressive_features_by_path
        staff_lyrics = scrape_lyrics(lyrics = track.lyrics) # get staff-level lyrics
        staff_lyrics[EXPRESSIVE_FEATURE_COLUMNS[1]] = staff_lyrics[EXPRESSIVE_FEATURE_COLUMNS[0]].apply(lambda time: music.metrical_time_to_absolute_time(time_steps = time)) # add time in seconds column
        staff_lyrics = staff_lyrics[EXPRESSIVE_FEATURE_COLUMNS] # reorder columns
        expressive_features = pd.concat(objs = (data, system_lyrics, staff_lyrics), axis = 0) # combine data and lyrics
        n_expressive_features_with_lyrics = len(expressive_features)
        n_expressive_features_by_path_with_lyrics += (n_expressive_features_with_lyrics - n_system_expressive_features_with_lyrics) # update n_expressive_features_by_path_with_lyrics

        # create current output (to be pickled)
        current_output = {
            "path" : path,
            "metadata" : metadata_path,
            "version" : version,
            "is_public_domain" : is_public_domain,
            "is_user_pro" : is_user_pro,
            "complexity" : complexity,
            "genres" : genres,
            "tags" : tags,
            "track" : i,
            "program" : track.program,
            "is_drum" : bool(track.is_drum),
            "resolution" : music.resolution,
            "track_length" : {
                "time_steps": music.song_length,
                "seconds": music.metrical_time_to_absolute_time(time_steps = music.song_length),
                "bars": len(music.barlines),
                "beats": len(music.beats)},
            "expressive_features" : expressive_features
        }

        ##################################################

    
        # OUTPUT
        ##################################################

        # output if it is worthwhile (at least some expressive features)
        if n_expressive_features_with_lyrics >= N_EXPRESSIVE_FEATURES_TO_STORE_THRESHOLD:

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
        write_to_file(info = dict(zip(OUTPUT_COLUMNS, (path, i, path_output, metadata_path, version, is_public_domain, True, is_user_pro, track.program, complexity, genres_string, tags_string, n_expressive_features, n_expressive_features_with_lyrics, n_expressive_features + n_notes, n_expressive_features_with_lyrics + n_notes, is_public_domain and (n_expressive_features > 0)))), columns = OUTPUT_COLUMNS, output_filepath = OUTPUT_FILEPATH)

        ##################################################

    
    # END STATS
    ##################################################

    # write to number expressive features per path
    write_to_file(info = dict(zip(OUTPUT_COLUMNS_BY_PATH, (path, metadata_path, version, is_public_domain, True, is_user_pro, complexity, genres_string, tags_string, n_tracks, n_expressive_features_by_path, n_expressive_features_by_path_with_lyrics, n_expressive_features_by_path + n_notes_by_path, n_expressive_features_by_path_with_lyrics + n_notes_by_path, is_public_domain and (n_expressive_features_by_path > 0)))), columns = OUTPUT_COLUMNS_BY_PATH, output_filepath = OUTPUT_FILEPATH_BY_PATH)

    # write to timing output
    end_time = perf_counter()
    total_time = end_time - start_time
    write_to_file(info = {"time": total_time}, output_filepath = TIMING_OUTPUT_FILEPATH)

    ##################################################


##################################################


# for multiprocessing, must use a main class
if __name__ == "__main__":

    # ARGS AND CONSTANTS
    ##################################################

    args = parse_args()
    if not exists(args.output_dir): # make output_dir if it doesn't yet exist
        subprocess.run(args = ["bash", f"{dirname(__file__)}/create_data_dir.sh", "-d", args.output_dir], check = True)

    # constant filepaths
    prefix = basename(args.output_dir)
    ERROR_MESSAGE_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.errors.csv"
    TIMING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.timing.txt"
    OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.csv"
    OUTPUT_FILEPATH_BY_PATH = f"{args.output_dir}/{prefix}.path.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = args.metadata_mapping, sep = ",", header = 0, index_col = False)
    METADATA = {data : (metadata if not pd.isna(metadata) else None) for data, metadata in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################

    # GET LIST OF FILES
    ##################################################

    # use glob to get all mscz files
    paths = glob.iglob(pathname = f"/data2/zachary/musescore/data/**", recursive = True) # glob filepaths recursively, generating an iterator object
    paths = tuple(path for path in paths if isfile(path) and path.endswith("mscz")) # filter out non-file elements that were globbed
    if exists(OUTPUT_FILEPATH):
        completed_paths = set(pd.read_csv(filepath_or_buffer = OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["path"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining already-completed paths") if path not in completed_paths)
        paths = tuple(random.sample(paths, len(paths)))

    # get prefix for output pickle files
    def get_path_output_prefixes(path: str) -> str:
        path = "/".join(path.split("/")[-3:]) # get the base name
        return f"{args.output_dir}/{path.split('.')[0]}"
    path_output_prefixes = tuple(map(get_path_output_prefixes, paths))

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
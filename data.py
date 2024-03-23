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
from os.path import exists, basename, dirname
from tqdm import tqdm
import logging
from time import perf_counter, strftime, gmtime
import multiprocessing
import random
from copy import copy
import subprocess

import utils
import representation
from encode import extract_data
from read_mscz.read_mscz import read_musescore, get_musescore_version
from read_mscz.classes import *

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore"
MSCZ_FILEPATHS = f"{INPUT_DIR}/relevant_mscz_files.txt"
OUTPUT_DIR = "/data2/pnlong/musescore/data"
OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n")

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Data", description = "Extract Notes and Expressive Features from MuseScore Data.")
    parser.add_argument("-p", "--paths", type = str, default = MSCZ_FILEPATHS, help = "List of (absolute) filepaths to MuseScore files whose data will be extracted")
    parser.add_argument("-i", "--input_dir", type = str, default = INPUT_DIR, help = "Directory containing all data tables needed as input")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-ed", "--explicit_duration", action = "store_true", help = "Whether or not to calculate the 'implied duration' of features without an explicitly-defined duration.")
    parser.add_argument("-v", "--velocity", action = "store_true", help = "Whether or not to include a velocity field that reflects expressive features.")
    parser.add_argument("-a", "--absolute_time", action = "store_true", help = "Whether or not to use absolute (seconds) or metrical (beats) time.")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# EXTRACTION FUNCTION (EXTRACT RELEVANT DATA FROM A GIVEN MUSESCORE FILE
##################################################

def extract(path: str, path_output_prefix: str, use_implied_duration: bool = True, include_velocity: bool = False, use_absolute_time: bool = False) -> tuple:
    """Extract relevant information from a .mscz file, output as tokens

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    path_output_prefix : str
        Prefix to path where tokenized information will be outputted

    Returns
    -------
    tuple: # of tracks processed, # of tokens processed
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
        music = read_musescore(path = path, timeout = 10)
        music.infer_velocity = True
        music.realize_expressive_features()
    except: # if that fails
        return # exit here

    # start timer
    start_time = perf_counter()

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################
    
    n_tokens = 0
    for i, track in enumerate(music.tracks):

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in representation.KNOWN_PROGRAMS:
            continue
        
        # create BetterMusic object with just one track (we are not doing multitrack)
        track_music = copy(x = music)
        track_music.tracks = [track,]
        data = extract_data(music = track_music, use_implied_duration = use_implied_duration, include_velocity = include_velocity, use_absolute_time = use_absolute_time)

        # create output path from path_output_prefix
        path_output = f"{path_output_prefix}.{i}.npy"

        # save encoded data
        np.save(file = path_output, arr = data)

        # create current output dictionary; OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n")
        current_output = {
            "path" : path_output,
            "musescore" : path,
            "track" : i,
            "metadata" : metadata_path,
            "version" : version,
            "n" : len(data)
        }

        # update n_tokens
        n_tokens += len(data)

        # write mapping
        utils.write_to_file(info = current_output, output_filepath = MAPPING_OUTPUT_FILEPATH, columns = OUTPUT_COLUMNS)

        ##################################################

    
    # END STATS
    ##################################################

    end_time = perf_counter()
    total_time = end_time - start_time

    utils.write_to_file(info = {"time": total_time}, output_filepath = TIMING_OUTPUT_FILEPATH)

    return len(music.tracks), n_tokens

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
        subprocess.run(args = ["bash", f"{dirname(__file__)}/create_data_dir.sh", "-d", args.output_dir], check = True)

    # some constants
    METADATA_MAPPING_FILEPATH = f"{args.input_dir}/metadata_to_data.csv"
    prefix = basename(args.output_dir)
    TIMING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.timing.txt"
    MAPPING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = METADATA_MAPPING_FILEPATH, sep = ",", header = 0, index_col = False)
    METADATA = {data_path : (metadata_path if not pd.isna(metadata_path) else None) for data_path, metadata_path in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # GET PATHS
    ##################################################

    # create list of paths if does not exist
    if not exists(args.paths):
        data = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/expressive_features/expressive_features.csv", sep = ",", header = 0, index_col = False) # load in data frame
        data = data[data["is_valid"] & data["is_public_domain"] & (data["n_expressive_features"] > 0)] # filter
        paths = pd.unique(values = data["path"]).tolist()
        with open(args.paths, "w") as file:
            file.write("\n".join(paths))
        del data, paths

    # load in paths
    with open(args.paths) as file:
        paths = [path.strip() for path in file.readlines()]
        # from random import sample
        # paths = sample(population = paths, k = int(0.1 * len(paths)))

    # see if I've already completed some path
    if exists(MAPPING_OUTPUT_FILEPATH):
        completed_paths = set(pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["musescore"].tolist())
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
                               iterable = tqdm(iterable = zip(
                                                              paths,
                                                              path_output_prefixes,
                                                              utils.rep(x = not bool(args.explicit_duration), times = len(paths)),
                                                              utils.rep(x = args.velocity, times = len(paths)),
                                                              utils.rep(x = args.absolute_time, times = len(paths))
                                                              ),
                                               desc = "Extracting Data from MuseScore Files", total = len(paths)),
                               chunksize = chunk_size)
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    logging.info(f"Total time: {total_time}")
    n_tracks, n_tokens = list(zip(*results))
    del results
    logging.info(f"Total Number of Tracks: {sum(n_tracks):,}")
    logging.info(f"Total Number of Tokens: {sum(n_tokens):,}")

    # make encoding file
    options = (["--velocity",] if args.velocity else []) + (["--absolute_time"] if args.absolute_time else [])
    subprocess.run(args = ["python", f"{dirname(__file__)}/representation.py", "--output_dir", args.output_dir] + options, check = True)

    # split into partitions
    subprocess.run(args = ["python", f"{dirname(__file__)}/split.py", "--input_filepath", MAPPING_OUTPUT_FILEPATH, "--output_dir", args.output_dir], check = True)

    ##################################################

##################################################
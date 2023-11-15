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
from extract_expressive_features import extract_expressive_features, MAPPING_OUTPUT_FILEPATH
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore/expressive_features"
##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description = "Extract expressive features from MuseScore files")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# for multiprocessing, must use a main class
if __name__ == "__main__":

    # ARGS
    ##################################################

    args = parse_args()

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

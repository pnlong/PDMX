# README
# Phillip Long
# August 11, 2024

# Create PDMX (Public Domain Music XML) Dataset.

# python /home/pnlong/model_musescore/pdmx.py

# IMPORTS
##################################################

import argparse
from os.path import exists, dirname
from os import makedirs, mkdir
from shutil import copyfile
import pandas as pd
from tqdm import tqdm
import multiprocessing
import logging

from make_dataset.full import DATASET_DIR_NAME, MUSESCORE_DIR, CHUNK_SIZE
from make_dataset.full import OUTPUT_DIR as DATASET_OUTPUT_DIR
from read_mscz.read_mscz import read_musescore
import utils

##################################################


# CONSTANTS
##################################################

# default output directory
OUTPUT_DIR = "/data1/pnlong/musescore"

# name of dataset
DATASET_NAME = "PDMX"

# whether to compress json music files
COMPRESS_JSON_MUSIC_FILES = False

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = DATASET_NAME, description = f"Create {DATASET_NAME} Dataset.")
    parser.add_argument("-d", "--dataset_filepath", type = str, default = f"{DATASET_OUTPUT_DIR}/{DATASET_DIR_NAME}.csv", help = "Filepath to full dataset.")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # ARGS AND CONSTANTS
    ##################################################

    # parse arguments
    args = parse_args()

    # filepaths
    output_dir = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(output_dir):
        makedirs(output_dir)
    output_filepath = f"{output_dir}/{DATASET_NAME}.csv"
    data_dir = f"{output_dir}/data"
    if not exists(data_dir):
        mkdir(data_dir)
    metadata_dir = f"{output_dir}/metadata"
    if not exists(metadata_dir):
        mkdir(metadata_dir)
    facets_dir = f"{output_dir}/subset_paths"
    if not exists(facets_dir):
        mkdir(facets_dir)

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    dataset["path_output"] = list(map(lambda path: data_dir + ".".join(path[len(f"{MUSESCORE_DIR}/data"):].split(".")[:-1]) + ".json", dataset["path"]))
    dataset["metadata_output"] = list(map(lambda path: metadata_dir + path[len(f"{MUSESCORE_DIR}/metadata"):] if (not pd.isna(path)) else None, dataset["metadata"]))

    # create necessary directory trees if required
    data_subdirectories = set(map(dirname, dataset["path_output"]))
    for data_subdirectory in data_subdirectories:
        makedirs(data_subdirectory, exist_ok = True)
    metadata_subdirectories = set(map(dirname, filter(lambda path: not pd.isna(path), dataset["metadata_output"])))
    for metadata_subdirectory in metadata_subdirectories:
        makedirs(metadata_subdirectory, exist_ok = True)
    del data_subdirectories, metadata_subdirectories # free up memory

    ##################################################


    # COPY OVER FILES
    ##################################################

    # helper function to save files
    def get_file(i: int) -> None:
        """
        Given the dataset index, copy over song as music object, and metadata if applicable.
        """

        # save as music object
        path_output = dataset.at[i, "path_output"]
        music = read_musescore(path = dataset.at[i, "path"], timeout = 10)
        music.save_json(path = path_output, compressed = COMPRESS_JSON_MUSIC_FILES) # save as music object
        dataset.at[i, "path"] = path_output # update path

        # copy over metadata path
        metadata_path = dataset.at[i, "metadata"]
        if metadata_path:
            metadata_path_new = dataset.at[i, "metadata_output"]
            copyfile(src = metadata_path, dst = metadata_path_new) # copy over metadata
            dataset.at[i, "metadata"] = metadata_path_new

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(func = get_file,
                                                     iterable = dataset.index,
                                                     chunksize = CHUNK_SIZE),
                            desc = f"Creating {DATASET_NAME}",
                            total = len(dataset)))
        
    # remove unnecessary columns
    dataset = dataset.drop(columns = ["path_output", "metadata_output"])
    facet_columns = list(filter(lambda column: column.startswith("facet:"), dataset.columns))
    subset_columns = list(map(lambda facet_column: facet_column.replace("facet", "subset"), facet_columns))
    dataset = dataset.rename(columns = dict(zip(facet_columns, subset_columns))) # rename facet to subset columns
    dataset.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # text files with paths for each facet
    for column in subset_columns:
        with open(f"{facets_dir}/{column.split(':')[-1]}.txt", "w") as output_file:
            output_file.write("\n".join(dataset[dataset[column]]["path"]))
    
    ##################################################

##################################################
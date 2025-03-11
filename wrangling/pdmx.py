# README
# Phillip Long
# August 11, 2024

# Create PDMX (Public Domain Music XML) Dataset.

# python /home/pnlong/model_musescore/wrangling/pdmx.py

# IMPORTS
##################################################

import argparse
from os.path import exists, dirname, basename
from os import makedirs, mkdir, environ
from shutil import copy, rmtree
import subprocess
import pandas as pd
from tqdm import tqdm
import multiprocessing
import tempfile
import json
from typing import Tuple

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import DATASET_DIR_NAME, MUSESCORE_DIR, CHUNK_SIZE
from full import OUTPUT_DIR as DATASET_OUTPUT_DIR
from reading.read_musescore import read_musescore
import utils

# environ["QT_QPA_PLATFORM"] = "offscreen" # for musescore 4 cli
# environ["QT_QPA_PLATFORM"] = "xcb" # for musescore 3 cli
# environ["DISPLAY"] = ":1" # for musescore 3 cli

##################################################


# CONSTANTS
##################################################

# default output directory
OUTPUT_DIR = "/data1/pnlong/musescore"

# path to MuseScore CLI application
MUSESCORE_APPLICATION_PATH = "/data3/pnlong/MuseScore-3.6.2.548021370-x86_64.AppImage" # MuseScore Command Line Options: https://musescore.org/en/handbook/3/command-line-options

# name of dataset
DATASET_NAME = "PDMX"

# the metadata says a song has a public domain license, but the internals of the MuseScore files say elsewise
LICENSE_DISCREPANCY_COLUMN_NAME = "license_conflict"

# whether to compress json music files
COMPRESS_JSON_MUSIC_FILES = False

# timeout for reading musescore files
TIMEOUT = 120

# mapping directory names from musescore scrape to more simple numbers
MUSESCORE_PARENT_SUBDIRECTORIES_MAP = {parent_subdirectory: i for i, parent_subdirectory in enumerate("abcdefNPQRSTUVWXYZ")}
MUSESCORE_CHILD_SUBDIRECTORIES_MAP = {child_subdirectory: i for i, child_subdirectory in enumerate("123456789aAbBcCdDeEfFgGhHijJkKLmMnNopPqQrRsStTuUvVwWxXyYzZ")}

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = DATASET_NAME, description = f"Create {DATASET_NAME} Dataset.")
    parser.add_argument("-df", "--dataset_filepath", type = str, default = f"{DATASET_OUTPUT_DIR}/{DATASET_DIR_NAME}.csv", help = "Filepath to full dataset")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-mf", "--musescore_filepath", type = str, default = MUSESCORE_APPLICATION_PATH, help = "Filepath to MuseScore CLI application (for generating extra files)")
    parser.add_argument("-g", "--gzip", action = "store_true", help = "GZIP the output directory of the dataset")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-rj", "--reset_json", action = "store_true", help = "Whether or not to recreate just MusicRender JSON files")
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
    if not exists(output_dir) or args.reset:
        makedirs(output_dir)
    output_filepath = f"{output_dir}/{DATASET_NAME}.csv"
    def directory_creator(directory: str):
        if not exists(directory) or args.reset:
            if exists(directory):
                rmtree(directory, ignore_errors = True)
            mkdir(directory)
    data_dir = f"{output_dir}/data"
    directory_creator(directory = data_dir)
    metadata_dir = f"{output_dir}/metadata"
    directory_creator(directory = metadata_dir)
    facets_dir = f"{output_dir}/subset_paths"
    directory_creator(directory = facets_dir)
    output_formats = ["mxl", "pdf", "mid"]
    output_format_dirs = [f"{output_dir}/{output_format}" for output_format in output_formats]
    for output_format_dir in output_format_dirs:
        directory_creator(directory = output_format_dir)

    # load in dataset
    output_format_output_columns = [f"{output_format}_output" for output_format in output_formats]
    output_columns = ["path_output", "metadata_output"] + output_format_output_columns
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    def get_path_output(path: str) -> str: # helper function to get the output path
        """Helper function to determine the output path given the input path."""
        path_output = data_dir + ".".join(path[len(f"{MUSESCORE_DIR}/data"):].split(".")[:-1]) + ".json"
        path_output = path_output.split("/")
        path_output[-3] = str(MUSESCORE_PARENT_SUBDIRECTORIES_MAP[path_output[-3]]) # replace zach's scraped parent subdirectory with an index
        path_output[-2] = str(MUSESCORE_CHILD_SUBDIRECTORIES_MAP[path_output[-2]]) # replace zach's scraped child subdirectory with an index
        path_output = "/".join(path_output)
        return path_output
    dataset["path_output"] = list(map(get_path_output, dataset["path"]))
    dataset["metadata_output"] = list(map(lambda path: metadata_dir + path[len(f"{MUSESCORE_DIR}/metadata"):] if (not pd.isna(path)) else None, dataset["metadata"]))
    for output_format_output_column, output_format in zip(output_format_output_columns, output_formats):
        dataset[output_format_output_column] = list(map(lambda path: f"{output_dir}/{output_format}/{path[(len(data_dir) + 1):-len('.json')]}.{output_format}", dataset["path_output"]))
    dataset[LICENSE_DISCREPANCY_COLUMN_NAME] = utils.rep(x = False, times = len(dataset)) # create license discrepancy column

    # update deduplication columns
    deduplication_columns_path_updater_helper = lambda path: "." + get_path_output(path = path)[len(output_dir):]
    dataset["best_path"] = list(map(deduplication_columns_path_updater_helper, dataset["best_path"]))
    dataset["best_arrangement"] = list(map(deduplication_columns_path_updater_helper, dataset["best_arrangement"]))
    dataset["best_unique_arrangement"] = list(map(deduplication_columns_path_updater_helper, dataset["best_unique_arrangement"]))
    del deduplication_columns_path_updater_helper

    # check if dataset exists already, and if it does, read it in to get license discrepancies
    need_to_calculate_license_discrepancy = True
    if exists(output_filepath):
        previous_dataset = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = [LICENSE_DISCREPANCY_COLUMN_NAME])
        if len(previous_dataset) == len(dataset):
            dataset[LICENSE_DISCREPANCY_COLUMN_NAME] = previous_dataset[LICENSE_DISCREPANCY_COLUMN_NAME]
            need_to_calculate_license_discrepancy = False
        del previous_dataset # free up memory

    # create necessary directory trees if required
    for column in output_columns:
        subdirectories = set(map(dirname, filter(lambda path: not pd.isna(path), dataset[column])))
        for subdirectory in subdirectories:
            makedirs(subdirectory, exist_ok = True)

    ##################################################


    # GENERATE MusicRender JSON FILES
    ##################################################

    # helper function to save files
    def get_json_metadata(i: int) -> Tuple[str, str, bool]:
        """
        Given the dataset index, copy over song as music object and metadata.
        Returns the output path, metadata output path, and whether the song 
        has an internal license discrepancy.
        """

        # musescore file path
        path = dataset.at[i, "path"]

        # save as music object
        path_output = dataset.at[i, "path_output"]
        need_to_save_music_json = not exists(path_output) or args.reset or args.reset_json
        if need_to_save_music_json or need_to_calculate_license_discrepancy: # save if necessary
            music = read_musescore(path = path, timeout = TIMEOUT)
            if need_to_save_music_json:
                music.save(path = path_output, compressed = COMPRESS_JSON_MUSIC_FILES) # save as music object
            has_license_discrepancy = (music.metadata.copyright is not None) # if the metadata says a song has a public domain license, but the internals of the MuseScore file say elsewise
            del music
        else:
            has_license_discrepancy = dataset.at[i, LICENSE_DISCREPANCY_COLUMN_NAME]            
        path_output = "." + path_output[len(output_dir):]

        # copy over metadata path
        metadata_path = dataset.at[i, "metadata"]
        metadata_path_output = dataset.at[i, "metadata_output"]
        if metadata_path is not None and (not exists(metadata_path_output) or args.reset):
            copy(src = metadata_path, dst = metadata_path_output) # copy over metadata
        if metadata_path_output is not None:
            metadata_path_output = "." + metadata_path_output[len(output_dir):]

        # return values
        return (path_output, metadata_path_output, has_license_discrepancy)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        dataset["path_output"], dataset["metadata_output"], dataset[LICENSE_DISCREPANCY_COLUMN_NAME] = list(zip(*list(
            pool.map(func = get_json_metadata,
                     iterable = tqdm(iterable = dataset.index,
                                     desc = "Generating JSON Files and Copying Metadata",
                                     total = len(dataset)),
                     chunksize = CHUNK_SIZE)
        )))
        
    ##################################################


    # GET EXTRA FILES
    ##################################################

    # helper function to save files with MuseScore CLI
    def get_output_formats(i: int) -> list:
        """
        Given a starting dataset index, generate output format 
        files with the MuseScore CLI.
        Returns the output paths of the generated files, in the order of `output_formats`.
        """

        # create json dict object
        json_dict = {"in": dataset.at[i, "path"], "out": []}
        output_format_path_outputs = [dataset.at[i, output_format_output_column] for output_format_output_column in output_format_output_columns]
        for output_format_path_output in output_format_path_outputs:
            if not exists(output_format_path_output) or args.reset:
                json_dict["out"].append(output_format_path_output)

        # avoid unnecessary calculations
        if len(json_dict["out"]) > 0:

            # create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:

                # write json dict to a json file
                json_path = f"{temp_dir}/job.json"
                with open(json_path, "w") as json_file:
                    json.dump(obj = [json_dict], fp = json_file)
                
                # run batch job with MuseScore CLI, add except clause for corrupted files
                try:
                    subprocess.run(
                        args = ["xvfb-run", "--auto-servernum", args.musescore_filepath, "--job", json_path],
                        check = True,
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL,
                        env = environ,
                        timeout = TIMEOUT, # wait for 60 seconds, if it's not done by then, then don't bother
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired): # if musescore file is corrupted
                    return [None] * len(output_format_path_outputs)
        
        # return output paths
        output_format_path_outputs = ["." + output_format_path_output[len(output_dir):] for output_format_path_output in output_format_path_outputs]
        return output_format_path_outputs
        
    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = list(zip(*list(
            pool.map(func = get_output_formats,
                     iterable = tqdm(iterable = dataset.index,
                                     desc = "Generating Extra Files",
                                     total = len(dataset)),
                     chunksize = CHUNK_SIZE)
        )))

    # update output filepaths in dataset
    for i, output_format_output_column in enumerate(output_format_output_columns):
        dataset[output_format_output_column] = results[i]
    del results # free up memory

    ##################################################


    # DO SOME WRANGLING TO MAKE DATASET NICER
    ##################################################

    # add license discrepancy free facet
    dataset[f"facet:no_{LICENSE_DISCREPANCY_COLUMN_NAME}"] = ~dataset[LICENSE_DISCREPANCY_COLUMN_NAME]
    dataset["facet:all_valid"] = list(map(all, zip(*[~pd.isna(dataset[output_format_output_column]) for output_format_output_column in output_format_output_columns])))
                                                        
    # rename facet columns
    facet_columns = list(filter(lambda column: column.startswith("facet:"), dataset.columns))
    subset_columns = list(map(lambda facet_column: facet_column.replace("facet", "subset"), facet_columns))
    dataset = dataset.rename(columns = dict(zip(facet_columns, subset_columns))) # rename facet to subset columns

    # rename columns that are filepaths
    dataset = dataset.drop(columns = ["path", "metadata", "has_metadata"])
    output_columns_new = [column.split("_")[0] for column in output_columns]
    dataset = dataset.rename(columns = dict(zip(output_columns, output_columns_new))) # rename output columns

    # reorder columns
    dataset_columns = output_columns_new + list(filter(lambda column: not ((column in subset_columns) or (column in output_columns_new) or (column == LICENSE_DISCREPANCY_COLUMN_NAME)), dataset.columns)) + subset_columns # place subset columns at end
    dataset_columns.insert(dataset_columns.index("license_url") + 1, LICENSE_DISCREPANCY_COLUMN_NAME) # add license discrepancy column back in
    dataset = dataset[dataset_columns] # reorder columns

    # save dataset to file
    dataset.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # text files with paths for each facet
    for column in subset_columns:
        with open(f"{facets_dir}/{column.split(':')[-1]}.txt", "w") as output_file:
            output_file.write("\n".join(dataset.loc[dataset[column], "path"]) + "\n") # ensure trailing newline so line count is correct

    ##################################################
    

    # GZIP
    ##################################################

    if args.gzip:
    
        # create directory for gzipped files
        gzip_dir = f"{args.output_dir}/{DATASET_NAME}_gzip"
        directory_creator(gzip_dir)

        # gzip subdirectories
        for directory in tqdm(iterable = [data_dir, metadata_dir, facets_dir] + output_format_dirs, desc = "Gzipping"):
            directory_basename = basename(directory)
            targz_output_path = f"{directory_basename}.tar.gz"
            subprocess.run(args = ["tar", "-zcf", targz_output_path, directory_basename], check = True, cwd = output_dir)
            subprocess.run(args = ["mv", targz_output_path, f"{gzip_dir}/{targz_output_path}"], check = True, cwd = output_dir)

        # copy over csv file
        copy(src = output_filepath, dst = f"{gzip_dir}/{basename(output_filepath)}")
    
    ##################################################

##################################################

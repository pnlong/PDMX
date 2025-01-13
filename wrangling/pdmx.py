# README
# Phillip Long
# August 11, 2024

# Create PDMX (Public Domain Music XML) Dataset.

# python /home/pnlong/model_musescore/pdmx.py

# IMPORTS
##################################################

import argparse
from os.path import exists, dirname, basename
from os import makedirs, mkdir, chdir, environ
from shutil import copy
import subprocess
import pandas as pd
from tqdm import tqdm
import multiprocessing
import logging
from typing import Tuple

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import DATASET_DIR_NAME, MUSESCORE_DIR, CHUNK_SIZE
from full import OUTPUT_DIR as DATASET_OUTPUT_DIR
from reading.read_musescore import read_musescore
import utils

environ["QT_QPA_PLATFORM"] = "offscreen" # for musescore cli

##################################################


# CONSTANTS
##################################################

# default output directory
OUTPUT_DIR = "/data1/pnlong/musescore"

# path to MuseScore CLI application
MUSESCORE_APPLICATION_PATH = "/data3/pnlong/MuseScore-Studio-4.4.4.243461245-x86_64.AppImage" # MuseScore Command Line Options: https://musescore.org/en/handbook/3/command-line-options

# name of dataset
DATASET_NAME = "PDMX"

# the metadata says a song has a public domain license, but the internals of the MuseScore files say elsewise
LICENSE_DISCREPANCY_COLUMN_NAME = "license_conflict"

# whether to compress json music files
COMPRESS_JSON_MUSIC_FILES = False

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
    parser.add_argument("-mf", "--musescore_filepath", type = str, default = MUSESCORE_APPLICATION_PATH, help = "Filepath to MuseScore CLI application (for generating MusicXML and PDF files)")
    parser.add_argument("-g", "--gzip", action = "store_true", help = "GZIP the output directory of the dataset")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
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
    def directory_creator(directory: str):
        if not exists(directory):
            mkdir(directory)
    data_dir = f"{output_dir}/data"
    directory_creator(directory = data_dir)
    metadata_dir = f"{output_dir}/metadata"
    directory_creator(directory = metadata_dir)
    mxl_dir = f"{output_dir}/mxl"
    directory_creator(directory = mxl_dir)
    pdf_dir = f"{output_dir}/pdf"
    directory_creator(directory = pdf_dir)
    facets_dir = f"{output_dir}/subset_paths"
    directory_creator(directory = facets_dir)

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # load in dataset
    output_columns = ["path_output", "metadata_output", "mxl_output", "pdf_output"]
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
    dataset["mxl_output"] = list(map(lambda path: mxl_dir + path[len(data_dir):-len(".json")] + ".mxl", dataset["path_output"]))
    dataset["pdf_output"] = list(map(lambda path: pdf_dir + path[len(data_dir):-len(".json")] + ".pdf", dataset["path_output"]))

    # create necessary directory trees if required
    for column in output_columns:
        subdirectories = set(map(dirname, filter(lambda path: not pd.isna(path), dataset[column])))
        for subdirectory in subdirectories:
            makedirs(subdirectory, exist_ok = True)

    ##################################################


    # COPY OVER FILES
    ##################################################

    # helper function to save files
    def get_file(i: int) -> Tuple[str, str, str, str, bool]:
        """
        Given the dataset index, copy over song as music object, and metadata, while generating other files.
        Returns tuple of JSONified data path, metadata path, MusicXML path, PDF path.
        """

        # musescore path
        path = dataset.at[i, "path"]

        # save as music object
        path_output = dataset.at[i, "path_output"]
        music = read_musescore(path = path, timeout = 10)
        license_discrepancy = music.metadata.copyright is not None # if the metadata says a song has a public domain license, but the internals of the MuseScore file say elsewise
        if not exists(path_output) or args.reset: # save if necessary
            music.save(path = path_output, compressed = COMPRESS_JSON_MUSIC_FILES) # save as music object
        del music
        path_output = "." + path_output[len(output_dir):]

        # copy over metadata path
        metadata_path = dataset.at[i, "metadata"]
        metadata_path_output = dataset.at[i, "metadata_output"]
        if metadata_path is not None and (not exists(metadata_path_output) or args.reset):
            copy(src = metadata_path, dst = metadata_path_output) # copy over metadata
        if metadata_path_output is not None:
            metadata_path_output = "." + metadata_path_output[len(output_dir):]

        # generate music xml file
        mxl_path_output = dataset.at[i, "mxl_output"]
        if not exists(mxl_path_output) or args.reset:
            subprocess.run(
                args = [args.musescore_filepath, "--export-to", mxl_path_output, path],
                check = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
        mxl_path_output = "." + mxl_path_output[len(output_dir):]

        # generate pdf file
        pdf_path_output = dataset.at[i, "pdf_output"]
        if not exists(pdf_path_output) or args.reset:
            subprocess.run(
                args = [args.musescore_filepath, "--export-to", pdf_path_output, path],
                check = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
        pdf_path_output = "." + pdf_path_output[len(output_dir):]

        # return paths
        return (path_output, metadata_path_output, mxl_path_output, pdf_path_output, license_discrepancy)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        dataset["path"], dataset["metadata"], dataset["mxl"], dataset["pdf"], dataset[LICENSE_DISCREPANCY_COLUMN_NAME] = list(zip(*list(pool.map(
            func = get_file,
            iterable = tqdm(iterable = dataset.index, desc = f"Creating {DATASET_NAME}", total = len(dataset)),
            chunksize = CHUNK_SIZE)
        )))

    ##################################################


    # DO SOME WRANGLING TO MAKE DATASET NICER
    ##################################################
                                                        
    # remove unnecessary columns
    dataset = dataset.drop(columns = output_columns)
    facet_columns = list(filter(lambda column: column.startswith("facet:"), dataset.columns))
    subset_columns_prefix = "subset"
    subset_columns = list(map(lambda facet_column: facet_column.replace("facet", subset_columns_prefix), facet_columns))
    dataset = dataset.rename(columns = dict(zip(facet_columns, subset_columns))) # rename facet to subset columns

    # add license conflict free subset
    no_license_discrepancy_subset_name = f"{subset_columns_prefix}:no_{LICENSE_DISCREPANCY_COLUMN_NAME}"
    subset_columns.append(no_license_discrepancy_subset_name)
    dataset[no_license_discrepancy_subset_name] = ~dataset[LICENSE_DISCREPANCY_COLUMN_NAME]

    # reorder columns
    dataset_columns = list(filter(lambda column: column not in subset_columns, dataset.columns)) + subset_columns # place subset columns at end
    dataset_columns.remove(LICENSE_DISCREPANCY_COLUMN_NAME) # remove license discrepancy column
    dataset_columns.insert(dataset_columns.index("license_url") + 1, LICENSE_DISCREPANCY_COLUMN_NAME) # add license discrepancy column back in
    dataset = dataset[dataset_columns] # reorder columns

    # save dataset to file
    dataset.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # text files with paths for each facet
    for column in subset_columns:
        with open(f"{facets_dir}/{column.split(':')[-1]}.txt", "w") as output_file:
            output_file.write("\n".join(dataset[dataset[column]]["path"]))

    # gzip if needed
    if args.gzip:
        chdir(dirname(output_dir))
        logging.info("Gzipping dataset.")
        subprocess.run(args = ["tar", "-zcf", f"{basename(output_dir)}.tar.gz", basename(output_dir)], check = True)
    
    ##################################################

##################################################

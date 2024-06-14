# README
# Phillip Long
# March 28, 2024

# Make the main dataset for the paper.

# python /home/pnlong/model_musescore/dataset_main.py


# IMPORTS
##################################################

from os.path import exists, basename, dirname
from os import makedirs, mkdir
import subprocess
from shutil import copy
import pandas as pd
from tqdm import tqdm
from time import perf_counter, strftime, gmtime
import multiprocessing
import argparse
import logging
from read_mscz.read_mscz import read_musescore
from train import NA_VALUE

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore/expressive_features"
OUTPUT_DIR = "/data2/pnlong/musescore"

DATASET_NAME = "ExpressionNet"

OUTPUT_COLUMNS = ["path", "metadata", "genres", "tags", "n_tracks", "n_expressive_features", "n_tokens"]

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore", description = "Extract expressive features from MuseScore files.")
    parser.add_argument("-i", "--input_dir", type = str, default = INPUT_DIR, help = "Output directory provided to `parse_mscz.py`")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory where the CSV mappings file and the directory with the actual dataset will be stored")
    parser.add_argument("-n", "--nested", action = "store_true", help = "Whether to replicate Herman's nested directory structure")
    parser.add_argument("-m", "--metadata", action = "store_true", help = "Whether to store metadata paths in the resulting CSV file")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# FUNCTION TO PRODUCE JSON FILES FOR DATASET
##################################################

# main function for getting data
def get_data(input_path: str, output_path: str, compressed: bool = False) -> str:
    """
    Given the input path, load that MuseScore file and save it in JSON format.
    If the file loads incorrectly, return None; otherwise, return the `output_path`.

    Parameters
    ----------
    input_path : str
        Path to the MuseScore file to read.
    output_path : str
        Path where the MusicExpress JSON file will be saved.
    compressed : bool
        Whether or not to compress the JSON output.

    Returns
    -------
    :str:
        The path to which the MusicExpress JSON file was successfully saved. If the file loads incorrectly, returns None.
    
    """

    # avoid doing if we don't need to
    if exists(output_path):
        return output_path

    # try to read in musescore object
    try:
        music = read_musescore(path = input_path, timeout = 10)
        music.save_json(path = output_path, compressed = compressed)
        return output_path
    
    # if that fails, return None
    except:
        return None
    

# function to help with copying metadata files
def get_metadata(input_path: str) -> str:
    """Copy over a metadata file, returning the copied filepath."""
    output_dir = f"{METADATA_DIR}" + (f"/{basename(dirname(input_path))}" if args.nested else "")
    if not exists(output_dir):
        try:
            mkdir(output_dir)
        except (FileExistsError):
            pass
    output_path = f"{output_dir}/{basename(input_path)}" # get output filepath
    copy(src = input_path, dst = output_path) # copy over file
    return output_path

##################################################


# for multiprocessing, must use a main class
if __name__ == "__main__":

    # ARGS AND CONSTANTS
    ##################################################

    args = parse_args()
    if not exists(args.output_dir): # make output_dir if it doesn't yet exist
        makedirs(args.output_dir)

    # output filepaths and directory
    OUTPUT_DIR_MAIN = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(OUTPUT_DIR_MAIN): # make output_dir if it doesn't yet exist
        mkdir(OUTPUT_DIR_MAIN)
    OUTPUT_PATH = f"{OUTPUT_DIR_MAIN}/{DATASET_NAME}.csv"
    DATA_DIR = f"{OUTPUT_DIR_MAIN}/data"
    if not exists(DATA_DIR):
        mkdir(DATA_DIR)
        if args.nested:
            subprocess.run(args = ["bash", f"{dirname(__file__)}/create_data_dir.sh", "-d", DATA_DIR], check = True)
    METADATA_DIR = f"{OUTPUT_DIR_MAIN}/metadata"
    if args.metadata and not exists(METADATA_DIR):
        mkdir(METADATA_DIR)

    # load in data frame
    dataset = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/{basename(args.input_dir)}.path.csv", sep = ",", header = 0, index_col = False)
    dataset = dataset[dataset["in_dataset"]].drop(columns = "in_dataset") # make sure we only have relevant files
    dataset = dataset[OUTPUT_COLUMNS] # get the right columns

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # GET LIST OF FILES
    ##################################################

    # use glob to get all mscz files
    input_paths = dataset["path"].tolist()

    # get output filepaths
    if args.nested:
        get_output_path = lambda path: f"{DATA_DIR}/{('/'.join(path.split('/')[-3:])).split('.')[0]}.json"
    else:
        get_output_path = lambda path: f"{DATA_DIR}/{basename(path).split('.')[0]}.json"
    output_paths = tuple(map(get_output_path, input_paths))

    ##################################################


    # MAKE DATASET
    ##################################################

    # some pre dataset building stuff
    logging.info(f"N_PATHS = {len(input_paths)}") # print number of paths to process
    chunk_size = 1
    make_path_relative = lambda path: path.replace(OUTPUT_DIR_MAIN, ".")
    start_time = perf_counter() # start the timer

    # use multiprocessing to generate dataset
    with multiprocessing.Pool(processes = args.jobs) as pool:

        # get json files
        output_paths = pool.starmap(
            func = get_data,
            iterable = tqdm(
                iterable = zip(input_paths, output_paths),
                desc = f"Creating {DATASET_NAME}",
                total = len(input_paths)),
            chunksize = chunk_size)
        dataset["path"] = output_paths
        dataset = dataset[~pd.isna(dataset["path"])]
        dataset["path"] = dataset["path"].apply(make_path_relative) # transform output paths into relative paths
        
        # copy metadata files over, or drop the metadata column altogether
        if args.metadata:
            valid_metadata_paths = ((~pd.isna(dataset["metadata"])) & dataset["metadata"].apply(exists))
            dataset.loc[~valid_metadata_paths, "metadata"] = None
            metadata_paths = list(tqdm(
                iterable = pool.imap(
                    func = get_metadata,
                    iterable = dataset["metadata"][valid_metadata_paths],
                    chunksize = chunk_size
                ),
                desc = "Copying over metadata",
                total = sum(valid_metadata_paths)))
            dataset.loc[valid_metadata_paths, "metadata"] = list(map(make_path_relative, metadata_paths))
        else:
            dataset = dataset.drop(columns = "metadata")    
    
    # output total time
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    logging.info(f"Total time: {total_time}")    

    # save dataset mapping
    dataset.to_csv(path_or_buf = OUTPUT_PATH, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")

    ##################################################

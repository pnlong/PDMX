# README
# Phillip Long
# June 1, 2025

# Remedy the bug in the malformed license discrepancy column.

# python /home/pnlong/model_musescore/wrangling/remedy_license_discrepancy_bug.py

# IMPORTS
##################################################

import argparse
from os.path import dirname, basename, exists
from os import chdir
from shutil import copy
import subprocess
import pandas as pd
from tqdm import tqdm
import multiprocessing

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from reading.music import load
import utils
from pdmx import DATASET_NAME, OUTPUT_DIR, LICENSE_DISCREPANCY_COLUMN_NAME

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Fix License Discrepancy Bug", description = f"Fixing License Conflict Bug in {DATASET_NAME} Dataset.")
    parser.add_argument("-df", "--dataset_filepath", type = str, default = f"{OUTPUT_DIR}/{DATASET_NAME}/{DATASET_NAME}.csv", help = "Absolute filepath to PDMX .csv file")
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

    # load in pdmx
    pdmx_dir = dirname(args.dataset_filepath)
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    # get facets directory for later
    facets_dir = f"{pdmx_dir}/subset_paths"

    # get gzip directory for later
    gzip_dir = f"{dirname(pdmx_dir)}/{DATASET_NAME}_gzip"

    # helper function for converting a path to absolute path
    convert_to_absolute_path = lambda path: pdmx_dir + path[1:]

    ##################################################


    # RECHECK LICENSE DISCREPANCIES
    ##################################################

    def check_license_discrepancy(i: int) -> bool:
        """
        Given the dataset index, load in the MusicRender JSON file and check for license discrepancy.
        Return whether or not there is a license discrepancy.
        """

        # load in as music object
        music = load(path = convert_to_absolute_path(dataset.at[i, "path"]))
        has_license_discrepancy = (music.metadata.copyright is not None) # check if there is license discrepancy
        return has_license_discrepancy

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        dataset[LICENSE_DISCREPANCY_COLUMN_NAME] = list(
            pool.map(func = check_license_discrepancy,
                     iterable = tqdm(iterable = dataset.index,
                                     desc = "Validating License Discrepancy",
                                     total = len(dataset)),
                     chunksize = 1)
        )
    
    # print license discrepancy info
    n_songs_with_license_discrepancy = sum(dataset[LICENSE_DISCREPANCY_COLUMN_NAME])
    print(f"{n_songs_with_license_discrepancy:,} songs with a license discrepancy. {len(dataset) - n_songs_with_license_discrepancy:,} songs without a license discrepancy.")
    _ = input("Can we proceed? Please press enter: ") # require user interaction to proceed before writing anything
    
    ##################################################


    # DO SOME WRANGLING TO MAKE DATASET NICER
    ##################################################

    # add license discrepancy free subset
    no_license_discrepancy_column_name = f"subset:no_{LICENSE_DISCREPANCY_COLUMN_NAME}"
    dataset[no_license_discrepancy_column_name] = ~dataset[LICENSE_DISCREPANCY_COLUMN_NAME]

    # save dataset to file
    dataset.to_csv(path_or_buf = args.dataset_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # rewrite subset path columns
    with open(f"{facets_dir}/{no_license_discrepancy_column_name.split(':')[-1]}.txt", "w") as output_file:
        output_file.write("\n".join(dataset.loc[dataset[no_license_discrepancy_column_name], "path"]) + "\n") # ensure trailing newline so line count is correct

    # regzip if necessary
    if exists(gzip_dir):
        copy(src = args.dataset_filepath, dst = f"{gzip_dir}/{basename(args.dataset_filepath)}") # copy new dataset in gzip directory
        targz_output_path = f"{basename(facets_dir)}.tar.gz"
        subprocess.run(args = ["tar", "-zcf", targz_output_path, basename(facets_dir)], check = True, cwd = pdmx_dir)
        subprocess.run(args = ["mv", targz_output_path, f"{gzip_dir}/{targz_output_path}"], check = True, cwd = pdmx_dir)

    ##################################################

##################################################



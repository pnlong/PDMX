# README
# Phillip Long
# September 26, 2023

# Correct the filepath mappings for musescore files and metadata.

# python /home/pnlong/model_musescore/remap_metadata_to_data.py

# IMPORTS
##################################################

import glob
from os.path import isfile, exists
from tqdm import tqdm
import multiprocessing
import json
import pandas as pd
import logging

from dataset_full import MUSESCORE_DIR, INPUT_DIR, CHUNK_SIZE
import utils

############################################### ###


# CONSTANTS
##################################################

OUTPUT_FILEPATH = f"{INPUT_DIR}/metadata_to_data.csv"
MAPPINGS_COLUMNS = ["metadata", "data", "metadata_path", "data_path"]

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Remap Metadata to Data", description = "Create data table mapping metadata to data files.")
    parser.add_argument("-m", "--musescore_dir", type = str, default = MUSESCORE_DIR, help = "Directory where MuseScore data and metadata is stored")
    parser.add_argument("-o", "--output_filepath", type = str, default = OUTPUT_FILEPATH, help = "Output filepath")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    if not exists(args.output_filepath):

        # SET UP
        ##################################################

        # parse arguments
        args = parse_args()

        # set up logging
        logging.basicConfig(level = logging.INFO, format = "%(message)s")

        ##################################################


        # GET FILEPATHS WITH GLOB + WRONG MAPPINGS
        ##################################################

        # create dictionary
        metadata_paths = glob.iglob(pathname = f"{args.musescore_dir}/metadata/**", recursive = True) # glob filepaths recursively, generating an iterator object
        metadata_paths = (metadata_path for metadata_path in metadata_paths if isfile(metadata_path)) # filter out non-file elements that were globbed
        metadata_paths = {metadata_path.split("/")[-1].split(".")[0] : metadata_path for metadata_path in metadata_paths} # turn into dictionary, will make finding files much faster

        # import mappings
        with open(f"{args.musescore_dir}/metadata2path.json", "r") as file:
            mappings = json.load(fp = file)
        mappings = pd.DataFrame.from_dict(data = mappings, orient = "index").reset_index(drop = False).rename(columns = {"index": "metadata", 0: "data_path"})
        mappings["data"] = mappings["data_path"].apply(lambda data_path: data_path.split("/")[-1].split(".")[0])
        mappings = mappings[MAPPINGS_COLUMNS[:2] + MAPPINGS_COLUMNS[3:]]

        ##################################################


        # ITERATE THROUGH MAPPINGS, CREATE A HELPER FUNCTION TO REMAP, OUTPUTTING EACH NEW FILE
        ##################################################

        # create remappings dataframe
        remappings = pd.DataFrame(columns = MAPPINGS_COLUMNS) # create dataframe
        remappings.to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

        def find_metadata_path(i: int):

            # get current mapping
            current = mappings.loc[i]

            # get metadata path
            try:
                metadata_path = metadata_paths[str(current["metadata"])]
            except KeyError: # if nothing was found, make NA
                metadata_path = None

            # add metadata path to list to output
            current = current.to_list() # convert to list
            current.insert(2, metadata_path) # add metadatapath to current

            # output current
            remappings_current = pd.DataFrame(data = [dict(zip(MAPPINGS_COLUMNS, current))], columns = MAPPINGS_COLUMNS)
            remappings_current.to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

    ##################################################


    # USE MULTIPROCESSING
    ##################################################

    # run with multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = list(tqdm(iterable = pool.imap_unordered(
                func = find_metadata_path,
                iterable = range(len(mappings)),
                chunksize = CHUNK_SIZE,
            ),
            desc = "Finding Filepaths",
            total = len(mappings)))

    ##################################################


    # OUTPUT STATISTIC(S)
    ##################################################

    # load in remappings
    remappings = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)

    # metadata not found
    logging.info("STATISTICS:")
    n_metadata_not_found = len(remappings[pd.isna(remappings["metadata_path"])])
    logging.info(f"{n_metadata_not_found} ({100 * (n_metadata_not_found / len(remappings)):.2f}%) of metadata files could not be located.")

    ##################################################

##################################################
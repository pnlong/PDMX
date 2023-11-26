# README
# Phillip Long
# September 26, 2023

# correct the filepath mappings for musescore files and metadata

# python ./remap_paths.py

# IMPORTS
##################################################
import glob
from os.path import isfile, exists
from tqdm import tqdm
import multiprocessing
import json
import pandas as pd
import logging
##################################################


# CONSTANTS
##################################################
MSCZ_DIR = "/data2/zachary/musescore"
ORIGINAL_FILEPATH = f"{MSCZ_DIR}/metadata2path.json"
OUTPUT_FILEPATH = "/data2/pnlong/musescore/metadata_to_data.csv"
##################################################


if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    if not exists(OUTPUT_FILEPATH):

        # GET FILEPATHS WITH GLOB + WRONG MAPPINGS
        ##################################################

        metadata_paths = glob.iglob(pathname = f"{MSCZ_DIR}/metadata/**", recursive = True) # glob filepaths recursively, generating an iterator object
        metadata_paths = (metadata_path for metadata_path in metadata_paths if isfile(metadata_path)) # filter out non-file elements that were globbed
        metadata_paths = {metadata_path.split("/")[-1].split(".")[0] : metadata_path for metadata_path in metadata_paths} # turn into dictionary, will make finding files much faster

        # import mappings
        with open(ORIGINAL_FILEPATH, "r") as file:
            mappings = json.load(fp = file)
        mappings = pd.DataFrame.from_dict(data = mappings, orient = "index").reset_index(drop = False).rename(columns = {"index": "metadata", 0: "data_path"})
        mappings["data"] = mappings["data_path"].apply(lambda data_path: data_path.split("/")[-1].split(".")[0])
        mappings_columns = ["metadata", "data", "metadata_path", "data_path"]
        mappings = mappings[mappings_columns[:2] + mappings_columns[3:]]

        ##################################################


        # ITERATE THROUGH MAPPINGS, CREATE A HELPER FUNCTION TO REMAP, OUTPUTTING EACH NEW FILE
        ##################################################

        # create remappings dataframe
        remappings = pd.DataFrame(columns = mappings_columns) # create dataframe
        remappings.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

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
            current.insert(2, metadata_path)# add metadatapath to current

            # output current
            remappings_current = pd.DataFrame(data = [dict(zip(mappings_columns, current))], columns = mappings_columns)
            remappings_current.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")

    ##################################################


    # USE MULTIPROCESSING
    ##################################################

    # run with multiprocessing
    chunk_size = 1
    with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
        results = list(tqdm(iterable = pool.imap_unordered(func = find_metadata_path, iterable = range(len(mappings)), chunksize = chunk_size),
                            desc = "Finding filepaths",
                            total = len(mappings)))

    ##################################################


    # OUTPUT STATISTIC(S)
    ##################################################

    # load in remappings
    remappings = pd.read_csv(filepath_or_buffer = OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)

    logging.info("STATISTICS:")

    # metadata not found
    n_metadata_not_found = len(remappings[pd.isna(remappings["metadata_path"])])
    logging.info(f"{n_metadata_not_found} ({100 * (n_metadata_not_found / len(remappings)):.2f}%) of metadata files could not be located.")

    ##################################################
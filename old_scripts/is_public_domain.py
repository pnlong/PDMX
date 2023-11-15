# README
# Phillip Long
# September 27, 2023

# determine what files in the musescore dataset are in the public domain. Two options: create Metadata objects and call Metadata.is_public_domain(), or load the JSON files as dictionaries x (x["data"]["is_public_domain"])

# python ./is_public_domain.py


# IMPORTS
##################################################
from os.path import exists
import json
from tqdm import tqdm
import multiprocessing
import pandas as pd
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore"
OUTPUT_FILEPATH = f"{OUTPUT_DIR}/public_domain.csv"
##################################################

# for multiprocessing
if __name__ == "__main__":

    if not exists(OUTPUT_FILEPATH):

        # LOAD IN DATA FRAME
        ##################################################
        metadata = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/metadata_to_data.csv", sep = ",", header = 0, index_col = False)
        metadata = metadata[~pd.isna(metadata["metadata_path"])] # remove NA entries
        
        out_df_cols = metadata.columns.to_list() + ["is_public_domain"]
        out_df = pd.DataFrame(columns = out_df_cols)
        out_df.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")
        ##################################################


        # HELPER FUNCTION TO DETERMINE PUBLIC DOMAIN
        ##################################################
        def is_public_domain(metadata_path: str) -> bool:

            # load JSON file as dictionary
            with open(metadata_path, "r") as metadata_file:
                metadata_for_path = json.load(fp = metadata_file)

            return bool(metadata_for_path["data"]["is_public_domain"])

        ##################################################


        # DEFINE HELPER FUNCTION FOR MULTIPROCESSING
        ##################################################

        def try_to_determine_public_domain(i: int):
            # try to get the version
            try:
                is_pd = is_public_domain(metadata_path = metadata.at[i, "metadata_path"])
            # if the file is corrupt or something, then whether or note it is public domain is NA
            except:
                is_pd = None

            # output row
            out_df = pd.DataFrame(data = [dict(zip(out_df_cols, metadata.loc[i].to_list() + [is_pd]))], columns = out_df_cols)
            out_df.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")
        
        ##################################################


        # PARSE THROUGH DATAFRAME, DETERMINE IF PUBLIC DOMAIN
        ##################################################

        chunk_size = 5    
        with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
            results = list(tqdm(iterable = pool.imap_unordered(func = try_to_determine_public_domain, iterable = metadata.index, chunksize = chunk_size),
                                desc = "Determining if Public Domains",
                                total = len(metadata)))

        ##################################################


    # OUTPUT STATISTICS
    ##################################################

    # load in df
    in_public_domain = pd.read_csv(filepath_or_buffer = OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)

    n_public_domain = len(in_public_domain[in_public_domain["is_public_domain"] == True]) # == True because of NA values
    print(f"{n_public_domain} of .mscz files ({100 * (n_public_domain / len(in_public_domain)):.2f}% of all files) are in the public domain.")

    ##################################################
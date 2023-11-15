# README
# Phillip Long
# September 27, 2023

# Get the version of all the musescore files, hopefully find a connection between invalid files + their version

# python /home/pnlong/parse_musescore/get_mscz_versions.py

# IMPORTS
##################################################
from os.path import exists
import multiprocessing
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from read_mscz.read_mscz import get_musescore_version
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore"
OUTPUT_FILEPATH = f"{OUTPUT_DIR}/versions.csv"
OUTPUT_FILEPATH_PLOT = f"{OUTPUT_DIR}/versions.png"
##################################################


if __name__ == "__main__":

    if not exists(OUTPUT_FILEPATH):

        # LOAD IN DATA FRAME
        ##################################################

        validities = pd.read_csv(filepath_or_buffer = f"{OUTPUT_DIR}/validity.csv", sep = ",", header = 0, index_col = False)
        
        out_df_cols = validities.columns.to_list() + ["version"]
        out_df = pd.DataFrame(columns = out_df_cols)
        out_df.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

        ##################################################


        # DEFINE A RUNNER FUNCTION
        ##################################################

        # get the musescore version and output to a file
        def check_musescore_version(i: int):

            # try to get the version
            try:
                version = get_musescore_version(path = validities.at[i, "path"])
            # if the file is corrupt or something, the version is NA
            except:
                version = None
            
            # output row
            out_df = pd.DataFrame(data = [dict(zip(out_df_cols, validities.loc[i].to_list() + [version]))], columns = out_df_cols)
            out_df.to_csv(path_or_buf = OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")

        ##################################################


        # PARSE THROUGH DATA FRAME, GETTING VERSION, WITH MULTIPROCESSING
        ##################################################

        chunk_size = 1
        with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
            results = list(tqdm(iterable = pool.imap_unordered(func = check_musescore_version, iterable = validities.index, chunksize = chunk_size),
                                desc = "Getting MuseScore Versions",
                                total = len(validities)))

        ##################################################


    # MAKE PLOT
    ##################################################

    # load in df
    versions = pd.read_csv(filepath_or_buffer = OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["bar_total", "bar_error"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Distribution of Musescore Versions", fontweight = "bold")

    # helper function to group by version
    def group_by_version(df: pd.DataFrame):

        df["version"] = df["version"].apply(lambda version: str(version).strip()[0] if version else "Corrupted") # switch out to base version
        df = df.groupby(by = "version").count() # sum over each version
        df = df.reset_index().rename(columns = {"index": "version", "is_valid": "count"}) # make error type into column
        return df[["version", "count"]] # select only subset of columns

    # total bar chart
    bar_total = group_by_version(df = versions)
    axes["bar_total"].barh(width = bar_total["count"], y = bar_total["version"]) # make bar chart
    axes["bar_total"].set_title("All Musescore Files")
    axes["bar_total"].set_xlabel("Count")

    # error bar chart
    bar_error = group_by_version(df = versions[~versions["is_valid"]].reset_index(drop = True))
    axes["bar_error"].barh(width = bar_error["count"], y = bar_error["version"]) # make bar chart
    axes["bar_error"].set_title("Invalid Files")
    axes["bar_error"].set_xlabel("Count")

    # save image
    fig.savefig(OUTPUT_FILEPATH_PLOT, dpi = 240) # save image
    print(f"Plot saved to {OUTPUT_FILEPATH_PLOT}.")

    ##################################################
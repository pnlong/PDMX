# README
# Phillip Long
# September 26, 2023

# Test the read_musescore function

# python /home/pnlong/parse_musescore/test_read_mscz.py

# IMPORTS
##################################################
import glob
from os.path import isfile, exists
import multiprocessing
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from read_mscz.read_mscz import read_musescore
##################################################


# CONSTANTS
##################################################
OUTPUT_DIR = "/data2/pnlong/musescore"
ERROR_MESSAGE_OUTPUT_FILEPATH = f"{OUTPUT_DIR}/errors.txt"
VALIDITY_COLUMNS = ("path", "is_valid")
VALIDITY_OUTPUT_FILEPATH = f"{OUTPUT_DIR}/validity.csv"
ERROR_RATES_COLUMNS = ("path", "zip_error", "keysig_error", "required_error", "neg2_error", "other_error")
ERROR_RATES_OUTPUT_FILEPATH = f"{OUTPUT_DIR}/errors.csv"
PLOT_OUTPUT_FILEPATH = f"{OUTPUT_DIR}/errors.png"
##################################################


# for multiprocessing
if __name__ == "__main__":

    # TESTING FUNCTION
    ##################################################

    # function to test each path for its validity
    def test_path(path: str):
        
        # initialize counters
        zip_error, keysig_error, required_error, other_error, neg2_error = (False,) * 5
        valid = True

        try:
            
            # try to read musescore
            _ = read_musescore(path = path)

        except Exception as exc:

            # print most recent error, determine error type
            error_message = str(exc)
            if "zip" in error_message:
                zip_error = True
            elif "KeySig" in error_message:
                keysig_error = True
            elif "required for" in error_message:
                required_error = True
            elif "-2" in error_message:
                neg2_error = True
            else:
                other_error = True

            # this path is invalid
            valid = False

            # output error message to file
            with open(ERROR_MESSAGE_OUTPUT_FILEPATH, "a") as error_message_output:
                error_message_output.write(f"{error_message}; PATH: {path}\n")

        # output validity
        validity_current = pd.DataFrame(data = [dict(zip(VALIDITY_COLUMNS, (path, valid)))], columns = VALIDITY_COLUMNS)
        validity_current.to_csv(path_or_buf = VALIDITY_OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = False, index = False, mode = "a") # append to file

        # output error
        error_rates_current = pd.DataFrame(data = [dict(zip(ERROR_RATES_COLUMNS, (path, zip_error, keysig_error, required_error, neg2_error, other_error)))], columns = ERROR_RATES_COLUMNS)
        error_rates_current.to_csv(path_or_buf = ERROR_RATES_OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = False, index = False, mode = "a") # append to file
    
    ##################################################


    # PARSE THROUGH ALL SUBDIRECTORIES
    ##################################################

    # calculate error rates if necessary
    if not exists(ERROR_RATES_OUTPUT_FILEPATH):

        # get all subdirectories
        paths = glob.iglob(pathname = f"/data2/zachary/musescore/data/**", recursive = True)
        paths = tuple(path for path in paths if (isfile(path) and path.endswith("mscz")))

        # create validity, write columns
        validity = pd.DataFrame(columns = VALIDITY_COLUMNS)
        validity.to_csv(path_or_buf = VALIDITY_OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w") # write columns

        # create error rates, write columns
        error_rates = pd.DataFrame(columns = ERROR_RATES_COLUMNS)
        error_rates.to_csv(path_or_buf = ERROR_RATES_OUTPUT_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False, mode = "w") # write columns

        # run with multiprocessing
        chunk_size = 1
        with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
            results = list(tqdm(iterable = pool.imap_unordered(func = test_path, iterable = paths, chunksize = chunk_size),
                                desc = "Testing Validities of MuseScore Files",
                                total = len(paths)))

    ##################################################


    # LOAD IN ERROR RATES, COMPUTE STATISTICS
    ##################################################

    # load in error rates
    validities = pd.read_csv(filepath_or_buffer = VALIDITY_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)
    error_rates = pd.read_csv(filepath_or_buffer = ERROR_RATES_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)

    n_errors = len(validities[validities["is_valid"]])
    n = len(validities)
    error_rate = n_errors / n

    # print output
    print(f"Total Error Rate: {n_errors} / {n} ; {100 * error_rate:.2f}%") 

    ##################################################


    # MAKE PLOT
    ##################################################

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["bar",]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Errors in MuseScore Data", fontweight = "bold")

    # make bar chart
    error_rates = pd.melt(frame = error_rates, id_vars = ERROR_RATES_COLUMNS[0], value_vars = ERROR_RATES_COLUMNS[1:], var_name = "error_type", value_name = "is_error") # reshape (tidy) dataframe
    error_rates = error_rates[error_rates["is_error"]] # remove non-errors
    error_rates = error_rates.groupby(by = "error_type").count() # sum over error type
    error_rates = error_rates.reset_index().rename(columns = {"index": "error_type"}) # make error type into column
    error_rates["error_type"] = error_rates["error_type"].apply(lambda error_type: error_type.split("_")[0].title()) # make error type look nicer
    axes["bar"].barh(width = error_rates["is_error"], y = error_rates["error_type"]) # make bar chart
    axes["bar"].set_title(f"Total Error Rate: {error_rate:.2f}%")
    axes["bar"].set_xlabel("Count")

    # save image
    fig.savefig(PLOT_OUTPUT_FILEPATH, dpi = 240) # save image
    print(f"Plot saved to {PLOT_OUTPUT_FILEPATH}.")

    ##################################################

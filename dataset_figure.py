# README
# Phillip Long
# March 28, 2024

# Make the dataset figure for the paper.

# python /home/pnlong/model_musescore/dataset_figure.py

# IMPORTS
##################################################

import pandas as pd
from numpy import percentile, log10, arange
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import makedirs
import multiprocessing
import argparse
import logging
from time import strftime, gmtime
from read_mscz.music import DIVIDE_BY_ZERO_CONSTANT
from utils import rep
from parse_mscz import LIST_FEATURE_JOIN_STRING
from parse_mscz_plots import INPUT_DIR, OUTPUT_DIR, OUTPUT_RESOLUTION_DPI, BAR_SHIFT_CONSTANT

plt.style.use("bmh")

##################################################


# CONSTANTS
##################################################



##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore Figures", description = "Make plots describing MuseScore dataset.")
    parser.add_argument("-i", "--input_dir", type = str, default = INPUT_DIR, help = "Directory that contains all data tables to be summarized (or where they will be created)")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # command line arguments
    args = parse_args()
    INPUT_FILEPATH = f"{args.input_dir}/expressive_features.csv"
    ERROR_FILEPATH = f"{args.input_dir}/expressive_features.errors.csv"
    TIMING_FILEPATH = f"{args.input_dir}/expressive_features.timing.txt"
    INPUT_FILEPATH_BY_PATH = f"{args.input_dir}/expressive_features.path.csv"

    # make sure directories exist
    if not exists(args.input_dir):
        makedirs(args.input_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # READ IN DATA FRAME
    ##################################################

    # get path- and track-grouped versions of data;
    data_by = {
        "path" : pd.read_csv(filepath_or_buffer = INPUT_FILEPATH_BY_PATH, sep = ",", header = 0, index_col = False),
        "track": pd.read_csv(filepath_or_buffer = INPUT_FILEPATH, sep = ",", header = 0, index_col = False)
    }

    # get plot output filepaths
    plot_output_filepath = f"{args.output_dir}/dataset.png"

    ##################################################


    # CREATE MAIN DATASET FIGURE
    ##################################################

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [[]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle(f"", fontweight = "bold")

    # get a cmap
    cmap = sns.color_palette(palette = "flare", as_cmap = True)

    ##################################################


    # 
    ##################################################


    ##################################################


    # SAVE FIGURE
    ##################################################

    fig.savefig(plot_output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Main Dataset plot saved to {plot_output_filepath}.")

    ##################################################

##################################################
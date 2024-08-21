# README
# Phillip Long
# July 23, 2024

# Analyze full dataset to see if there are differences between different facets.

# python /home/pnlong/model_musescore/make_dataset/ratings.py

# IMPORTS
##################################################

import argparse
import pandas as pd
from os.path import exists, dirname, realpath
from os import mkdir
import matplotlib.pyplot as plt
import logging

import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import MMT_STATISTIC_COLUMNS, OUTPUT_DIR, DATASET_DIR_NAME
from quality import discretize_rating, group_by, RATING_ROUND_TO_THE_NEAREST, PLOTS_DIR_NAME

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

GREY = "#7b7d7b"
LIGHT_GREY = "#bfbfbf"

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Ratings", description = "Create ratings plot for paper.")
    parser.add_argument("-d", "--dataset_filepath", type = str, default = f"{OUTPUT_DIR}/{DATASET_DIR_NAME}.csv", help = "Filepath to full dataset.")
    parser.add_argument("-c", "--column", action = "store_true", help = "Whether plot is a column or a row.")
    parser.add_argument("-eb", "--error_bars", action = "store_true", help = "Whether to add error bars.")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # LOAD DATASET, ARRANGE
    ##################################################

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    # deal with ratings column
    dataset["rating"] = list(map(discretize_rating, dataset["rating"]))

    ##################################################


    # PRINT DATA TABLES
    ##################################################

    # group datasets by arguments
    df = group_by(df = dataset, by = "rating")
    df = df.sort_index(ascending = True)
    logging.info(df.to_string()) # print data frame
    df = df.drop(index = [0.0])

    ##################################################


    # MAKE PLOT
    ##################################################

    # create plot
    mosaic = list(zip(MMT_STATISTIC_COLUMNS)) if args.column else [MMT_STATISTIC_COLUMNS]
    figsize = (4, 6) if args.column else (8, 2.5)
    fig, axes = plt.subplot_mosaic(mosaic = mosaic, constrained_layout = True, figsize = figsize)
    plt.set_loglevel("WARNING")

    # get current data frame
    width_proportion = 0.92 # proportion of 0.1 each bar is wide
    margin_proportion = 0.05 # what fraction of the range do we extend on both sides
    x_values = df.index.to_list()

    # make plots
    for mmt_statistic_column in MMT_STATISTIC_COLUMNS[::-1]:

        # variables
        statistic_fancy = " ".join(mmt_statistic_column.split("_")).title() # stylize the name of the mmt statistic

        # little bit of data wrangling
        data_mmt_statistic = df[mmt_statistic_column]
        # data_mmt_statistic = data_mmt_statistic[~pd.isna(data_mmt_statistic["sem"])] # no na values

        # plot
        axes[mmt_statistic_column].bar(x = x_values, height = data_mmt_statistic["mean"], width = RATING_ROUND_TO_THE_NEAREST * width_proportion, align = "center", color = LIGHT_GREY if args.error_bars else GREY)
        if args.error_bars:
            axes[mmt_statistic_column].errorbar(x = x_values, y = data_mmt_statistic["mean"], yerr = data_mmt_statistic["sem"], fmt = "o", color = GREY)            

        # y and x axis labels
        if (not args.column) or (args.column and (mmt_statistic_column == MMT_STATISTIC_COLUMNS[-1])):
            axes[mmt_statistic_column].set_xlabel("Rating")
        elif args.column and (mmt_statistic_column != MMT_STATISTIC_COLUMNS[-1]):
            axes[mmt_statistic_column].sharex(other = axes[MMT_STATISTIC_COLUMNS[-1]])
            # axes[mmt_statistic_column].set_xticklabels([])
        axes[mmt_statistic_column].set_ylabel(statistic_fancy)
        # axes[mmt_statistic_column].grid()

        # set range of x axis to avoid weird outliers
        axes[mmt_statistic_column].set_xlim(left = 3.0 - (0.5 * RATING_ROUND_TO_THE_NEAREST), right = 5.0 + (0.5 * RATING_ROUND_TO_THE_NEAREST))

        # improve range of y axis
        if args.error_bars:
            min_val, max_val = min(data_mmt_statistic["mean"] - data_mmt_statistic["sem"]), max(data_mmt_statistic["mean"] + data_mmt_statistic["sem"])
        else:
            min_val, max_val = min(data_mmt_statistic["mean"]), max(data_mmt_statistic["mean"])
        margin = margin_proportion * (max_val - min_val)
        axes[mmt_statistic_column].set_ylim(bottom = min_val - margin, top = max_val + margin)

    # save image
    plots_dir = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}"
    output_filepath = f"{plots_dir}/rating.pdf" # get output filepath
    if not exists(plots_dir): # make sure output directory exists
        mkdir(plots_dir)
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath}.")

    ##################################################

##################################################

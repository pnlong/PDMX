# README
# Phillip Long
# July 23, 2024

# Analyze full dataset to see if there are differences between different facets.

# python /home/pnlong/model_musescore/wrangling/rating.py

# IMPORTS
##################################################

import argparse
import pandas as pd
import numpy as np
from os.path import exists, dirname
from os import mkdir
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import MMT_STATISTIC_COLUMNS, OUTPUT_DIR, DATASET_DIR_NAME
from quality import discretize_rating, group_by, PLOTS_DIR_NAME
from genres import FACETS_FOR_PLOTTING

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

GREY = "#7b7d7b"
LIGHT_GREY = "#bfbfbf"
COLORS = list(TABLEAU_COLORS.keys())[(len(FACETS_FOR_PLOTTING)):]

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Ratings", description = "Create ratings plot for paper.")
    parser.add_argument("-df", "--dataset_filepath", type = str, default = f"{OUTPUT_DIR}/{DATASET_DIR_NAME}.csv", help = "Filepath to full dataset.")
    parser.add_argument("-c", "--column", action = "store_true", help = "Whether plot is a column or a row.")
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

    # wrangle columns
    dataset["rating"] = list(map(discretize_rating, dataset["rating"]))
    for mmt_statistic_column in MMT_STATISTIC_COLUMNS[1:]:
        dataset[mmt_statistic_column] *= 100 # convert consistency columns to percentages

    ##################################################


    # GET PERCENTILES
    ##################################################

    # group datasets by arguments
    count_by_rating = group_by(df = dataset, by = "rating")
    count_by_rating = count_by_rating.sort_index(ascending = True)
    logging.info(count_by_rating.to_string()) # print data frame
    count_by_rating = count_by_rating.drop(index = [0.0])

    # get only rated songs
    n_total_songs = len(dataset)
    dataset = dataset[dataset["rating"] > 0][MMT_STATISTIC_COLUMNS + ["rating"]].reset_index(drop = True) # get rid of non-rated songs
    n_unrated = n_total_songs - len(dataset)
    logging.info(f"{n_unrated:,} ({100 * (n_unrated / n_total_songs):.2f}%) unrated songs. Therefore, there are {len(dataset):,} ({100 * (len(dataset) / n_total_songs):.2f}%) rated songs.")

    # map each discrete rating to a percentile
    rating_percentiles = (100 * (np.cumsum(a = count_by_rating["n"], axis = 0) / len(dataset))).to_dict()
    dataset["percentile"] = dataset["rating"].replace(to_replace = rating_percentiles)
    dataset = dataset.sort_values(by = "percentile", axis = 0, ascending = True, ignore_index = True)

    # group by percentile to get one point per percentile (for a line plot)
    df = dataset.groupby(by = "percentile").mean()

    ##################################################


    # MAKE PLOT
    ##################################################

    # create plot
    column_subplot_proportions = (1, 1, 1) # (3, 2, 2)
    mosaic = list(zip(sum([[mmt_statistic_column] * column_subplot_proportion for mmt_statistic_column, column_subplot_proportion in zip(MMT_STATISTIC_COLUMNS, column_subplot_proportions)], []))) if args.column else [MMT_STATISTIC_COLUMNS]
    figsize = (4, 2.85) if args.column else (8, 2.5)
    fig, axes = plt.subplot_mosaic(mosaic = mosaic, constrained_layout = True, figsize = figsize)
    plt.set_loglevel("WARNING")
    axis_tick_fontsize = "x-small"
    axes_label_fontsize = "small"
    margin_proportions = (0.1, 0.07, 0.04)
    ratings_for_plot_step = 0.1
    rating_labels_for_plot_rotation = 30
    major_ratings_for_plot_step = 0.25
    major_ratings_linewidth = 0.5

    # get x values
    x_values = df.index.to_list()

    # get a discrete list of ratings, and the percentiles at which they fall
    ratings_for_plot = [4.0] + np.arange(start = 4.5, stop = 5.0 + ratings_for_plot_step, step = ratings_for_plot_step).tolist()
    calculate_percentile_given_rating = lambda rating: 100 * (dataset["rating"] <= rating).mean()
    rating_ticks_for_plot = list(map(calculate_percentile_given_rating, ratings_for_plot))
    rating_labels_for_plot = list(map(lambda rating_for_plot: f"{rating_for_plot:.1f}\u2605", ratings_for_plot))
    # major_ratings_for_plot = ratings_for_plot # np.arange(start = 4.0, stop = 5.0, step = major_ratings_for_plot_step)
    major_rating_ticks_for_plot = rating_ticks_for_plot

    # make plots
    for i in np.arange(start = len(MMT_STATISTIC_COLUMNS) - 1, stop = -1, step = -1):

        # set tick label parameters
        mmt_statistic_column = MMT_STATISTIC_COLUMNS[i]
        axes[mmt_statistic_column].tick_params(axis = "both", which = "major", labelsize = axis_tick_fontsize) # set tick label font size

        # get y limits
        y_values = df[mmt_statistic_column].to_list()
        min_y_val, max_y_val, mean_y_val = min(y_values), max(y_values), (sum(y_values) / len(y_values))
        ylim_bottom, ylim_top = (1 - margin_proportions[i]) * mean_y_val, (1 + margin_proportions[i]) * mean_y_val

        # plot
        axes[mmt_statistic_column].vlines(x = major_rating_ticks_for_plot, ymin = ylim_bottom, ymax = ylim_top, linestyle = "solid", linewidth = major_ratings_linewidth, color = GREY)
        axes[mmt_statistic_column].plot(x_values, df[mmt_statistic_column], linestyle = "solid", color = COLORS[i])
        
        # y and x axis labels
        if (not args.column) or (args.column and (mmt_statistic_column == MMT_STATISTIC_COLUMNS[-1])):
            rating_axis = axes[mmt_statistic_column].secondary_xaxis(location = -0.235)
            rating_axis.tick_params(axis = "x", length = 0)
            rating_axis.spines["bottom"].set_linewidth(0)
            rating_axis.set_xticks(ticks = rating_ticks_for_plot, labels = rating_labels_for_plot, fontsize = axis_tick_fontsize, rotation = rating_labels_for_plot_rotation)
            rating_axis.set_xlabel("Rating Percentile (%)", fontsize = axes_label_fontsize)
        elif args.column and (mmt_statistic_column != MMT_STATISTIC_COLUMNS[-1]):
            # axes[mmt_statistic_column].sharex(other = axes[MMT_STATISTIC_COLUMNS[-1]])
            axes[mmt_statistic_column].set_xticklabels([])
        axes[mmt_statistic_column].set_ylabel("".join(map(lambda word: word[0], mmt_statistic_column.split("_"))).upper() + (" (%)" if ("consistency" in mmt_statistic_column) else ""), fontsize = axes_label_fontsize)
        # axes[mmt_statistic_column].grid()

        # set range of x axis to avoid weird outliers
        axes[mmt_statistic_column].set_xlim(left = 0, right = 100)

        # improve y axis range
        
        axes[mmt_statistic_column].set_ylim(bottom = ylim_bottom, top = ylim_top)

    # save image
    plots_dir = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}"
    output_filepath = f"{plots_dir}/rating.pdf" # get output filepath
    if not exists(plots_dir): # make sure output directory exists
        mkdir(plots_dir)
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath}.")

    ##################################################

##################################################

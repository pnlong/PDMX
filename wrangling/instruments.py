# README
# Phillip Long
# August 11, 2024

# Create instrument distribution plot for paper.

# python /home/pnlong/model_musescore/wrangling/instruments.py

# IMPORTS
##################################################

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging
from os.path import dirname, exists
from os import mkdir
import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import OUTPUT_DIR, DATASET_DIR_NAME
from quality import PLOTS_DIR_NAME, make_facet_name_fancy
from genres import TOP_N, FACETS_FOR_PLOTTING, FACET_COLORS

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################


##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Instruments", description = "Create instruments plot for paper.")
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
    bar_width = 100

    ##################################################


    # LOAD IN DATASET, GET INSTRUMENTATION INFORMATION
    ##################################################

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset[["tracks"] + list(map(lambda facet: f"facet:{facet}", FACETS_FOR_PLOTTING))] # extract only necessary columns

    # get the top-n most common instrumentations in entire dataset
    def get_counts(df: pd.DataFrame = dataset):
        """Get the counts of each instrumentation."""
        instrumentations = df.loc[~pd.isna(df["tracks"]), "tracks"].values.tolist()
        counts = {instrumentation: instrumentations.count(instrumentation) for instrumentation in set(instrumentations)} # count each instrumentation
        instrumentations = sorted(list(counts.keys()), key = lambda instrumentation: counts[instrumentation])[::-1] # get instrumentations ordered from most common to least
        counts = {instrumentation: counts[instrumentation] for instrumentation in instrumentations}
        return counts
    convert_counts = lambda counts: {instrumentation: 100 * (counts[instrumentation] / sum(counts.values())) for instrumentation in counts.keys()} # convert counts to a percentage
    data = dict()
    counts = get_counts(df = dataset)
    logging.info(f"{len(counts):,} distinct instrumentations.")
    instrumentations = list(counts.keys())[:TOP_N] # get the instrumentations from most common to n-th most common
    counts = convert_counts(counts = {instrumentation: counts[instrumentation] for instrumentation in instrumentations}) # get top n results
    data[FACETS_FOR_PLOTTING[0]] = list(counts.values())

    # for other facets, get the fraction of each instrumentation
    for facet in FACETS_FOR_PLOTTING[1:]:
        counts = convert_counts(counts = get_counts(df = dataset[dataset[f"facet:{facet}"]]))
        data[facet] = list(map(lambda instrumentation: counts[instrumentation] if (instrumentation in counts.keys()) else 0.0, instrumentations))

    # wrangle instrumentation names
    fancy_instrumentation_name = {
        "0": "Piano",
        "0-0": "Two Pianos",
        "0-91": "Piano-Voice",
        "52-52-52-52": "SATB Choir",
        "0-0-0-0": "Four Pianos",
        "0-0-0": "Three Pianos",
        "40": "Violin",
        "52-52-52-52-52": "Five-Part Choral",
        "0-68": "Piano-Oboe",
        "19": "Organ",
    }
    instrumentations = list(map(lambda instrumentation: fancy_instrumentation_name[instrumentation], instrumentations)) # specific case

    ##################################################


    # PLOT
    ##################################################

    # create figure
    figsize = (5, 2.7) if args.column else (8, 2.2)
    fig, axes = plt.subplot_mosaic(mosaic = [["instrumentations"]], constrained_layout = True, figsize = figsize)
    xlabel, ylabel = "Instrumentation", "Percent of Songs (%)"
    xaxis_tick_label_rotation = 0

    # plot hyperparameters
    axis_tick_fontsize = "small"
    legend_fontsize = "x-small"
    total_width = 0.8
    width = total_width / len(FACETS_FOR_PLOTTING)
    offset = np.arange(start = 0.5 * (width - total_width), stop = 0.5 * total_width , step = width) # offsets
    xticks = np.arange(len(instrumentations))
    yticks = 10 ** np.arange(start = 0, stop = 3, step = 1)
    bar_edgecolor = "0.2"
    bar_edgewidth = 0.45

    # plot by facet
    for i, facet in enumerate(FACETS_FOR_PLOTTING):
        if args.column:
            axes["instrumentations"].barh(y = xticks + offset[i], width = data[facet], height = width, align = "center", log = True, label = facet, color = FACET_COLORS[facet], edgecolor = bar_edgecolor, linewidth = bar_edgewidth)
        else:
            axes["instrumentations"].bar(x = xticks + offset[i], height = data[facet], width = width, align = "center", log = True, label = facet, color = FACET_COLORS[facet], edgecolor = bar_edgecolor, linewidth = bar_edgewidth)
    if args.column:
        # axes["instrumentations"].set_ylabel(xlabel, fontsize = axis_tick_fontsize)
        axes["instrumentations"].set_yticks(ticks = xticks, labels = instrumentations, fontsize = axis_tick_fontsize, rotation = xaxis_tick_label_rotation) # get instrumentation names
        axes["instrumentations"].invert_yaxis() # so most common instrumentations are on top
        axes["instrumentations"].set_xlabel(ylabel, fontsize = axis_tick_fontsize)
        # axes["instrumentations"].xaxis.grid(True)
        axes["instrumentations"].set_xticks(ticks = yticks, labels = yticks, fontsize = axis_tick_fontsize)
    else:
        # axes["instrumentations"].set_xlabel(xlabel, fontsize = axis_tick_fontsize)
        axes["instrumentations"].set_xticks(ticks = xticks, labels = instrumentations, fontsize = axis_tick_fontsize, rotation = xaxis_tick_label_rotation) # get instrumentation names
        axes["instrumentations"].set_ylabel(ylabel, fontsize = axis_tick_fontsize)
        # axes["instrumentations"].yaxis.grid(True)
        axes["instrumentations"].set_yticks(ticks = yticks, labels = yticks, fontsize = axis_tick_fontsize)

    # add legend
    handles, labels = axes["instrumentations"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["instrumentations"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          fontsize = legend_fontsize, alignment = "center",
                          ncol = 1, title = "Subset", title_fontproperties = {"size": legend_fontsize, "weight": "bold"},
                          fancybox = True, shadow = True)

    # save image
    output_filepath = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}/instruments.pdf" # get output filepath
    if not exists(dirname(output_filepath)): # make sure output directory exists
        mkdir(dirname(output_filepath))
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight") # save image
    logging.info(f"Instruments plot saved to {output_filepath}.")

    ##################################################

##################################################
# README
# Phillip Long
# August 11, 2024

# Create genres plot for paper.

# python /home/pnlong/model_musescore/dataset_full_genres.py

# IMPORTS
##################################################

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging
from os.path import dirname, exists
from os import mkdir
import numpy as np

import dataset_full
import dataset_full_analysis
from dataset_deduplicate import FACETS
import remi_evaluate_analysis
from pdmx import DATASET_NAME

plt.style.use("default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

# display the top `n` genres
TOP_N = 10

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Genres", description = "Create genres plot for paper.")
    parser.add_argument("-d", "--dataset_filepath", type = str, default = f"{dataset_full.OUTPUT_DIR}/{dataset_full.DATASET_DIR_NAME}.csv", help = "Filepath to full dataset.")
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


    # LOAD IN DATASET, GET GENRES INFORMATION
    ##################################################

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    # wrangle genres column
    dataset = dataset[["genres"] + list(map(lambda facet: f"facet:{facet}", FACETS))] # extract only necessary columns
    dataset["genres"] = list(map(lambda genres_string: genres_string.split(dataset_full.LIST_FEATURE_JOIN_STRING)[0] if not pd.isna(genres_string) else None, dataset["genres"]))

    # get some statistics
    logging.info("Percent of songs with at least one genre, by facet:")
    for facet in FACETS:
        logging.info(f"- {remi_evaluate_analysis.make_facet_name_fancy(facet = facet)}: {100 * (sum((~pd.isna(dataset['genres'])) * dataset[f'facet:{facet}']) / sum(dataset[f'facet:{facet}'])):.2f}%")

    # get the top-n most common genres in entire dataset
    convert_counts = lambda counts: 100 * (counts / sum(counts)) # convert counts to a percentage
    data = dict()
    counts = dataset["genres"].value_counts(sort = True, ascending = False, dropna = True)
    logging.info(f"{len(counts)} distinct genres.")
    counts = convert_counts(counts = counts.head(n = TOP_N)) # get top n results
    genres = list(counts.index) # get the genres from most common to n-th most common
    data[FACETS[0]] = list(counts.values)

    # for other facets, get the fraction of each genre
    for facet in FACETS[1:]:
        counts = convert_counts(counts = dataset[dataset[f"facet:{facet}"]]["genres"].value_counts())
        data[facet] = list(map(lambda genre: counts[genre] if (genre in counts.index) else 0.0, genres))

    # wrangle genre names
    genres = list(map(lambda genre: genre.replace("music", "").title(), genres))

    ##################################################


    # PLOT
    ##################################################

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["genres"]], constrained_layout = True, figsize = (8, 2.5))

    # plot by facet
    axis_tick_fontsize = "small"
    total_width = 0.8
    width = total_width / len(FACETS)
    offset = np.arange(start = 0.5 * (width - total_width), stop = 0.5 * total_width , step = width) # offsets
    xticks = np.arange(len(genres))
    yticks = 10 ** np.arange(start = 0, stop = 3, step = 1)
    for i, facet in enumerate(FACETS):
        axes["genres"].bar(x = xticks + offset[i], height = data[facet], width = width, align = "center", log = True, label = facet)
    axes["genres"].set_xlabel(f"Top {TOP_N} Genres")
    axes["genres"].set_xticks(ticks = xticks, labels = list(map(lambda i: genres[i], xticks)), fontsize = axis_tick_fontsize, rotation = 0) # get genre names
    axes["genres"].set_ylabel("Percent of Songs (%)")
    # axes["genres"].yaxis.grid(True)
    axes["genres"].set_yticks(ticks = yticks, labels = yticks, fontsize = axis_tick_fontsize)

    # add legend
    handles, labels = axes["genres"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["genres"].legend(handles = by_label.values(), labels = list(map(remi_evaluate_analysis.make_facet_name_fancy, by_label.keys())),
                          fontsize = "small", title_fontsize = "medium", alignment = "center",
                          ncol = 1, title = "Subset")

    # save image
    output_filepath = f"{dirname(args.dataset_filepath)}/{dataset_full_analysis.PLOTS_DIR_NAME}/genres.pdf" # get output filepath
    if not exists(dirname(output_filepath)): # make sure output directory exists
        mkdir(dirname(output_filepath))
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight") # save image
    logging.info(f"Genres plot saved to {output_filepath}.")

    ##################################################

##################################################
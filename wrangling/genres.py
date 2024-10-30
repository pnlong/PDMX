# README
# Phillip Long
# August 11, 2024

# Create genres plot for paper.

# python /home/pnlong/model_musescore/wrangling/genres.py

# IMPORTS
##################################################

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import logging
from os.path import dirname, exists
from os import mkdir
import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from full import OUTPUT_DIR, DATASET_DIR_NAME, LIST_FEATURE_JOIN_STRING
from quality import PLOTS_DIR_NAME, make_facet_name_fancy
from deduplicate import FACETS

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

# display the top `n` genres
TOP_N = 10

# facets for plotting, as the order is different in paper
FACETS_FOR_PLOTTING = sorted(FACETS.copy())

# colors for plotting
FACET_COLORS = dict(zip(FACETS_FOR_PLOTTING, list(TABLEAU_COLORS.keys())[:len(FACETS_FOR_PLOTTING)]))

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Genres", description = "Create genres plot for paper.")
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


    # LOAD IN DATASET, GET GENRES INFORMATION
    ##################################################

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset[["genres"] + list(map(lambda facet: f"facet:{facet}", FACETS_FOR_PLOTTING))] # extract only necessary columns

    # get some statistics
    logging.info("Percent of songs with at least one genre, by facet:")
    for facet in FACETS_FOR_PLOTTING:
        logging.info(f"- {make_facet_name_fancy(facet = facet)}: {100 * (sum((~pd.isna(dataset['genres'])) * dataset[f'facet:{facet}']) / sum(dataset[f'facet:{facet}'])):.2f}%")
        
    # # get the top-n most common genres in entire dataset
    # dataset["genres"] = list(map(lambda genres_string: genres_string.split(LIST_FEATURE_JOIN_STRING)[0] if not pd.isna(genres_string) else None, dataset["genres"]))
    # convert_counts = lambda counts: 100 * (counts / sum(counts)) # convert counts to a percentage
    # data = dict()
    # counts = dataset["genres"].value_counts(sort = True, ascending = False, dropna = True)
    # logging.info(f"{len(counts)} distinct genres.")
    # counts = convert_counts(counts = counts.head(n = TOP_N)) # get top n results
    # genres = list(counts.index) # get the genres from most common to n-th most common
    # data[FACETS_FOR_PLOTTING[0]] = list(counts.values)

    # # for other facets, get the fraction of each genre
    # for facet in FACETS_FOR_PLOTTING[1:]:
    #     counts = convert_counts(counts = dataset[dataset[f"facet:{facet}"]]["genres"].value_counts())
    #     data[facet] = list(map(lambda genre: counts[genre] if (genre in counts.index) else 0.0, genres))

    # get the top-n most common genres in entire dataset
    def get_counts(df: pd.DataFrame = dataset):
        """Get the counts of each genre."""
        genres = sum(list(map(lambda genre: genre.split(LIST_FEATURE_JOIN_STRING), df.loc[~pd.isna(df["genres"]), "genres"].values)), []) # get all genres
        counts = {genre: genres.count(genre) for genre in set(genres)} # count each genre
        genres = sorted(list(counts.keys()), key = lambda genre: counts[genre])[::-1] # get genres ordered from most common to least
        counts = {genre: counts[genre] for genre in genres}
        return counts
    convert_counts = lambda counts: {genre: 100 * (counts[genre] / sum(counts.values())) for genre in counts.keys()} # convert counts to a percentage
    data = dict()
    counts = get_counts(df = dataset)
    logging.info(f"{len(counts):,} distinct genres.")
    genres = list(counts.keys())[:TOP_N] # get the genres from most common to n-th most common
    counts = convert_counts(counts = {genre: counts[genre] for genre in genres}) # get top n results
    data[FACETS_FOR_PLOTTING[0]] = list(counts.values())

    # for other facets, get the fraction of each genre
    for facet in FACETS_FOR_PLOTTING[1:]:
        counts = convert_counts(counts = get_counts(df = dataset[dataset[f"facet:{facet}"]]))
        data[facet] = list(map(lambda genre: counts[genre] if (genre in counts.keys()) else 0.0, genres))

    # wrangle genre names
    genres = list(map(lambda genre: "Funk/Soul" if genre == "rbfunksoul" else genre, genres)) # specific case
    genres = list(map(lambda genre: genre.replace("music", "").title(), genres))

    ##################################################


    # PLOT
    ##################################################

    # create figure
    figsize = (4, 2.7) if args.column else (8, 2.2)
    fig, axes = plt.subplot_mosaic(mosaic = [["genres"]], constrained_layout = True, figsize = figsize)
    xlabel, ylabel = "Genre", "Percent of Songs (%)"
    xaxis_tick_label_rotation = 0

    # plot hyperparameters
    axis_tick_fontsize = "small"
    legend_fontsize = "x-small"
    total_width = 0.8
    width = total_width / len(FACETS_FOR_PLOTTING)
    offset = np.arange(start = 0.5 * (width - total_width), stop = 0.5 * total_width , step = width) # offsets
    xticks = np.arange(len(genres))
    yticks = 10 ** np.arange(start = 0, stop = 3, step = 1)
    bar_edgecolor = "0.2"
    bar_edgewidth = 0.45

    # plot by facet
    for i, facet in enumerate(FACETS_FOR_PLOTTING):
        if args.column:
            axes["genres"].barh(y = xticks + offset[i], width = data[facet], height = width, align = "center", log = True, label = facet, color = FACET_COLORS[facet], edgecolor = bar_edgecolor, linewidth = bar_edgewidth)
        else:
            axes["genres"].bar(x = xticks + offset[i], height = data[facet], width = width, align = "center", log = True, label = facet, color = FACET_COLORS[facet], edgecolor = bar_edgecolor, linewidth = bar_edgewidth)
    if args.column:
        # axes["genres"].set_ylabel(xlabel, fontsize = axis_tick_fontsize)
        axes["genres"].set_yticks(ticks = xticks, labels = genres, fontsize = axis_tick_fontsize, rotation = xaxis_tick_label_rotation) # get genre names
        axes["genres"].invert_yaxis() # so most common genres are on top
        axes["genres"].set_xlabel(ylabel, fontsize = axis_tick_fontsize)
        # axes["genres"].xaxis.grid(True)
        axes["genres"].set_xticks(ticks = yticks, labels = yticks, fontsize = axis_tick_fontsize)
    else:
        # axes["genres"].set_xlabel(xlabel, fontsize = axis_tick_fontsize)
        axes["genres"].set_xticks(ticks = xticks, labels = genres, fontsize = axis_tick_fontsize, rotation = xaxis_tick_label_rotation) # get genre names
        axes["genres"].set_ylabel(ylabel, fontsize = axis_tick_fontsize)
        # axes["genres"].yaxis.grid(True)
        axes["genres"].set_yticks(ticks = yticks, labels = yticks, fontsize = axis_tick_fontsize)

    # add legend
    handles, labels = axes["genres"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["genres"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          fontsize = legend_fontsize, alignment = "center",
                          ncol = 1, title = "Subset", title_fontproperties = {"size": legend_fontsize, "weight": "bold"},
                          fancybox = True, shadow = True)

    # save image
    output_filepath = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}/genres.pdf" # get output filepath
    if not exists(dirname(output_filepath)): # make sure output directory exists
        mkdir(dirname(output_filepath))
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight") # save image
    logging.info(f"Genres plot saved to {output_filepath}.")

    ##################################################

##################################################
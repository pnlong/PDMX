# README
# Phillip Long
# July 23, 2024

# Analyze full dataset to see if there are differences between different facets.

# python /home/pnlong/model_musescore/dataset_full_analysis.py


# IMPORTS
##################################################

import pandas as pd
from typing import Union, List
from utils import rep
from os.path import exists, dirname
from os import makedirs, getcwd
import matplotlib.pyplot as plt
import argparse

from dataset_full import MMT_STATISTIC_COLUMNS, OUTPUT_DIR

##################################################


# CONSTANTS
##################################################

STATISTICS_TO_CALCULATE = ["size", "mean", "std", "median", "max", "min"]
DEFAULT_STATISTIC = STATISTICS_TO_CALCULATE[1]

##################################################


# GROUP DATASET BY SOME FACET
##################################################

# group dataset by a certain column(s) to see differences in data quality
def group_dataset_by(by: Union[str, List[str]]) -> pd.DataFrame:
    """
    Function to help facilitate testing differences in data quality by various facets
    """

    # don't operate on the actual dataset
    df = dataset.copy()

    # only select relevant columns
    if isinstance(by, str):
        by = [by]
    df = df[by + MMT_STATISTIC_COLUMNS]

    # perform groupby
    agg_dict = dict(zip(MMT_STATISTIC_COLUMNS, rep(x = STATISTICS_TO_CALCULATE, times = len(MMT_STATISTIC_COLUMNS))))
    df = df.groupby(by = by).agg(agg_dict)

    # sort indicies
    df = df.sort_index(ascending = True)

    # remove nested parts from index
    if (len(by) > 1):
        by_string = ", ".join(by)
        df[by_string] = list(map(lambda *args: ", ".join((str(arg) for arg in args[0])), df.index))
        df = df.set_index(keys = by_string, drop = True)

    # return df
    return df

##################################################


# VISUALIZE GROUPING OF DATASET
##################################################

# given a dataframe of the dataset grouped by some facet, visualize it
def visualize_grouping(df: pd.DataFrame, statistic: str = DEFAULT_STATISTIC, output_filepath: str = None):
    """
    Given a dataframe that has been grouped by some facet, visualize it.
    """

    # determine string for how to refer the facet
    facet_name = df.index.name

    # infer output_filepath if necessary
    if (output_filepath is None):
        output_filepath = f"{getcwd()}/plots/{facet_name.replace(', ', '-')}.{statistic}.pdf"
    if (not exists(dirname(output_filepath))): # make sure output directory exists
        makedirs(dirname(output_filepath))

    # ensure statistic is valid
    if statistic not in STATISTICS_TO_CALCULATE:
        raise ValueError(f"Invalid `statistic` argument. Valid options include: [{', '.join(STATISTICS_TO_CALCULATE)}]")

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = [MMT_STATISTIC_COLUMNS], constrained_layout = True, figsize = (12, 4))
    fig.suptitle(statistic.title(), fontweight = "bold")

    # make plots
    for column in MMT_STATISTIC_COLUMNS:
        axes[column].barh(y = df.index, width = df[column][statistic])
        axes[column].set_yticks(sorted(pd.unique(df.index)))
        axes[column].set_ylabel(" ".join(facet_name.split("_")).title())
        axes[column].set_xlabel(" ".join(column.split("_")).title())
        axes[column].grid()

    # rotate y-axis ticks if necessary
    # if (facet_name.count(", ") > 0):
    #     for column in MMT_STATISTIC_COLUMNS:
    #         axes[column].set_yticks(axes[column].get_xticks())
    #         axes[column].set_yticklabels(axes[column].get_xticklabels(), rotation = 20, ha = "right")

    # save image
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight")
    print(f"Saved figure to {output_filepath}.")

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore", description = "Analyze full dataset for music-quality differences within variables.")
    parser.add_argument("-d", "--dataset_filepath", type = str, default = f"{OUTPUT_DIR}/dataset.full.csv", help = "Filepath to full dataset.")
    parser.add_argument("-s", "--statistic", type = str, default = DEFAULT_STATISTIC, help = "Relevant statistic to display in plots.")
    parser.add_argument("-b", "--by", action = "store", type = str, nargs = "+", help = "Variable(s) on which to facet.")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # get output directory
    output_dir = dirname(args.dataset_filepath)

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    # some filterings of the dataset
    dataset["rating"] = list(map(round, dataset["rating"]))

    # group dataset by arguments
    df = group_dataset_by(by = args.by)

    # visualize
    output_filepath = f"{output_dir}/plots/{df.index.name.replace(', ', '-')}.{args.statistic}.pdf"
    visualize_grouping(df = df, statistic = args.statistic, output_filepath = output_filepath)

##################################################

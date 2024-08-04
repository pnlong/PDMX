# README
# Phillip Long
# August 3, 2024

# Analyze the evaluation a REMI-Style model.

# python /home/pnlong/model_musescore/remi_evaluate_analysis.py

# IMPORTS
##################################################

import argparse
import logging
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

from dataset_full import MMT_STATISTIC_COLUMNS, CHUNK_SIZE
from remi_dataset import FACETS, OUTPUT_DIR
from remi_evaluate import OUTPUT_COLUMNS
import utils

##################################################


# CONSTANTS
##################################################

COLUMNS = ["facet"] + OUTPUT_COLUMNS

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Evaluate Analysis", description = "Analyze the evaluation a REMI-Style Model.")
    parser.add_argument("-i", "--input_dir", default = OUTPUT_DIR, type = str, help = "Directory containing facets (as subdirectories) to evaluate")
    parser.add_argument("-m", "--model", default = None, type = str, help = "Name of the model to evaluate for each different facet")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # create full dataset
    dataset = pd.DataFrame(columns = COLUMNS)
    for facet in FACETS:
        data = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/{facet}/evaluation.csv", sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False)
        data["facet"] = utils.rep(x = facet, times = len(data))
        dataset = pd.concat(objs = (dataset, data[COLUMNS]), axis = 0, ignore_index = True)
    del data

    # determine model to analyze; assumes the same models have been created for each facet
    models = set(pd.unique(values = dataset["model"]))
    model = (str(max(map(lambda model: int(model[:-1]), models))) + "M") if args.model is None else args.model
    if model not in models:
        raise RuntimeError(f"`{model}` is not a valid model.")

    ##################################################


    # PLOT
    ##################################################

    # create plot
    plot_to_legend_ratio = 3 # plots are `plot_to_legend_ratio` times taller than the legend
    fig, axes = plt.subplot_mosaic(mosaic = [MMT_STATISTIC_COLUMNS[:-1], [MMT_STATISTIC_COLUMNS[-1], "legend"]], constrained_layout = True, figsize = (6, 6))
    fig.suptitle(f"Evaluating {model} Model Performance", fontweight = "bold")

    # helper function to plot
    def plot_mmt_statistic(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        #

    # plot plots
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = pool.map(func = plot_mmt_statistic, iterable = MMT_STATISTIC_COLUMNS, chunksize = dataset_full.CHUNK_SIZE)

    # plot legend
    handles, labels = axes[MMT_STATISTIC_COLUMNS[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    make_facet_name_fancy = lambda facet: facet.title().replace("_", " and ")
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())), loc = "center", fontsize = "xlarge", title_fontsize = "xxlarge", alignment = "center", ncol = 1, title = "Facet", mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{args.input_dir}/model_comparison.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################

##################################################
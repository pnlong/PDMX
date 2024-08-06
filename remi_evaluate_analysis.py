# README
# Phillip Long
# August 3, 2024

# Analyze the evaluation a REMI-Style model.

# python /home/pnlong/model_musescore/remi_evaluate_analysis.py

# IMPORTS
##################################################

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataset_full import MMT_STATISTIC_COLUMNS
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
    plot_title = f"Evaluating {model} Model Performance"

    # remove part of dataset we don't need
    dataset = dataset[dataset["model"] == model]

    ##################################################


    # PLOTTING CONSTANTS
    ##################################################

    # plotting constants
    n_bins = 12
    range_multiplier_constant = 1.001
    plot_mosaic = [MMT_STATISTIC_COLUMNS[:-1], [MMT_STATISTIC_COLUMNS[-1], "legend"]]
    make_facet_name_fancy = lambda facet: facet.title().replace("_", " and ")
    legent_title = "Facet"
    x_axis_label = "Value"
    output_filepath_prefix = "model_comparison"
    legend_title_fontsize = "large"
    legend_fontsize = "medium"
    colors = ("black", "blue", "orange", "green") # FACETS = ["all", "rated", "deduplicated", "rated_deduplicated"]

    ##################################################


    # PLOT LINE PLOT
    ##################################################

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = plot_mosaic, constrained_layout = True, figsize = (8, 6))
    fig.suptitle(plot_title, fontweight = "bold")

    # helper function to plot
    def plot_mmt_statistic(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # left side will be fraction, right will be count
        count_axes = axes[mmt_statistic].twinx()

        # loop through facets
        for i, facet in enumerate(FACETS):
            
            # get histogram values
            data = dataset[dataset["facet"] == facet][mmt_statistic]
            min_data, max_data = min(data), max(data)
            data_range = max_data - min_data
            margin = ((range_multiplier_constant - 1) / 2) * data_range
            bin_width = (range_multiplier_constant * data_range) / n_bins
            bins = np.arange(start = min_data - margin, stop = max_data + margin + (bin_width / 2), step = bin_width)
            data, bins = np.histogram(a = data, bins = bins) # create histogram
            bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] # get centerpoints of each bin

            # plot
            axes[mmt_statistic].plot(bin_centers, data / sum(data), color = colors[i], label = facet) # fraction
            count_axes.plot(bin_centers, data, color = colors[i], label = facet) # count

        # axes labels and such
        axes[mmt_statistic].set_xlabel(x_axis_label)
        axes[mmt_statistic].set_ylabel("Fraction")
        count_axes.set_ylabel("Count")
        axes[mmt_statistic].set_title(mmt_statistic.replace("_", " ").title())
        axes[mmt_statistic].grid()

    # plot plots
    for mmt_statistic_column in MMT_STATISTIC_COLUMNS:
        plot_mmt_statistic(mmt_statistic = mmt_statistic_column)

    # plot legend
    handles, labels = axes[MMT_STATISTIC_COLUMNS[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          loc = "center", fontsize = legend_fontsize, title_fontsize = legend_title_fontsize, alignment = "center",
                          ncol = 1, title = legent_title, mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{args.input_dir}/{output_filepath_prefix}.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################


    # PLOT STACKED PLOT
    ##################################################

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = plot_mosaic, constrained_layout = True, figsize = (8, 8))
    fig.suptitle(plot_title, fontweight = "bold")

    # helper function to plot
    def plot_mmt_statistic_stacked(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # get the range of data
        min_data, max_data = dataset[mmt_statistic].min(), dataset[mmt_statistic].max()
        data_range = max_data - min_data
        margin = ((range_multiplier_constant - 1) / 2) * data_range
        bin_width = (range_multiplier_constant * data_range) / n_bins
        bins = np.arange(start = min_data - margin, stop = max_data + margin + (bin_width / 2), step = bin_width)
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] # get centerpoints of each bin
        
        # get histograms and convert to fraction
        data = np.array(list(map(lambda facet: np.histogram(a = dataset[dataset["facet"] == facet][mmt_statistic], bins = bins)[0], FACETS)))
        bin_sums = np.sum(a = data, axis = 0)
        bin_sums += (bin_sums == 0) # replace 0s with 1s to avoid divide by zero error
        data = data / bin_sums

        # loop through facets and plot
        for i, facet in enumerate(FACETS):
            axes[mmt_statistic].bar(bin_centers, data[i], width = bin_width, bottom = np.sum(a = data[:i, :], axis = 0), color = colors[i], label = facet)

        # axes labels and such
        axes[mmt_statistic].set_xlabel(x_axis_label)
        axes[mmt_statistic].set_xlim(left = bins[0], right = bins[-1])
        axes[mmt_statistic].set_ylabel("")
        axes[mmt_statistic].set_ylim(bottom = 0, top = 1)
        axes[mmt_statistic].set_title(mmt_statistic.replace("_", " ").title())
        axes[mmt_statistic].grid()

    # plot plots
    for mmt_statistic_column in MMT_STATISTIC_COLUMNS:
        plot_mmt_statistic_stacked(mmt_statistic = mmt_statistic_column)

    # plot legend
    handles, labels = axes[MMT_STATISTIC_COLUMNS[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          loc = "center", fontsize = legend_fontsize, title_fontsize = legend_title_fontsize, alignment = "center",
                          ncol = 1, title = legent_title, mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{args.input_dir}/{output_filepath_prefix}.stacked.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################

##################################################


# HELPER FUNCTIONS
##################################################

# given a generated codes path, convert into audio
def generated_to_audio(path: str, output_path: str = None) -> None:
    """
    Given the path to generated codes, convert those codes into audio.
    """

    # imports
    import remi_representation
    import utils

    # get variables
    encoding = remi_representation.get_encoding() # load the encoding
    vocabulary = utils.inverse_dict(remi_representation.Indexer(data = encoding["event_code_map"]).get_dict()) # for decoding

    # load codes
    codes = np.load(file = path)

    # convert codes to a music object
    music = remi_representation.decode(codes = codes, encoding = encoding, vocabulary = vocabulary) # convert to a MusicExpress object

    # output codes as audio
    if output_path is None:
        output_path = "/home/pnlong/model_musescore/" + path[len("/home/pnlong/musescore/remi/"):].replace("/", ".")
    music.write(path = output_path)
    print(f"Saved to {output_path}.")

##################################################

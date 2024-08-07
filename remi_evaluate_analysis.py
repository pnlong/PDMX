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

import dataset_full
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
    parser.add_argument("-d", "--dataset_filepath", default = f"{dataset_full.OUTPUT_DIR}/{dataset_full.DATASET_DIR_NAME}_full.csv", type = str, help = "Dataset from which facets are derived")
    parser.add_argument("-m", "--model", default = None, type = str, help = "Name of the model to evaluate for each different facet")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# HELPER FUNCTIONS
##################################################

# convert matrix of histograms (as rows) to fractions
def convert_to_fraction(data: np.array) -> np.array:
    """Helper function to convert histograms (as rows) to fractions of the sum of each column."""
    bin_sums = np.sum(a = data, axis = 0)
    bin_sums += (bin_sums == 0) # replace 0s with 1s to avoid divide by zero error
    data_matrix = data / bin_sums
    return data_matrix

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

    # remove part of dataset we don't need
    dataset = dataset[dataset["model"] == model]

    ##################################################


    # PLOTTING CONSTANTS
    ##################################################

    # plotting constants
    n_bins = 12
    range_multiplier_constant = 1.001
    make_facet_name_fancy = lambda facet: facet.title().replace("_", " and ")
    legend_title = "Facet"
    output_filepath_prefix = "model_comparison"
    legend_title_fontsize = "large"
    legend_fontsize = "medium"
    colors = ("black", "blue", "orange", "green") # FACETS = ["all", "rated", "deduplicated", "rated_deduplicated"]

    ##################################################


    # PLOT LINE PLOT
    ##################################################

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = [dataset_full.MMT_STATISTIC_COLUMNS[:-1], [dataset_full.MMT_STATISTIC_COLUMNS[-1], "legend"]], constrained_layout = True, figsize = (8, 6))
    fig.suptitle(f"Evaluating {model} Model Performance", fontweight = "bold")

    # plotting function
    def plot_mmt_statistic(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # left side will be fraction, right will be count
        count_axes = axes[mmt_statistic].twinx()

        # loop through facets
        for i, facet in enumerate(FACETS):
            
            # get histogram values
            data_values = dataset[dataset["facet"] == facet][mmt_statistic]
            min_data, max_data = min(data_values), max(data_values)
            data_range = max_data - min_data
            margin = ((range_multiplier_constant - 1) / 2) * data_range
            bin_width = (range_multiplier_constant * data_range) / n_bins
            bins = np.arange(start = min_data - margin, stop = max_data + margin + (bin_width / 2), step = bin_width)
            data, bins = np.histogram(a = data_values, bins = bins) # create histogram
            bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] # get centerpoints of each bin

            # plot
            axes[mmt_statistic].plot(bin_centers, data / sum(data), color = colors[i], label = facet) # fraction
            count_axes.plot(bin_centers, data, color = colors[i], label = facet) # count

        # axes labels and such
        axes[mmt_statistic].set_xlabel("Value")
        axes[mmt_statistic].set_ylabel("Fraction")
        count_axes.set_ylabel("Count")
        axes[mmt_statistic].set_title(mmt_statistic.replace("_", " ").title())
        axes[mmt_statistic].grid()

    # plot plots
    for mmt_statistic_column in dataset_full.MMT_STATISTIC_COLUMNS:
        plot_mmt_statistic(mmt_statistic = mmt_statistic_column)

    # plot legend
    handles, labels = axes[dataset_full.MMT_STATISTIC_COLUMNS[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          loc = "center", fontsize = legend_fontsize, title_fontsize = legend_title_fontsize, alignment = "center",
                          ncol = 1, title = legend_title, mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{args.input_dir}/{output_filepath_prefix}.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################


    # PLOT STACKED PLOT
    ##################################################

    # load in dataset
    dataset_real = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    dataset_real = dataset_real.merge(
        right = pd.read_csv(
            filepath_or_buffer = f"{args.dataset_filepath[:-len('_full.csv')]}_deduplicated.csv", # add deduplication information to dataset
            sep = ",",
            header = 0,
            index_col = False),
        how = "inner",
        on = "path"
    )

    # create plot
    realness_names = ["actual", "generated"]
    plot_to_legend_ratio = 3
    fig, axes = plt.subplot_mosaic(
        mosaic = (
            utils.rep(x = list(map(lambda mmt_statistic: f"{realness_names[0]}.{mmt_statistic}", dataset_full.MMT_STATISTIC_COLUMNS)), times = plot_to_legend_ratio) +
            utils.rep(x = list(map(lambda mmt_statistic: f"{realness_names[1]}.{mmt_statistic}", dataset_full.MMT_STATISTIC_COLUMNS)), times = plot_to_legend_ratio) +
            [utils.rep(x = "legend", times = len(dataset_full.MMT_STATISTIC_COLUMNS))]
        ),
        constrained_layout = True, figsize = (10, 7))
    fig.suptitle(f"Comparing Facets in Actual versus Generated Music")

    # plotting function
    def plot_mmt_statistic_stacked(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # get the range of data
        data_values = pd.concat(objs = (dataset[mmt_statistic], dataset_real[mmt_statistic]), axis = 0)
        min_data, max_data = min(data_values), max(data_values)
        data_range = max_data - min_data
        margin = ((range_multiplier_constant - 1) / 2) * data_range
        bin_width = (range_multiplier_constant * data_range) / n_bins
        bins = np.arange(start = min_data - margin, stop = max_data + margin + (bin_width / 2), step = bin_width)
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] # get centerpoints of each bin
        
        # get histograms and convert to fraction
        def get_facet_values_for_real_dataset(facet: str) -> np.array:
            """Helper function for getting data values for a facet of the real dataset."""
            data = dataset_real
            if "rate" in facet:
                data = data[data["rating"] > 0]
            if "deduplicate" in facet:
                data = data[data["is_best_unique_arrangement"]]
            return data[mmt_statistic] # return the data for tbe MMT statistic
        data = {
            realness_names[0]: np.array(list(map(lambda facet: np.histogram(a = get_facet_values_for_real_dataset(facet = facet), bins = bins)[0], FACETS))), # real
            realness_names[1]: np.array(list(map(lambda facet: np.histogram(a = dataset[dataset["facet"] == facet][mmt_statistic], bins = bins)[0], FACETS))), # generated
        }
        data = {realness_name: data_values / np.sum(a = data_values, axis = 1).reshape(-1, 1) for realness_name, data_values in data.items()} # normalize data such that it's like every facet has the same number of songs
        data = {realness_name: convert_to_fraction(data = data_values) for realness_name, data_values in data.items()} # convert to fraction

        # loop through facets and plot
        axes_names = list(map(lambda realness_name: f"{realness_name}.{mmt_statistic}", realness_names))
        for realness_name, axes_name in zip(realness_names, axes_names):
            for i, facet in enumerate(FACETS):
                axes[axes_name].bar(bin_centers, data[realness_name][i], width = bin_width, bottom = np.sum(a = data[realness_name][:i, :], axis = 0), color = colors[i], label = facet)
            axes[axes_name].set_xlabel(mmt_statistic.replace("_", " ").title())
            axes[axes_name].set_xlim(left = bins[0], right = bins[-1])
            axes[axes_name].set_ylabel("")
            axes[axes_name].set_ylim(bottom = 0, top = 1)
            axes[axes_name].grid()

        # add title if needed
        if mmt_statistic == dataset_full.MMT_STATISTIC_COLUMNS[1]:
            axes[axes_names[0]].set_title("\nActual Data\n", fontweight = "bold")
            axes[axes_names[1]].set_title(f"\nGenerated by {model} Model\n", fontweight = "bold")

    # plot plots
    for mmt_statistic in dataset_full.MMT_STATISTIC_COLUMNS:
        plot_mmt_statistic_stacked(mmt_statistic = mmt_statistic)

    # plot legend
    handles, labels = axes[f"{realness_names[0]}.{dataset_full.MMT_STATISTIC_COLUMNS[0]}"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          loc = "center", fontsize = legend_fontsize, title_fontsize = legend_title_fontsize, alignment = "center",
                          ncol = len(FACETS), title = legend_title, mode = "expand")
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
    import numpy as np

    # get variables
    encoding = remi_representation.get_encoding() # load the encoding
    vocabulary = utils.inverse_dict(remi_representation.Indexer(data = encoding["event_code_map"]).get_dict()) # for decoding

    # load codes
    codes = np.load(file = path)

    # convert codes to a music object
    music = remi_representation.decode(codes = codes, encoding = encoding, vocabulary = vocabulary) # convert to a MusicExpress object

    # output codes as audio
    if output_path is None:
        output_path = "/home/pnlong/model_musescore/" + path[len("/home/pnlong/musescore/remi/"):-len(".npy")].replace("/", ".") + ".wav"
    music.write(path = output_path)
    print(f"Saved to {output_path}.")

##################################################

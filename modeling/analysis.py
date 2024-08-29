# README
# Phillip Long
# August 3, 2024

# Analyze the evaluation a REMI-Style model.

# python /home/pnlong/model_musescore/modeling/analysis.py

# IMPORTS
##################################################

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from os import mkdir

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from wrangling.full import DATASET_DIR_NAME, MMT_STATISTIC_COLUMNS
from wrangling.full import OUTPUT_DIR as DATASET_OUTPUT_DIR
from wrangling.deduplicate import FACETS
from wrangling.quality import make_facet_name_fancy, PLOTS_DIR_NAME
from dataset import OUTPUT_DIR
from train import RELEVANT_PARTITIONS
from evaluate import OUTPUT_COLUMNS, loss_to_perplexity
import utils

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

COLUMNS = ["facet"] + OUTPUT_COLUMNS

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


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Evaluate Analysis", description = "Analyze the evaluation a REMI-Style Model.")
    parser.add_argument("-i", "--input_dir", default = OUTPUT_DIR, type = str, help = "Directory containing facets (as subdirectories) to evaluate")
    parser.add_argument("-d", "--dataset_filepath", default = f"{DATASET_OUTPUT_DIR}/{DATASET_DIR_NAME}.csv", type = str, help = "Dataset from which facets are derived")
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

    # create output directory
    output_dir = f"{args.input_dir}/{PLOTS_DIR_NAME}"
    if not exists(output_dir):
        mkdir(output_dir)

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # create full dataset
    output_filepath_dataset = f"{args.input_dir}/evaluation.csv"
    if exists(output_filepath_dataset):
        dataset = pd.read_csv(filepath_or_buffer = output_filepath_dataset, sep = ",", header = 0, index_col = False)
    else:
        dataset = pd.DataFrame(columns = COLUMNS)
        for facet in FACETS:
            data = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/{facet}/evaluation.csv", sep = ",", header = 0, index_col = False)
            data["facet"] = utils.rep(x = facet, times = len(data))
            dataset = pd.concat(objs = (dataset, data[COLUMNS]), axis = 0, ignore_index = True)
        del data
        dataset = dataset.sort_values(by = ["facet", "model"], axis = 0, ascending = True, ignore_index = True)
        dataset.to_csv(path_or_buf = output_filepath_dataset, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w") # output dataset

    # output mmt statistics and perplexity
    bar_width = 100
    correct_model = list(map(lambda model: model.startswith(args.model), dataset["model"]))
    float_formatter = lambda num: f"{num:.2f}"
    logging.info(f"\n{' MMT STATISTICS ':=^{bar_width}}\n") # mmt statistics
    dataset[MMT_STATISTIC_COLUMNS[1]] *= 100 # convert scale consistency to percentage
    dataset[MMT_STATISTIC_COLUMNS[2]] *= 100 # convert groove consistency to percentage
    mmt_statistics = dataset[["facet", "model"] + MMT_STATISTIC_COLUMNS][correct_model].groupby(by = ["model", "facet"]).agg(["mean", "sem"])
    logging.info(mmt_statistics.to_string(float_format = float_formatter))
    logging.info("\n" + "".join(("=" for _ in range(bar_width))) + "\n")
    for facet, model in mmt_statistics.index:
        logging.info(" & ".join((f"${mmt_statistics.at[(facet, model), (mmt_statistic, 'mean')]:.2f} \pm {mmt_statistics.at[(facet, model), (mmt_statistic, 'sem')]:.2f}$" for mmt_statistic in MMT_STATISTIC_COLUMNS)))
    logging.info(f"\n{' PERPLEXITY ':=^{bar_width}}\n") # perplexity
    loss_facet_columns = list(filter(lambda column: column.startswith("loss:"), dataset.columns))
    perplexity = dataset[["facet", "model"] + loss_facet_columns][correct_model].groupby(by = ["model", "facet"]).agg(loss_to_perplexity) # group by model and facet
    perplexity = perplexity.rename(columns = dict(zip(loss_facet_columns, map(lambda loss_facet_column: loss_facet_column[len("loss:"):].replace(f"{FACETS[-1]}", "").replace("-", "").replace("_", ""), loss_facet_columns)))) # rename columns
    logging.info(perplexity.to_string(float_format = float_formatter))
    logging.info("\n" + "".join(("=" for _ in range(bar_width))) + "\n")
    del correct_model, mmt_statistics, perplexity

    # load in dataset
    dataset_real = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    # determine model to analyze; assumes the same models have been created for each facet
    models = set(pd.unique(values = dataset["model"]))
    model = (str(max(map(lambda model: int(model.split("_")[0][:-1]), models))) + "M") if args.model is None else args.model
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
    realness_names = ["actual", "generated"]
    legend_title = "Facet"
    legend_title_fontsize = "large"
    legend_fontsize = "medium"
    plot_to_legend_ratio = 3
    output_filepath_prefix = f"{output_dir}/evaluation.{model}"

    ##################################################


    # PLOT LINE PLOT
    ##################################################

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = [MMT_STATISTIC_COLUMNS[:-1], [MMT_STATISTIC_COLUMNS[-1], "legend"]], constrained_layout = True, figsize = (8, 6))
    fig.suptitle(f"Evaluating {model} Model Performance", fontweight = "bold")

    # plotting function
    def plot_mmt_statistic(mmt_statistic: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # left side will be fraction, right will be count
        count_axes = axes[mmt_statistic].twinx()

        # loop through facets
        for facet in FACETS:
            
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
            axes[mmt_statistic].plot(bin_centers, data / sum(data), label = facet) # fraction
            count_axes.plot(bin_centers, data, label = facet) # count

        # axes labels and such
        axes[mmt_statistic].set_xlabel("Value")
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
                          ncol = 1, title = legend_title, mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{output_filepath_prefix}.lines.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################


    # PLOT STACKED PLOT
    ##################################################

    # create plot
    fig, axes = plt.subplot_mosaic(
        mosaic = (
            utils.rep(x = list(map(lambda mmt_statistic: f"{realness_names[0]}.{mmt_statistic}", MMT_STATISTIC_COLUMNS)), times = plot_to_legend_ratio) +
            utils.rep(x = list(map(lambda mmt_statistic: f"{realness_names[1]}.{mmt_statistic}", MMT_STATISTIC_COLUMNS)), times = plot_to_legend_ratio) +
            [utils.rep(x = "legend", times = len(MMT_STATISTIC_COLUMNS))]
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
        data = {
            realness_names[0]: np.array(list(map(lambda facet: np.histogram(a = dataset_real[dataset_real[f"facet:{facet}"]][mmt_statistic], bins = bins)[0], FACETS))), # real
            realness_names[1]: np.array(list(map(lambda facet: np.histogram(a = dataset[dataset["facet"] == facet][mmt_statistic], bins = bins)[0], FACETS))), # generated
        }
        data = {realness_name: data_values / np.sum(a = data_values, axis = 1).reshape(-1, 1) for realness_name, data_values in data.items()} # normalize data such that it's like every facet has the same number of songs
        data = {realness_name: convert_to_fraction(data = data_values) for realness_name, data_values in data.items()} # convert to fraction

        # loop through facets and plot
        axes_names = list(map(lambda realness_name: f"{realness_name}.{mmt_statistic}", realness_names))
        for realness_name, axes_name in zip(realness_names, axes_names):
            for i, facet in enumerate(FACETS):
                axes[axes_name].bar(x = bin_centers, height = data[realness_name][i], width = bin_width, bottom = np.sum(a = data[realness_name][:i, :], axis = 0), label = facet)
            axes[axes_name].set_xlabel(mmt_statistic.replace("_", " ").title())
            axes[axes_name].set_xlim(left = bins[0], right = bins[-1])
            axes[axes_name].set_ylabel("")
            axes[axes_name].set_ylim(bottom = 0, top = 1)
            axes[axes_name].grid()

        # add title if needed
        if mmt_statistic == MMT_STATISTIC_COLUMNS[1]:
            axes[axes_names[0]].set_title("\nActual Data\n", fontweight = "bold")
            axes[axes_names[1]].set_title(f"\nGenerated by {model} Model\n", fontweight = "bold")

    # plot plots
    for mmt_statistic in MMT_STATISTIC_COLUMNS:
        plot_mmt_statistic_stacked(mmt_statistic = mmt_statistic)

    # plot legend
    handles, labels = axes[f"{realness_names[0]}.{MMT_STATISTIC_COLUMNS[0]}"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                          loc = "center", fontsize = legend_fontsize, title_fontsize = legend_title_fontsize, alignment = "center",
                          ncol = len(FACETS), title = legend_title, mode = "expand")
    axes["legend"].axis("off")

    # save image
    output_filepath_plot = f"{output_filepath_prefix}.stacked.pdf"
    fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
    logging.info(f"Saved figure to {output_filepath_plot}.")

    ##################################################


    # DIFFERENT LINES PLOT
    ##################################################

    # plotting function
    def plot_mmt_statistic_faceted(facet: str) -> None:
        """Plot information on MMT-style statistic in evaluations."""

        # create plot
        fig, axes = plt.subplot_mosaic(
            mosaic = utils.rep(x = MMT_STATISTIC_COLUMNS, times = plot_to_legend_ratio * 2) + [utils.rep(x = "legend", times = len(MMT_STATISTIC_COLUMNS))],
            constrained_layout = True,
            figsize = (12, 5))
        fig.suptitle(f"Comparing {make_facet_name_fancy(facet = facet)} Facet in Actual versus Generated Music")

        # get faceted dataset
        dataset_facet = dataset[dataset["facet"] == facet]
        dataset_real_facet = dataset_real[dataset_real[f"facet:{facet}"]]

        # go through different mmt statistics
        for mmt_statistic in MMT_STATISTIC_COLUMNS:

            # # get the range of data
            # data_values = pd.concat(objs = (dataset_facet[mmt_statistic], dataset_real_facet[mmt_statistic]), axis = 0)
            # min_data, max_data = min(data_values), max(data_values)
            # data_range = max_data - min_data
            # margin = ((range_multiplier_constant - 1) / 2) * data_range
            # bin_width = (range_multiplier_constant * data_range) / n_bins
            # bins = np.arange(start = min_data - margin, stop = max_data + margin + (bin_width / 2), step = bin_width)
            # bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)] # get centerpoints of each bin
            
            # # get histograms and convert to fraction
            # data = {
            #     realness_names[0]: np.histogram(a = dataset_real_facet[mmt_statistic], bins = bins)[0], # real
            #     realness_names[1]: np.histogram(a = dataset_facet[mmt_statistic], bins = bins)[0], # generated
            # }
            # data = {realness_name: data_values / sum(data_values) for realness_name, data_values in data.items()} # normalize data such that it's like every facet has the same number of songs

            # plot
            alpha = 0.5
            axes[mmt_statistic].hist(dataset_real_facet[mmt_statistic], label = realness_names[0], density = True, alpha = alpha)
            axes[mmt_statistic].hist(dataset_facet[mmt_statistic], label = realness_names[1], density = True, alpha = alpha)
            axes[mmt_statistic].set_xlabel(mmt_statistic.replace("_", " ").title())
            axes[mmt_statistic].grid()
            if mmt_statistic == MMT_STATISTIC_COLUMNS[0]: # add y label if necessary
                axes[mmt_statistic].set_ylabel("Density")

        # plot legend
        handles, labels = axes[MMT_STATISTIC_COLUMNS[0]].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes["legend"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                              loc = "center", fontsize = legend_fontsize, alignment = "center", ncol = len(realness_names))
        axes["legend"].axis("off")

        # save image
        output_filepath_plot = f"{output_filepath_prefix}.lines.{facet}.pdf"
        fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
        logging.info(f"Saved figure to {output_filepath_plot}.")

    # plot plots
    for facet in FACETS:
        plot_mmt_statistic_faceted(facet = facet)

    ##################################################


    # TRAIN LOSS PLOT
    ##################################################

    # helper function to plot loss plot
    def plot_loss(partition: str = RELEVANT_PARTITIONS[-1]) -> None:
        """
        Plot the loss curves (different dataset facets) for a given partition.
        """

        # create plot
        fig, axes = plt.subplot_mosaic(mosaic = [["loss"]], constrained_layout = True, figsize = (4, 4))
        fig.suptitle("Loss", fontweight = "bold")

        # loop through facets
        step_by = 1000 # step axis tick labels will be in units of step_by
        for facet in FACETS:

            # get data
            data = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/{facet}/{model}/loss.csv", sep = ",", header = 0, index_col = False)
            data = data[data["partition"] == partition] # filter to correct partition

            # plot data
            axes["loss"].plot(data["step"] / step_by, data["loss"], label = facet)

        # add axis titles and such
        axes["loss"].set_xlabel(f"Step (in {step_by:,}s)")
        axes["loss"].set_ylabel("Loss")
        axes["loss"].grid()

        # plot legend
        handles, labels = axes["loss"].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes["loss"].legend(handles = by_label.values(), labels = list(map(make_facet_name_fancy, by_label.keys())),
                            alignment = "center", ncol = 1, title = legend_title)

        # save image
        output_filepath_plot = f"{output_dir}/loss.{model}.{partition}.pdf"
        fig.savefig(output_filepath_plot, dpi = 200, transparent = True, bbox_inches = "tight")
        logging.info(f"Saved figure to {output_filepath_plot}.")

    # plot plots
    for partition in RELEVANT_PARTITIONS:
        plot_loss(partition = partition)

    ##################################################

##################################################

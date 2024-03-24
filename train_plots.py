# README
# Phillip Long
# January 25, 2024

# Create plots to describe training.

# python /home/pnlong/model_musescore/train_plots.py


# IMPORTS
##################################################

import argparse
from os.path import exists
from os import mkdir

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import utils
import train
import expressive_features_plots
import evaluate_baseline

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/home/pnlong/musescore/datav"
MODELS_FILEPATH = f"{DATA_DIR}/models.txt"
OUTPUT_DIR = "/home/pnlong/musescore/datav"

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default = MODELS_FILEPATH, type = str, help = ".txt file with a list of directory names for each model")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# HELPER FUNCTION THAT MAKES A PLOT
##################################################

def make_model_name_fancy(model: str) -> str:
    """Return a prettified version of the model name for legend."""
    if model == evaluate_baseline.TRUTH_DIR_STEM:
        return "Truth"
    model_name = model.split("_")
    model_name = ("uni" if "unidimensional" in model else "") + model_name[0].title() + (" (C)" if "conditional" in model else "") + ": " + model_name[-1] # model_name[-1] is model size
    return model_name

def make_plot(partition: str, metric: str, mask: str, output_dir: str):
    """Make plot."""

    current_performance = performance[(performance["metric"] == metric) & (performance["partition"] == partition) & (performance["mask"] == mask)] # get subset of performance relevant to this plot

    # create figure
    n_cols = 4
    fig, axes = plt.subplot_mosaic(mosaic = [fields[:n_cols], fields[n_cols:] + ["legend"]], constrained_layout = True, figsize = (12, 8))
    figure_title = partition.title() + ("ing" if partition == "train" else "ation") + " " + metric.title() + (f" for {mask.title()}s" if mask != train.ALL_STRING else "")
    fig.suptitle(figure_title, fontweight = "bold")

    # plot values
    for i, field in enumerate(fields):
        for j, model in enumerate(models):
            current_performance_subset = current_performance[(current_performance["field"] == field) & (current_performance["model"] == model)]
            axes[field].plot(current_performance_subset["step"], current_performance_subset["value"], label = model, color = expressive_features_plots.LINE_COLORS[j])
        if i % n_cols == 0: # if this is a leftmost plot, add y labels
            axes[field].set_ylabel(metric.title())
        elif (i % n_cols != 0) and ("acc" in metric): # if this is a non left most plot and it is for accuracy
            axes[field].set_yticklabels([])
        if "acc" in metric: # make sure accuracy scales from 0 to 1
            axes[field].set_ylim(0, 1)
        # axes[field].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{count:,.2f}")) # add commas, but this also causes decimals to get ruined
        if i >= n_cols: # if this is a bottom plot, add x labels
            axes[field].set_xlabel("Step")
            axes[field].ticklabel_format(axis = "x", style = "scientific", scilimits = (0, 1))
        else: # is not a bottom plot
            # axes[field].set_xticks([]) # will keep xticks for now
            axes[field].set_xticklabels([])
        axes[field].set_title(field.title())
        axes[field].grid() # add gridlines

    # get legend
    handles, labels = axes[fields[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "xsmall", title_fontsize = "small", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")
    
    # save image
    fig.savefig(f"{output_dir}/{partition}_{metric}_{mask}.png", dpi = expressive_features_plots.OUTPUT_RESOLUTION_DPI) # save image

##################################################


# MAIN METHOD
##################################################
if __name__ == "__main__":

    # PROCESS DATA
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # get model names
    with open(args.models, "r") as models_output: # read in list of trained models
        models = [model.strip() for model in models_output.readlines()] # use a set because better for `in` operations

    # create/read in big data frame of performance
    performance_output_filepath = f"{args.output_dir}/performance.csv"
    if exists(performance_output_filepath):
        performance = pd.read_csv(filepath_or_buffer = performance_output_filepath, sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in full performance
    else:
        PERFORMANCE_COLUMNS = ["model"] + train.PERFORMANCE_OUTPUT_COLUMNS
        performance = pd.DataFrame(columns = PERFORMANCE_COLUMNS)
        for model in models:
            model_performance = pd.read_csv(filepath_or_buffer = f"{args.output_dir}/{model}/performance.csv", sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in performance
            model_performance["model"] = utils.rep(x = model, times = len(model_performance)) # add model column
            model_performance = model_performance[PERFORMANCE_COLUMNS] # reorder columns
            performance = pd.concat(objs = (performance, model_performance), axis = 0)
        del model, model_performance # free up memory
        performance["unidimensional"] = performance["model"].apply(lambda model: "unidimensional" in model) # boolean value for unidimensionality
        performance["model_size"] = performance["model"].apply(lambda model: model.split("_")[-1])
        performance.to_csv(path_or_buf = performance_output_filepath, sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")

    # get list of fields
    fields = pd.unique(values = performance["field"]).tolist()
    
    ##################################################
        
    
    # CREATE PLOTS
    ##################################################

    plots_dir = f"{args.output_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    for metric in train.PERFORMANCE_METRICS:
        for partition in train.RELEVANT_PARTITIONS:
            for mask in train.MASKS:
                make_plot(metric = metric, partition = partition, mask = mask, output_dir = plots_dir)

    ##################################################

##################################################

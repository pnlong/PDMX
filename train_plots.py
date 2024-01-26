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

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
MODELS_FILEPATH = f"{DATA_DIR}/models.txt"
OUTPUT_DIR = "/data2/pnlong/musescore/data"

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

def make_plot(partition: str, metric: str, mask: str, output_dir: str):
    """Make plot."""

    current_performance = performance[(performance["metric"] == metric) & (performance["partition"] == partition) & (performance["mask"] == mask)] # get subset of performance relevant to this plot

    # create figure
    n_cols = 4
    fig, axes = plt.subplot_mosaic(mosaic = [fields[:n_cols], fields[n_cols:] + ["legend"]], constrained_layout = True, figsize = (12, 8))
    figure_title = partition.title() + ("ing" if partition == "train" else "ation") + " " + metric.title() + (f"for {mask.title()}s" if mask != train.ALL_STRING else "")
    fig.suptitle(figure_title, fontweight = "bold")

    # plot values
    for i, field in enumerate(fields):
        for j, model in enumerate(models):
            current_performance_subset = current_performance[(current_performance["field"] == field) & (current_performance["model"] == model)]
            model_name = model.split("_")
            model_name = model_name[0].title() + (", Conditional" if "conditional" in model else "") + ": " + model_name[-1]
            axes[field].plot(current_performance_subset["step"], current_performance_subset["value"], label = model_name, color = expressive_features_plots.LINE_COLORS[j])
        if i % n_cols == 0: # if this is a leftmost plot, add y labels
            axes[field].set_ylabel(metric.title())
        else: # is not a leftmost plot
            axes[field].set_yticklabels([])
        axes[field].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}")) # add commas
        if i >= n_cols: # if this is a bottom plot, add x labels
            axes[field].set_xlabel("Step")
        else: # is not a bottom plot
            # axes[field].set_xticks([]) # will keep xticks for now
            axes[field].set_xticklabels([])
        axes[field].set_title(field.title())
        axes[field].grid() # add gridlines

    # get legend
    handles, labels = axes[models[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = by_label.keys(), loc = "center", fontsize = "medium", title_fontsize = "large", alignment = "center", ncol = 2, title = "Model", mode = "expand")
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
        models = {model.strip() for model in models_output.readlines()} # use a set because better for `in` operations

    # create big data frame
    PERFORMANCE_COLUMNS = ["model"] + train.PERFORMANCE_OUTPUT_COLUMNS
    performance = pd.DataFrame(columns = PERFORMANCE_COLUMNS)
    for model in models:
        model_performance = pd.read_csv(filepath_or_buffer = f"{args.output_dir}/{model}", sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in performance
        model_performance["model"] = utils.rep(x = model, times = len(model_performance)) # add model column
        model_performance = model_performance[PERFORMANCE_COLUMNS] # reorder columns
        performance = pd.concat(objs = (performance, model_performance), axis = 0)
    del model, model_performance # free up memory
    performance_output_filepath = f"{args.output_dir}/performance.csv"
    if not exists(performance_output_filepath): # save model if needed
        performance.to_csv(path_or_buf = performance_output_filepath, sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")

    # get list of fields
    fields = pd.unique(values = performance["field"])
    
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

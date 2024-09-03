# README
# Phillip Long
# August 16, 2024

# Generate audio samples for listening test.

# python /home/pnlong/model_musescore/modeling/listening_test.py

# IMPORTS
##################################################

import argparse
import logging
from os.path import exists, basename, dirname
from os import mkdir, makedirs, chdir, remove
import sys
from shutil import rmtree
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import random
from itertools import product
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
import seaborn as sns

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from wrangling.full import CHUNK_SIZE
from wrangling.genres import FACETS_FOR_PLOTTING, FACET_COLORS
from wrangling.quality import PLOTS_DIR_NAME, make_facet_name_fancy
from dataset import OUTPUT_DIR as DATASET_OUTPUT_DIR
from train import FINE_TUNING_SUFFIX
from representation import Indexer, get_encoding
from generated_to_audio import generated_to_audio
import utils

plt.style.use("default")
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

OUTPUT_DIR = f"{DATASET_OUTPUT_DIR}/listening_test" # where to output generated samples
MODEL_SIZE = "65M" # model size to evaluate
N_SAMPLES_PER_GROUP = 10

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Listening Test", description = "Generate audio samples for a listening test.")
    parser.add_argument("-d", "--dataset_filepath", default = f"{DATASET_OUTPUT_DIR}/evaluation.csv", type = str, help = "Dataset with evaluated samples for all subsets and models.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory where audio samples will be stored.")
    parser.add_argument("-m", "--model_size", default = MODEL_SIZE, type = str, help = "Model size from which to generate listening samples.")
    parser.add_argument("-n", "--n_samples_per_group", default = N_SAMPLES_PER_GROUP, type = int, help = "Number of samples per group to generate.")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate data files.")
    parser.add_argument("-bp", "--bar_plot", action = "store_true", help = "Whether or not to make a bar plot (violin plot is default).")
    parser.add_argument("-eb", "--error_bars", action = "store_true", help = "Whether to add error bars to bar plot (irrelevant if --bar_plot is not selected).")
    parser.add_argument("-c", "--combined", action = "store_true", help = "Combine bar and violin plot.")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of jobs.")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # set random seed
    random.seed(0)

    ##################################################

    
    # CHECK IF WE NEED TO CREATE LISTENING TEST
    ##################################################

    if (not exists(args.output_dir)) or args.reset:

        # OUTPUT DIRECTORY STUFF
        ##################################################

        # deal with output directory
        output_dir_targz = f"{args.output_dir}.tar.gz"
        if exists(args.output_dir): # clear directory
            rmtree(args.output_dir)
            remove(output_dir_targz)
            mkdir(args.output_dir)
        else:
            makedirs(args.output_dir)
        
        # get variables
        encoding = get_encoding() # load the encoding
        vocabulary = utils.inverse_dict(Indexer(data = encoding["event_code_map"]).get_dict()) # for decoding

        # load in dataset
        dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
        dataset = dataset[dataset["model"].map(lambda model: model.split("_")[0]) == args.model_size] # get only relevant samples (correct model size)

        ##################################################


        # FUNCTION THAT DETERMINES OUTPUT PATH FROM INPUT PATH
        ##################################################

        def get_output_path_prefix(path: str) -> str:
            """
            Given the input path, return the output audio path prefix.
            """

            path_info = path[:-len(".npy")].split("/")[-4:] # /home/pnlong/musescore/remi/all/20M/eval/0.npy
            output_dir = args.output_dir
            if not exists(output_dir):
                mkdir(output_dir)
            output_path_prefix = f"{output_dir}/{path_info[1]}.{path_info[0]}" # output_dir/model.facet
            return output_path_prefix
        
        ##################################################


        # GENERATE AUDIO SAMPLES
        ##################################################

        # get different instrumentations
        groups = list(product(set(dataset["model"]), set(dataset["facet"])))
        top_instrumentations = dataset["tracks"].value_counts(ascending = False).index.to_numpy()
        def minimum_common_amount_per_group(instrumentation: str) -> int:
            """Determines if an instrumentation is a valid instrumentation (at least one sample per group)."""
            amount_per_group = dataset[dataset["tracks"] == instrumentation].groupby(by = ["model", "facet"]).size()
            if len(amount_per_group) != len(groups):
                return 0
            else:
                return min(amount_per_group)
        with multiprocessing.Pool(processes = args.jobs) as pool:
            top_instrumentations_mask = np.array(list(tqdm(iterable = pool.map(func = minimum_common_amount_per_group, iterable = top_instrumentations, chunksize = CHUNK_SIZE),
                                                        desc = "Choosing Different Instrumentations",
                                                        total = len(top_instrumentations))))
        top_instrumentations = top_instrumentations[top_instrumentations_mask > 0] # only include where there is an example of that instrumentation in all groups
        n_different_instrumentations_other_than_top = min(len(top_instrumentations), args.n_samples_per_group) - 1
        n_samples_per_instrumentation = [args.n_samples_per_group - n_different_instrumentations_other_than_top] + utils.rep(x = 1, times = n_different_instrumentations_other_than_top)
        logging.info(f"{n_different_instrumentations_other_than_top + 1} different instrumentations.")

        # get paths
        paths, output_paths = [], []
        for i in range(len(n_samples_per_instrumentation)):
            dataset_instrumentation = dataset[dataset["tracks"] == top_instrumentations[i]]
            starting_index = sum(n_samples_per_instrumentation[:i])
            for model, facet in groups:
                sampled_paths = dataset_instrumentation[(dataset_instrumentation["model"] == model) & (dataset_instrumentation["facet"] == facet)]["path"].to_list()
                sampled_paths = random.sample(population = sampled_paths, k = n_samples_per_instrumentation[i])
                paths.extend(sampled_paths)
                output_paths.extend(map(lambda j: f"{args.output_dir}/{model}.{facet}.{starting_index + j}.wav", range(len(sampled_paths))))

        # use multiprocessing
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = list(tqdm(iterable = pool.starmap(func = generated_to_audio,
                                                iterable = zip(
                                                    paths,
                                                    output_paths,
                                                    utils.rep(x = encoding, times = len(paths)),
                                                    utils.rep(x = vocabulary, times = len(paths)),
                                                ),
                                                chunksize = CHUNK_SIZE),
                        desc = f"Generating Audio",
                        total = len(paths)))

        # tar and gzip the output directory
        chdir(dirname(args.output_dir))
        subprocess.run(args = ["tar", "-zcf", basename(output_dir_targz), basename(args.output_dir)], check = True)

        ##################################################

    ##################################################


    # MEAN OPINION SCORE PLOT
    ##################################################

    # load in data
    listening_test_results_filepath = f"{dirname(args.output_dir)}/listening_test.csv"
    if not exists(listening_test_results_filepath):
        sys.exit()
    listening_test = pd.read_csv(filepath_or_buffer = listening_test_results_filepath, sep = ",", header = 0, index_col = 0)
    listening_test = listening_test.rename(columns = {"page": "model", "cond": "facet", "is_ft": "fine_tuned"}).drop(columns = ["project", "time"]) # rename columns to work in our infrastructure
    listening_test["model"] = list(map(lambda model: model.split("-")[0], listening_test["model"])) # remove page number
    listening_test = listening_test.sort_values(by = ["model", "facet"], axis = 0, ascending = True, ignore_index = True) # sort in correct order
    mos = listening_test[["model", "facet", "rating"]].groupby(by = ["model", "facet"]).agg(["mean", "sem"])["rating"] # mean opinion scores by model and facet

    # create plot
    mosaic = [["bar"], ["violin"]] if args.combined else ([["bar"]] if args.bar_plot else [["violin"]])
    fig, axes = plt.subplot_mosaic(mosaic = mosaic, constrained_layout = True, figsize = (5, 2.6 * (1.6 if args.combined else 1)))

    # plot hyperparameters
    axis_label_fontsize = "small"
    xlabel = "Subset"
    ylabel = "Rating"
    axis_tick_fontsize = "x-small"
    total_bar_width = 0.8
    individual_bar_width = total_bar_width / 2
    half_bar_width = 0.5 * individual_bar_width
    alpha_for_fine_tune = (1.0, 0.6)
    bar_plot_margin = 0.02 # set None if there is no margin and matplotlib automatically decides
    xticks = np.arange(len(FACETS_FOR_PLOTTING))

    # plot a bar plot
    if args.bar_plot or args.combined:
        for xtick, facet in zip(xticks, FACETS_FOR_PLOTTING):
            for fine_tuned in (False, True):
                axes["bar"].bar(x = xtick + ((1 if fine_tuned else -1) * half_bar_width),
                                height = mos.at[(args.model_size + (f"_{FINE_TUNING_SUFFIX}" if fine_tuned else ""), facet), "mean"],
                                width = individual_bar_width,
                                align = "center",
                                label = "Fine Tuned" if fine_tuned else "Base",
                                color = FACET_COLORS[facet],
                                alpha = alpha_for_fine_tune[fine_tuned])
        if args.error_bars: # error bars if needed
            mos_by_fine_tuned = mos.droplevel(level = "facet", axis = 0)
            for fine_tuned in (False, True):
                model = args.model_size + (f"_{FINE_TUNING_SUFFIX}" if fine_tuned else "")
                axes["bar"].errorbar(x = xticks + ((1 if fine_tuned else -1) * half_bar_width),
                                     y = mos_by_fine_tuned.loc[model, "mean"],
                                     yerr = mos_by_fine_tuned.loc[model, "sem"],
                                     fmt = "o",
                                     color = "0.4")
        if bar_plot_margin is not None:
            low, high = min(mos["mean"] - (mos["sem"] if args.error_bars else 0)), max(mos["mean"] + (mos["sem"] if args.error_bars else 0))
            axes["bar"].set_ylim(bottom = (1 - bar_plot_margin) * low, top = (1 + bar_plot_margin) * high) # add limits
                
    # plot a violion plot
    if (not args.bar_plot) or args.combined:
        violin_parts = sns.violinplot(
                       data = listening_test,
                       x = "facet",
                       y = "rating",
                       hue = "fine_tuned",
                       split = True,
                       inner = "quart",
                       fill = True, linewidth = 0,
                       ax = axes["violin"]
                       )
        for i, patch in enumerate(violin_parts.findobj(PolyCollection)):
            patch.set_facecolor(FACET_COLORS[FACETS_FOR_PLOTTING[i // 2]])
            patch.set_alpha(alpha_for_fine_tune[i % 2 == 1])
        axes["violin"].legend_.remove() # remove legend
        # listening_test = listening_test.set_index(keys = ["model", "facet"], drop = True)["rating"]
        # plotting_indicies = listening_test.sort_index(level = "facet").index
        # violin_plot_data = list(map(lambda model_facet: listening_test.loc[model_facet].to_list(), plotting_indicies))
        # violin_parts = axes["violin"].violinplot(
        #     dataset = violin_plot_data,
        #     positions = sorted(np.concatenate((xticks - half_bar_width, xticks + half_bar_width), axis = 0)),
        #     vert = True,
        #     width = 0.5,
        #     showmeans = False,
        #     showextrema = False,
        #     showmedians = False,
        #     quantiles = None, # alternatively, [0.0, 0.25, 0.5, 0.75, 1.0]
        # )
        # for patch, model, facet in zip(violin_parts["bodies"], *zip(*plotting_indicies)): # set colors
        #     patch.set_facecolor(FACET_COLORS[facet]) # set color of each violin
        #     patch.set_edgecolor("black") # set the edgecolor
        #     patch.set_alpha(alpha_for_fine_tune[FINE_TUNING_SUFFIX in model]) # set alpha
    
    if args.combined:
        top_plot_type, bottom_plot_type = list(zip(*mosaic))[0]
        # axes[top_plot_type].set_xlabel(xlabel, fontsize = axis_label_fontsize)
        axes[bottom_plot_type].set_xlabel(xlabel, fontsize = axis_label_fontsize)
        axes[top_plot_type].set_xticks(ticks = [], labels = []) # no x tick labels
        axes[bottom_plot_type].set_xticks(ticks = xticks, labels = list(map(make_facet_name_fancy, FACETS_FOR_PLOTTING)), fontsize = axis_tick_fontsize, rotation = 0) # get subset names
        axes[top_plot_type].set_ylabel(ylabel, fontsize = axis_label_fontsize)
        axes[bottom_plot_type].set_ylabel(ylabel, fontsize = axis_label_fontsize)
    else:
        plot_type = "bar" if args.bar_plot else "violin"
        axes[plot_type].set_xticks(ticks = xticks, labels = list(map(make_facet_name_fancy, FACETS_FOR_PLOTTING)), fontsize = axis_tick_fontsize, rotation = 0)
        axes[plot_type].set_xlabel(xlabel, fontsize = axis_label_fontsize)
        axes[plot_type].set_ylabel(ylabel, fontsize = axis_label_fontsize)

    # save plot
    output_filepath = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}/listening_test.pdf" # get output filepath
    if not exists(dirname(output_filepath)): # make sure output directory exists
        mkdir(dirname(output_filepath))
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight") # save image
    logging.info(f"MOS plot saved to {output_filepath}.")

    ##################################################

##################################################

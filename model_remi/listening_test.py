# README
# Phillip Long
# August 16, 2024

# Generate audio samples for listening test.

# python /home/pnlong/model_musescore/model_remi/listening_test.py

# IMPORTS
##################################################

import argparse
import logging
from os.path import exists, basename, dirname
from os import mkdir, makedirs, chdir, remove
from shutil import rmtree
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import random
from itertools import product
import subprocess
import matplotlib.pyplot as plt

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from make_dataset.full import CHUNK_SIZE
from make_dataset.genres import FACETS_FOR_PLOTTING, FACET_COLORS
from make_dataset.quality import PLOTS_DIR_NAME
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
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate data files")
    parser.add_argument("-eb", "--error_bars", action = "store_true", help = "Whether to add error bars.")
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
    listening_test = pd.read_csv(filepath_or_buffer = "", sep = ",", header = 0, index_col = False)

    # wrangle data, expecting multiindex (facet: str, fine_tuned: bool)
    paths = list(map(basename, listening_test["path"]))
    paths_info = np.array(list(map(lambda path: path.split("."), paths)))[:, :-1] # remove wav suffix
    listening_test["facet"] = paths_info[:, 1]
    listening_test["fine_tuned"] = list(map(lambda model: FINE_TUNING_SUFFIX in model, paths_info[:, 0]))
    listening_test = listening_test.groupby(by = ("facet", "fine_tuned")).agg(["mean", "sem"])

    # create plot
    fig, axes = plt.subplot_mosaic(mosaic = [["plot"]], constrained_layout = True, figsize = (4, 4))

    # plot
    axis_tick_fontsize = "small"
    total_bar_width = 0.8
    individual_bar_width = total_bar_width / 2
    half_bar_width = 0.5 * individual_bar_width
    alpha_for_fine_tune = (0.75, 1.0)
    xticks = np.arange(len(FACETS_FOR_PLOTTING))
    for i, facet in enumerate(FACETS_FOR_PLOTTING):
        axes["plot"].bar(x = i - half_bar_width, height = listening_test.at[(facet, False), "mean"], width = individual_bar_width, align = "center", label = "Base", color = FACET_COLORS[facet], alpha = alpha_for_fine_tune[0]) # not fine tuned
        axes["plot"].bar(x = i + half_bar_width, height = listening_test.at[(facet, True), "mean"], width = individual_bar_width, align = "center", label = "Fine Tuned", color = FACET_COLORS[facet], alpha = alpha_for_fine_tune[1]) # fine tuned
    
    # error bars if needed
    if args.error_bars:
        error_bar_fmt = "o"
        error_bar_color = "0"
        listening_test_by_fine_tuned = listening_test.droplevel(level = "facet", axis = 0)
        axes["plot"].errorbar(x = xticks - half_bar_width, y = listening_test_by_fine_tuned.loc[False, "mean"], yerr = listening_test_by_fine_tuned.loc[False, "sem"], fmt = error_bar_fmt, color = error_bar_color) # not fine tuned
        axes["plot"].errorbar(x = xticks - half_bar_width, y = listening_test_by_fine_tuned.loc[True, "mean"], yerr = listening_test_by_fine_tuned.loc[True, "sem"], fmt = error_bar_fmt, color = error_bar_color) # fine tuned
    
    # axes["plot"].set_xlabel("Subset", fontsize = axis_tick_fontsize)
    axes["plot"].set_xticks(ticks = xticks, labels = FACETS_FOR_PLOTTING, fontsize = axis_tick_fontsize, rotation = 0) # get subset names
    axes["plot"].set_ylabel("Mean Opinion Score", fontsize = axis_tick_fontsize)
    axes["plot"].legend(fontsize = axis_tick_fontsize)

    # save plot
    output_filepath = f"{dirname(args.dataset_filepath)}/{PLOTS_DIR_NAME}/listening_test.pdf" # get output filepath
    if not exists(dirname(output_filepath)): # make sure output directory exists
        mkdir(dirname(output_filepath))
    fig.savefig(output_filepath, dpi = 200, transparent = True, bbox_inches = "tight") # save image
    logging.info(f"MOS plot saved to {output_filepath}.")

    ##################################################

##################################################

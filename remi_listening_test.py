# README
# Phillip Long
# August 16, 2024

# Generate audio samples for listening test.

# python /home/pnlong/model_musescore/remi_listening_test.py

# IMPORTS
##################################################

import argparse
from os.path import exists
from os import mkdir, makedirs
from shutil import rmtree
import pandas as pd
import multiprocessing
from tqdm import tqdm
from typing import List
import random

from dataset_full import CHUNK_SIZE
import remi_dataset
import remi_representation
from remi_generated_to_audio import generated_to_audio
import utils

##################################################


# CONSTANTS
##################################################

OUTPUT_DIR = f"{remi_dataset.OUTPUT_DIR}/listening_test" # where to output generated samples
MODEL_SIZE = "65M" # model size to evaluate
NUMBER_OF_SAMPLES_PER_GROUP = 10

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Listening Test", description = "Generate audio samples for a listening test.")
    parser.add_argument("-d", "--dataset_filepath", default = f"{remi_dataset.OUTPUT_DIR}/evaluation.csv", type = str, help = "Dataset with evaluated samples for all subsets and models.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory where audio samples will be stored.")
    parser.add_argument("-m", "--model_size", default = MODEL_SIZE, type = str, help = "Model size from which to generate listening samples.")
    parser.add_argument("-n", "--number_of_samples_per_group", default = NUMBER_OF_SAMPLES_PER_GROUP, type = int, help = "Number of samples per group to generate.")
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

    # deal with output directory
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    else: # clear directory
        rmtree(args.output_dir)
        mkdir(args.output_dir)
    
    # get variables
    encoding = remi_representation.get_encoding() # load the encoding
    vocabulary = utils.inverse_dict(remi_representation.Indexer(data = encoding["event_code_map"]).get_dict()) # for decoding

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset[dataset["model"].map(lambda model: model.split("_")[0]) == args.model_size] # get only relevant samples (correct model size)

    # set random seed
    random.seed(0)

    ##################################################


    # FUNCTION THAT DETERMINES OUTPUT PATH FROM INPUT PATH
    ##################################################

    def get_output_path(path: str) -> str:
        """
        Given the input path, return the output audio path.
        """

        path_info = path[:-len(".npy")].split("/")[-4:] # /home/pnlong/musescore/remi/all/20M/eval/0.npy
        output_dir = args.output_dir
        if not exists(output_dir):
            mkdir(output_dir)
        output_path = f"{output_dir}/{path_info[1]}.{path_info[0]}.{path_info[-1]}.wav" # output_dir/model.facet.n.wav
        return output_path
    
    ##################################################


    # GENERATE AUDIO SAMPLES
    ##################################################

    # get best paths
    dataset = dataset[dataset["tracks"] == dataset["tracks"].mode().item()] # get only the most common instrumentation
    def get_paths(model: str, facet: str) -> List[str]: # get paths for each model facet combination
        """
        Get paths for a given model and facet combination.
        """
        subset = dataset[(dataset["model"] == model) & (dataset["facet"] == facet)]
        return random.sample(population = subset["path"].to_list(), k = args.number_of_samples_per_group)
    paths = []
    for model in set(dataset["model"]):
        for facet in set(dataset["facet"]):
            paths.extend(get_paths(model = model, facet = facet))

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.starmap(func = generated_to_audio,
                                              iterable = zip(
                                                paths,
                                                map(get_output_path, paths),
                                                utils.rep(x = encoding, times = len(paths)),
                                                utils.rep(x = vocabulary, times = len(paths)),
                                              ),
                                              chunksize = CHUNK_SIZE),
                      desc = f"Generating Audio",
                      total = len(paths)))

    ##################################################

##################################################

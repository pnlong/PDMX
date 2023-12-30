# README
# Phillip Long
# November 22, 2023

# Split data into train, cross-validate, and test partitions.

# python /home/pnlong/model_musescore/split.py


# IMPORTS
##################################################

import argparse
import logging
import random
import pandas as pd
from os.path import dirname

import utils

##################################################


# CONSTANTS
##################################################

INPUT_FILEPATH = "/data2/pnlong/musescore/data/data.csv"
PARTITIONS = {"train": 0.8, "valid": 0.1, "test": 0.1}

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Split", description = "Partition data into training-validation-testing sets.")
    parser.add_argument("-i", "--input_filepath", default = INPUT_FILEPATH, type = str, help = "List of filepaths to semi-encoded data files")
    parser.add_argument("-o", "--output_dir", default = None, type = str, help = "Output directory")
    parser.add_argument("-v", "--ratio_valid", default = PARTITIONS["valid"], type = float, help = "Ratio of validation files")
    parser.add_argument("-t", "--ratio_test", default = PARTITIONS["test"], type = float, help = "Ratio of test files")
    parser.add_argument("-s", "--seed", default = 0, help = "Random Seed")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # read in arguments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # SPLIT UP DATA
    ##################################################

    # set random seed
    random.seed(args.seed)

    # get filepaths
    logging.info("Loading names...")
    paths = list(set(pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)["path"].tolist())) # make sure paths are unique
    logging.info(f"Loaded {len(paths):,} names.")

    # sample validation and test names
    n_valid = int(args.ratio_valid * len(paths)) # number in validation set
    n_test = int(args.ratio_test * len(paths)) # number in testing set
    sampled = random.sample(population = paths, k = n_valid + n_test) # extract validation and testing partitions
    valid_paths = sampled[n_valid:] # get validation set
    test_paths = sampled[:n_valid] # get testing set
    sampled = set(sampled) # convert to set to test if values are in sampled
    train_paths = [path for path in paths if path not in sampled] #  get training set

    ##################################################


    # OUTPUT
    ##################################################

    if args.output_dir is not None:
        base_dir = args.output_dir
    else: # infer output dir
        base_dir = dirname(args.input_filepath)

    # training paths
    train_path = f"{base_dir}/train.txt"
    utils.save_txt(filepath = train_path, data = train_paths)
    logging.info(f"Collected {len(train_paths):,} files for training. Saved training set to {train_path}")

    # validation paths
    valid_path = f"{base_dir}/valid.txt"
    utils.save_txt(filepath = valid_path, data = valid_paths)
    logging.info(f"Collected {len(valid_paths):,} files for validation. Saved validation set to {valid_path}")

    # test paths
    test_path = f"{base_dir}/test.txt"
    utils.save_txt(filepath = test_path, data = test_paths)
    logging.info(f"Collected {len(test_paths):,} files for testing. Saved testing set to {test_path}")

    ##################################################

##################################################

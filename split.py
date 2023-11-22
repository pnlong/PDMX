# README
# Phillip Long
# November 22, 2023

# Split data into train, cross-validate, and test partitions.

# python /home/pnlong/model_musescore/split.py


# IMPORTS
##################################################

import argparse
import logging
import pathlib
import pprint
import random
import sys

import utils
from utils import DATA_DIRECTORY

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices = ("sod", "lmd", "lmd_full", "snd"),
        required = True,
        help = "dataset key",
    )
    parser.add_argument("-n", "--names", type = pathlib.Path, help = "input names")
    parser.add_argument(
        "-o", "--out_dir", type = pathlib.Path, help = "output directory"
    )
    parser.add_argument(
        "-v",
        "--ratio_valid",
        default = 0.1,
        type = float,
        help = "ratio of validation files",
    )
    parser.add_argument(
        "-t",
        "--ratio_test",
        default = 0.1,
        type = float,
        help = "ratio of test files",
    )
    parser.add_argument("-s", "--seed", default = 0, help = "random seed")
    parser.add_argument(
        "-q", "--quiet", action = "store_true", help = "show warnings only"
    )
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # read in arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"{DATA_DIRECTORY}data/{args.dataset}/processed/names.txt"
            )
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"{DATA_DIRECTORY}data/{args.dataset}/processed/")

    # Set up the logger
    logging.basicConfig(
        stream = sys.stdout,
        level = logging.ERROR if args.quiet else logging.INFO,
        format = "%(levelname)-8s %(message)s",
    )

    ##################################################


    # SPLIT UP DATA
    ##################################################

    # Set random seed
    random.seed(args.seed)

    # Get filenames
    logging.info("Loading names...")
    names = utils.load_txt(args.names)
    logging.info(f"Loaded {len(names)} names.")

    # Sample training and test names
    n_valid = int(len(names) * args.ratio_valid)
    n_test = int(len(names) * args.ratio_test)
    sampled = random.sample(names, n_valid + n_test)
    valid_names = sampled[:n_valid]
    test_names = sampled[n_valid:]
    train_names = [name for name in names if name not in sampled]

    ##################################################


    # OUTPUT
    ##################################################

    utils.save_txt(args.out_dir / "train-names.txt", train_names)
    logging.info(f"Collected {len(train_names)} files for training.")

    utils.save_txt(args.out_dir / "valid-names.txt", valid_names)
    logging.info(f"Collected {len(valid_names)} files for validation.")

    utils.save_txt(args.out_dir / "test-names.txt", test_names)
    logging.info(f"Collected {len(test_names)} files for test.")

    ##################################################

##################################################

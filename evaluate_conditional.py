# README
# Phillip Long
# November 27, 2023

# Evaluate a model with notes conditional on expressive features.

# python /home/pnlong/model_musescore/evaluate_conditional.py


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, basename
from os import mkdir
from typing import Union
import math

import muspy
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from read_mscz.music import BetterMusic
import dataset
import music_x_transformers
import representation
import encode
import decode
import utils
from train import PARTITIONS, NA_VALUE

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"
OUTPUT_DIR = "/data2/pnlong/musescore/data"
EVAL_SUBSETS = ["truth", "unconditioned"]
EVAL_METRICS = ["pitch_class_entropy", "scale_consistency", "groove_consistency"]
OUTPUT_COLUMNS = ["i", "original_path", "is_truth", "stem",] + EVAL_METRICS

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", default = PATHS, type = str, help = ".txt file with absolute filepaths to testing dataset.")
    parser.add_argument("-e", "--encoding", default = ENCODING_FILEPATH, type = str, help = ".json file with encoding information.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-ns", "--n_samples", type = int, help = "Number of samples to evaluate")
    # model
    parser.add_argument("--model_steps", type = int, help = "Step of the trained model to load (default to the best model)")
    parser.add_argument("--seq_len", default = 1024, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 0, type = int, help = "Number of jobs")
    parser.add_argument("-r", "--resume", action = "store_true", help = "Whether or not to resume evaluation from previous run")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATE FUNCTION
##################################################

def evaluate(data: Union[np.array, torch.tensor], encoding: dict, stem: str, eval_dir: str) -> dict:
    """Evaluate the results."""

    # save results
    np.save(file = f"{eval_dir}/npy/{stem}.npy", arr = data) # save as a numpy array
    encode.save_csv_codes(filepath = f"{eval_dir}/csv/{stem}.csv", data = data) # save as a .csv file
    music = decode.decode(codes = data, encoding = encoding) # convert to a BetterMusic object
    music.trim(end = music.resolution * 64) # trim the music
    music.save_json(path = f"{eval_dir}/json/{stem}.json") # save as a BetterMusic .json file

    # return a dictionary
    if len(music.tracks) == 0:
        return {eval_metric: np.nan for eval_metric in EVAL_METRICS}
    else:
        return {
            EVAL_METRICS[0]: pitch_class_entropy(music = music),
            EVAL_METRICS[1]: scale_consistency(music = music),
            EVAL_METRICS[2]: groove_consistency(music = music, measure_resolution = 4 * music.resolution)
        }

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # create eval_dir if necessary
    EVAL_DIR = f"{args.output_dir}/eval_conditional"
    if not exists(EVAL_DIR):
        mkdir(EVAL_DIR)

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = f"{EVAL_DIR}/evaluate.log", mode = "a"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{EVAL_DIR}/evaluate_args.json"
    logging.info(f"Saved arguments to {args_output_filepath}")
    utils.save_args(filepath = args_output_filepath, args = args)
    del args_output_filepath

    ##################################################


    # LOAD IN STUFF
    ##################################################

    # load training configurations
    train_args_filepath = f"{args.output_dir}/train_args.json"
    logging.info(f"Loading training arguments from: {train_args_filepath}")
    train_args = utils.load_json(filepath = train_args_filepath)
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
    del train_args_filepath

    # make sure the output directory exists
    for key in EVAL_SUBSETS:
        key_base_dir = f"{EVAL_DIR}/{key}"
        if not exists(key_base_dir):
            mkdir(key_base_dir) # create base_dir if necessary
        for subdir in ("npy", "csv", "json"):
            if not exists(key_base_dir_subdir := f"{key_base_dir}/{subdir}"):
                mkdir(key_base_dir_subdir) # create base_dir with subdir if necessary
        del subdir, key_base_dir_subdir
    del key, key_base_dir # clear up memory

    # get the specified device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = train_args["max_seq_len"])
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate, batch_size = 1, shuffle = False)

    # create the model
    logging.info(f"Creating the model...")
    model = music_x_transformers.MusicXTransformer(
        dim = train_args["dim"],
        encoding = encoding,
        depth = train_args["layers"],
        heads = train_args["heads"],
        max_seq_len = train_args["max_seq_len"],
        max_beat = train_args["max_beat"],
        rotary_pos_emb = train_args["rel_pos_emb"],
        use_abs_pos_emb = train_args["abs_pos_emb"],
        emb_dropout = train_args["dropout"],
        attn_dropout = train_args["dropout"],
        ff_dropout = train_args["dropout"],
    ).to(device)

    # load the checkpoint
    CHECKPOINT_DIR = f"{args.output_dir}/checkpoints"
    if args.model_steps is None:
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.{PARTITIONS[1]}.pth"
    else:
        checkpoint_filepath = f"{CHECKPOINT_DIR}/model_{args.model_steps}.pth"
    model.load_state_dict(state_dict = torch.load(f = checkpoint_filepath, map_location = device))
    logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
    model.eval()

    # to output evaluation metrics
    output_filepath = f"{EVAL_DIR}/{basename(EVAL_DIR)}.csv"
    output_columns_must_be_written = not (exists(output_filepath) and args.resume)
    if output_columns_must_be_written: # if column names need to be written
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
    i_start = 0 # index from which to start evaluating
    if not output_columns_must_be_written:
        previous = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # load in previous values
        if len(previous) > 0:
            i_start = int(previous["i"].max(axis = 0)) + 1 # update start index
            print(f"Resuming from index {i_start}.")
        del previous
    
    # get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    ##################################################


    # EVALUATE
    ##################################################

    # instantiate iterable
    test_iter = iter(test_data_loader)

    # iterate over the dataset
    with torch.no_grad():
        for i in tqdm(iterable = range(i_start, len(test_data_loader) if args.n_samples is None else args.n_samples), desc = "Evaluating"):

            # get new batch
            batch = next(test_iter)
            stem = f"{i}_0"
            result = {eval_subset: None for eval_subset in EVAL_SUBSETS}

            # GROUND TRUTH
            ##################################################

            # get ground truth
            truth_np = batch["seq"].numpy()

            # add to results
            result[EVAL_SUBSETS[0]] = evaluate(data = truth_np[0], encoding = encoding, stem = stem, eval_dir = f"{EVAL_DIR}/{EVAL_SUBSETS[0]}")

            ##################################################

            # UNCONDITIONED GENERATION
            ##################################################

            # get output start tokens
            tgt_start = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long, device = device).reshape(1, 1, len(encoding["dimensions"]))

            # generate new samples
            generated = model.generate(
                seq_in = tgt_start,
                seq_len = args.seq_len,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat")
            )
            generated_seq = torch.cat(tensors = (tgt_start, generated), dim = 1).cpu().numpy()

            # add to results
            result[EVAL_SUBSETS[1]] = evaluate(data = generated_seq[0], encoding = encoding, stem = stem, eval_dir = f"{EVAL_DIR}/{EVAL_SUBSETS[1]}")

            ##################################################

            # OUTPUT STATISTICS
            ##################################################

            pd.DataFrame(data = [
                [i, batch["path"][0], True, stem,] + list(result[EVAL_SUBSETS[0]].values()),
                [i, batch["path"][0], False, stem,] + list(result[EVAL_SUBSETS[1]].values()),
                ], columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")
            
            ##################################################

    # log statistics
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # load in previous values
    for eval_subset in EVAL_SUBSETS:
        logging.info(f"{eval_subset.upper()}:")
        results_subset = results[results["subset"] == eval_subset]
        for eval_metric in EVAL_METRICS:
            logging.info(f"  - {eval_metric}: mean = {np.nanmean(a = results_subset[eval_metric], axis = 0):.4f}, std = {np.nanstd(a = results_subset[eval_metric], axis = 0):.4f}")

    ##################################################

##################################################
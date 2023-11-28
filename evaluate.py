# README
# Phillip Long
# November 27, 2023

# Evaluate a neural network.

# python /home/pnlong/model_musescore/evaluate.py

# Absolute positional embedding (APE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/ape -ns 100 -g 0

# Relative positional embedding (RPE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/rpe -ns 100 -g 0

# No positional embedding (NPE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/npe -ns 100 -g 0


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
import os
from os.path import exists
from collections import defaultdict

import muspy
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import dataset
import music_x_transformers
import representation
import utils

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"
OUTPUT_DIR = "/data2/pnlong/musescore/data"

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
    parser.add_argument("--sequence_length", default = 1024, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 0, type = int, help = "Number of jobs")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATE FUNCTION
##################################################

def evaluate(data: np.array, encoding: dict, stem: str, eval_dir: str):
    """Evaluate the results."""

    # save as a numpy array
    np.save(file = f"{eval_dir}/npy/{stem}.npy", arr = data)

    # save as a CSV file
    representation.save_csv_codes(filepath = f"{eval_dir}/csv/{stem}.csv", data = data)

    # convert to a BetterMusic object
    music = representation.decode(codes = data, encoding = encoding)

    # trim the music
    music.trim(end = music.resolution * 64)

    # save as a BetterMusic JSON file
    music.save(path = f"{eval_dir}/json/{stem}.json")

    if len(music.tracks) == 0:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan
        }
    else:
        return {
            "pitch_class_entropy": muspy.pitch_class_entropy(music = music),
            "scale_consistency": muspy.scale_consistency(music = music),
            "groove_consistency": muspy.groove_consistency(music = music, measure_resolution = 4 * music.resolution)
        }

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(f"{args.output_dir}/evaluate.log", "w"), logging.StreamHandler(sys.stdout)])

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{args.output_dir}/evaluate-args.json"
    logging.info(f"Saved arguments to {args_output_filepath}")
    utils.save_args(filepath = args_output_filepath, args = args)
    del args_output_filepath

    ##################################################


    # LOAD IN STUFF
    ##################################################

    # load training configurations
    train_args_filepath = f"{args.output_dir}/train-args.json"
    logging.info(f"Loading training arguments from: {train_args_filepath}")
    train_args = utils.load_json(filepath = train_args_filepath)
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
    del train_args_filepath

    # make sure the output directory exists
    EVAL_DIR = f"{args.output_dir}/eval"
    if not exists(args.output_dir): os.makedirs(args.output_dir)
    if not exists(EVAL_DIR): os.mkdir(EVAL_DIR)
    for key in ("truth", "unconditioned"):
        key_basedir = f"{EVAL_DIR}/{key}"
        if not exists(key_basedir): os.mkdir(key_basedir)
        if not exists(key_basedir_npy := f"{key_basedir}/npy"): os.mkdir(key_basedir_npy)
        if not exists(key_basedir_csv := f"{key_basedir}/csv"): os.mkdir(key_basedir_csv)
        if not exists(key_basedir_json := f"{key_basedir}/json"): os.mkdir(key_basedir_json)
    del key_basedir, key_basedir_npy, key_basedir_csv, key_basedir_json # clear up memory

    # get the specified device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_sequence_length = train_args["max_sequence_length"])
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate)

    # create the model
    logging.info(f"Creating the model...")
    model = music_x_transformers.MusicXTransformer(
        dim = train_args["dim"],
        encoding = encoding,
        depth = train_args["layers"],
        heads = train_args["heads"],
        max_sequence_length = train_args["max_sequence_length"],
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
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.pth"
    else:
        checkpoint_filepath = f"{CHECKPOINT_DIR}/model_{args.model_steps}.pth"
    model.load_state_dict(state_dict = torch.load(f = checkpoint_filepath, map_location = device))
    logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
    model.eval()

    # get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    ##################################################


    # EVALUATE
    ##################################################

    # instantiate results and iterables
    results = defaultdict(list)
    test_iter = iter(test_data_loader)

    # iterate over the dataset
    with torch.no_grad():
        for i in tqdm(iterable = range(len(test_data_loader) if args.n_samples is None else args.n_samples), desc = "Evaluating"):

            batch = next(test_iter)

            # GROUND TRUTH
            ##################################################

            # get ground truth
            truth_np = batch["sequence"].numpy()

            # add to results
            result = evaluate(data = truth_np[0], encoding = encoding, stem = f"{i}_0", eval_dir = EVAL_DIR / "truth")
            results["truth"].append(result)

            ##################################################

            # UNCONDITIONED GENERATION
            ##################################################

            # get output start tokens
            tgt_start = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long, device = device).reshape(1, 1, len(encoding["dimensions"]))

            # generate new samples
            generated = model.generate(
                sequence_in = tgt_start,
                sequence_length = args.sequence_length,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat")
            )
            generated_sequence = torch.cat(tensors = (tgt_start, generated), dim = 1).cpu().numpy()

            # add to results
            result = evaluate(data = generated_sequence[0], encoding = encoding, stem = f"{i}_0", eval_dir = EVAL_DIR / "unconditioned")
            results["unconditioned"].append(result)

            ##################################################

    # output statistics
    for exp, result in results.items():
        logging.info(f"{exp.upper()}:")
        for key in result[0]:
            logging.info(f"  - {key}: mean = {np.nanmean(a = [r[key] for r in result]):.4f}, std = {np.nanstd(a = [r[key]for r in result]):.4f}")

    ##################################################

##################################################
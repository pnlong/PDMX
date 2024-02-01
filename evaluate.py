# README
# Phillip Long
# January 25, 2024

# Evaluate the distribution of expressive features.

# python /home/pnlong/model_musescore/evaluate.py


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, dirname
from os import makedirs
from typing import Union
import math

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
import parse_mscz
import expressive_features_plots
import train
import evaluate_baseline

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"
OUTPUT_DIR = "/data2/pnlong/musescore/data"
EVAL_STEM = "eval"

DEFAULT_MAX_PREFIX_LEN = 50

CONDITIONAL_TYPES = train.MASKS
GENERATION_TYPES = train.MASKS[:-1]
EVAL_TYPES = ["joint",] + [f"conditional_{conditional_type}_{generation_type}" for conditional_type in CONDITIONAL_TYPES for generation_type in GENERATION_TYPES]
PLOT_TYPES = ["baseline", "n_expressive_features", "density", "summary", "sparsity", "loss_for_perplexity"]

BASELINE_COLUMNS = evaluate_baseline.OUTPUT_COLUMNS[2:]
N_EXPRESSIVE_FEATURES_COLUMNS = ["path", "n"]

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
    parser.add_argument("--seq_len", default = train.DEFAULT_MAX_SEQ_LEN, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    parser.add_argument("--prefix_len", default = DEFAULT_MAX_PREFIX_LEN, type = int, help = "Number of notes in prefix sequence for generation")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 0, type = int, help = "Number of jobs")
    parser.add_argument("-t", "--truth", action = "store_true", help = "Whether or not to run the evaluation on the paths provided")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATION METRICS
##################################################

def baseline(music: BetterMusic, stem: str, output_filepath: str):
    """Calculate baseline metrics just for consistency."""
    if len(music.tracks) == 0:
        result = {eval_metric: np.nan for eval_metric in evaluate_baseline.EVAL_METRICS}
    else:
        result = {
            evaluate_baseline.EVAL_METRICS[0]: evaluate_baseline.pitch_class_entropy(music = music),
            evaluate_baseline.EVAL_METRICS[1]: evaluate_baseline.scale_consistency(music = music),
            evaluate_baseline.EVAL_METRICS[2]: evaluate_baseline.groove_consistency(music = music, measure_resolution = 4 * music.resolution)
        }
    pd.DataFrame(data = [[stem,] + list(result.values())], columns = BASELINE_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a")

def n_expressive_features(expressive_features: pd.DataFrame, stem: str, output_filepath: str):
    """Given a dataframe with expressive features, output the number of expressive features to a csv file."""
    pd.DataFrame(data = [dict(zip(N_EXPRESSIVE_FEATURES_COLUMNS, (stem, len(expressive_features))))], columns = N_EXPRESSIVE_FEATURES_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output

def density(expressive_features: pd.DataFrame, music: BetterMusic, stem: str, output_filepath: str):
    """Given a dataframe with expressive features, calculate the density of expressive features and output to a csv file."""
    density = {
        "time_steps": (music.song_length / len(expressive_features)) if len(expressive_features) != 0 else 0,
        "seconds": (music.metrical_time_to_absolute_time(time_steps = music.song_length) / len(expressive_features)) if len(expressive_features) != 0 else 0
        } # time unit per expressive feature
    density["path"] = stem
    density = pd.DataFrame(data = [density], columns = expressive_features_plots.DENSITY_COLUMNS)
    density.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output

def summary(expressive_features: pd.DataFrame, stem: str, output_filepath: str):
    """Given a dataframe with expressive features, summarize the types of features present and output to a csv file."""
    summary = expressive_features[["type", "value"]].groupby(by = "type").size().reset_index(drop = False).rename(columns = {0: expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS[-1]}) # group by type
    summary["path"] = utils.rep(x = stem, times = len(summary))
    summary = summary[expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS] # ensure we have just the columns we need
    summary.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output

def sparsity(expressive_features: pd.DataFrame, music: BetterMusic, stem: str, output_filepath: str):
    """Given a dataframe with expressive features, calculate the sparsity of expressive features and output to a csv file."""
    sparsity = expressive_features[["type", "value", "time", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME]]
    sparsity = sparsity.rename(columns = {"time": "time_steps", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME: "seconds"})
    sparsity["path"] = utils.rep(x = stem, times = len(sparsity))
    sparsity["beats"] = sparsity["time_steps"] / music.resolution
    sparsity["fraction"] = sparsity["time_steps"] / (music.song_length + expressive_features_plots.DIVIDE_BY_ZERO_CONSTANT)
    for successive_distance_column, distance_column in zip(expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS, expressive_features_plots.DISTANCE_COLUMNS): # add successive times columns
        sparsity[successive_distance_column] = sparsity[distance_column]
    sparsity = sparsity[expressive_features_plots.SPARSITY_COLUMNS].sort_values("time_steps").reset_index(drop = True) # sort by increasing times
    sparsity = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity, columns = expressive_features_plots.DISTANCE_COLUMNS) # calculate distances
    expressive_feature_types = pd.unique(expressive_features["type"]) # get types of expressive features
    distance = pd.DataFrame(columns = expressive_features_plots.SPARSITY_COLUMNS)
    for expressive_feature_type in expressive_feature_types: # get distances between successive features of the same type
        distance_for_expressive_feature_type = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity[sparsity["type"] == expressive_feature_type], columns = expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS) # calculate sparsity for certain feature type
        distance = pd.concat(objs = (distance, distance_for_expressive_feature_type), axis = 0, ignore_index = False) # append to overall distance
    distance = distance.sort_index(axis = 0) # sort by index (return to original index)
    distance.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output

def loss_for_perplexity(model, seq: torch.tensor, mask: torch.tensor, stem: str, loss_for_perplexity_columns: str, output_filepath: str):
    """Calculate loss values for perplexity and output to a csv file."""

    # get and wrangle losses
    _, losses_for_perplexity = model(seq = seq, mask = mask, return_list = True, reduce = False, return_output = False)
    losses_for_perplexity = torch.sum(input = losses_for_perplexity.cpu().squeeze(dim = 0), dim = 0).tolist() # get rid of batch size dimension, then sum across sequence length

    # convert to dataframe and output
    losses_for_perplexity = pd.DataFrame(data = [[stem, sum(losses_for_perplexity)] + losses_for_perplexity], columns = loss_for_perplexity_columns) # convert to dataframe
    losses_for_perplexity.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output


##################################################


# EVALUATE FUNCTION
##################################################

def evaluate(
        data: Union[np.array, torch.tensor],
        encoding: dict,
        stem: str,
        eval_dir: str,
        output_filepaths: list,
        calculate_loss_for_perplexity: bool = False,
        model: music_x_transformers.MusicXTransformer = None, # for loss for perplexity
        seq: torch.tensor = None, # for loss for perplexity
        mask: torch.tensor = None, # for loss for perplexity
        loss_for_perplexity_columns: list = None # for loss for perplexity
        ):
    """
    Evaluate the results by calculating expressive feature density and sparsity as well as what types of features are present.
    Outputs values to csv, returns nothing.
    """

    # save results
    path = f"{eval_dir}/{stem}"
    np.save(file = f"{path}.npy", arr = data) # save as a numpy array
    encode.save_csv_codes(filepath = f"{path}.csv", data = data) # save as a .csv file
    music = decode.decode(codes = data, encoding = encoding) # convert to a BetterMusic object
    music.trim(end = music.resolution * 64) # trim the music
    music.save_json(path = f"{path}.json") # save as a BetterMusic .json file

    # extract data
    data = encode.extract_data(music = music, use_implied_duration = False, include_annotation_class_name = True) # duration doesn't matter for our purposes here
    data = data[data[:, 0] == representation.EXPRESSIVE_FEATURE_TYPE_STRING] # filter down to just expressive features
    expressive_features = pd.DataFrame(data = data[:, [representation.DIMENSIONS.index("time"), data.shape[1] - 1, representation.DIMENSIONS.index("value")]], columns = parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[:1] + parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[2:]) # create pandas data frame
    del data
    expressive_features[parse_mscz.TIME_IN_SECONDS_COLUMN_NAME] = expressive_features["time"].apply(lambda time: music.metrical_time_to_absolute_time(time_steps = time)) # add time in seconds column
    expressive_features = expressive_features[parse_mscz.EXPRESSIVE_FEATURE_COLUMNS] # reorder columns
    expressive_features["type"] = expressive_features["type"].apply(lambda expressive_feature_type: expressive_feature_type.replace("Spanner", ""))

    # baseline metrics
    baseline(music = music, stem = stem, output_filepath = output_filepaths[0])

    # number of expressive features
    n_expressive_features(expressive_features = expressive_features, stem = stem, output_filepath = output_filepaths[1])

    # density
    density(expressive_features = expressive_features, music = music, stem = stem, output_filepath = output_filepaths[2])

    # feature types summary
    summary(expressive_features = expressive_features, stem = stem, output_filepath = output_filepaths[3])

    # distance/sparsity
    sparsity(expressive_features = expressive_features, music = music, stem = stem, output_filepath = output_filepaths[4])

    if calculate_loss_for_perplexity:
        if any((var is None for var in (model, seq, mask, loss_for_perplexity_columns))): # make sure arguments are valid
            raise ValueError("To calculate loss values for perplexity, valid `model`, `seq`, `mask`, and `loss_for_perplexity_columns` arguments must be supplied.")
        loss_for_perplexity(model = model, seq = seq, mask = mask, stem = stem, loss_for_perplexity_columns = loss_for_perplexity_columns, output_filepath = output_filepaths[5])

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()
    # args.output_dir += "/baseline_aug_ape_20M" # for debugging

    # create eval_dir if necessary
    EVAL_DIR = (f"{dirname(args.paths)}/{evaluate_baseline.TRUTH_DIR_STEM}" if args.truth else args.output_dir) + f"/{EVAL_STEM}"
    if not exists(EVAL_DIR):
        makedirs(EVAL_DIR)
    # make sure the output directory exists
    if args.truth:
        eval_output_dir = f"{EVAL_DIR}/data"
        if not exists(eval_output_dir):
            makedirs(eval_output_dir) # create if necessary
    else:
        eval_output_dirs = {eval_type: f"{EVAL_DIR}/{eval_type}/data" for eval_type in EVAL_TYPES}
        for eval_output_dir in eval_output_dirs.values():
            if not exists(eval_output_dir):
                makedirs(eval_output_dir) # create if necessary

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # determine columns
    if not args.truth:
        LOSS_FOR_PERPLEXITY_COLUMNS = ["path"] + [f"loss_{field}" for field in [train.ALL_STRING] + encoding["dimensions"]]

    # output filepaths for data used in plots
    if args.truth:
        output_filepaths = [f"{EVAL_DIR}/eval_{plot_type}.csv" for plot_type in PLOT_TYPES]
    else:
        output_filepaths = {eval_type: [f"{dirname(eval_output_dirs[eval_type])}/eval_{plot_type}.csv" for plot_type in PLOT_TYPES] for eval_type in EVAL_TYPES}
        for eval_type in output_filepaths.keys():
            if not all(exists(path) for path in output_filepaths[eval_type][:-1]):
                pd.DataFrame(columns = BASELINE_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][0], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")
                pd.DataFrame(columns = N_EXPRESSIVE_FEATURES_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][1], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # n expressive features
                pd.DataFrame(columns = expressive_features_plots.DENSITY_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][2], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # density
                pd.DataFrame(columns = expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][3], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # features summary
                pd.DataFrame(columns = expressive_features_plots.SPARSITY_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][4], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # sparsity
                if not args.truth:
                    pd.DataFrame(columns = LOSS_FOR_PERPLEXITY_COLUMNS).to_csv(path_or_buf = output_filepaths[eval_type][5], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # sparsity

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

    if args.truth:

        # create the dataset
        logging.info(f"Creating the data loader...")
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = train.DEFAULT_MAX_SEQ_LEN, max_beat = train.DEFAULT_MAX_BEAT, use_augmentation = False)

    # load model if necessary
    else:

        # load training configurations
        train_args_filepath = f"{args.output_dir}/train_args.json"
        logging.info(f"Loading training arguments from: {train_args_filepath}")
        train_args = utils.load_json(filepath = train_args_filepath)
        logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
        del train_args_filepath

        # get the specified device
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
        logging.info(f"Using device: {device}")

        # create the dataset
        logging.info(f"Creating the data loader...")
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = train_args["max_seq_len"], max_beat = train_args["max_beat"], use_augmentation = False, is_baseline = ("baseline" in args.output_dir))

        # create the model
        logging.info(f"Creating the model...")
        max_beat = train_args["max_beat"]
        model = music_x_transformers.MusicXTransformer(
            dim = train_args["dim"],
            encoding = encoding,
            depth = train_args["layers"],
            heads = train_args["heads"],
            max_seq_len = train_args["max_seq_len"],
            max_beat = max_beat,
            rotary_pos_emb = train_args["rel_pos_emb"],
            use_abs_pos_emb = train_args["abs_pos_emb"],
            emb_dropout = train_args["dropout"],
            attn_dropout = train_args["dropout"],
            ff_dropout = train_args["dropout"],
        ).to(device)

        # load the checkpoint
        CHECKPOINT_DIR = f"{args.output_dir}/checkpoints"
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.{train.PARTITIONS[1]}.pth"
        model.load_state_dict(state_dict = torch.load(f = checkpoint_filepath, map_location = device))
        logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
        model.eval()
        
        # get special tokens
        sos = encoding["type_code_map"]["start-of-song"]
        eos = encoding["type_code_map"]["end-of-song"]

    ##################################################


    # EVALUATE
    ##################################################

    # create data loader and instantiate iterable
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate, batch_size = 1, shuffle = False)
    test_iter = iter(test_data_loader)

    # iterate over the dataset
    with torch.no_grad():
        for i in tqdm(iterable = range(len(test_data_loader) if args.n_samples is None else args.n_samples), desc = "Evaluating"):

            # get new batch
            batch = next(test_iter)
            stem = i

            if args.truth:

                # GROUND TRUTH
                ##################################################

                # evaluate
                evaluate(
                    data = batch["seq"].squeeze(dim = 0).numpy(),
                    encoding = encoding,
                    stem = stem,
                    eval_dir = eval_output_dir,
                    output_filepaths = output_filepaths,
                    calculate_loss_for_perplexity = False
                    )

                ##################################################

            else:

                # DIFFERENT EVALUATION TYPES
                ##################################################

                # default prefix sequence
                prefix_default = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long, device = device).reshape(1, 1, len(encoding["dimensions"])).to(device)
                n_notes_so_far = 0
                last_prefix_index = -1
                for j in range(batch["seq"].shape[1]):
                    if n_notes_so_far > args.prefix_len: # make sure the prefix isn't too long
                        break
                    if batch["seq"][:, j, 0] in (encoding["type_code_map"]["note"], encoding["type_code_map"]["grace-note"]): # if event is a note
                        n_notes_so_far += 1 # increment
                        last_prefix_index = j # update last prefix index
                prefix_conditional_default = batch["seq"][:, :last_prefix_index + 1, :].to(device)
                if prefix_conditional_default.shape[1] == 0: # make sure the prefix conditional default is not just empty
                    prefix_conditional_default = prefix_default

                for eval_type in EVAL_TYPES:

                    # DETERMINE PREFIX SEQUENCE AND ANY RELEVANT VARIABLES
                    ##################################################

                    if eval_type == EVAL_TYPES[0]: # joint
                        notes_only = False
                        prefix = prefix_default
                    else: # conditional
                        conditional_type, generation_type = eval_type.split("_")[1:]
                        notes_only = (generation_type != train.ALL_STRING)
                        if conditional_type == CONDITIONAL_TYPES[1]: # conditional on notes only
                            prefix = prefix_conditional_default[:, (prefix_conditional_default[0, :, 0] != encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]), :]
                        elif conditional_type == CONDITIONAL_TYPES[2]: # conditional on expressive features only
                            prefix = prefix_conditional_default[:, ((prefix_conditional_default[0, :, 0] != encoding["type_code_map"]["note"]) & (prefix_conditional_default[0, :, 0] != encoding["type_code_map"]["grace-note"])), :]
                        else: # conditional on everything
                            prefix = prefix_conditional_default

                    ##################################################

                    # GENERATION
                    ##################################################

                    # generate new samples
                    generated = model.generate(
                        seq_in = prefix,
                        seq_len = args.seq_len,
                        eos_token = eos,
                        temperature = args.temperature,
                        filter_logits_fn = args.filter,
                        filter_thres = args.filter_threshold,
                        monotonicity_dim = ("type", "beat"),
                        notes_only = notes_only
                    )
                    generated = torch.cat(tensors = (prefix, generated), dim = 1).cpu().squeeze(dim = 0).numpy() # wrangle a bit

                    # evaluate
                    evaluate(
                        data = generated,
                        encoding = encoding,
                        stem = stem,
                        eval_dir = eval_output_dirs[eval_type],
                        output_filepaths = output_filepaths[eval_type],
                        calculate_loss_for_perplexity = True,
                        model = model, seq = batch["seq"].to(device), mask = batch["mask"].to(device), loss_for_perplexity_columns = LOSS_FOR_PERPLEXITY_COLUMNS
                        )
                    
                    ##################################################

                ##################################################  

    ##################################################


    # MAKE PLOTS
    ##################################################

    # output perplexity if available
    if not args.truth:
        logging.info("".join(("=" for _ in range(25))) + " PERPLEXITY " + "".join(("=" for _ in range(25))))
        for eval_type in EVAL_TYPES:
            eval_type_fancy = eval_type.title() if eval_type == EVAL_TYPES[0] else (eval_type.split("_")[2].title() + "s conditional on " + eval_type.split("_")[1].title() + "s:")
            logging.info("\n" + eval_type_fancy)
            losses_for_perplexity = pd.read_csv(filepath_or_buffer = output_filepaths[eval_type][-1], sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # load in previous values
            for field in losses_for_perplexity.columns[1:]:
                logging.info(f"  - {field.replace('loss_', '').title()}: {math.exp(-math.log(np.sum(losses_for_perplexity[field]))):.4f}")

    ##################################################

##################################################
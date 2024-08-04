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
from typing import Union, Callable
import math
import multiprocessing
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from read_mscz.music import MusicExpress
import dataset
import music_x_transformers
import representation
from representation import ENCODING_FILEPATH
import encode
import decode
import utils
import parse_mscz
import expressive_features_plots
import train
import evaluate_baseline
from evaluate_baseline import pad, unpad_prefix # for padding batches

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
OUTPUT_DIR = "/data2/pnlong/musescore/data"
EVAL_STEM = "eval"

DEFAULT_MAX_PREFIX_LEN = 50

CONDITIONAL_TYPES = train.MASKS
GENERATION_TYPES = train.MASKS
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
    parser.add_argument("-bs", "--batch_size", default = 8, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 4, type = int, help = "Number of jobs")
    parser.add_argument("-t", "--truth", action = "store_true", help = "Whether or not to run the evaluation on the paths provided")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATION METRICS
##################################################

def baseline(music: MusicExpress, stem: str, output_filepath: str):
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

def density(expressive_features: pd.DataFrame, music: MusicExpress, stem: str, output_filepath: str):
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

def sparsity(expressive_features: pd.DataFrame, music: MusicExpress, stem: str, output_filepath: str):
    """Given a dataframe with expressive features, calculate the sparsity of expressive features and output to a csv file."""
    sparsity = expressive_features[["type", "value", "time", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME]]
    sparsity = sparsity.rename(columns = {"time": "time_steps", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME: "seconds"})
    sparsity["path"] = utils.rep(x = stem, times = len(sparsity))
    sparsity["beats"] = sparsity["time_steps"] / music.resolution
    sparsity["fraction"] = sparsity["time_steps"] / (music.song_length + expressive_features_plots.DIVIDE_BY_ZERO_CONSTANT)
    for successive_distance_column, distance_column in zip(expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS, expressive_features_plots.DISTANCE_COLUMNS): # add successive times columns
        sparsity[successive_distance_column] = sparsity[distance_column]
    sparsity = sparsity[expressive_features_plots.SPARSITY_COLUMNS].sort_values("time_steps").reset_index(drop = True) # sort by increasing times
    expressive_feature_types = pd.unique(expressive_features["type"]) # get types of expressive features
    distance = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity, columns = expressive_features_plots.DISTANCE_COLUMNS) # calculate distances
    for expressive_feature_type in expressive_feature_types: # get distances between successive features of the same type
        distance_for_expressive_feature_type = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity[sparsity["type"] == expressive_feature_type], columns = expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS) # calculate sparsity for certain feature type
        distance.loc[distance_for_expressive_feature_type.index, expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS] = distance_for_expressive_feature_type[expressive_features_plots.SUCCESSIVE_DISTANCE_COLUMNS]
    distance.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = False, index = False, mode = "a") # output

def loss_for_perplexity(model, seq: torch.tensor, mask: torch.tensor, conditional_mask: torch.tensor, stem: str, loss_for_perplexity_columns: str, output_filepath: str):
    """Calculate loss values for perplexity and output to a csv file."""

    # get and wrangle losses
    _, losses_for_perplexity = model(seq = seq, mask = mask, return_list = True, reduce = False, return_output = False, conditional_mask = conditional_mask)
    losses_for_perplexity = torch.mean(input = losses_for_perplexity.cpu().squeeze(dim = 0), dim = 0).tolist() # get rid of batch size dimension, then sum across sequence length

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
        conditional_mask: torch.tensor = None, # for loss for perplexity
        loss_for_perplexity_columns: list = None, # for loss for perplexity
        unidimensional_decoding_function: Callable = representation.get_unidimensional_coding_functions(encoding = encode.DEFAULT_ENCODING)[-1],
        ):
    """
    Evaluate the results by calculating expressive feature density and sparsity as well as what types of features are present.
    Outputs values to csv, returns nothing.
    """

    # save results
    path = f"{eval_dir}/{stem}"
    np.save(file = f"{path}.npy", arr = data) # save as a numpy array
    # encode.save_csv_codes(filepath = f"{path}.csv", data = data) # save as a .csv file
    music = decode.decode(codes = data, encoding = encoding, infer_metrical_time = True, unidimensional_decoding_function = unidimensional_decoding_function) # convert to a MusicExpress object
    # music.trim(end = music.resolution * 64) # trim the music

    # extract data
    data = encode.extract_data(music = music, use_implied_duration = False, include_annotation_class_name = True) # duration doesn't matter for our purposes here
    data = data[data[:, 0] == representation.EXPRESSIVE_FEATURE_TYPE_STRING] # filter down to just expressive features
    expressive_features = pd.DataFrame(data = data[:, [representation.DIMENSIONS.index("time"), data.shape[1] - 1, representation.DIMENSIONS.index("value")]], columns = parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[:1] + parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[2:]) # create pandas data frame
    del data
    expressive_features[parse_mscz.TIME_IN_SECONDS_COLUMN_NAME] = expressive_features["time"].apply(lambda time: music.metrical_time_to_absolute_time(time_steps = time)) # add time in seconds column
    expressive_features = expressive_features[parse_mscz.EXPRESSIVE_FEATURE_COLUMNS] # reorder columns
    expressive_features["type"] = expressive_features["type"].apply(lambda expressive_feature_type: expressive_feature_type.replace("Spanner", ""))

    # for scenarios with no time or key signature changes
    time_signatures = (expressive_features["type"] == "TimeSignature")
    if sum(time_signatures) <= 1:
        expressive_features = expressive_features[~time_signatures]
    key_signatures = (expressive_features["type"] == "KeySignature")
    if sum(key_signatures) <= 1:
        expressive_features = expressive_features[~key_signatures]
    del time_signatures, key_signatures

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
        if any((var is None for var in (model, seq, mask, conditional_mask, loss_for_perplexity_columns))): # make sure arguments are valid
            raise ValueError("To calculate loss values for perplexity, valid `model`, `seq`, `mask`, `conditional_mask`, and `loss_for_perplexity_columns` arguments must be supplied.")
        loss_for_perplexity(model = model, seq = seq, mask = mask, conditional_mask = conditional_mask, stem = stem, loss_for_perplexity_columns = loss_for_perplexity_columns, output_filepath = output_filepaths[5])

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

    # helper function for writing column names
    def write_column_names(output_filepaths: list):
        pd.DataFrame(columns = BASELINE_COLUMNS).to_csv(path_or_buf = output_filepaths[0], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")
        pd.DataFrame(columns = N_EXPRESSIVE_FEATURES_COLUMNS).to_csv(path_or_buf = output_filepaths[1], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # n expressive features
        pd.DataFrame(columns = expressive_features_plots.DENSITY_COLUMNS).to_csv(path_or_buf = output_filepaths[2], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # density
        pd.DataFrame(columns = expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS).to_csv(path_or_buf = output_filepaths[3], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # features summary
        pd.DataFrame(columns = expressive_features_plots.SPARSITY_COLUMNS).to_csv(path_or_buf = output_filepaths[4], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # sparsity
        if not args.truth:
            pd.DataFrame(columns = LOSS_FOR_PERPLEXITY_COLUMNS).to_csv(path_or_buf = output_filepaths[5], sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w") # sparsity

    # output filepaths for data used in plots
    if args.truth:
        output_filepaths = [f"{EVAL_DIR}/eval_{plot_type}.csv" for plot_type in PLOT_TYPES]
        write_column_names(output_filepaths = output_filepaths)
    else:
        output_filepaths = {eval_type: [f"{dirname(eval_output_dirs[eval_type])}/eval_{plot_type}.csv" for plot_type in PLOT_TYPES] for eval_type in EVAL_TYPES}
        for output_filepaths_at_eval_type in output_filepaths.values():
            write_column_names(output_filepaths = output_filepaths_at_eval_type)
        
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
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = train.DEFAULT_MAX_SEQ_LEN, use_augmentation = False, unidimensional = False)

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
        max_seq_len = train_args["max_seq_len"]
        conditioning = train_args["conditioning"]
        unidimensional = train_args.get("unidimensional", False)
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, conditioning = conditioning, max_seq_len = max_seq_len, use_augmentation = False, is_baseline = ("baseline" in args.output_dir), unidimensional = unidimensional, for_generation = True)

        # create the model
        logging.info(f"Creating the model...")
        use_absolute_time = encoding["use_absolute_time"]
        model = music_x_transformers.MusicXTransformer(
            dim = train_args["dim"],
            encoding = encoding,
            depth = train_args["layers"],
            heads = train_args["heads"],
            max_seq_len = max_seq_len,
            max_temporal = encoding["max_" + ("time" if use_absolute_time else "beat")],
            rotary_pos_emb = train_args["rel_pos_emb"],
            use_abs_pos_emb = train_args["abs_pos_emb"],
            emb_dropout = train_args["dropout"],
            attn_dropout = train_args["dropout"],
            ff_dropout = train_args["dropout"],
            unidimensional = unidimensional,
        ).to(device)

        # load the checkpoint
        CHECKPOINT_DIR = f"{args.output_dir}/checkpoints"
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.{train.PARTITIONS[1]}.pth"
        model_state_dict = torch.load(f = checkpoint_filepath, map_location = device)
        model.load_state_dict(state_dict = model_state_dict)
        logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
        model.eval()
        
        # get special tokens
        if unidimensional:
            unidimensional_encoding_order = encoding["unidimensional_encoding_order"]
        type_dim = (unidimensional_encoding_order if unidimensional else encoding["dimensions"]).index("type")
        sos = encoding["type_code_map"]["start-of-song"]
        eos = encoding["type_code_map"]["end-of-song"]
        note_token, grace_note_token = encoding["type_code_map"]["note"], encoding["type_code_map"]["grace-note"]
        expressive_feature_token = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]
        conditional_on_controls = (bool(train_args.get("conditional", False)) or bool(train_args.get("econditional", False)))
        notes_are_controls = bool(train_args.get("econditional", False))
        print(f"Notes are controls: {notes_are_controls}")
        event_tokens = torch.tensor(data = (note_token, grace_note_token) if (not notes_are_controls) else (expressive_feature_token,), device = device)
        control_tokens = torch.tensor(data = (note_token, grace_note_token) if notes_are_controls else (expressive_feature_token,), device = device)
        is_anticipation = (conditioning == encode.CONDITIONINGS[-1])
        sigma = train_args["sigma"]
        unidimensional_encoding_function, unidimensional_decoding_function = representation.get_unidimensional_coding_functions(encoding = encoding)
        get_type_field = lambda prefix_conditional: prefix_conditional[type_dim::model.decoder.net.n_tokens_per_event] if unidimensional else prefix_conditional[:, type_dim] # helper function for quickly accessing the type field

    ##################################################


    # EVALUATE
    ##################################################

    # create data loader and instantiate iterable
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = test_dataset.collate, batch_size = args.batch_size, shuffle = False)
    test_iter = iter(test_data_loader)
    chunk_size = int(args.batch_size / 2)

    # iterate over the dataset
    with torch.no_grad():
        n_iterations = (int((args.n_samples - 1) / args.batch_size) + 1) if args.n_samples is not None else len(test_data_loader)
        for i in tqdm(iterable = range(n_iterations), desc = "Evaluating"):

            # get new batch
            batch = next(test_iter)
            stem = i
            if (args.n_samples is not None) and (i == n_iterations - 1): # if last iteration
                n_samples = args.n_samples % args.batch_size
                if (n_samples == 0):
                    n_samples = args.batch_size
                batch["seq"] = batch["seq"][:n_samples]
                batch["mask"] = batch["mask"][:n_samples]
                batch["seq_len"] = batch["seq_len"][:n_samples]
                batch["path"] = batch["path"][:n_samples]

            if args.truth:

                # GROUND TRUTH
                ##################################################

                # get ground truth
                truth = batch["seq"].numpy()

                # add to results
                def evaluate_helper(j: int):
                    return evaluate(data = truth[j],
                                    encoding = encoding,
                                    stem = f"{stem}_{j}",
                                    eval_dir = eval_output_dir,
                                    output_filepaths = output_filepaths,
                                    calculate_loss_for_perplexity = False)
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    results = pool.map(func = evaluate_helper, iterable = range(len(truth)), chunksize = chunk_size)

                ##################################################

            else:

                # DIFFERENT EVALUATION TYPES
                ##################################################

                # default prefix sequence
                prefix_default = torch.repeat_interleave(input = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long).reshape(1, 1, len(encoding["dimensions"])), repeats = batch["seq"].shape[0], dim = 0).cpu().numpy()
                if unidimensional:
                    for dimension_index in range(prefix_default.shape[-1]):
                        prefix_default[..., dimension_index] = unidimensional_encoding_function(code = prefix_default[..., dimension_index], dimension_index = dimension_index)
                    prefix_default = prefix_default[..., unidimensional_encoding_order].reshape(prefix_default.shape[0], -1)
                prefix_default = torch.from_numpy(prefix_default).to(device)
                n_events_so_far = utils.rep(x = 0, times = len(batch["seq"]))
                last_sos_token_indicies, last_prefix_indicies = utils.rep(x = -1, times = len(batch["seq"])), utils.rep(x = -1, times = len(batch["seq"]))
                for seq_index in range(len(last_prefix_indicies)):
                    for j in range(type_dim, batch["seq"].shape[1], model.decoder.net.n_tokens_per_event):
                        current_event_type = batch["seq"][seq_index, j + type_dim] if unidimensional else batch["seq"][seq_index, j, type_dim]
                        if (n_events_so_far[seq_index] > args.prefix_len) or (current_event_type == eos): # make sure the prefix isn't too long, or if end of song token, no end of song tokens in prefix
                            last_prefix_indicies[seq_index] = j - model.decoder.net.n_tokens_per_event
                            break
                        elif current_event_type in event_tokens: # if an event
                            n_events_so_far[seq_index] += 1 # increment
                            last_prefix_indicies[seq_index] = j # update last prefix index
                        elif current_event_type == sos:
                            last_sos_token_indicies[seq_index] = j
                prefix_conditional_default = [batch["seq"][seq_index, last_sos_token_indicies[seq_index]:(last_prefix_indicies[seq_index] + model.decoder.net.n_tokens_per_event)] for seq_index in range(len(last_prefix_indicies))] # truncate to last prefix for each sequence in batch
                for seq_index in range(len(prefix_conditional_default)):
                    if len(prefix_conditional_default[seq_index]) == 0: # make sure the prefix conditional default is not just empty
                        prefix_conditional_default[seq_index] = prefix_default[0]
                prefix_conditional_default = pad(data = prefix_conditional_default).to(device) # pad

                for eval_type in EVAL_TYPES:

                    # DETERMINE PREFIX SEQUENCE AND ANY RELEVANT VARIABLES
                    ##################################################

                    joint = (eval_type == EVAL_TYPES[0])
                    if joint: # joint
                        prefix = prefix_default
                    else: # conditional
                        conditional_type, generation_type = eval_type.split("_")[1:]
                        if conditional_type == CONDITIONAL_TYPES[1]: # conditional on notes only
                            prefix = pad(data = [prefix_conditional[(get_type_field(prefix_conditional = prefix_conditional) != expressive_feature_token)] for prefix_conditional in prefix_conditional_default]).to(device)
                        elif conditional_type == CONDITIONAL_TYPES[2]: # conditional on expressive features only
                            prefix = pad(data = [prefix_conditional[torch.logical_and(input = (get_type_field(prefix_conditional = prefix_conditional) != note_token), other = (get_type_field(prefix_conditional = prefix_conditional) != grace_note_token))] for prefix_conditional in prefix_conditional_default]).to(device)
                        else: # conditional on everything
                            prefix = prefix_conditional_default

                    # skip irrelevant eval types
                    if conditional_on_controls and (
                        joint or 
                        (((notes_are_controls) and (generation_type in (GENERATION_TYPES[0], GENERATION_TYPES[1]))) or # notes are controls and generation is notes
                         ((not notes_are_controls) and (generation_type == (GENERATION_TYPES[0], GENERATION_TYPES[2]))) # expressive features are controls and generation is expressive features
                        )):
                        continue

                    ##################################################

                    # GENERATION
                    ##################################################

                    eval_dir = eval_output_dirs[eval_type]
                    generated_output_filepaths = glob(pathname = f"{eval_dir}/{stem}_*.npy")
                    generated_output_filepaths_exist = tuple(map(exists, generated_output_filepaths))
                    if (not all(generated_output_filepaths_exist)) or (len(generated_output_filepaths_exist) == 0):

                        # generate new samples
                        generated = model.generate(
                            seq_in = prefix,
                            seq_len = args.seq_len,
                            eos_token = eos,
                            temperature = args.temperature,
                            filter_logits_fn = args.filter,
                            filter_thres = args.filter_threshold,
                            monotonicity_dim = ("type", "time" if use_absolute_time else "beat"),
                            joint = joint,
                            notes_are_controls = notes_are_controls,
                            is_anticipation = is_anticipation,
                            sigma = sigma
                        )

                        # concatenate generation to prefix
                        generated = torch.cat(tensors = (prefix, generated), dim = 1).cpu().numpy() # wrangle a bit

                    else:

                        # load in previously generated samples
                        try:
                            generated = list(map(np.load, generated_output_filepaths))
                        except EOFError:
                            generated = [None] * len(generated_output_filepaths)
                            for j, generated_output_filepath in enumerate(generated_output_filepaths):
                                try:
                                    generated[j] = np.load(file = generated_output_filepath)
                                except EOFError:
                                    pass
                            generated = [generation for generation in generated if generation is not None] # remove None values
                            if len(generated) == 0:
                                continue
                        generated = dataset.pad(data = generated, front = True)

                    # add to results
                    def evaluate_helper(j: int):
                        # setup for loss for perplexity calculations
                        seq = batch["seq"][j].unsqueeze(dim = 0).to(device)
                        mask = batch["mask"][j].unsqueeze(dim = 0).to(device)
                        conditional_mask = torch.ones(size = seq.shape if unidimensional else seq.shape[:-1], dtype = torch.bool, device = device)
                        if not joint:
                            conditional_mask = torch.isin(seq[:, 0::model.decoder.net.n_tokens_per_event] if unidimensional else seq[..., 0], test_elements = event_tokens, invert = False) # filter to just events for conditional masking
                        conditional_mask = conditional_mask[:, model.decoder.net.n_tokens_per_event:]
                        # evaluate
                        evaluate(data = unpad_prefix(prefix = generated[j], sos_token = sos, pad_value = model.decoder.pad_value, n_tokens_per_event = model.decoder.net.n_tokens_per_event),
                                 encoding = encoding,
                                 stem = f"{stem}_{j}",
                                 eval_dir = eval_dir,
                                 output_filepaths = output_filepaths[eval_type],
                                 calculate_loss_for_perplexity = True,
                                 model = model, seq = seq, mask = mask, conditional_mask = conditional_mask, loss_for_perplexity_columns = LOSS_FOR_PERPLEXITY_COLUMNS,
                                 unidimensional_decoding_function = unidimensional_decoding_function
                                 )
                    for j in range(len(generated)):
                        evaluate_helper(j = j)
                    
                    ##################################################

                ##################################################  

    ##################################################


    # MAKE PLOTS
    ##################################################

    # output perplexity if available
    if not args.truth:
        logging.info("\n" + "".join(("=" for _ in range(25))) + " PERPLEXITY " + "".join(("=" for _ in range(25))))
        for eval_type in EVAL_TYPES:
            eval_type_fancy = eval_type.title() if eval_type == EVAL_TYPES[0] else (eval_type.split("_")[2].title() + "s conditional on " + eval_type.split("_")[1].title() + "s:")
            logging.info("\n" + eval_type_fancy)
            losses_for_perplexity = pd.read_csv(filepath_or_buffer = output_filepaths[eval_type][-1], sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # load in previous values
            for field in losses_for_perplexity.columns[1:]:
                loss_by_field = np.sum(losses_for_perplexity[field])
                logging.info(f"  - {field.replace('loss_', '').title()}: " + (f"{math.exp(-math.log(loss_by_field)):.4f}" if loss_by_field != 0 else "NaN"))

    ##################################################

##################################################
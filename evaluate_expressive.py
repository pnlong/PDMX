# README
# Phillip Long
# January 25, 2024

# Evaluate the distribution of expressive features.

# python /home/pnlong/model_musescore/evaluate_expressive.py


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, dirname
from os import mkdir, makedirs
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.nn.functional import cross_entropy
from tqdm import tqdm

import dataset
import music_x_transformers
import representation
import encode
import decode
import utils
import parse_mscz
import expressive_features_plots
from train import PARTITIONS, NA_VALUE, DEFAULT_MAX_SEQ_LEN

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"
OUTPUT_DIR = "/data2/pnlong/musescore/data"

DISTANCE_COLUMNS = expressive_features_plots.SPARSITY_COLUMNS[expressive_features_plots.SPARSITY_COLUMNS.index("time_steps"):expressive_features_plots.SPARSITY_COLUMNS.index("time_steps" + expressive_features_plots.SPARSITY_SUCCESSIVE_SUFFIX)]
SUCCESSIVE_DISTANCE_COLUMNS = [distance_column + expressive_features_plots.SPARSITY_SUCCESSIVE_SUFFIX for distance_column in DISTANCE_COLUMNS]

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
    parser.add_argument("-t", "--truth", action = "store_true", help = "Whether or not to run the evaluation on the paths provided")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATE FUNCTION
##################################################

def evaluate(data: Union[np.array, torch.tensor], encoding: dict, stem: str, eval_dir: str):
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
    data = pd.DataFrame(data = data[:, [representation.DIMENSIONS.index("time"), data.shape[1] - 1, representation.DIMENSIONS.index("value")]], columns = parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[:1] + parse_mscz.EXPRESSIVE_FEATURE_COLUMNS[2:]) # create pandas data frame
    expressive_features[parse_mscz.TIME_IN_SECONDS_COLUMN_NAME] = expressive_features["time"].apply(lambda time: music.metrical_time_to_absolute_time(time_steps = time)) # add time in seconds column
    expressive_features = expressive_features[parse_mscz.EXPRESSIVE_FEATURE_COLUMNS] # reorder columns
    expressive_features["type"] = expressive_features["type"].apply(lambda expressive_feature_type: expressive_feature_type.replace("Spanner", ""))

    # density
    density = {
        "time_steps": music.song_length / len(expressive_features),
        "seconds": music.metrical_time_to_absolute_time(time_steps = music.song_length) / len(expressive_features)
        } # time unit per expressive feature
    density["path"] = stem
    density = pd.DataFrame(data = [density], columns = expressive_features_plots.DENSITY_COLUMNS)
    density.to_csv(path_or_buf = list(PLOT_DATA_OUTPUT_FILEPATHS.values())[0], sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a") # output

    # feature types summary
    feature_types_summary = expressive_features[["type", "value"]].groupby(by = "type").size().reset_index(drop = False).rename(columns = {0: expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS[-1]}) # group by type
    feature_types_summary["path"] = utils.rep(x = stem, times = len(feature_types_summary))
    feature_types_summary = feature_types_summary[expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS] # ensure we have just the columns we need
    feature_types_summary.to_csv(path_or_buf = list(PLOT_DATA_OUTPUT_FILEPATHS.values())[1], sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a") # output

    # distance/sparsity, add some columns, set up for calculation of distance
    sparsity = expressive_features[["type", "value", "time", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME]]
    sparsity = sparsity.rename(columns = {"time": "time_steps", parse_mscz.TIME_IN_SECONDS_COLUMN_NAME: "seconds"})
    sparsity["path"] = utils.rep(x = stem, times = len(sparsity))
    sparsity["beats"] = sparsity["time_steps"] / music.resolution
    sparsity["fraction"] = sparsity["time_steps"] / (music.song_length + expressive_features_plots.DIVIDE_BY_ZERO_CONSTANT)
    for successive_distance_column, distance_column in zip(SUCCESSIVE_DISTANCE_COLUMNS, DISTANCE_COLUMNS): # add successive times columns
        sparsity[successive_distance_column] = sparsity[distance_column]
    sparsity = sparsity[expressive_features_plots.SPARSITY_COLUMNS].sort_values("time_steps").reset_index(drop = True) # sort by increasing times
    sparsity = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity, columns = DISTANCE_COLUMNS) # calculate distances
    expressive_feature_types = pd.unique(expressive_features["type"]) # get types of expressive features
    distance = pd.DataFrame(columns = expressive_features_plots.SPARSITY_COLUMNS)
    for expressive_feature_type in expressive_feature_types: # get distances between successive features of the same type
        distance_for_expressive_feature_type = expressive_features_plots.calculate_difference_between_successive_entries(df = sparsity[sparsity["type"] == expressive_feature_type], columns = SUCCESSIVE_DISTANCE_COLUMNS) # calculate sparsity for certain feature type
        distance = pd.concat(objs = (distance, distance_for_expressive_feature_type), axis = 0, ignore_index = False) # append to overall distance
    distance = distance.sort_index(axis = 0) # sort by index (return to original index)
    distance.to_csv(path_or_buf = list(PLOT_DATA_OUTPUT_FILEPATHS.values())[2], sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a") # output

##################################################


# PERPLEXITY
##################################################

def perplexity(predicted: np.array, expected: np.array, stem: str):
    """Calculate perplexity. Outputs values to csv."""

    losses = cross_entropy(input = predicted, target = expected, reduction = "none") # calculate losses
    perplexities = torch.exp(input = losses) # calculate perplexities
    perplexities_total = torch.sum(input = losses, dim = 1).exp() # calculate perplexity per event
    perplexities = torch.cat(tensors = (perplexities, perplexities_total), dim = 1) # make into a single matrix
    perplexities = pd.DataFrame(data = perplexities, columns = PERPLEXITY_COLUMNS[1:]) # convert to dataframe
    perplexities["path"] = utils.rep(x = stem, times = len(perplexities)) # add path column
    perplexities = perplexities[PERPLEXITY_COLUMNS] # reorder columns
    perplexities.to_csv(path_or_buf = list(PLOT_DATA_OUTPUT_FILEPATHS.values())[3], sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a") # output

##################################################

# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # create eval_dir if necessary
    EVAL_DIR = (f"{dirname(args.paths)}/eval_truth" if args.truth else args.output_dir) + "/eval_expressive"
    if not exists(EVAL_DIR):
        makedirs(EVAL_DIR)
    # make sure the output directory exists
    eval_output_dir = f"{EVAL_DIR}/data"
    if not exists(eval_output_dir):
        mkdir(eval_output_dir) # create if necessary

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # determine perplexity columns
    if not args.truth:
        PERPLEXITY_COLUMNS = ["path"] + [f"ppl_{field}" for field in ["total"] + encoding["dimensions"]]

    # output filepaths for data used in plots
    PLOT_DATA_OUTPUT_FILEPATHS = {plot_type : f"{EVAL_DIR}/eval_{plot_type}.csv" for plot_type in ("density", "summary", "sparsity", "perplexity")}
    if not any(exists(path) for path in tuple(PLOT_DATA_OUTPUT_FILEPATHS.values())[:3]):
        pd.DataFrame(columns = expressive_features_plots.DENSITY_COLUMNS).to_csv(path_or_buf = PLOT_DATA_OUTPUT_FILEPATHS[list(PLOT_DATA_OUTPUT_FILEPATHS.keys())[0]], sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w") # density
        pd.DataFrame(columns = expressive_features_plots.FEATURE_TYPES_SUMMARY_COLUMNS).to_csv(path_or_buf = PLOT_DATA_OUTPUT_FILEPATHS[list(PLOT_DATA_OUTPUT_FILEPATHS.keys())[1]], sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w") # features summary
        pd.DataFrame(columns = expressive_features_plots.SPARSITY_COLUMNS).to_csv(path_or_buf = PLOT_DATA_OUTPUT_FILEPATHS[list(PLOT_DATA_OUTPUT_FILEPATHS.keys())[2]], sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w") # sparsity
        if not args.truth:
            pd.DataFrame(columns = PERPLEXITY_COLUMNS).to_csv(path_or_buf = PLOT_DATA_OUTPUT_FILEPATHS[list(PLOT_DATA_OUTPUT_FILEPATHS.keys())[3]], sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w") # sparsity

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
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = DEFAULT_MAX_SEQ_LEN)

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
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = train_args["max_seq_len"])

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

                # get ground truth
                truth = batch["seq"].unsqueeze(dim = 0).numpy()

                # evaluate
                evaluate(data = truth, encoding = encoding, stem = stem, eval_dir = eval_output_dir)

                ##################################################

            else:

                # GENERATION
                ##################################################

                # get output start tokens
                prefix = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long, device = device).reshape(1, 1, len(encoding["dimensions"]))

                # generate new samples
                generated = model.generate(
                    seq_in = prefix,
                    seq_len = args.seq_len,
                    eos_token = eos,
                    temperature = args.temperature,
                    filter_logits_fn = args.filter,
                    filter_thres = args.filter_threshold,
                    monotonicity_dim = ("type", "beat"),
                    notes_only = False
                )
                generated = torch.cat(tensors = (prefix, generated), dim = 1).unsqueeze(dim = 0).cpu()

                # evaluate
                evaluate(data = generated.numpy(), encoding = encoding, stem = stem, eval_dir = eval_output_dir)

                # calculate perplexity
                perplexity(predicted = generated, expected = batch["seq"], stem = stem)

                ##################################################

    ##################################################


    # MAKE PLOTS
    ##################################################
                
    # make sure the plots directory exists
    plot_dir = f"{EVAL_DIR}/plots"
    if not exists(plot_dir):
        mkdir(plot_dir)

    # make plots    
    plot_output_filepaths = [f"{plot_dir}/{plot_type}.png" for plot_type in PLOT_DATA_OUTPUT_FILEPATHS.keys()]
    _ = expressive_features_plots.make_density_plot(output_filepath = plot_output_filepaths[0])
    expressive_feature_types = expressive_features_plots.make_feature_summary_plot(output_filepath = plot_output_filepaths[1])
    _ = expressive_features_plots.make_sparsity_plot(output_filepath_prefix = plot_output_filepaths[2].split(".")[0], expressive_feature_types = expressive_feature_types[::-1])

    # output perplexity if available
    if not args.truth:
        perplexities = pd.read_csv(filepath_or_buffer = list(PLOT_DATA_OUTPUT_FILEPATHS.values())[3], sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # load in previous values
        logging.info("PERPLEXITY:")
        for field in perplexities.columns[1:]:
            logging.info(f"  - {field.replace('ppl_', '').title()}: mean = {np.nanmean(a = perplexities[field], axis = 0):.4f}, std = {np.nanstd(a = perplexities[field], axis = 0):.4f}")

    ##################################################

##################################################
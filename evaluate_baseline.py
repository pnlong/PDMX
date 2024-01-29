# README
# Phillip Long
# November 27, 2023

# Evaluate a model same as the MMT paper. Metric functions copied from
# https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html

# python /home/pnlong/model_musescore/evaluate_baseline.py


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, basename, dirname
from os import mkdir, makedirs
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
from train import PARTITIONS, NA_VALUE, DEFAULT_MAX_SEQ_LEN

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS = f"{DATA_DIR}/test.txt"
EVAL_STEM = "eval_baseline"
TRUTH_DIR_STEM = "eval_truth"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"
OUTPUT_DIR = "/data2/pnlong/musescore/data"
EVAL_METRICS = ["pitch_class_entropy", "scale_consistency", "groove_consistency"]
OUTPUT_COLUMNS = ["i", "original_path", "path",] + EVAL_METRICS

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
    parser.add_argument("--seq_len", default = 1024, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_thres", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 0, type = int, help = "Number of jobs")
    parser.add_argument("-r", "--resume", action = "store_true", help = "Whether or not to resume evaluation from previous run")
    parser.add_argument("-t", "--truth", action = "store_true", help = "Whether or not to run the evaluation on the paths provided")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# EVALUATION METRICS
##################################################

def pitch_class_entropy(music: BetterMusic) -> float:
    """Return the entropy of the normalized note pitch class histogram.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#pitch_class_entropy

    The pitch class entropy is defined as the Shannon entropy of the
    normalized note pitch class histogram. Drum tracks are ignored.
    Return NaN if no note is found. This metric is used in [1].

    .. math::
        pitch\_class\_entropy = -\sum_{i = 0}^{11}{
            P(pitch\_class=i) \times \log_2 P(pitch\_class=i)}

    Parameters
    ----------
    music : :class:`read_mscz.BetterMusic`
        Music object to evaluate.

    Returns
    -------
    float
        Pitch class entropy.

    See Also
    --------
    :func:`muspy.pitch_entropy` :
        Compute the entropy of the normalized pitch histogram.

    References
    ----------
    1. Shih-Lun Wu and Yi-Hsuan Yang, "The Jazz Transformer on the Front
       Line: Exploring the Shortcomings of AI-composed Music through
       Quantitative Measures”, in Proceedings of the 21st International
       Society for Music Information Retrieval Conference, 2020.

    """
    counter = np.zeros(12)
    for track in music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            counter[note.pitch % 12] += 1
    denominator = counter.sum()
    if denominator < 1:
        return math.nan
    prob = counter / denominator
    return muspy.metrics.metrics._entropy(prob = prob)


def scale_consistency(music: BetterMusic) -> float:
    """Return the largest pitch-in-scale rate.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#scale_consistency

    The scale consistency is defined as the largest pitch-in-scale rate
    over all major and minor scales. Drum tracks are ignored. Return NaN
    if no note is found. This metric is used in [1].

    .. math::
        scale\_consistency = \max_{root, mode}{
            pitch\_in\_scale\_rate(root, mode)}

    Parameters
    ----------
    music : :class:`read_mscz.BetterMusic`
        Music object to evaluate.

    Returns
    -------
    float
        Scale consistency.

    See Also
    --------
    :func:`muspy.pitch_in_scale_rate` :
        Compute the ratio of pitches in a certain musical scale.

    References
    ----------
    1. Olof Mogren, "C-RNN-GAN: Continuous recurrent neural networks
       with adversarial training," in NeuIPS Workshop on Constructive
       Machine Learning, 2016.

    """
    max_in_scale_rate = 0.0
    for mode in ("major", "minor"):
        for root in range(12):
            rate = muspy.metrics.metrics.pitch_in_scale_rate(music = music, root = root, mode = mode)
            if math.isnan(rate):
                return math.nan
            if rate > max_in_scale_rate:
                max_in_scale_rate = rate
    return max_in_scale_rate


def groove_consistency(music: BetterMusic, measure_resolution: int) -> float:
    """Return the groove consistency.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#groove_consistency

    The groove consistency is defined as the mean hamming distance of
    the neighboring measures.

    .. math::
        groove\_consistency = 1 - \frac{1}{T - 1} \sum_{i = 1}^{T - 1}{
            d(G_i, G_{i + 1})}

    Here, :math:`T` is the number of measures, :math:`G_i` is the binary
    onset vector of the :math:`i`-th measure (a one at position that has
    an onset, otherwise a zero), and :math:`d(G, G')` is the hamming
    distance between two vectors :math:`G` and :math:`G'`. Note that
    this metric only works for songs with a constant time signature.
    Return NaN if the number of measures is less than two. This metric
    is used in [1].

    Parameters
    ----------
    music : :class:`read_mscz.BetterMusic`
        Music object to evaluate.
    measure_resolution : int
        Time steps per measure.

    Returns
    -------
    float
        Groove consistency.

    References
    ----------
    1. Shih-Lun Wu and Yi-Hsuan Yang, "The Jazz Transformer on the Front
       Line: Exploring the Shortcomings of AI-composed Music through
       Quantitative Measures”, in Proceedings of the 21st International
       Society for Music Information Retrieval Conference, 2020.

    """

    length = max(track.get_end_time() for track in music.tracks)
    if measure_resolution < 1:
        raise ValueError("Measure resolution must be a positive integer.")

    n_measures = int(length / measure_resolution) + 1
    if n_measures < 2:
        return math.nan

    groove_patterns = np.zeros(shape = (n_measures, measure_resolution), dtype = bool)

    for track in music.tracks:
        for note in track.notes:
            measure, position = divmod(int(note.time), int(measure_resolution)) # ensure these values are integers, as they will be used for indexing
            if not groove_patterns[measure, position]:
                groove_patterns[measure, position] = 1

    hamming_distance = np.count_nonzero(a = (groove_patterns[:-1] != groove_patterns[1:]))

    return 1 - (hamming_distance / (measure_resolution * (n_measures - 1)))

##################################################


# EVALUATE FUNCTION
##################################################

def evaluate(data: Union[np.array, torch.tensor], encoding: dict, stem: str, eval_dir: str) -> dict:
    """Evaluate the results."""

    # save results
    np.save(file = f"{eval_dir}/{stem}.npy", arr = data) # save as a numpy array
    encode.save_csv_codes(filepath = f"{eval_dir}/{stem}.csv", data = data) # save as a .csv file
    music = decode.decode(codes = data, encoding = encoding) # convert to a BetterMusic object
    music.trim(end = music.resolution * 64) # trim the music
    music.save_json(path = f"{eval_dir}/{stem}.json") # save as a BetterMusic .json file

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
    EVAL_DIR = (f"{dirname(args.paths)}/{TRUTH_DIR_STEM}" if args.truth else args.output_dir) + f"/{EVAL_STEM}"
    if not exists(EVAL_DIR):
        makedirs(EVAL_DIR)
    # make sure the output directory exists
    eval_output_dir = f"{EVAL_DIR}/data"
    if not exists(eval_output_dir):
        mkdir(eval_output_dir) # create if necessary

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

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

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
        # kwargs = {"depth": train_args["layers"], "heads": train_args["heads"], "max_seq_len": train_args["max_seq_len"], "max_beat": train_args["max_beat"], "rotary_pos_emb": train_args["rel_pos_emb"], "use_abs_pos_emb": train_args["abs_pos_emb"], "emb_dropout": train_args["dropout"], "attn_dropout": train_args["dropout"], "ff_dropout": train_args["dropout"]} # for debugging

        # load the checkpoint
        CHECKPOINT_DIR = f"{args.output_dir}/checkpoints"
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.{PARTITIONS[1]}.pth"
        model.load_state_dict(state_dict = torch.load(f = checkpoint_filepath, map_location = device))
        logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
        model.eval()
        
        # get special tokens
        sos = encoding["type_code_map"]["start-of-song"]
        eos = encoding["type_code_map"]["end-of-song"]

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

    ##################################################


    # EVALUATE
    ##################################################

    # create data loader and instantiate iterable
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate, batch_size = 1, shuffle = False)
    test_iter = iter(test_data_loader)

    # iterate over the dataset
    with torch.no_grad():
        for i in tqdm(iterable = range(i_start, len(test_data_loader) if args.n_samples is None else args.n_samples), desc = "Evaluating"):

            # get new batch
            batch = next(test_iter)
            stem = i

            if args.truth:

                # GROUND TRUTH
                ##################################################

                # get ground truth
                truth = batch["seq"].squeeze(dim = 0).numpy()
                truth = truth[truth[:, 0] != encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]] # filter out expressive features

                # add to results
                result = evaluate(data = truth, encoding = encoding, stem = stem, eval_dir = eval_output_dir)

                ##################################################

            else:

                # UNCONDITIONED GENERATION
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
                    filter_thres = args.filter_thres,
                    monotonicity_dim = ("type", "beat"),
                    notes_only = True
                )
                # kwargs = {"eos_token": eos, "temperature": args.temperature, "filter_logits_fn": args.filter, "filter_thres": args.filter_thres, "monotonicity_dim": ("type", "beat")}
                generated = torch.cat(tensors = (prefix, generated), dim = 1).squeeze(dim = 0).cpu().numpy()

                # add to results
                result = evaluate(data = generated, encoding = encoding, stem = stem, eval_dir = eval_output_dir)

                ##################################################

            # OUTPUT STATISTICS
            ##################################################

            pd.DataFrame(data = [[i, batch["path"][0], stem,] + list(result.values())], columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")
            
            ##################################################

    # log statistics
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # load in previous values
    for eval_metric in EVAL_METRICS:
        logging.info(f"- {eval_metric}: mean = {np.nanmean(a = results[eval_metric], axis = 0):.4f}, std = {np.nanstd(a = results[eval_metric], axis = 0):.4f}")

    ##################################################

##################################################
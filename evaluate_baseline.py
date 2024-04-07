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
from typing import Union, List, Callable
import math
import multiprocessing
from glob import glob

import muspy
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
from train import PARTITIONS, NA_VALUE, DEFAULT_MAX_SEQ_LEN

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/home/pnlong/musescore/datav"
PATHS = f"{DATA_DIR}/test.txt"
EVAL_STEM = "eval_baseline"
TRUTH_DIR_STEM = "eval_truth"
OUTPUT_DIR = f"{DATA_DIR}/model"
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
    parser.add_argument("-bs", "--batch_size", default = 8, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 4, type = int, help = "Number of jobs")
    parser.add_argument("-r", "--resume", action = "store_true", help = "Whether or not to resume evaluation from previous run")
    parser.add_argument("-t", "--truth", action = "store_true", help = "Whether or not to run the evaluation on the paths provided")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# HELPER PADDING FUNCTION
##################################################

def pad(data: List[torch.tensor]) -> torch.tensor:
    max_seq_len = max(len(seq) for seq in data) # get the longest length of a sequence
    padded = [torch.from_numpy(np.pad(array = seq.cpu(), pad_width = [(max_seq_len - len(seq), 0)] + [() if (len(seq.shape) == 1) else (0, 0)])) for seq in data] # pad so all of same length
    padded = torch.stack(tensors = padded, dim = 0) # combine into a single tensor
    return padded

def unpad_prefix(prefix: np.array, sos_token: int, pad_value: float = dataset.PAD_VALUE, n_tokens_per_event: int = 1) -> np.array:
    """Unpad the prefix"""
    
    # if unidimensional, reshape to multidimensional
    unidimensional = (len(prefix.shape) == 1)
    if unidimensional:
        # make sure prefix is correct length
        if (len(prefix) % n_tokens_per_event) > 0:
            prefix = np.pad(array = prefix, pad_width = [0, n_tokens_per_event - (len(prefix) % n_tokens_per_event)], mode = "constant", constant_values = pad_value)
        prefix = prefix.reshape(int(len(prefix) / n_tokens_per_event), n_tokens_per_event)

    # mask out pad values
    prefix_mask = ~np.all(a = (prefix == pad_value), axis = -1)
    if (not unidimensional) and (sos_token == pad_value):
        prefix_mask[np.argmax(a = prefix_mask, axis = 0) - 1] = True
    prefix = prefix[prefix_mask, :]

    # reflatten if unidimensional
    if unidimensional:
        prefix = prefix.flatten()
    
    # return the unpadded tensor
    return prefix

##################################################


# EVALUATION METRICS
##################################################

def pitch_class_entropy(music: MusicExpress) -> float:
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
    music : :class:`read_mscz.MusicExpress`
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


def scale_consistency(music: MusicExpress) -> float:
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
    music : :class:`read_mscz.MusicExpress`
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


def groove_consistency(music: MusicExpress, measure_resolution: int) -> float:
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
    music : :class:`read_mscz.MusicExpress`
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

def evaluate(data: Union[np.array, torch.tensor], encoding: dict, stem: str, eval_dir: str, unidimensional_decoding_function: Callable = representation.get_unidimensional_coding_functions(encoding = encode.DEFAULT_ENCODING)[-1]) -> dict:
    """Evaluate the results."""

    # save results
    path = f"{eval_dir}/{stem}"
    np.save(file = f"{path}.npy", arr = data) # save as a numpy array
    # encode.save_csv_codes(filepath = f"{path}.csv", data = data) # save as a .csv file
    music = decode.decode(codes = data, encoding = encoding, unidimensional_decoding_function = unidimensional_decoding_function) # convert to a MusicExpress object
    # music.trim(end = music.resolution * 64) # trim the music

    # return a dictionary
    if len(music.tracks) == 0:
        return {"stem": stem, EVAL_METRICS[0]: np.nan, EVAL_METRICS[1]: np.nan, EVAL_METRICS[2]: np.nan}
    else:
        return {
            "stem": stem,
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
        test_dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_seq_len = DEFAULT_MAX_SEQ_LEN, use_augmentation = False, unidimensional = False)
        expressive_feature_token = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]

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
        # kwargs = {"depth": train_args["layers"], "heads": train_args["heads"], "max_seq_len": max_seq_len, "max_temporal": encoding["max_" + ("time" if use_absolute_time else "beat")], "rotary_pos_emb": train_args["rel_pos_emb"], "use_abs_pos_emb": train_args["abs_pos_emb"], "emb_dropout": train_args["dropout"], "attn_dropout": train_args["dropout"], "ff_dropout": train_args["dropout"]} # for debugging

        # load the checkpoint
        CHECKPOINT_DIR = f"{args.output_dir}/checkpoints"
        checkpoint_filepath = f"{CHECKPOINT_DIR}/best_model.{PARTITIONS[1]}.pth"
        model_state_dict = torch.load(f = checkpoint_filepath, map_location = device)
        model.load_state_dict(state_dict = model_state_dict)
        logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
        model.eval()
        
        # get special tokens
        if unidimensional:
            unidimensional_encoding_order = encoding["unidimensional_encoding_order"]
        sos = encoding["type_code_map"]["start-of-song"]
        eos = encoding["type_code_map"]["end-of-song"]
        is_anticipation = (conditioning == encode.CONDITIONINGS[-1])
        sigma = train_args["sigma"] # if use_absolute_time else encode.SIGMA_METRICAL
        unidimensional_encoding_function, unidimensional_decoding_function = representation.get_unidimensional_coding_functions(encoding = encoding)
        
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
    test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, num_workers = args.jobs, collate_fn = test_dataset.collate, batch_size = args.batch_size, shuffle = False)
    test_iter = iter(test_data_loader)
    chunk_size = int(args.batch_size / 2)

    # iterate over the dataset
    with torch.no_grad():
        n_iterations = int((args.n_samples - 1) / args.batch_size) + 1
        for i in tqdm(iterable = range(i_start, len(test_data_loader) if (args.n_samples is None) else n_iterations), desc = "Evaluating"):

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
                truth = batch["seq"]
                truth = pad(data = [truth[j][truth[j, :, 0] != expressive_feature_token] for j in range(len(truth))]) # filter out expressive features

                # add to results
                def evaluate_helper(j: int) -> dict:
                    return evaluate(data = truth[j], encoding = encoding, stem = f"{stem}_{j}", eval_dir = eval_output_dir)
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    results = pool.map(func = evaluate_helper, iterable = range(len(truth)), chunksize = chunk_size)

                ##################################################

            else:

                # UNCONDITIONED GENERATION
                ##################################################

                eval_dir = eval_output_dir
                generated_output_filepaths = glob(pathname = f"{eval_dir}/{stem}_*.npy")
                generated_output_filepaths_exist = tuple(map(exists, generated_output_filepaths))
                if (not all(generated_output_filepaths_exist)) or (len(generated_output_filepaths_exist) == 0):

                    # get output start tokens
                    prefix = torch.repeat_interleave(input = torch.tensor(data = [sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long).reshape(1, 1, len(encoding["dimensions"])), repeats = len(batch["seq"]), dim = 0).cpu().numpy()
                    if unidimensional:
                        for dimension_index in range(prefix.shape[-1]):
                            prefix[..., dimension_index] = unidimensional_encoding_function(code = prefix[..., dimension_index], dimension_index = dimension_index)
                        prefix = prefix[..., unidimensional_encoding_order].flatten(start_dim = 1)
                    prefix = torch.from_numpy(prefix).to(device)

                    # generate new samples
                    generated = model.generate(
                        seq_in = prefix,
                        seq_len = args.seq_len,
                        eos_token = eos,
                        temperature = args.temperature,
                        filter_logits_fn = args.filter,
                        filter_thres = args.filter_thres,
                        monotonicity_dim = ("type", "time" if use_absolute_time else "beat"),
                        joint = False,
                        notes_are_controls = False,
                        is_anticipation = is_anticipation,
                        sigma = sigma
                    )
                    # kwargs = {"eos_token": eos, "temperature": args.temperature, "filter_logits_fn": args.filter, "filter_thres": args.filter_thres, "monotonicity_dim": ("type", "time" if use_absolute_time else "beat")}
                    generated = torch.cat(tensors = (prefix, generated), dim = 1).cpu().numpy()

                else:
                
                    # load in previously generated samples
                    generated = dataset.pad(data = list(map(np.load, generated_output_filepaths)), front = True)

                # add to results
                def evaluate_helper(j: int) -> dict:
                    return evaluate(data = unpad_prefix(prefix = generated[j], sos_token = sos, pad_value = model.decoder.pad_value, n_tokens_per_event = model.decoder.net.n_tokens_per_event), encoding = encoding, stem = f"{stem}_{j}", eval_dir = eval_dir, unidimensional_decoding_function = unidimensional_decoding_function)
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    results = pool.map(func = evaluate_helper, iterable = range(len(generated)), chunksize = chunk_size)

                ##################################################

            # OUTPUT STATISTICS
            ##################################################

            pd.DataFrame(data = [[i, batch["path"][j]] + list(results[j].values()) for j in range(len(results))], columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")
            
            ##################################################

    # log statistics
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # load in previous values
    for eval_metric in EVAL_METRICS:
        logging.info(f"- {eval_metric}: mean = {np.nanmean(a = results[eval_metric], axis = 0):.4f}, std = {np.nanstd(a = results[eval_metric], axis = 0):.4f}")

    ##################################################

##################################################
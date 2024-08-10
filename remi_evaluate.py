# README
# Phillip Long
# August 3, 2024

# Evaluate a REMI-Style model.

# python /home/pnlong/model_musescore/remi_evaluate.py

# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, basename, isdir
from os import mkdir, listdir
from typing import Union, List
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import x_transformers

from dataset_full import pitch_class_entropy, scale_consistency, groove_consistency, MMT_STATISTIC_COLUMNS, CHUNK_SIZE
import remi_representation
from remi_train import OUTPUT_DIR, BATCH_SIZE
import utils

##################################################


# CONSTANTS
##################################################

# default number of samples to evaluate
N_SAMPLES = 1000

# evaluation constants
SEQ_LEN = 1024
TEMPERATURE = 1.0
FILTER = "top_k"

# output columns
OUTPUT_COLUMNS = ["model", "path"] + MMT_STATISTIC_COLUMNS

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate a REMI-Style Model.")
    parser.add_argument("-i", "--input_dir", default = OUTPUT_DIR, type = str, help = "Dataset facet directory containing the model(s) (as subdirectories) to evaluate")
    parser.add_argument("-ns", "--n_samples", default = N_SAMPLES, type = int, help = "Number of samples to evaluate")
    # model
    parser.add_argument("--seq_len", default = SEQ_LEN, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = TEMPERATURE, type = float, help = f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--filter", nargs = "+", default = FILTER, type = str, help = f"Sampling filter (default: '{FILTER}')")
    # others
    parser.add_argument("-bs", "--batch_size", default = BATCH_SIZE, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of workers for data loading")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to regenerate samples")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # get directories to eval
    model_dirs = list(filter(lambda path: isdir(path) and path.endswith("M"), map(lambda base: f"{args.input_dir}/{base}", listdir(args.input_dir))))
    models = list(map(basename, model_dirs))

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = f"{args.input_dir}/evaluate.log", mode = "a"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{args.input_dir}/evaluate_args.json"
    logging.info(f"Saved arguments to {args_output_filepath}")
    utils.save_args(filepath = args_output_filepath, args = args)
    del args_output_filepath

    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = remi_representation.get_encoding()

    # load the indexer
    indexer = remi_representation.Indexer(data = encoding["event_code_map"])
    vocabulary = utils.inverse_dict(indexer.get_dict()) # for decoding

    # get special tokens
    sos = indexer["start-of-song"]
    eos = indexer["end-of-song"]

    # get the logits filter function
    if args.filter == "top_k":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_k
    elif args.filter == "top_p":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_p
    elif args.filter == "top_a":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_a
    else:
        raise ValueError("Unknown logits filter.")

    # output file
    output_filepath = f"{args.input_dir}/evaluation.csv"
    output_columns_must_be_written = (not exists(output_filepath)) or args.reset
    n_lines_in_output = sum(1 for _ in open(output_filepath, "r")) if exists(output_filepath) else 0
    if output_columns_must_be_written or (n_lines_in_output == 1): # if column names need to be written
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        n_samples_to_calculate = utils.rep(x = args.n_samples, times = len(model_dirs)) # number of samples to calculate
        starting_indicies = utils.rep(x = 0, times = len(model_dirs))
    else:
        previous = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False) # load in previous values
        n_samples_to_calculate = previous.groupby(by = "model").size().to_dict() # calculate number of samples already generated
        n_samples_to_calculate = list(map(lambda model: max(args.n_samples - n_samples_to_calculate.get(model, 0), 0), models)) # determine number of samples to calculate
        starting_indicies = list(map(lambda model: max(list(map(lambda path: int(path[len(f"{args.input_dir}/{model}/eval/"):-len(f".npy")]), previous[previous["model"] == model]["path"])) + [-1]) + 1, models)) # get the starting index for generation names for each model
        del previous
    del output_columns_must_be_written, n_lines_in_output # free up memory

    ##################################################


    # HELPER FUNCTION FOR EVALUATING
    ##################################################

    # helper function for evaluating a generated sequence
    def evaluate(codes: Union[np.array, torch.tensor]) -> List[float]:
        """Evaluate the results."""

        # convert codes to a music object
        music = remi_representation.decode(codes = codes, encoding = encoding, vocabulary = vocabulary) # convert to a MusicExpress object

        # return a dictionary
        if len(music.tracks) == 0:
            return utils.rep(x = np.nan, times = len(MMT_STATISTIC_COLUMNS))
        else:
            return [pitch_class_entropy(music = music), scale_consistency(music = music), groove_consistency(music = music)]

    ##################################################


    # REPEAT WITH EACH MODEL IN INPUT DIRECTORY
    ##################################################

    for model_name, model_dir, n_samples, starting_index in zip(models, model_dirs, n_samples_to_calculate, starting_indicies):

        # LOAD MODEL
        ##################################################

        # get evaluation directory (where to output generations)
        eval_dir = f"{model_dir}/eval"
        if not exists(eval_dir):
            mkdir(eval_dir)

        # load training configurations
        train_args_filepath = f"{model_dir}/train_args.json"
        train_args = utils.load_json(filepath = train_args_filepath)
        del train_args_filepath       

        # create the model
        model = x_transformers.TransformerWrapper(
            num_tokens = len(indexer),
            max_seq_len = train_args["max_seq_len"],
            attn_layers = x_transformers.Decoder(
                dim = train_args["dim"],
                depth = train_args["layers"],
                heads = train_args["heads"],
                rotary_pos_emb = train_args["rel_pos_emb"],
                emb_dropout = train_args["dropout"],
                attn_dropout = train_args["dropout"],
                ff_dropout = train_args["dropout"],
            ),
            use_abs_pos_emb = train_args["abs_pos_emb"],
        ).to(device)
        model = x_transformers.AutoregressiveWrapper(net = model)

        # load the checkpoint
        checkpoint_filepath = f"{model_dir}/checkpoints/best_model.valid.pth"
        model_state_dict = torch.load(f = checkpoint_filepath, map_location = device)
        model.load_state_dict(state_dict = model_state_dict)
        model.eval()
        del checkpoint_filepath, model_state_dict # free up memory

        ##################################################


        # EVALUATE
        ##################################################

        # iterate over the dataset
        batch_size_evenly_divides_n_samples = ((n_samples % args.batch_size) == 0) # does batch size evenly divide n_samples
        n_batches = int(n_samples / args.batch_size) + int(not batch_size_evenly_divides_n_samples) # n batches
        with torch.no_grad():
            for i in tqdm(iterable = range(n_batches), desc = f"Evaluating the {model_name} Model"):

                # get number of samples to calculate
                n_samples_in_batch = (n_samples % args.batch_size) if ((i == (n_batches - 1)) and (not batch_size_evenly_divides_n_samples)) else args.batch_size

                # get start tokens
                prefix = torch.ones(size = (n_samples_in_batch, 1), dtype = torch.long, device = device) * sos

                # generate new samples; unconditioned generation
                generated = model.generate(
                    prompts = prefix,
                    seq_len = args.seq_len,
                    eos_token = eos,
                    temperature = args.temperature,
                    filter_logits_fn = filter_logits_fn,
                )
                generated = torch.cat(tensors = (prefix, generated), dim = 1).cpu().numpy()

                # save generation
                generated_output_filepaths = list(map(lambda j: f"{eval_dir}/{starting_index + j}.npy", range(len(generated))))
                starting_index += len(generated) # update starting index
                for j in range(len(generated)):
                    np.save(file = generated_output_filepaths[j], arr = generated[j]) # save generation to file

                # analyze 
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    results = pool.map(func = evaluate, iterable = generated, chunksize = CHUNK_SIZE)

                # write results to file
                results = pd.DataFrame(data = map(lambda j: [model_name, generated_output_filepaths[j]] + results[j], range(len(results))), columns = OUTPUT_COLUMNS)
                results.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")
                
        ##################################################

    ##################################################


    # LOG STATISTICS
    ##################################################

    # log statistics
    bar_width = 104
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False) # load in previous values
    for model in sorted(models, key = lambda model: int(model[:-1])):
        results_model = results[results["model"] == model]
        logging.info(f"\n{f' {model} ':=^{bar_width}}")
        for mmt_statistic in MMT_STATISTIC_COLUMNS:
            logging.info(f"{mmt_statistic.replace('_', ' ').title()}: mean = {np.nanmean(a = results_model[mmt_statistic], axis = 0):.4f}, std = {np.nanstd(a = results_model[mmt_statistic], axis = 0):.4f}")
    print("\n")

    ##################################################

##################################################
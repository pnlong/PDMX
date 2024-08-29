# README
# Phillip Long
# August 3, 2024

# Evaluate a REMI-Style model.

# python /home/pnlong/model_musescore/modeling/evaluate.py

# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, dirname, basename, isdir
from os import mkdir, listdir
from typing import Union, List
import multiprocessing
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import x_transformers

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from wrangling.full import MMT_STATISTIC_COLUMNS, CHUNK_SIZE, pitch_class_entropy, scale_consistency, groove_consistency, get_tracks_string
from wrangling.deduplicate import FACETS
from dataset import FACETS_HQ, MusicDataset, pad
from train import OUTPUT_DIR, FINE_TUNING_SUFFIX
from train import BATCH_SIZE as TRAIN_BATCH_SIZE
from representation import Indexer, get_encoding, encode_notes, decode
import utils

##################################################


# CONSTANTS
##################################################

# default number of samples to evaluate
BATCH_SIZE = TRAIN_BATCH_SIZE # 1
N_SAMPLES = 1200
N_BATCHES = int((N_SAMPLES - 1) / BATCH_SIZE) + 1

# evaluation constants
SEQ_LEN = 1024
TEMPERATURE = 1.0
FILTER = "top_k"

# facets to use as loss sets
LOSS_FACETS = [FACETS[0]] + FACETS_HQ

# output columns
OUTPUT_COLUMNS = ["model", "path"] + MMT_STATISTIC_COLUMNS + ["tracks"] + list(map(lambda loss_facet: f"loss:{loss_facet}", LOSS_FACETS))

##################################################


# HELPER FUNCTIONS
##################################################

# get the base stem of a filepath
basestem = lambda path: ".".join(basename(path).split(".")[:-1])

# perplexity function
perplexity_function = lambda loss: math.exp(-math.log(loss))

# convert a list of losses into a single perplexity value
def loss_to_perplexity(losses: List[float]) -> float:
    return perplexity_function(loss = sum(losses) / N_BATCHES)

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate a REMI-Style Model.")
    parser.add_argument("-i", "--input_dir", default = OUTPUT_DIR, type = str, help = "Dataset facet directory containing the model(s) (as subdirectories) to evaluate")
    # model
    parser.add_argument("--seq_len", default = SEQ_LEN, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = TEMPERATURE, type = float, help = f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--filter", nargs = "+", default = FILTER, type = str, help = f"Sampling filter (default: '{FILTER}')")
    # others
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
    model_dirs = list(filter(lambda path: isdir(path) and basename(path).split("_")[0].endswith("M"), map(lambda base: f"{args.input_dir}/{base}", listdir(args.input_dir))))
    model_dirs = sorted(model_dirs, key = lambda model_dir: int(basename(model_dir).split("_")[0][:-1]) + (0.5 if FINE_TUNING_SUFFIX in basename(model_dir) else 0)) # order from least to greatest
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

    # data paths filepath
    data_paths_dirs = list(map(lambda loss_facet: f"{dirname(args.input_dir)}/{loss_facet}", LOSS_FACETS))
    data_paths_filepaths = list(map(lambda data_paths_dir: data_paths_dir + "/" + ("test" if exists(f"{data_paths_dir}/test.txt") else "valid") + ".txt", data_paths_dirs))

    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = get_encoding()

    # load the indexer
    indexer = Indexer(data = encoding["event_code_map"])
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
    
    # path basename to rating mapping
    # path_to_rating = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    # path_to_rating = path_to_rating.set_index(keys = "path", drop = True)["rating"] # convert to series
    # path_to_rating.index = path_to_rating.index.map(basestem, na_action = "ignore") # remove filetype, keep just basename
    # path_to_rating = path_to_rating.to_dict() # convert to dictionary

    # output file
    output_filepath = f"{args.input_dir}/evaluation.csv"
    if (not exists(output_filepath)) or args.reset: # if column names need to be written
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    ##################################################


    # EVALUATE IF WE HAVEN'T YET
    ##################################################

    if sum(1 for _ in open(output_filepath, "r")) < ((len(models) * N_SAMPLES) + 1): # if the output is complete, based on number of lines

        # HELPER FUNCTION FOR EVALUATING
        ##################################################

        # helper function for evaluating a generated sequence
        def evaluate(codes: Union[np.array, torch.tensor]) -> List[float]:
            """Evaluate the results."""

            # convert codes to a music object
            music = decode(codes = codes, encoding = encoding, vocabulary = vocabulary) # convert to a MusicRender object

            # return a dictionary
            if len(music.tracks) == 0:
                return utils.rep(x = np.nan, times = len(MMT_STATISTIC_COLUMNS)) + [""]
            else:
                return [
                    pitch_class_entropy(music = music),
                    scale_consistency(music = music),
                    groove_consistency(music = music),
                    get_tracks_string(tracks = music.tracks),
                ]

        ##################################################


        # REPEAT WITH EACH MODEL IN INPUT DIRECTORY
        ##################################################

        for model_name, model_dir in zip(models, model_dirs):

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

            # load dataset and data loader
            datasets = [MusicDataset(
                paths = data_paths_filepath,
                encoding = encoding,
                indexer = indexer,
                encode_fn = encode_notes,
                max_seq_len = train_args["max_seq_len"],
                max_beat = train_args["max_beat"],
                use_augmentation = train_args["aug"],
            ) for data_paths_filepath in data_paths_filepaths]
            data_loaders = [torch.utils.data.DataLoader(
                dataset = dataset,
                num_workers = args.jobs,
                collate_fn = dataset.collate,
                batch_size = BATCH_SIZE,
                shuffle = False
            ) for dataset in datasets]
            data_iters = [iter(data_loader) for data_loader in data_loaders]

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
            model_state_dict = torch.load(f = checkpoint_filepath, map_location = device, weights_only = True)
            model.load_state_dict(state_dict = model_state_dict)
            model.eval()
            del checkpoint_filepath, model_state_dict # free up memory

            ##################################################


            # EVALUATE
            ##################################################

            # iterate over the dataset
            with torch.no_grad():
                for i in tqdm(iterable = range(N_BATCHES), desc = f"Evaluating the {model_name} Model"):

                    # get number of samples to calculate
                    n_samples_in_batch = (((N_SAMPLES - 1) % BATCH_SIZE) + 1) if (i == (N_BATCHES - 1)) else BATCH_SIZE

                    # GENERATE, EVALUATE MMT STATISTICS
                    ##################################################

                    # get output filepaths for generated sequences
                    generated_output_filepaths = list(map(lambda j: f"{eval_dir}/{(i * BATCH_SIZE) + j}.npy", range(n_samples_in_batch)))

                    # generate if needed
                    if (not all(map(exists, generated_output_filepaths))) or args.reset:

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
                        for j in range(len(generated)):
                            np.save(file = generated_output_filepaths[j], arr = generated[j]) # save generation to file

                    # reload generated files
                    else:

                        # load in generated content
                        generated = pad(data = list(map(np.load, generated_output_filepaths)))

                    # analyze
                    with multiprocessing.Pool(processes = args.jobs) as pool:
                        results = pool.map(func = evaluate, iterable = generated, chunksize = CHUNK_SIZE)

                    ##################################################


                    # LOSS FOR PERPLEXITY
                    ##################################################

                    # initialize loss_batch array
                    loss_batch = utils.rep(x = 0.0, times = len(LOSS_FACETS))

                    # calculate loss for each loss facet
                    for j in range(len(LOSS_FACETS)):

                        # get batch
                        try:
                            batch = next(data_iters[j])
                        except (StopIteration):
                            data_iters[j] = iter(data_loaders[j]) # reinitialize dataset iterator if necessary
                            batch = next(data_iters[j])

                        # get loss value through forward pass
                        loss_batch_facet = model(
                            x = batch["seq"][:n_samples_in_batch].to(device),
                            return_outputs = False,
                            mask = batch["mask"][:n_samples_in_batch].to(device),
                        )

                        # convert to non-torch number
                        loss_batch[j] = float(loss_batch_facet) / n_samples_in_batch # divide by number of samples so we get per sample loss, instead of per batch loss
                        del loss_batch_facet # free up memory

                        # get ratings
                        # if j == len(LOSS_FACETS) - 1:
                        #     ratings = list(map(lambda path: path_to_rating.get(basestem(path), 0), batch["name"][:n_samples_in_batch]))

                    ##################################################


                    # OUTPUT RESULTS
                    ##################################################

                    # write results to file
                    results = pd.DataFrame(data = map(lambda j: [model_name, generated_output_filepaths[j]] + results[j] + loss_batch, range(n_samples_in_batch)), columns = OUTPUT_COLUMNS)
                    results.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")
                    
                    ##################################################

            ##################################################

            # free up memory
            del model, datasets, data_loaders, data_iters

        ##################################################

    ##################################################


    # LOG STATISTICS
    ##################################################

    # log statistics
    bar_width = 50
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False) # load in previous values
    for model in models:
        results_model = results[results["model"] == model]
        logging.info(f"\n{f' {model} ':=^{bar_width}}")
        for mmt_statistic in MMT_STATISTIC_COLUMNS:
            logging.info(f"{mmt_statistic.replace('_', ' ').title()}: mean = {np.nanmean(a = results_model[mmt_statistic], axis = 0):.4f}, std = {np.nanstd(a = results_model[mmt_statistic], axis = 0):.4f}")
        logging.info(f"Perplexity (All): {loss_to_perplexity(losses = results_model[f'loss:all']):.4f}")
    print("")

    ##################################################

##################################################
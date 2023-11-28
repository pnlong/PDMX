# README
# Phillip Long
# November 27, 2023

# Generate music from a neural network.

# python /home/pnlong/model_musescore/generate.py

# Absolute positional embedding (APE):
# python /home/pnlong/mmt/generate.py -d sod -o /data2/pnlong/mmt/exp/sod/ape -g 0

# Relative positional embedding (RPE):
# python /home/pnlong/mmt/generate.py -d sod -o /data2/pnlong/mmt/exp/sod/rpe -g 0

# No positional embedding (NPE):
# python /home/pnlong/mmt/generate.py -d sod -o /data2/pnlong/mmt/exp/sod/npe -g 0


# IMPORTS
##################################################

import argparse
import logging
import pprint
import subprocess
from typing import Union, Tuple
from os.path import exists
from os import makedirs, mkdir
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from read_mscz.music import BetterMusic
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
    parser.add_argument("-s", "--shuffle", action = "store_true", help = "Whether to shuffle the test data")
    parser.add_argument("--model_steps", type = int, help = "Step of the trained model to load (default to the best model)")
    # sampling
    parser.add_argument("--sequence_length", default = 1024, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 1, type = int, help = "Number of jobs")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# SAVE THE RESULT
##################################################

# helper function to save piano roll
def save_pianoroll(filepath: str, music: BetterMusic, size: Tuple[int] = None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label = "program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filepath)
    plt.close()


# save the result
def save_result(stem: str, data: Union[np.array, torch.tensor], sample_dir: str, encoding: dict):
    """Save the results in multiple formats."""

    # save results
    np.save(file = f"{sample_dir}/npy/{stem}.npy", arr = data) # save as a numpy array
    representation.save_csv_codes(filepath = f"{sample_dir}/csv/{stem}.csv", data = data) # save as a .csv file
    representation.save_txt(filepath = f"{sample_dir}/txt/{stem}.txt", data = data, encoding = encoding) # save as a .txt file
    music = representation.decode(codes = data, encoding = encoding) # convert to a BetterMusic object
    music.save(path = f"{sample_dir}/json/{stem}.json") # save as a BetterMusic .json file
    save_pianoroll(filepath = f"{sample_dir}/png/{stem}.png", music = music, size = (20, 5), preset = "frame") # save as a piano roll
    music.write(path = f"{sample_dir}/mid/{stem}.mid") # save as a .mid file
    music.write(path = f"{sample_dir}/wav/{stem}.wav", options = "-o synth.polyphony = 4096") # save as a .wav file
    subprocess.check_output(args = ["ffmpeg", "-loglevel", "error", "-y", "-i", f"{sample_dir}/wav/{stem}.wav", "-b:a", "192k", f"{sample_dir}/mp3/{stem}.mp3"]) # save also as a .mp3 file

    # trim and save
    music.trim(end = music.resolution * 64) # trim the music
    save_pianoroll(filepath = f"{sample_dir}/png-trimmed/{stem}.png", music = music, size = (10, 5)) # save the trimmed version as a piano roll
    music.write(path = f"{sample_dir}/wav-trimmed/{stem}.wav", options = "-o synth.polyphony = 4096") # save as a .wav file
    subprocess.check_output(args = ["ffmpeg", "-loglevel", "error", "-y", "-i", f"{sample_dir}/wav-trimmed/{stem}.wav", "-b:a", "192k", f"{sample_dir}/mp3-trimmed/{stem}.mp3"]) # save also as a .mp3 file

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(f"{args.output_dir}/generate.log", "w"), logging.StreamHandler(sys.stdout)])

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{args.output_dir}/generate-args.json"
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

    # make sure the sample directory exists
    SAMPLE_DIR = f"{args.output_dir}/samples"
    if not exists(args.output_dir): makedirs(args.output_dir) # create output_dir if necessary
    if not exists(SAMPLE_DIR): mkdir(SAMPLE_DIR) # create sample_dir if necessary
    for subdir in (f"{SAMPLE_DIR}/{subdir}" for subdir in ("npy", "csv", "txt", "json", "png", "mid", "wav", "mp3", "png-trimmed", "wav-trimmed", "mp3-trimmed")):
        if not exists(subdir): mkdir(subdir) # create subdir if necessary

    # get the specified device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    logging.info(f"Creating the data loader...")
    dataset = dataset.MusicDataset(paths = args.paths, encoding = encoding, max_sequence_length = train_args["max_sequence_length"], max_beat = train_args["max_beat"])
    data_loader = torch.utils.data.DataLoader(dataset = dataset, shuffle = args.shuffle, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate)

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
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    ##################################################


    # GENERATE
    ##################################################

    # iterate over the dataset
    with torch.no_grad():

        # instantiate iterable
        data_iter = iter(data_loader)
        for i in tqdm(iterable = range(args.n_samples), desc = "Generating"):

            # get new batch
            batch = next(data_iter)

            # GROUND TRUTH
            ##################################################

            # get ground truth
            truth_np = batch["sequence"][0].numpy()

            # save the results
            save_result(stem = f"{i}_truth", data = truth_np, sample_dir = SAMPLE_DIR, encoding = encoding)

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

            # save the results
            save_result(stem = f"{i}_unconditioned", data = generated_sequence[0], sample_dir = SAMPLE_DIR, encoding = encoding)

            ##################################################

            # INSTRUMENT-INFORMED GENERATION
            ##################################################

            # get output start tokens
            prefix_length = int(np.argmax(a = batch["sequence"][0, :, 1] >= beat_0))
            tgt_start = batch["sequence"][:1, :prefix_length].to(device)

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

            # save the results
            save_result(stem = f"{i}_instrument-informed", data = generated_sequence[0], sample_dir = SAMPLE_DIR, encoding = encoding)

            ##################################################

            # 4-beat continuation
            ##################################################

            # get output start tokens
            cond_length = int(np.argmax(a = batch["sequence"][0, :, 1] >= beat_4))
            tgt_start = batch["sequence"][:1, :cond_length].to(device)

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

            # save the results
            save_result(stem = f"{i}_4-beat-continuation", data = generated_sequence[0], sample_dir = SAMPLE_DIR, encoding = encoding)

            ##################################################

            # 16-BEAT CONTINUATION
            ##################################################

            # get output start tokens
            cond_length = int(np.argmax(a = batch["sequence"][0, :, 1] >= beat_16))
            tgt_start = batch["sequence"][:1, :cond_length].to(device)

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

            # save the results
            save_result(stem = f"{i}_16-beat-continuation", data = generated_sequence[0], sample_dir = SAMPLE_DIR, encoding = encoding)

            ##################################################
    
    ##################################################

##################################################
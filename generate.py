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
import pathlib
import pprint
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
import representation
import utils

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--names", type = pathlib.Path, help = "input names")
    parser.add_argument("-i", "--in_dir", type = pathlib.Path, help = "input data directory")
    parser.add_argument("-o", "--out_dir", type = pathlib.Path, help = "output directory")
    parser.add_argument("-ns", "--n_samples", default = 50, type = int, help = "number of samples to generate")
    # model
    parser.add_argument("-s", "--shuffle", action = "store_true", help = "whether to shuffle the test data")
    parser.add_argument("--model_steps", type = int, help = "step of the trained model to load (default to the best model)")
    # sampling
    parser.add_argument("--seq_len", default = 1024, type = int, help = "sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "sampling filter threshold (default: 0.9)")
    # others
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 1, type = int, help = "Number of jobs")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# SAVE PIANO ROLL
##################################################

def save_pianoroll(filepath, music, size = None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label = "program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filepath)
    plt.close()

##################################################


# SAVE THE RESULT
##################################################

def save_result(filepath, data, sample_dir, encoding):
    """Save the results in multiple formats."""
    # Save as a numpy array
    np.save(sample_dir / "npy" / f"{filepath}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(sample_dir / "csv" / f"{filepath}.csv", data)

    # Save as a TXT file
    representation.save_txt(
        sample_dir / "txt" / f"{filepath}.txt", data, encoding
    )

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding)

    # Save as a MusPy JSON file
    music.save(sample_dir / "json" / f"{filepath}.json")

    # Save as a piano roll
    save_pianoroll(
        sample_dir / "png" / f"{filepath}.png", music, (20, 5), preset = "frame"
    )

    # Save as a MIDI file
    music.write(sample_dir / "mid" / f"{filepath}.mid")

    # Save as a WAV file
    music.write(
        sample_dir / "wav" / f"{filepath}.wav",
        options = "-o synth.polyphony = 4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav" / f"{filepath}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3" / f"{filepath}.mp3")]
    )

    # Trim the music
    music.trim(music.resolution * 64)

    # Save the trimmed version as a piano roll
    save_pianoroll(
        sample_dir / "png-trimmed" / f"{filepath}.png", music, (10, 5)
    )

    # Save as a WAV file
    music.write(
        sample_dir / "wav-trimmed" / f"{filepath}.wav",
        options = "-o synth.polyphony = 4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav-trimmed" / f"{filepath}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3-trimmed" / f"{filepath}.mp3")]
    )

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"{DATA_DIRECTORY}data/{args.dataset}/processed/test-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"{DATA_DIRECTORY}data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    # Set up the logger
    logging.basicConfig(
        level = logging.ERROR if args.quiet else logging.INFO,
        format = "%(message)s",
        handlers = [
            logging.FileHandler(args.out_dir / "generate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'generate-args.json'}")
    utils.save_args(args.out_dir / "generate-args.json", args)

    ##################################################


    # LOAD IN STUFF
    ##################################################

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "samples"
    sample_dir.mkdir(exist_ok = True)
    (sample_dir / "npy").mkdir(exist_ok = True)
    (sample_dir / "csv").mkdir(exist_ok = True)
    (sample_dir / "txt").mkdir(exist_ok = True)
    (sample_dir / "json").mkdir(exist_ok = True)
    (sample_dir / "png").mkdir(exist_ok = True)
    (sample_dir / "mid").mkdir(exist_ok = True)
    (sample_dir / "wav").mkdir(exist_ok = True)
    (sample_dir / "mp3").mkdir(exist_ok = True)
    (sample_dir / "png-trimmed").mkdir(exist_ok = True)
    (sample_dir / "wav-trimmed").mkdir(exist_ok = True)
    (sample_dir / "mp3-trimmed").mkdir(exist_ok = True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding,
        max_seq_len = train_args["max_seq_len"],
        max_beat = train_args["max_beat"],
        use_csv = args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle = args.shuffle,
        num_workers = args.jobs,
        collate_fn = dataset.MusicDataset.collate,
    )

    # Create the model
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

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filepath = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filepath = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filepath, map_location = device))
    logging.info(f"Loaded the model weights from: {checkpoint_filepath}")
    model.eval()

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    ##################################################


    # GENERATE
    ##################################################

    # Iterate over the dataset
    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(args.n_samples), ncols = 80):
            batch = next(data_iter)

            # ------------
            # Ground truth
            # ------------
            truth_np = batch["seq"][0].numpy()
            save_result(f"{i}_truth", truth_np, sample_dir, encoding)

            # ------------------------
            # Unconditioned generation
            # ------------------------

            # Get output start tokens
            tgt_start = torch.zeros((1, 1, 6), dtype = torch.long, device = device)
            tgt_start[:, 0, 0] = sos

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_unconditioned", generated_np[0], sample_dir, encoding
            )

            # ------------------------------
            # Instrument-informed generation
            # ------------------------------

            # Get output start tokens
            prefix_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_0))
            tgt_start = batch["seq"][:1, :prefix_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_instrument-informed",
                generated_np[0],
                sample_dir,
                encoding,
            )

            # -------------------
            # 4-beat continuation
            # -------------------

            # Get output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_4))
            tgt_start = batch["seq"][:1, :cond_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_4-beat-continuation",
                generated_np[0],
                sample_dir,
                encoding,
            )

            # --------------------
            # 16-beat continuation
            # --------------------

            # Get output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1] >= beat_16))
            tgt_start = batch["seq"][:1, :cond_len].to(device)

            # Generate new samples
            generated = model.generate(
                tgt_start,
                args.seq_len,
                eos_token = eos,
                temperature = args.temperature,
                filter_logits_fn = args.filter,
                filter_thres = args.filter_threshold,
                monotonicity_dim = ("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save results
            save_result(
                f"{i}_16-beat-continuation",
                generated_np[0],
                sample_dir,
                encoding,
            )
    
    ##################################################

##################################################
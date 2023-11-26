# Absolute positional embedding (APE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/ape -ns 100 -g 0

# Relative positional embedding (RPE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/rpe -ns 100 -g 0

# No positional embedding (NPE):
# python /home/pnlong/mmt/evaluate.py -d sod -o /data2/pnlong/mmt/exp/sod/npe -ns 100 -g 0

import argparse
import logging
import pathlib
import pprint
import sys
from collections import defaultdict

import muspy
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
import representation
import utils
from utils import DATA_DIRECTORY


@utils.resolve_paths
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices = ("sod", "lmd", "lmd_full", "snd"),
        required = True,
        help = "dataset key",
    )
    parser.add_argument("-n", "--names", type = pathlib.Path, help = "input names")
    parser.add_argument(
        "-i", "--in_dir", type = pathlib.Path, help = "input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type = pathlib.Path, help = "output directory"
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        type = int,
        help = "number of samples to evaluate",
    )
    # Data
    parser.add_argument(
        "--use_csv",
        action = "store_true",
        help = "whether to save outputs in CSV format (default to NPY format)",
    )
    # Model
    parser.add_argument(
        "--model_steps",
        type = int,
        help = "step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "--seq_len", default = 1024, type = int, help = "sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs = "+",
        default = 1.0,
        type = float,
        help = "sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs = "+",
        default = "top_k",
        type = str,
        help = "sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs = "+",
        default = 0.9,
        type = float,
        help = "sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type = int, help = "gpu number")
    parser.add_argument(
        "-j", "--jobs", default = 0, type = int, help = "number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action = "store_true", help = "show warnings only"
    )
    return parser.parse_args(args = args, namespace = namespace)


def evaluate(data, encoding, filename, eval_dir):
    """Evaluate the results."""
    # Save as a numpy array
    np.save(eval_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(eval_dir / "csv" / f"{filename}.csv", data)

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding)

    # Trim the music
    music.trim(music.resolution * 64)

    # Save as a MusPy JSON file
    music.save(eval_dir / "json" / f"{filename}.json")

    if not music.tracks:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }

    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "scale_consistency": muspy.scale_consistency(music),
        "groove_consistency": muspy.groove_consistency(
            music, 4 * music.resolution
        ),
    }


def main():
    """Main function."""
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
            logging.FileHandler(args.out_dir / "evaluate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'evaluate-args.json'}")
    utils.save_args(args.out_dir / "evaluate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the output directory exists
    eval_dir = args.out_dir / "eval"
    eval_dir.mkdir(exist_ok = True)
    for key in ("truth", "unconditioned"):
        (eval_dir / key).mkdir(exist_ok = True)
        (eval_dir / key / "npy").mkdir(exist_ok = True)
        (eval_dir / key / "csv").mkdir(exist_ok = True)
        (eval_dir / key / "json").mkdir(exist_ok = True)

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
        use_csv = args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location = device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    results = defaultdict(list)

    n_samples = len(test_loader) if args.n_samples is None else args.n_samples
    test_iter = iter(test_loader)

    # Iterate over the dataset
    with torch.no_grad():
        for idx in enumerate(tqdm.tqdm(range(n_samples), ncols = 120)):

            batch = next(test_iter)

            # ------------
            # Ground truth
            # ------------

            truth_np = batch["seq"].numpy()
            result = evaluate(
                truth_np[0], encoding, f"{idx}_0", eval_dir / "truth"
            )
            results["truth"].append(result)

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

            # Evaluate the results
            result = evaluate(
                generated_np[0],
                encoding,
                f"{idx}_0",
                eval_dir / "unconditioned",
            )
            results["unconditioned"].append(result)

    for exp, result in results.items():
        logging.info(exp)
        for key in result[0]:
            logging.info(
                f"{key}: mean = {np.nanmean([r[key] for r in result]):.4f}, "
                f"steddev = {np.nanstd([r[key]for r in result]):.4f}"
            )


if __name__ == "__main__":
    main()

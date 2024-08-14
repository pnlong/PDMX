# README
# Phillip Long
# August 1, 2024

# Data Loader for REMI-Style Encoding.

# python /home/pnlong/model_musescore/remi_dataset.py

# IMPORTS
##################################################

import argparse
import logging
import multiprocessing
from tqdm import tqdm
from typing import List, Callable
from os.path import exists, basename
from os import makedirs, mkdir
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import dataset_full
from dataset_deduplicate import FACETS
from read_mscz.music import MusicExpress
from read_mscz.read_mscz import read_musescore
import remi_representation
import utils

##################################################


# CONSTANTS
##################################################

# output directory
OUTPUT_DIR = "/data1/pnlong/musescore/remi"

# facets of the dataset
HQ_RATING_THRESHOLDS = [3.0, 3.5, 4.0, 4.5]
FACETS_HQ = list(map(lambda hq_rating_threshold: f"{FACETS[-1]}-{hq_rating_threshold:.1f}", HQ_RATING_THRESHOLDS)) # high quality facet names

# partition names
PARTITIONS = {"train": 0.9, "valid": 0.1, "test": 0.0} # no test partition

# value for padding
PAD_VALUE = 0

# at what rating (strictly greater than) do we consider a song high quality?
HIGH_QUALITY_RATING_THRESHOLD = 4.0

##################################################


# HELPER FUNCTIONS
##################################################

# pad a sequence
def pad(data: np.array, maxlen: int = None) -> np.array:
    """Pad a sequence."""

    # determine max sequence length
    if maxlen is None:
        max_len = max(len(seq) for seq in data)
    else:
        for seq in data:
            assert len(seq) <= max_len

    # pad
    if data[0].ndim == 1:
        padded = [np.pad(array = seq, pad_width = (0, max_len - len(seq)), mode = "constant", constant_values = PAD_VALUE) for seq in data]
    elif data[0].ndim == 2:
        padded = [np.pad(array = seq, pad_width = ((0, max_len - len(seq)), (0, 0)), mode = "constant", constant_values = PAD_VALUE) for seq in data]
    else:
        raise ValueError("Got 3D data.")

    # return padded array
    return np.stack(arrays = padded, axis = 0)

# get mask for data
def get_mask(data: np.array) -> torch.tensor:
    """Get a boolean mask to cover part of data."""
    max_seq_len = max(len(seq) for seq in data)
    mask = torch.zeros(size = (len(data), max_seq_len), dtype = torch.bool)
    for i, seq in enumerate(data):
        mask[i, :len(seq)] = True # mask values
    mask = mask[:, :(max_seq_len - 1)] # because we do autoregression autoregression, we are predicting data[:, 1:], so mask must be max_seq_len - 1 to accomodate
    return mask # return the mask

##################################################


# DATASET CLASS
##################################################

class MusicDataset(torch.utils.data.Dataset):

    # intializer
    def __init__(
        self,
        paths: str, # path to file with filepaths to semi-encoded song representations
        encoding: dict = remi_representation.get_encoding(), # encoding dictionary
        indexer: remi_representation.Indexer = remi_representation.Indexer(), # indexer
        encode_fn: Callable = remi_representation.encode_notes, # encoding function
        max_seq_len: int = None, # max sequence length
        max_beat: int = remi_representation.MAX_BEAT, # max beat
        use_augmentation: bool = False, # use data augmentation?
    ):
        super().__init__()
        with open(paths, "r") as file:
            self.paths = [line.strip() for line in file if line]
        self.encoding = encoding
        self.indexer = indexer
        self.encode_fn = encode_fn
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = self.paths[0].endswith("csv")
        self.use_augmentation = use_augmentation

    # length attribute
    def __len__(self) -> int:
        return len(self.paths)

    # obtain an item
    def __getitem__(self, index: int) -> dict:

        # get the name
        path = self.paths[index]

        # load data
        if self.use_csv:
            notes = utils.load_csv(filepath = path)
        else:
            notes = np.load(file = path)

        # check the shape of the loaded notes
        assert notes.shape[1] == 5

        # data augmentation
        if self.use_augmentation:

            # shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(low = -5, high = 7)
            notes[:, 2] = np.clip(a = notes[:, 2] + pitch_shift, a_min = 0, a_max = 127)

            # randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                while trial < 10:
                    start_beat = np.random.randint(low = 0, high = n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[(notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)]
                    if len(sliced_notes) > 10: # avoid section with too few notes
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat # make beats start at 0
                notes = sliced_notes

        # trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]

        # encode the notes
        seq = self.encode_fn(notes = notes, encoding = self.encoding, indexer = self.indexer)

        # trim sequence to max_seq_len
        if (self.max_seq_len is not None) and (len(seq) > self.max_seq_len):
            seq = np.concatenate((seq[:(self.max_seq_len - 2)], seq[(-2):]))

        return {"name": path, "seq": seq}

    # collate method
    @classmethod
    def collate(cls, data: List[dict]) -> dict:
        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(data = seq), dtype = torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype = torch.long),
            "mask": get_mask(data = seq),
        }

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Dataset", description = "Create and test PyTorch Dataset for MuseScore data.")
    parser.add_argument("-d", "--dataset_filepath", default = f"{dataset_full.OUTPUT_DIR}/{dataset_full.DATASET_DIR_NAME}.csv", type = str, help = "Filepath to full dataset")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory for any relevant files")
    parser.add_argument("-u", "--use_csv", action = "store_true", help = "Whether to save outputs in CSV format (default to NPY format)")
    parser.add_argument("-rv", "--ratio_valid", default = PARTITIONS["valid"], type = float, help = "Ratio of validation files")
    parser.add_argument("-rt", "--ratio_test", default = PARTITIONS["test"], type = float, help = "Ratio of test files")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate data files")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # deal with output directories
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    DATA_DIR = f"{args.output_dir}/data"
    if not exists(DATA_DIR):
        mkdir(DATA_DIR)

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # save random seed so this can be replicated
    random.seed(0)

    ##################################################


    # LOAD IN DATA
    ##################################################

    # load in dataset
    logging.info("Loading in Dataset.")
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    ##################################################


    # EXTRACT NOTES FROM MUSESCORE FILES
    ##################################################

    # helper function for extracting notes and saving those events to a file
    def extract_notes(path: str) -> str:
        """
        Helper function for extracting notes from MuseScore of music object files.
        Given a path, extract notes and return the output path.
        """

        # determine output path early to avoid computations if possible
        output_path = f"{DATA_DIR}/{'.'.join(basename(path).split('.')[:-1])}.{'csv' if args.use_csv else 'npy'}"
        if exists(output_path) and (not args.reset): # avoid computations if possible
            return output_path

        # load music object
        if path.endswith("mscz"): # musescore file
            music = read_musescore(path = path, resolution = remi_representation.RESOLUTION)
        elif path.endswith("json"): # music object file
            music = MusicExpress().load_json(path = path)
        else:
            raise ValueError(f"Unknown filetype `{path.split('.')[-1]}`.")

        # extract notes
        notes = remi_representation.extract_notes(music = music, resolution = remi_representation.RESOLUTION)

        # output
        if args.use_csv:
            remi_representation.save_csv_notes(filepath = output_path, data = notes)
        else:
            np.save(file = output_path, arr = notes)

        # return path to which we outputted
        return output_path

    # use multiprocessing to extract notes
    with multiprocessing.Pool(processes = args.jobs) as pool:
        dataset["output_path"] = list(pool.map(func = extract_notes, iterable = tqdm(iterable = dataset["path"], desc = "Extracting Notes", total = len(dataset)), chunksize = dataset_full.CHUNK_SIZE))

    ##################################################


    # PARTITION
    ##################################################

    # get high quality facet
    for hq_rating_threshold, facet_hq in zip(HQ_RATING_THRESHOLDS, FACETS_HQ):
        dataset[f"facet:{facet_hq}"] = (dataset[f"facet:{FACETS[-1]}"] & (dataset["rating"] > hq_rating_threshold))

    # get partitions set up
    partitions = dict(zip(PARTITIONS.keys(), (1 - args.ratio_valid - args.ratio_test, args.ratio_valid, args.ratio_test)))

    # helper function for saving to a file
    def save_paths_to_file(paths: List[str], output_filepath: str) -> None:
        """Given a list of paths, save to a file."""
        with open(output_filepath, "w") as output_file:
            output_file.write("\n".join(paths))

    # go through the different facets
    for facet in (FACETS + FACETS_HQ):

        # filter dataset
        data = dataset[dataset[f"facet:{facet}"]]["output_path"].to_list() # filter down to only necessary column, output_path

        # create subdirectory
        output_dir = f"{args.output_dir}/{facet}"
        if not exists(output_dir):
            mkdir(output_dir)

        # partition files
        n_valid, n_test = int(partitions["valid"] * len(data)), int(partitions["test"] * len(data)) # get the validation and test partitions from the ratios
        n_train = len(data) - n_valid - n_test # as to not exclude any files, the train partition is simply what's not in the validation or test partition
        paths = random.sample(population = data, k = len(data)) # shuffle paths
        save_paths_to_file(paths = paths[:n_train], output_filepath = f"{output_dir}/train.txt") # train partition
        save_paths_to_file(paths = paths[n_train:(n_train + n_valid)], output_filepath = f"{output_dir}/valid.txt") # validation partition
        if n_test > 0:
            save_paths_to_file(paths = paths[(n_train + n_valid):], output_filepath = f"{output_dir}/test.txt") # test partition
        
    # update
    logging.info("Partitioned data.")

    ##################################################


    # TEST DATALOADER
    ##################################################

    # load the encoding
    encoding = remi_representation.get_encoding()

    # get the indexer
    indexer = remi_representation.Indexer(data = encoding["event_code_map"])

    # create the dataset and data loader
    dataset = MusicDataset(
        paths = f"{args.output_dir}/{FACETS[0]}/valid.txt",
        encoding = encoding,
        indexer = indexer,
        encode_fn = remi_representation.encode_notes,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 8,
        shuffle = True,
        collate_fn = MusicDataset.collate,
    )

    # iterate over the data loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    for i, batch in enumerate(data_loader):

        # update tracker variables
        n_batches += 1
        n_samples += len(batch["name"])
        seq_lens.extend(int(l) for l in batch["seq_len"])

        # print example on first batch
        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")

    # print how many batches were loaded
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

    ##################################################


    # STATISTICS
    ##################################################

    # print sequence length statistics
    logging.info(f"Average sequence length: {np.mean(seq_lens):2f}")
    logging.info(f"Minimum sequence length: {min(seq_lens)}")
    logging.info(f"Maximum sequence length: {max(seq_lens)}")

    ##################################################

##################################################

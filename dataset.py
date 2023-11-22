# README
# Phillip Long
# November 1, 2023

# Create torch dataset object for training a neural network.

# python /home/pnlong/model_musescore/dataset.py


# IMPORTS
##################################################

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
import sys
import argparse
import pprint
from tqdm import tqdm
from typing import List

import representation

#################################################


# HELPER FUNCTIONS
##################################################

# padder function
def pad(data: np.array, maxlen: int = None) -> np.array:

    # deal with maxlen argument
    if maxlen is None:
        max_len = max(len(sample) for sample in data)
    else:
        for sample in data:
            assert len(sample) <= max_len
    
    # check dimensionality of data
    if data[0].ndim == 1:
        padded = [np.pad(array = sample, pad_width = (0, max_len - len(sample))) for sample in data]
    elif data[0].ndim == 2:
        padded = [np.pad(array = sample, pad_width = ((0, max_len - len(sample)), (0, 0))) for sample in data]
    else:
        raise ValueError("Got data in higher than two dimensions.")
    
    # return
    return np.stack(arrays = padded)

# masking function
def get_mask(data: np.array) -> np.array:
    max_sequence_len = max(len(sample) for sample in data) # get the maximum sequence length
    mask = torch.zeros(size = (data.shape[0], max_sequence_len), dtype = torch.bool) # instantiate mask
    for i, sequence in enumerate(data):
        mask[i, :len(sequence)] = 1 # mask values
    return mask # return the mask

##################################################


# DATASET
##################################################

class MusicDataset(Dataset):

    # CONSTRUCTOR
    ##################################################

    def __init__(self, filename: str, data_dir: str, encoding: dict, max_sequence_len: int = None, max_beat: int = None, use_augmentation: bool = False):
        super().__init__()
        self.data_dir = data_dir
        with open(filename) as f:
            self.paths = [line.strip() for line in f if line]
        self.encoding = encoding
        self.max_sequence_len = max_sequence_len
        self.max_beat = max_beat
        self.use_augmentation = use_augmentation

    ##################################################

    # GET LENGTH
    ##################################################

    def __len__(self):
        return len(self.paths)
    
    ##################################################

    # GET ITEM
    ##################################################

    def __getitem__(self, index: int):

        # Get the name
        path = self.paths[index]

        # load data
        data = np.load(file = path, allow_pickle = True)

        # get number of beats
        n_beats = data[-1, representation.DIMENSIONS.index("beat")] + 1
        beat_column = representation.DIMENSIONS.index("beat")

        # data augmentation
        if self.use_augmentation:

            # shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(low = -5, high = 7)
            value_column = representation.DIMENSIONS.index("value")
            data[:, value_column] = [np.clip(a = data[i, value_column] + (pitch_shift if (data[i, representation.DIMENSIONS.index("type")] in ("note", "grace-note")) else 0), a_min = 0, a_max = 127) for i in range(data.shape[0])] # apply pitch shift to pitches only
            del value_column, pitch_shift

            # randomly select a starting beat
            if n_beats > self.max_beat: # make sure sequence isn't too long
                trial = 0
                while trial < 10: # avoid section with too few notes
                    start_beat = np.random.randint(n_beats - self.max_beat) # randomly generate a start beat
                    end_beat = start_beat + self.max_beat # get end beat from start_beat
                    data_slice = data[(data[:, beat_column] >= start_beat) & (data[:, beat_column] < end_beat)]
                    if len(data_slice) > 10: # get a sufficiently large slice of values
                        break
                    trial += 1 # iterate trial
                data_slice[:, beat_column] = data_slice[:, beat_column] - start_beat # make sure slice beats start at 0
                data = data_slice
                del data_slice

        # trim sequence to max_beat
        elif self.max_beat is not None:
            if n_beats > self.max_beat:
                data = data[data[:, beat_column] < self.max_beat]
        
        # encode the data
        sequence = representation.encode(data = data, encoding = self.encoding, conditioning = self.conditioning, sigma = self.sigma)

        # trim sequence to max_sequence_len
        if self.max_sequence_len is not None and len(sequence) > self.max_sequence_len:
            sequence = np.delete(arr = sequence, obj = range(self.max_sequence_len - 1, sequence.shape[0] - 1), axis = 0)

        return {"name": path, "seq": sequence}

    ##################################################

    # CLASS METHODS
    ##################################################

    @classmethod
    def collate(cls, data: List[dict]):
        sequence = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(data = sequence), dtype = torch.long),
            "seq_len": torch.tensor([len(s) for s in sequence], dtype = torch.long),
            "mask": get_mask(data = sequence),
        }

    ##################################################


##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = f"data/{args.dataset}/processed/names.txt"
        if args.in_dir is None:
            args.in_dir = f"data/{args.dataset}/processed/notes"
    if args.jobs is None:
        args.jobs = min(args.batch_size, 8)

    # Set up the logger
    logging.basicConfig(stream = sys.stdout, level = logging.ERROR if args.quiet else logging.INFO, format = "%(message)s")

    ##################################################


    # LOAD UP DATA
    ##################################################
    
    encoding = representation.load_encoding(filename = args.in_dir / "encoding.json")

    # Create the dataset and data loader
    dataset = MusicDataset(
        args.names,
        args.in_dir,
        encoding=encoding,
        max_sequence_len=args.max_sequence_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        use_augmentation=args.aug,
    )
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, True, collate_fn = MusicDataset.collate)

    ##################################################


    # TEST DATALOADER
    ##################################################

    n_batches = 0
    n_samples = 0
    seq_lens = []
    for i, batch in enumerate(tqdm(iterable = data_loader)):
        n_batches += 1
        n_samples += len(batch["name"])
        seq_lens.extend(int(l) for l in batch["seq_len"])
        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")
    ##################################################


    # STATISTICS
    ##################################################

    logging.info(f"Avg sequence length: {np.mean(seq_lens):2f}")
    logging.info(f"Min sequence length: {min(seq_lens)}")
    logging.info(f"Max sequence length: {max(seq_lens)}")

    ##################################################


##################################################
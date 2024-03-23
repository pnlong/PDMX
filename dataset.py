# README
# Phillip Long
# November 1, 2023

# Create torch dataset object for training a neural network.

# python /home/pnlong/model_musescore/dataset.py


# IMPORTS
##################################################

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from tqdm import tqdm
from typing import List
import math
from os.path import exists

import representation
import encode

#################################################


# HELPER FUNCTIONS
##################################################

# padder function
def pad(data: np.array, max_length: int = None) -> np.array:

    # deal with maxlen argument
    if max_length is None:
        max_length = max(len(seq) for seq in data)
    else:
        for seq in data:
            assert len(seq) <= max_length
    
    # check dimensionality of data
    if data[0].ndim == 1:
        padded = [np.pad(array = seq, pad_width = (0, max_length - len(seq))) for seq in data]
    elif data[0].ndim == 2:
        padded = [np.pad(array = seq, pad_width = ((0, max_length - len(seq)), (0, 0))) for seq in data]
    else:
        raise ValueError("Received data in higher than two dimensions.")
    
    # return
    return np.stack(arrays = padded)

# masking function
def get_mask(data: List[np.array]) -> np.array:
    max_seq_len = max(len(seq) for seq in data) # get the maximum seq length
    mask = torch.zeros(size = (len(data), max_seq_len), dtype = torch.bool) # instantiate mask
    for i, seq in enumerate(data):
        mask[i, :len(seq)] = True # mask values
    return mask # return the mask

##################################################


# DATASET
##################################################

class MusicDataset(Dataset):

    # CONSTRUCTOR
    ##################################################

    def __init__(self, paths: str, encoding: dict, conditioning: str = encode.DEFAULT_CONDITIONING, sigma: float = encode.SIGMA, is_baseline: bool = False, max_seq_len: int = None, use_augmentation: bool = False, unidimensional: bool = False):
        super().__init__()
        with open(paths) as file:
            self.paths = [line.strip() for line in file if line]
        self.encoding = encoding
        self.conditioning = conditioning
        self.sigma = sigma
        self.is_baseline = is_baseline
        self.max_seq_len = max_seq_len
        self.use_augmentation = use_augmentation
        self.use_absolute_time = encoding["use_absolute_time"]
        temporal = "time" if self.use_absolute_time else "beat"
        self.max_temporal = self.encoding[f"max_{temporal}"]
        self.temporal_dim = self.encoding["dimensions"].index(temporal)
        self.value_dim = self.encoding["dimensions"].index("value")
        self.unidimensional = unidimensional
        self.n_tokens_per_event = len(self.encoding["dimensions"]) if unidimensional else 1
        self.unidimensional_encoding_function, _ = representation.get_unidimensional_coding_functions(encoding = self.encoding)
        if self.unidimensional:
            self.value_dim_unidimensional = self.encoding["unidimensional_encoding_order"].index("value")

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
        n_temporals = data[-1, self.temporal_dim] + 1 if data.shape[0] > 0 else -1 # deal with 0 length data

        # data augmentation
        if self.use_augmentation:

            # shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(low = -5, high = 7)
            pitches = np.any(a = (data[:, 0] == "note", data[:, 0] == "grace-note"), axis = 0) # get rows of pitches (notes and grace notes)
            data[pitches, self.value_dim] = np.clip(a = data[pitches, self.value_dim].astype(encode.ENCODING_ARRAY_TYPE) + pitch_shift, a_min = 0, a_max = 127) # clip values
            del pitch_shift

            # randomly select a starting beat
            if n_temporals > self.max_temporal: # make sure seq isn't too long
                trial = 0
                while trial < 10: # avoid section with too few notes
                    start = (math.floor((np.random.rand() * (n_temporals - self.max_temporal)) / representation.TIME_STEP) * representation.TIME_STEP) if self.use_absolute_time else np.random.randint(0, n_temporals - self.max_temporal) # randomly generate a start beat
                    end = start + self.max_temporal # get end beat from start_beat
                    data_slice = data[(data[:, self.temporal_dim] >= start) & (data[:, self.temporal_dim] < end)]
                    if len(data_slice) > 10: # get a sufficiently large slice of values
                        break
                    trial += 1 # iterate trial
                data_slice[:, self.temporal_dim] = data_slice[:, self.temporal_dim] - start # make sure slice beats start at 0
                data = data_slice
                del data_slice

        # trim seq to max_temporal
        elif self.max_temporal is not None:
            if n_temporals > self.max_temporal:
                data = data[data[:, self.temporal_dim] < self.max_temporal]
        
        # encode the data
        seq = encode.encode_data(data = data[data[:, 0] != representation.EXPRESSIVE_FEATURE_TYPE_STRING] if self.is_baseline else data,
                                 encoding = self.encoding,
                                 conditioning = self.conditioning,
                                 sigma = self.sigma,
                                 unidimensional = self.unidimensional,
                                 unidimensional_encoding_function = self.unidimensional_encoding_function)

        # trim seq to max_seq_len
        if (self.max_seq_len is not None) and (len(seq) > self.max_seq_len):
            seq = np.delete(
                arr = seq,
                obj = range(self.max_seq_len - (self.max_seq_len % self.n_tokens_per_event) - self.n_tokens_per_event, seq.shape[0] - self.n_tokens_per_event),
                axis = 0
            )

        return {"path": path, "seq": seq}

    ##################################################

    # CLASS METHODS
    ##################################################

    @classmethod
    def collate(cls, data: List[dict]):
        seq = [sample["seq"] for sample in data]
        return {
            "path": [sample["path"] for sample in data],
            "seq": torch.tensor(pad(data = seq), dtype = torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype = torch.long),
            "mask": get_mask(data = seq),
        }

    ##################################################


##################################################


# CONSTANTS
##################################################

DEFAULT_PATHS = "/data2/pnlong/musescore/data/valid.txt"

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Dataset", description = "Create and test PyTorch Dataset for MuseScore data.")
    parser.add_argument("-p", "--paths", type = str, default = DEFAULT_PATHS, help = "Filepath to list of paths to data")
    parser.add_argument("-e", "--encoding", type = str, default = representation.ENCODING_FILEPATH, help = "Filepath to encoding .json file")
    parser.add_argument("-bs", "--batch_size", type = int, default = 8, help = "Batch size")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # load arugments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # LOAD UP DATA
    ##################################################
    
    # load encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    dataset = MusicDataset(paths = args.paths, encoding = encoding, conditioning = encode.DEFAULT_CONDITIONING, max_seq_len = None, max_temporal = encoding["max_" + ("time" if "max_time" in encoding.keys() else "beat")], use_augmentation = False)
    data_loader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = True, collate_fn = MusicDataset.collate)

    ##################################################


    # TEST DATALOADER
    ##################################################

    # instantiate
    n_batches, n_samples = 0, 0
    seq_lens = []

    # loop through data loader
    example = "" # store example as a string
    for i, batch in enumerate(tqdm(iterable = data_loader, desc = "Combing through dataset")):

        # update counters
        n_batches += 1
        n_samples += len(batch["path"])
        seq_lens.extend(int(l) for l in batch["seq_len"])

        # show example on first iteration
        if i == 0:
            example += f"EXAMPLE:\n  - Path: {batch['path'][0]}\n"
            for key, value in batch.items():
                if key == "path":
                    continue
                example += f"  - Shape of {key}: {value.shape}\n"

    # output number of batches and example
    logging.info(example)
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

    ##################################################


    # STATISTICS
    ##################################################

    logging.info(f"Average sequence length: {np.mean(seq_lens):2f}")
    logging.info(f"Minimum sequence length: {min(seq_lens)}")
    logging.info(f"Maximum sequence length: {max(seq_lens)}")

    ##################################################


##################################################
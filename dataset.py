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

import representation

#################################################


# HELPER FUNCTIONS
##################################################

# padder function
def pad(data: np.array, max_length: int = None) -> np.array:

    # deal with maxlen argument
    if max_length is None:
        max_length = max(len(sequence) for sequence in data)
    else:
        for sequence in data:
            assert len(sequence) <= max_length
    
    # check dimensionality of data
    if data[0].ndim == 1:
        padded = [np.pad(array = sequence, pad_width = (0, max_length - len(sequence))) for sequence in data]
    elif data[0].ndim == 2:
        padded = [np.pad(array = sequence, pad_width = ((0, max_length - len(sequence)), (0, 0))) for sequence in data]
    else:
        raise ValueError("Received data in higher than two dimensions.")
    
    # return
    return np.stack(arrays = padded)

# masking function
def get_mask(data: List[np.array]) -> np.array:
    max_sequence_length = max(len(sequence) for sequence in data) # get the maximum sequence length
    mask = torch.zeros(size = (len(data), max_sequence_length), dtype = torch.bool) # instantiate mask
    for i, sequence in enumerate(data):
        mask[i, :len(sequence)] = 1 # mask values
    return mask # return the mask

##################################################


# DATASET
##################################################

class MusicDataset(Dataset):

    # CONSTRUCTOR
    ##################################################

    def __init__(self, paths: str, encoding: dict, conditioning: str = representation.DEFAULT_CONDITIONING, sigma: float = representation.SIGMA, max_sequence_length: int = None, max_beat: int = None, use_augmentation: bool = False):
        super().__init__()
        with open(paths) as file:
            self.paths = [line.strip() for line in file if line]
        self.encoding = encoding
        self.conditioning = conditioning
        self.sigma = sigma
        self.max_sequence_length = max_sequence_length
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
        n_beats = data[-1, representation.DIMENSIONS.index("beat")] + 1 if data.shape[0] > 0 else -1 # deal with 0 length data
        beat_column = representation.DIMENSIONS.index("beat")

        # data augmentation
        if self.use_augmentation:

            # shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(low = -5, high = 7)
            value_column, type_column = representation.DIMENSIONS.index("value"), representation.DIMENSIONS.index("type")
            pitches = np.any(a = (data[:, type_column] == "note", data[:, type_column] == "grace-note"), axis = 0) # get rows of pitches (notes and grace notes)
            data[pitches, value_column] = np.clip(a = data[pitches, value_column].astype(representation.ENCODING_ARRAY_TYPE) + pitch_shift, a_min = 0, a_max = 127) # clip values
            del value_column, type_column, pitch_shift

            # randomly select a starting beat
            if n_beats > self.max_beat: # make sure sequence isn't too long
                trial = 0
                while trial < 10: # avoid section with too few notes
                    start_beat = np.random.randint(n_beats - self.max_beat) # randomly generate a start beat
                    end_beat = start_beat + self.max_beat # get end beat from start_beat
                    data_slice = data[(data[:, beat_column].astype(representation.ENCODING_ARRAY_TYPE) >= start_beat) & (data[:, beat_column].astype(representation.ENCODING_ARRAY_TYPE) < end_beat)]
                    if len(data_slice) > 10: # get a sufficiently large slice of values
                        break
                    trial += 1 # iterate trial
                data_slice[:, beat_column] = data_slice[:, beat_column].astype(representation.ENCODING_ARRAY_TYPE) - start_beat # make sure slice beats start at 0
                data = data_slice
                del data_slice

        # trim sequence to max_beat
        elif self.max_beat is not None:
            if n_beats > self.max_beat:
                data = data[data[:, beat_column].astype(representation.ENCODING_ARRAY_TYPE) < self.max_beat]
        
        # encode the data
        sequence = representation.encode_data(data = data, encoding = self.encoding, conditioning = self.conditioning, sigma = self.sigma)

        # FOR NOW, TRIM OFF UNKNOWN TEXT (-1)
        sequence = sequence[sequence[:, representation.DIMENSIONS.index("value")] != representation.DEFAULT_VALUE_CODE]

        # trim sequence to max_sequence_length
        if self.max_sequence_length is not None and len(sequence) > self.max_sequence_length:
            sequence = np.delete(arr = sequence, obj = range(self.max_sequence_length - 1, sequence.shape[0] - 1), axis = 0)

        return {"path": path, "sequence": sequence}

    ##################################################

    # CLASS METHODS
    ##################################################

    @classmethod
    def collate(cls, data: List[dict]):
        sequence = [sample["sequence"] for sample in data]
        return {
            "path": [sample["path"] for sample in data],
            "sequence": torch.tensor(pad(data = sequence), dtype = torch.long),
            "sequence_length": torch.tensor([len(s) for s in sequence], dtype = torch.long),
            "mask": get_mask(data = sequence),
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
    parser.add_argument("-e", "--encoding", type = str, default = None, help = "Filepath to encoding .json file")
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
    if args.encoding is not None:
        encoding = representation.load_encoding(filepath = args.encoding)
    else:
        encoding = representation.get_encoding()

    # create the dataset and data loader
    dataset = MusicDataset(paths = args.paths, encoding = encoding, conditioning = "sort", max_sequence_length = None, max_beat = encoding["max_beat"], use_augmentation = False)
    data_loader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = True, collate_fn = MusicDataset.collate)

    ##################################################


    # TEST DATALOADER
    ##################################################

    # instantiate
    n_batches, n_samples = 0, 0
    sequence_lengths = []

    # loop through data loader
    example = "" # store example as a string
    for i, batch in enumerate(tqdm(iterable = data_loader, desc = "Combing through dataset")):

        # update counters
        n_batches += 1
        n_samples += len(batch["path"])
        sequence_lengths.extend(int(l) for l in batch["sequence_length"])

        # show example on first iteration
        if i == 0:
            example += f"EXAMPLE\nPath: {batch['path'][0]}"
            for key, value in batch.items():
                if key == "path":
                    continue
                example += f"Shape of {key}: {value.shape}\n"

    # output number of batches and example
    logging.info(example)
    logging.info(f"Successfully loaded {n_batches} batches ({n_samples} samples).")

    ##################################################


    # STATISTICS
    ##################################################

    logging.info(f"Average sequence length: {np.mean(sequence_lengths):2f}")
    logging.info(f"Minimum sequence length: {min(sequence_lengths)}")
    logging.info(f"Maximum sequence length: {max(sequence_lengths)}")

    ##################################################


##################################################
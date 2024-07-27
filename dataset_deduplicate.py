# README
# Phillip Long
# July 26, 2024

# Dataset deduplication efforts. Outputs a list of filepaths.
# Each filepath is the 'best' version of a unique song.
# In other words, we group songs with duplicate artists and titles together,
# and from that grouping, we choose the best version within that group.

# python /home/pnlong/model_musescore/dataset_deduplicate.py

# IMPORTS
##################################################

import argparse
import pandas as pd
import numpy as np
from re import sub
from os.path import dirname
from tqdm import tqdm
import multiprocessing
from typing import List
import torch
import logging

from sentence_transformers import SentenceTransformer

from dataset_full import OUTPUT_DIR, CHUNK_SIZE

##################################################


# CONSTANTS
##################################################

# default batch size for encoding song titles as sentence embeddings
DEFAULT_BATCH_SIZE = 32

# column names in dataset from which we can create a description of the song that can be used for deduplication
DESCRIPTOR_COLUMNS = ["song_name", "title", "subtitle", "artist_name", "composer_name"]

# column names for how to determine the best version of a song
BEST_VERSION_METRIC_COLUMNS = ["rating", "n_ratings", "n_notes", "n_tokens"]

# minimum similarity (0 to 1) between two song titles for them to be considered duplicates
SIMILARITY_THRESHOLD = 0.99

##################################################


# FUNCTION THAT CREATES SONG DESCRIPTORS
##################################################

def get_song_descriptor_from_index(i: int) -> str:
    """
    Given the index of a row in the dataset, generate a song 'descriptor'
    based off the songs title and composer, as well as any other relevant attributes.
    """

    # extract relevant attributes from the dataset, replacing NA values with empty strings
    song_name, title, subtitle, artist_name, composer_name = dataset.loc[i, DESCRIPTOR_COLUMNS].fillna(value = "")

    # deal with song name
    song = ""
    if (song_name.lower() == title.lower()) or ((len(song_name) > 0) and (len(title) == 0)):
        song = song_name
    elif (len(song_name) > 0) and (len(title) > 0):
        song = f"{song_name}, also known as {title}"
    elif len(title) > 0:
        song = title
    if len(subtitle) > 0:
        song += f": {subtitle}"

    # deal with artist name
    artist = ""
    if (len(artist_name) > 0) and (len(composer_name) > 0): # if both are defined
        artist = f"{artist_name}, and composed by {composer_name}"
    elif len(artist_name) > 0:
        artist = artist_name
    elif len(composer_name) > 0:
        artist = composer_name

    # create descriptor, while doing some string processing
    descriptor = f"{song}; by {artist}"
    if descriptor[-1] not in ".?!": # add punctuation to the end if there isn't any
        descriptor += "."
    descriptor = sub(pattern = r'[^ \w0-9,.?!;:()&-]', repl = " ", string = descriptor)
    descriptor = " ".join(descriptor.split()) # remove wierd whitespace

    # return the descriptor
    return descriptor

##################################################


# TOP SONG CHOOSER
##################################################

def choose_best_song_from_indicies(indicies: List[int]) -> int:
    """
    Given a set of indicies in the dataset, return the index representing the 'best' song
    from those indicies. Our definition of best is the song with the highest ratings.
    """

    # default best index
    best_index = indicies[0]

    # avoid unnecessary computations if possible
    if len(indicies) == 1: # if a song has no duplicates
        return best_index # simply return that song's index
    
    # get all the duplicates in one data frame
    duplicates = dataset.loc[indicies, BEST_VERSION_METRIC_COLUMNS]

    # determine best version of song
    duplicates = duplicates.sort_values(
        by = BEST_VERSION_METRIC_COLUMNS,
        axis = 0,
        ascending = False,
        na_position = "last",
        ignore_index = False
    )
    best_index = duplicates.index[0] # the top index is the best index

    # return the best index
    return best_index


##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore", description = "Analyze full dataset for music-quality differences within variables.")
    parser.add_argument("-d", "--dataset_filepath", type = str, default = f"{OUTPUT_DIR}/dataset.full.csv", help = "Filepath to full dataset")
    parser.add_argument("-s", "--similarity_threshold", default = SIMILARITY_THRESHOLD, type = float, help = "Similarity Threshold")
    parser.add_argument("-bs", "--batch_size", default = DEFAULT_BATCH_SIZE, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # BASIC SETUP
    ##################################################

    # parse arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    ##################################################


    # GENERATE DESCRIPTORS
    ##################################################

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        descriptors = list(tqdm(iterable = pool.map(func = get_song_descriptor_from_index, iterable = dataset.index, chunksize = CHUNK_SIZE),
                                desc = "Generating Song Descriptors",
                                total = len(dataset)))
    
    ##################################################


    # USE EMBEDDING MODEL TO ENCODE SONG DESCRIPTORS FOR SIMILARITY CALCULATIONS
    ##################################################
    # we use Sentence-BERT to embed song titles as vectors
    # https://github.com/UKPLab/sentence-transformers

    # update on progress
    logging.info(f"Computing Similarities Between Song titles.")

    # load in Sentence-BERT model
    device = f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu" # set up device for embeddings
    model = SentenceTransformer(model_name_or_path = "all-MiniLM-L6-v2",
                                device = device,
                                similarity_fn_name = "cosine")

    # retrieve embeddings for each song descriptor
    embeddings = model.encode(sentences = descriptors,
                              batch_size = args.batch_size,
                              show_progress_bar = True,
                              output_value = "sentence_embedding",
                              device = device)
    del descriptors # free up memory

    # calculate similarity matrix
    similarities = model.similarity(embeddings1 = embeddings, embeddings2 = embeddings)
    similarities = (similarities >= args.similarity_threshold).cpu().numpy() # convert to booleans, where two songs are 'duplicates' if true
    del model, device, embeddings # free up memory

    ##################################################


    # GROUP TOGETHER SIMILAR SONG NAMES INTO A DICTIONARY
    ##################################################

    # stores lists of indicies, where each list represents a song
    songs = []

    # group duplicates together
    songs_already_grouped = set() # store indicies of songs that have already been placed in a group
    for i in tqdm(iterable = dataset.index, desc = "Grouping Duplicates Together", total = len(dataset)):

        # don't deal with songs that have already been grouped
        if i in songs_already_grouped:
            continue

        # otherwise, create a song group
        song = np.where(similarities[i])[0] # get indicies of duplicates for the `i`th song
        song = list(filter(lambda index: index not in songs_already_grouped, song)) # remove indicies that have already been grouped with another song
        songs.append(song) # add song group to songs
        songs_already_grouped.update(song) # all these indicies have already been grouped

    # free up memory
    del similarities

    ##################################################


    # ASSEMBLE DEDUPLICATED DATASET
    ##################################################

    # use multiprocessing to get deduplicated indicies
    with multiprocessing.Pool(processes = args.jobs) as pool:
        deduplicated_indicies = list(tqdm(iterable = pool.map(func = choose_best_song_from_indicies, iterable = songs, chunksize = CHUNK_SIZE),
                                          desc = "Choosing the Best Version of Each Song",
                                          total = len(songs)))

    # get and output deduplicated paths
    paths = dataset.loc[deduplicated_indicies, "path"] # obtain the filepath of each top choice per song
    output_filepath = f"{dirname(args.dataset_filepath)}/paths.deduplicated.txt" # get output filepath
    with open(output_filepath, "w") as output_file:
        output_file.write("\n".join(paths))

    ##################################################

##################################################
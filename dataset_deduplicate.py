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
import os
from os.path import dirname
from tqdm import tqdm
import multiprocessing
from typing import List
import torch
import logging

from sentence_transformers import SentenceTransformer

from dataset_full import OUTPUT_DIR, CHUNK_SIZE

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    logging.info("Loading in Dataset.")
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
    logging.info("Computing Vector Embeddings of Song Titles.")

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

    # calculate similarities
    similarities = model.similarity(embeddings, embeddings).cpu().numpy()
    similarities = (similarities >= args.threshold)
    del embeddings, model, device

    ##################################################


    # GROUP TOGETHER SIMILAR SONG NAMES INTO A DICTIONARY
    ##################################################

    # # for calculating cosine similarity, calculate the magnitude of each song title vector embedding; use multiprocessing because why not
    # with multiprocessing.Pool(processes = args.jobs) as pool:
    #     magnitudes = list(pool.map(func = np.linalg.norm, iterable = embeddings, chunksize = CHUNK_SIZE))

    # stores lists of indicies, where each list represents a song
    songs = []
    songs_already_grouped = set() # store indicies of songs that have already been placed in a group
    
    # group duplicates together
    for i in tqdm(iterable = dataset.index, desc = "Grouping Duplicates Together", total = len(dataset)):

        # don't deal with songs that have already been grouped
        if i in songs_already_grouped:
            continue

        # # helper function for calculating similarity
        # def similarity_fn(j: int) -> bool:
        #     """
        #     Helper function for calculating similarity between vector embeddings `i` and `j`.
        #     Uses cosine similarity. Returns a boolean representing whether the vectors are 'duplicates'.
        #     """

        #     # avoid unnecessary calculations if possible
        #     if j in songs_already_grouped:
        #         return False

        #     # calculate similarity
        #     cosine_similarity = np.dot(embeddings[i], embeddings[j]) / (magnitudes[i] * magnitudes[j]) # calculate cosine similarity
        #     similarity = (cosine_similarity + 1) / 2 # normalize such that similarity is between 0 and 1
        #     return (similarity >= args.similarity_threshold) # convert to a boolean value

        # # calculate similarities (boolean values representing whether or not two songs are considered duplicates)
        # with multiprocessing.Pool(processes = args.jobs) as pool:
        #     similarities = list(pool.map(func = similarity_fn,
        #                                  iterable = range(len(embeddings)),
        #                                  chunksize = CHUNK_SIZE))

        # # old similarity calculations
        # dot_products = np.matmul(embeddings, embeddings[i]) # need dot products for cosine similarity
        # cosine_similarities = dot_products / (magnitudes * magnitudes[i]) # calculate cosine similarity
        # similarities = (cosine_similarities + 1) / 2 # normalize such that similarities are between 0 and 1
        # similarities = (similarities >= args.similarity_threshold) # turn into booleans, True when two songs are duplicate
        # song = np.where(similarities)[0] # get indicies of duplicates for the `i`th song

        # otherwise, create a song group
        song = np.where(similarities[i])[0] # get indicies of duplicates for the `i`th song
        song = list(filter(lambda index: index not in songs_already_grouped, song)) # remove indicies that have already been grouped with another song; not needed anymore, as this is done in similarity function calculations
        songs.append(song) # add song group to songs
        songs_already_grouped.update(song) # all these indicies have already been grouped        

    # free up memory
    del similarities, songs_already_grouped

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
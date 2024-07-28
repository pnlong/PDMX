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
from os.path import dirname, exists
from tqdm import tqdm
import multiprocessing
from typing import List
import torch
import logging
from utils import rep

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
    parser = argparse.ArgumentParser(prog = "Deduplicate", description = "Deduplicate songs in full dataset.")
    parser.add_argument("-d", "--dataset_filepath", default = f"{OUTPUT_DIR}/dataset.full.csv", type = str, help = "Filepath to full dataset")
    parser.add_argument("-s", "--similarity_threshold", default = SIMILARITY_THRESHOLD, type = float, help = "Similarity Threshold")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate intermediate data tables")
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

    # directory to output files
    output_dir = dirname(args.dataset_filepath)
    extra_output_dir = f"{output_dir}/deduplicate_intermediate_data"
    if not exists(extra_output_dir):
        os.mkdir(extra_output_dir)

    # output filepaths
    output_filepath_embeddings = f"{extra_output_dir}/embeddings.csv"
    output_filepath_magnitudes = f"{extra_output_dir}/magnitudes.csv"
    output_filepath_similarities = f"{extra_output_dir}/similarities.csv"
    output_filepath = f"{output_dir}/paths.deduplicated.txt"

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # load in dataset
    logging.info("Loading in Dataset.")
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)

    ##################################################


    # DO WE NEED TO CALCULATE SIMILARITIES, OR ARE THEY ALREADY CALCUALTED?
    ##################################################

    if (not exists(output_filepath_similarities)) or args.reset:

        # USE EMBEDDING MODEL TO ENCODE SONG DESCRIPTORS
        ##################################################
        # we use Sentence-BERT to embed song titles as vectors
        # https://github.com/UKPLab/sentence-transformers

        # load in Sentence-BERT model
        device = f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu" # set up device for embeddings
        model = SentenceTransformer(model_name_or_path = "all-MiniLM-L6-v2",
                                    device = device,
                                    similarity_fn_name = "cosine")

        # do we need to generate the embeddings here?
        if (not exists(output_filepath_embeddings)) or args.reset:

            # update on progress
            logging.info("Computing Vector Embeddings.")

            # generate descriptors
            with multiprocessing.Pool(processes = args.jobs) as pool:
                descriptors = list(pool.map(func = get_song_descriptor_from_index, iterable = dataset.index, chunksize = CHUNK_SIZE))

            # retrieve embeddings for each song descriptor
            embeddings = model.encode(sentences = descriptors,
                                    batch_size = args.batch_size,
                                    show_progress_bar = True,
                                    output_value = "sentence_embedding",
                                    device = device)
            del descriptors, model, device # free up memory

            # write embeddings to file for future use
            pd.DataFrame(data = embeddings).to_csv(path_or_buf = output_filepath_embeddings, sep = ",", header = False, index = False, mode = "w")

        # can we load them instead
        else :

            # update on progress
            logging.info("Loading Vector Embeddings.")

            # read in embeddings and turn into numpy array
            embeddings = np.array(pd.read_csv(filepath_or_buffer = output_filepath_embeddings, sep = ",", header = None, index_col = False))

        ##################################################


        # CALCULATE MAGNITUDES OF EMBEDDINGS
        ##################################################

        # do we need to calculate embeddings
        if (not exists(output_filepath_magnitudes)) or args.reset:

            # for calculating cosine similarity, calculate the magnitude of each song title vector embedding
            with multiprocessing.Pool(processes = args.jobs) as pool:
                magnitudes = np.array(list(tqdm(iterable = pool.map(func = np.linalg.norm, iterable = embeddings, chunksize = CHUNK_SIZE),
                                                desc = "Computing Embedding Magnitudes",
                                                total = len(embeddings))))
            
            # write magnitudes to file for future use
            pd.DataFrame(data = magnitudes).to_csv(path_or_buf = output_filepath_magnitudes, sep = ",", header = False, index = False, mode = "w")

        # can we load them in instead
        else:

            # update on progress
            logging.info("Loading Embedding Magnitudes.")

            # read in magnitudes and turn into numpy array
            magnitudes = np.array(pd.read_csv(filepath_or_buffer = output_filepath_magnitudes, sep = ",", header = None, index_col = False))

        ##################################################


        # CALCULATE SIMILARITY MATRIX
        ##################################################

        # helper function for calculating similarity
        def similarity_fn(i: int, j: int) -> float:
            """
            Helper function for calculating similarity between vector embeddings `i` and `j`.
            Uses cosine similarity, but normalizes after.
            """
            similarity = np.dot(embeddings[i], embeddings[j]) / (magnitudes[i] * magnitudes[j]) # calculate cosine similarity
            similarity = (similarity + 1) / 2 # normalize such that similarity is between 0 and 1
            return similarity

        # compute similarities matrix line by line
        with open(output_filepath_similarities, "w") as similarities_matrix_file:
            for i in tqdm(iterable = range(len(dataset) - 1), desc = "Computing Similarity Matrix", total = len(dataset)):

                # calculate similarities (boolean values representing whether or not two songs are considered duplicates)
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    similarities = list(pool.starmap(func = similarity_fn,
                                                    iterable = zip(rep(x = i, times = len(dataset) - i - 1), range(i + 1, len(dataset))),
                                                    chunksize = CHUNK_SIZE))

                # output similarities to file
                similarities_matrix_file.write(",".join(map(str, similarities)) + "\n")

        ##################################################

    else:

        # INFORM THAT SIMILARITIES MATRIX IS ALREADY COMPUTED
        ##################################################

        # give progress update
        logging.info("Similarity Matrix Already Computed.")

        ##################################################

    ##################################################


    # GROUP TOGETHER SIMILAR SONG NAMES INTO A DICTIONARY
    ##################################################

    # stores lists of indicies, where each list represents a song
    songs = []
    songs_already_grouped = set() # store indicies of songs that have already been placed in a group
    
    # group duplicates together, reading in similarities line by line
    with open(output_filepath_similarities, "r") as similarities_matrix_file:
        for i, line in tqdm(iterable = enumerate(similarities_matrix_file), desc = "Grouping Duplicates Together", total = len(dataset)):

            # don't deal with songs that have already been grouped
            if i in songs_already_grouped:
                continue

            # read in line of similarities matrix, converting to floats
            similarities = np.array(list(map(float, line.split(","))))

            # create a song group
            song = np.where(similarities >= args.threshold)[0] + (i + 1) # get indicies of duplicates for the `i`th song, add `i` + 1 to account for the fact the matrix is a triangle
            song = list(filter(lambda index: index not in songs_already_grouped, song)) # remove indicies that have already been grouped with another song; not needed anymore, as this is done in similarity function calculations
            song.append(i) # a song is similar to itself

            # add song group to lists
            songs.append(song) # add song group to songs
            songs_already_grouped.update(song) # all these indicies have already been grouped

    # account for the last song, i.e. if it hasn't been grouped yet
    last_song_index = len(dataset) - 1
    if last_song_index not in songs_already_grouped:
        song = [last_song_index]
        songs.append(song)
        songs_already_grouped.update(song)

    # free up memory
    del songs_already_grouped

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
    with open(output_filepath, "w") as output_file:
        output_file.write("\n".join(paths))

    ##################################################

##################################################
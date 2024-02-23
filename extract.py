# README
# Phillip Long
# February 22, 2024

# Given the path to a .mscz file, extracts expressive features

# python /home/pnlong/model_musescore/extract.py


# IMPORTS
##################################################

import numpy as np
from copy import copy

import representation
from encode import extract_data
from read_mscz.read_mscz import read_musescore
from read_mscz.classes import *

##################################################


# EXTRACTION FUNCTION (EXTRACT RELEVANT DATA FROM A GIVEN MUSESCORE FILE
##################################################

def extract(path: str, use_implied_duration: bool = True) -> np.array:
    """Extract relevant information from a .mscz file into a numpy array

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.

    Returns
    -------
    np.array: extracted data
    """
    
    # LOAD IN MSCZ FILE, CONSTANTS
    ##################################################

    try:
        music = read_musescore(path = path, timeout = 10)
        music.realize_expressive_features()
    except: # if that fails
        return # exit here

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################

    relevant_column_indicies = [representation.DIMENSIONS.index("time"), representation.DIMENSIONS.index("duration"), representation.DIMENSIONS.index("value")]
    output = np.empty(shape = (0, len(relevant_column_indicies)), dtype = np.object_)

    for track in music.tracks:

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in representation.KNOWN_PROGRAMS:
            continue
        
        # create BetterMusic object with just one track (we are not doing multitrack)
        track_music = copy(x = music)
        track_music.tracks = [track,]
        data = extract_data(music = track_music, use_implied_duration = use_implied_duration, include_velocity = False, use_absolute_time = True)
        data = data[data[0, :] == representation.EXPRESSIVE_FEATURE_TYPE_STRING] # filter to just expressive features
        data = data[:, relevant_column_indicies] # extract only relevant columns
        output = np.concatenate((output, data), axis = 0) # concatenate

    return output

##################################################

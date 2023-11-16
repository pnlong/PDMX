# README
# Phillip Long
# October 3, 2023

# child class of Muspy.Music better suited for storing expressive features from musescore
# Music class -- a universal container for symbolic music.
# copied mostly from https://github.com/salu133445/muspy/blob/b2d4265c6279e730903d8abe9dddda8484511903/muspy/music.py
# Classes
# -------
# - Music
# Variables
# ---------
# - DEFAULT_RESOLUTION

# ./music.py


# IMPORTS
##################################################
import muspy
from .classes import (Annotation, Barline, Beat, KeySignature, Lyric, Metadata, Tempo, TimeSignature, Track)
from collections import OrderedDict
from typing import List
from re import sub
import yaml # for printing
import pickle
##################################################


# CONSTANTS
##################################################

DIVIDE_BY_ZERO_CONSTANT = 1e-10

##################################################


# BETTER MUSIC CLASS
##################################################

class BetterMusic(muspy.music.Music):
    """A universal container for symbolic music, better suited for storing expressive features than MusPy.

    Attributes
    ----------
    metadata : :class:`read_mscz.Metadata`, default: `Metadata()`
        Metadata.
    resolution : int, default: `muspy.DEFAULT_RESOLUTION` (24)
        Time steps per quarter note.
    tempos : list of :class:`read_mscz.Tempo`, default: []
        Tempo changes.
    key_signatures : list of :class:`read_mscz.KeySignature`, default: []
        Key signatures changes.
    time_signatures : list of :class:`read_mscz.TimeSignature`, default: []
        Time signature changes.
    barlines : list of :class:`read_mscz.Barline`, default: []
        Barlines.
    beats : list of :class:`read_mscz.Beat`, default: []
        Beats.
    lyrics : list of :class:`read_mscz.Lyric`, default: []
        Lyrics.
    annotations : list of :class:`read_mscz.Annotation`, default: []
        Annotations.
    tracks : list of :class:`read_mscz.Track`, default: []
        Music tracks.

    Note
    ----
    Indexing a Music object returns the track of a certain index. That
    is, ``music[idx]`` returns ``music.tracks[idx]``. Length of a Music
    object is the number of tracks. That is, ``len(music)``  returns
    ``len(music.tracks)``.

    """


    _attributes = OrderedDict([("metadata", Metadata), ("resolution", int), ("tempos", Tempo), ("key_signatures", KeySignature), ("time_signatures", TimeSignature), ("barlines", Barline), ("beats", Beat), ("lyrics", Lyric), ("annotations", Annotation), ("tracks", Track)])
    _optional_attributes = ["metadata", "resolution", "tempos", "key_signatures", "time_signatures", "barlines", "beats", "lyrics", "annotations", "tracks"]
    _list_attributes = ["tempos", "key_signatures", "time_signatures", "barlines", "beats", "lyrics", "annotations", "tracks"]


    # INITIALIZER
    ##################################################

    def __init__(self, metadata: Metadata = None, resolution: int = None, tempos: List[Tempo] = None, key_signatures: List[KeySignature] = None, time_signatures: List[TimeSignature] = None, barlines: List[Barline] = None, beats: List[Beat] = None, lyrics: List[Lyric] = None, annotations: List[Annotation] = None, tracks: List[Track] = None):
        self.metadata = metadata if metadata is not None else Metadata()
        self.resolution = resolution if resolution is not None else muspy.DEFAULT_RESOLUTION
        self.tempos = tempos if tempos is not None else []
        self.key_signatures = key_signatures if key_signatures is not None else []
        self.time_signatures = time_signatures if time_signatures is not None else []
        self.beats = beats if beats is not None else []
        self.barlines = barlines if barlines is not None else []
        self.lyrics = lyrics if lyrics is not None else []
        self.annotations = annotations if annotations is not None else []
        self.tracks = tracks if tracks is not None else []

    ##################################################


    # PRETTY PRINTING
    ##################################################

    def print(self, output_filepath: str = None, remove_empty_lines: bool = True):
        """Print the BetterMusic object in a pretty way.
        
        Arguments
        ---------
        output_filepath : str, optional
            If provided, outputs the yaml to the provided filepath. If not provided, prints to stdout
        remove_empty_lines : bool, optional, default: True
            Whether or not to remove empty lines from the output

        """

        # instantiate output
        output = ""
        divider = "".join(("=" for _ in range(100))) + "\n" # divider

        # loop through fields to maintain order
        for attribute in self.__dict__.keys():
            
            output += divider

            # yaml dump normally
            if attribute != "resolution":
                output += f"{attribute.upper()}\n"
                output += sub(pattern = "!!python/object:.*classes.", repl = "", string = yaml.dump(data = self.__dict__[attribute]))

            # resolution is special, since it's just a number
            else:
                output += f"{attribute.upper()}: {self.__dict__[attribute]}\n"
        output += divider

        # remove_empty_lines
        if remove_empty_lines:

            output_filtered = "" # instantiate
            for line in output.splitlines():
                if not any((hitword in line for hitword in ("null", "[]"))): # if line contains null or empty list
                    output_filtered += line + "\n"

            output = output_filtered # reset output
                
        # output
        if output_filepath:
            with open(output_filepath, "w") as file:
                file.write(output)
        else:
            print(output)
    
    ##################################################


    # CONVERSIONS BETWEEN MUSPY TIME SYSTEM AND REAL TIME / MEASURES
    ##################################################
    
    def metrical_time_to_absolute_time(self, time_steps: int) -> float:
        """Convert from MusPy time (in time steps) to Absolute time (in seconds)."""

        # create list of temporal features (tempo and timesig)
        temporal_features = sorted(self.time_signatures + self.tempos, key = lambda obj: obj.time)
        temporal_features.insert(0, TimeSignature(time = 0, measure = 1, numerator = 4, denominator = 4)) # add default starting timesig
        temporal_features.insert(1, Tempo(time = 0, measure = 1, qpm = 60)) # add default starting tempo
        largest_possible_timestep = self.get_song_length()
        temporal_features.append(TimeSignature(time = largest_possible_timestep, measure = 1, numerator = 4, denominator = 4)) # add default ending timesig

        # initialize some variables
        time_signature_idx = 0 # keep track of most recent timesig
        tempo_idx = 1 # keep track of most recent tempo
        most_recent_time_step = 0 # keep track of most recent time step
        reached_time_steps = False # check if we ever reached time_steps, if we didnt, raise an error
        time = 0.0 # running count of time elapsed

        # loop through temporal features
        for i in range(2, len(temporal_features)):

            # check if we reached time_steps
            if time_steps <= temporal_features[i].time:
                end = time_steps
                reached_time_steps = True # update boolean flag
            else:
                end = temporal_features[i].time
            period_length = end - most_recent_time_step
            
            # update time elapsed
            quarters_per_minute_at_tempo = temporal_features[tempo_idx].qpm + DIVIDE_BY_ZERO_CONSTANT # to avoid divide by zero error
            time_signature_denominator = temporal_features[time_signature_idx].denominator + DIVIDE_BY_ZERO_CONSTANT
            time += (period_length / self.resolution) * (60 / quarters_per_minute_at_tempo) * (4 / time_signature_denominator)

            # break if reached time steps
            if reached_time_steps:
                return time

            # update most recent time step
            most_recent_time_step += period_length

            # check for temporal feature type
            if type(temporal_features[i]) is TimeSignature:
                time_signature_idx = i
            elif type(temporal_features[i]) is Tempo:
                tempo_idx = i

        # return end time if we never reached time steps
        return time
    
        # if still haven't reached time steps by the time we've parsed through the whole song
        # if not reached_time_steps: # go on with pace at the end
        #     period_length = time_steps - largest_possible_timestep
        #     time += (period_length / self.resolution) * (60 / temporal_features[tempo_idx].qpm) * (4 / temporal_features[time_signature_idx].denominator)
        #     return time

    ##################################################


    # GET THE LENGTH OF THE SONG IN TIME STEPS
    ##################################################

    def get_song_length(self) -> int:
        if len(self.barlines) >= 2:
            return int((2 * self.beats[-1].time) - self.beats[-2].time)
        else:
            return 0

    ##################################################


    # SAVE THE BETTERMUSIC OBJECT AS PICKLE FILE
    ##################################################

    def save(self, path: str) -> str: # returns string of filepath where it was saved to
        
        with open(path, "wb") as pickle_file:
            pickle.dump(obj = path, file = pickle_file, protocol = pickle.HIGHEST_PROTOCOL)

        return path

    ##################################################

##################################################

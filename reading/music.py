# README
# Phillip Long
# October 3, 2023

# child class of Muspy.Music better suited for storing expressive features from musescore
# Music class -- a universal container for symbolic music.
# copied mostly from https://github.com/salu133445/muspy/blob/b2d4265c6279e730903d8abe9dddda8484511903/muspy/music.py
# Classes
# -------
# - Music

# python /home/pnlong/model_musescore/reading/music.py


# IMPORTS
##################################################

import muspy
from collections import OrderedDict
from typing import List
from re import sub
import yaml # for printing
import json
import gzip
import utils
import numpy as np
from warnings import warn
from copy import deepcopy

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from classes import *
from output import write_midi, write_audio, write_musicxml, get_expressive_features_per_note, VELOCITY_INCREASE_FACTOR, ACCENT_VELOCITY_INCREASE_FACTOR, PEDAL_DURATION_CHANGE_FACTOR, STACCATO_DURATION_CHANGE_FACTOR, FERMATA_TEMPO_SLOWDOWN

##################################################


# CONSTANTS
##################################################

DIVIDE_BY_ZERO_CONSTANT = 1e-10

##################################################


# HELPER FUNCTIONS
##################################################

def to_dict(obj) -> dict:
    """Convert an object into a dictionary (for .json output)."""

    # base case
    if isinstance(obj, bool) or isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float) or obj is None:
        return obj

    # deal with lists
    elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
        return [to_dict(obj = value) for value in obj]
    
    # deal with dictionaries
    elif isinstance(obj, dict):
        return {key: to_dict(obj = value) for key, value in obj.items()}

    # deal with objects
    else:
        return {key: to_dict(obj = value) for key, value in ([("__class__.__name__", obj.__class__.__name__)] + list(vars(obj).items()))}

##################################################

# BETTER MUSIC CLASS
##################################################

class MusicRender(muspy.music.Music):
    """A universal container for symbolic music, better suited for storing expressive features than MusPy.

    Attributes
    ----------
    metadata : :class:`read_musescore.Metadata`, default: `Metadata()`
        Metadata.
    resolution : int, default: `muspy.DEFAULT_RESOLUTION` (24)
        Time steps per quarter note.
    tempos : list of :class:`read_musescore.Tempo`, default: []
        Tempo changes.
    key_signatures : list of :class:`read_musescore.KeySignature`, default: []
        Key signatures changes.
    time_signatures : list of :class:`read_musescore.TimeSignature`, default: []
        Time signature changes.
    barlines : list of :class:`read_musescore.Barline`, default: []
        Barlines.
    beats : list of :class:`read_musescore.Beat`, default: []
        Beats.
    lyrics : list of :class:`read_musescore.Lyric`, default: []
        Lyrics.
    annotations : list of :class:`read_musescore.Annotation`, default: []
        Annotations.
    tracks : list of :class:`read_musescore.Track`, default: []
        Music tracks.
    song_length : int
        The length of the song (in time steps).
    infer_velocity : bool
        Should the velocity values of notes be altered to reflect expressive features?
    absolute_time : bool
        Are the times of different objects in seconds?

    Note
    ----
    Indexing a MusicRender object returns the track of a certain index. That is, ``music[idx]`` returns ``music.tracks[idx]``. Length of a Music object is the number of tracks. That is, ``len(music)`` returns ``len(music.tracks)``.

    """


    _attributes = OrderedDict([("metadata", Metadata), ("resolution", int), ("tempos", Tempo), ("key_signatures", KeySignature), ("time_signatures", TimeSignature), ("barlines", Barline), ("beats", Beat), ("lyrics", Lyric), ("annotations", Annotation), ("tracks", Track), ("song_length", int), ("infer_velocity", bool), ("absolute_time", bool)])
    _optional_attributes = ["metadata", "resolution", "tempos", "key_signatures", "time_signatures", "barlines", "beats", "lyrics", "annotations", "tracks", "song_length", "infer_velocity", "absolute_time"]
    _list_attributes = ["tempos", "key_signatures", "time_signatures", "barlines", "beats", "lyrics", "annotations", "tracks"]


    # INITIALIZER
    ##################################################

    def __init__(
            self,
            metadata: Metadata = None,
            resolution: int = None,
            tempos: List[Tempo] = None,
            key_signatures: List[KeySignature] = None,
            time_signatures: List[TimeSignature] = None,
            barlines: List[Barline] = None,
            beats: List[Beat] = None,
            lyrics: List[Lyric] = None,
            annotations: List[Annotation] = None,
            tracks: List[Track] = None,
            song_length: int = None,
            infer_velocity: bool = True,
            absolute_time: bool = False
        ):
        self.metadata = metadata if metadata is not None else Metadata()
        self.resolution = resolution if resolution is not None else muspy.DEFAULT_RESOLUTION
        self.tempos = tempos if tempos is not None else []
        if not any((tempo.time == 0 for tempo in self.tempos)):
            self.tempos.insert(0, Tempo(time = 0, qpm = DEFAULT_QPM))
        self.key_signatures = key_signatures if key_signatures is not None else []
        self.time_signatures = time_signatures if time_signatures is not None else []
        self.beats = beats if beats is not None else []
        self.barlines = barlines if barlines is not None else []
        self.lyrics = lyrics if lyrics is not None else []
        self.annotations = annotations if annotations is not None else []
        self.tracks = tracks if tracks is not None else []
        self.song_length = self.get_song_length() if song_length is None else song_length
        self.infer_velocity = infer_velocity
        self.absolute_time = absolute_time

    ##################################################


    # PRETTY PRINTING
    ##################################################

    def print(self, output_filepath: str = None, remove_empty_lines: bool = True):
        """Print the MusicRender object in a pretty way.
        
        Parameters
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
        """Convert from MusPy metrical time (in time steps) to absolute time (in seconds).
        
        Parameters
        ---------
        time_steps : int
            The time steps value to convert
        
        Returns
        ---------
        `time_steps` in seconds into the song
        """

        if self.absolute_time:
            warn("Time values are already in absolute time (`absolute_time` = True). Returning unaltered time value.", RuntimeWarning)
            return time_steps

        # create list of temporal features (tempo and time_signature)
        temporal_features = sorted(self.time_signatures + self.tempos, key = lambda obj: obj.time)
        temporal_features.insert(0, TimeSignature(time = 0, measure = 1, numerator = 4, denominator = 4)) # add default starting time_signature
        temporal_features.insert(1, Tempo(time = 0, measure = 1, qpm = 60)) # add default starting tempo
        temporal_features.append(TimeSignature(time = self.song_length, measure = 1, numerator = 4, denominator = 4)) # add default ending time_signature

        # initialize some variables
        time_signature_idx = 0 # keep track of most recent time_signature
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
            if isinstance(temporal_features[i], TimeSignature):
                time_signature_idx = i
            elif isinstance(temporal_features[i], Tempo):
                tempo_idx = i

        # return end time if we never reached time steps
        return time
    
        # if still haven't reached time steps by the time we've parsed through the whole song
        # if not reached_time_steps: # go on with pace at the end
        #     period_length = time_steps - self.song_length
        #     time += (period_length / self.resolution) * (60 / temporal_features[tempo_idx].qpm) * (4 / temporal_features[time_signature_idx].denominator)
        #     return time

    def convert_from_metrical_to_absolute_time(self):
        """
        Convert this object from metrical (time steps) to absolute (seconds) time.
        """

        # do we even need to convert
        if self.absolute_time:
            warn("Time values are already in absolute time (`absolute_time` = True).", RuntimeWarning)
            return
        
        # convert
        for i in range(len(self.beats)):
            self.beats[i].time = self.metrical_time_to_absolute_time(time_steps = self.beats[i].time)
        for i in range(len(self.barlines)):
            self.barlines[i].time = self.metrical_time_to_absolute_time(time_steps = self.barlines[i].time)
        for i in range(len(self.lyrics)):
            self.lyrics[i].time = self.metrical_time_to_absolute_time(time_steps = self.lyrics[i].time)
        for i in range(len(self.annotations)):
            time_time_steps = self.annotations[i].time
            self.annotations[i].time = self.metrical_time_to_absolute_time(time_steps = time_time_steps)
            if hasattr(self.annotations[i].annotation, "duration"):
                self.annotations[i].annotation.duration = self.metrical_time_to_absolute_time(time_steps = time_time_steps + self.annotations[i].annotation.duration) - self.annotations[i].time
        for i in range(len(self.tracks)):
            for j in range(len(self.tracks[i].notes)):
                time_time_steps = self.tracks[i].notes[j].time
                self.tracks[i].notes[j].time = self.metrical_time_to_absolute_time(time_steps = time_time_steps)
                self.tracks[i].notes[j].duration = self.metrical_time_to_absolute_time(time_steps = time_time_steps + self.tracks[i].notes[j].duration) - self.tracks[i].notes[j].time
            for j in range(len(self.tracks[i].chords)):
                time_time_steps = self.tracks[i].chords[j].time
                self.tracks[i].chords[j].time = self.metrical_time_to_absolute_time(time_steps = time_time_steps)
                self.tracks[i].chords[j].duration = self.metrical_time_to_absolute_time(time_steps = time_time_steps + self.tracks[i].chords[j].duration) - self.tracks[i].chords[j].time
            for j in range(len(self.tracks[i].annotations)):
                time_time_steps = self.tracks[i].annotations[j].time
                self.tracks[i].annotations[j].time = self.metrical_time_to_absolute_time(time_steps = time_time_steps)
                if hasattr(self.tracks[i].annotations[j].annotation, "duration"):
                    self.tracks[i].annotations[j].annotation.duration = self.metrical_time_to_absolute_time(time_steps = time_time_steps + self.tracks[i].annotations[j].annotation.duration) - self.tracks[i].annotations[j].time
            for j in range(len(self.tracks[i].lyrics)):
                self.tracks[i].lyrics[j].time = self.metrical_time_to_absolute_time(time_steps = self.tracks[i].lyrics[j].time)
        for i in range(len(self.tempos)):
            self.tempos[i].time = self.metrical_time_to_absolute_time(time_steps = self.tempos[i].time)
        for i in range(len(self.key_signatures)):
            self.key_signatures[i].time = self.metrical_time_to_absolute_time(time_steps = self.key_signatures[i].time)
        for i in range(len(self.time_signatures)):
            self.time_signatures[i].time = self.metrical_time_to_absolute_time(time_steps = self.time_signatures[i].time)
        
        # update
        self.absolute_time = True

    ##################################################


    # CONVERT FROM ABSOLUTE TO METRICAL TIME
    ##################################################

    def absolute_time_to_metrical_time(self, seconds: float) -> int:
        """Convert from absolute time (in seconds) to MusPy metrical time (in time steps).
        
        Parameters
        ---------
        seconds : float
            The seconds value to convert
        
        Returns
        ---------
        `seconds` in time steps into the song
        """

        if (not self.absolute_time):
            warn("Time values are already in metrical time (`absolute_time` = False). Returning unaltered time value.", RuntimeWarning)
            return seconds

        # get temporal features
        temporal_features = sorted(self.tempos + list(filter(lambda annotation: isinstance(annotation.annotation, (Fermata, TempoSpanner)), self.annotations)), key = lambda obj: obj.time)
        temporal_features = list(filter(lambda temporal_feature: temporal_feature.time <= seconds, temporal_features)) # filter such that the times are less than the given time

        # helper function for converting absolute to metrical time
        def get_absolute_to_metrical_time_func(start_time: int = 0, start_time_seconds: float = 0.0, qpm: float = DEFAULT_QPM) -> int:
            """Helper function that returns a function object that converts absolute to metrical time.
            Logic:
                - add start time (in metrical time)
                - convert time from seconds to minutes
                - multiply by qpm value to convert to quarter beats since start time
                - multiply by MusicRender resolution
            """
            return lambda time: int(start_time + ((((time - start_time_seconds) / 60) * qpm) * self.resolution))

        # prepare for adding the notes and expressive features
        current_time = 0.0
        tempo_obj = Tempo(time = 0, qpm = DEFAULT_QPM)
        absolute_to_metrical_time_func = get_absolute_to_metrical_time_func(start_time = tempo_obj.time, start_time_seconds = current_time, qpm = tempo_obj.qpm) # self.tempos[tempo_idx] is the default tempo (time == 0 and qpm == DEFAULT_QPM)
        fermata_on, fermata_time = False, -1
        
        # iterate through temporal features
        for temporal_feature in temporal_features:
            
            # get the current time and duration in time steps
            time, duration = temporal_feature.time, 0.0
            if hasattr(temporal_feature, "annotation"):
                if hasattr(temporal_feature.annotation, "duration"): # get duration if there
                    duration = temporal_feature.annotation.duration
            time_seconds, duration_seconds = time, duration # store the time and duration in seconds
            time = int(absolute_to_metrical_time_func(time = time_seconds)) # get time in time steps
            duration = int(absolute_to_metrical_time_func(time = duration_seconds)) # get duration in time steps
            if fermata_on and (time_seconds != fermata_time): # update tempo function if necessary
                absolute_to_metrical_time_func = get_absolute_to_metrical_time_func(start_time = time, start_time_seconds = time_seconds, qpm = tempo_obj.qpm)
                fermata_on = False
   
            # update variables if necessary
            if isinstance(temporal_feature, Tempo):
                absolute_to_metrical_time_func = get_absolute_to_metrical_time_func(start_time = time, start_time_seconds = time_seconds, qpm = temporal_feature.qpm)
                tempo_obj = temporal_feature
            elif isinstance(temporal_feature.annotation, Fermata):
                absolute_to_metrical_time_func = get_absolute_to_metrical_time_func(start_time = time, start_time_seconds = time_seconds, qpm = tempo_obj.qpm / FERMATA_TEMPO_SLOWDOWN)
                fermata_on, fermata_time = True, time_seconds
            elif isinstance(temporal_feature.annotation, TempoSpanner): # currently not implemented
                pass     

        # get time steps           
        time_steps = absolute_to_metrical_time_func(time = seconds)
        return time_steps
    
    def convert_from_absolute_to_metrical_time(self):
        """
        Convert this object from absolute (seconds) to metrical (time steps) time.
        """

        # do we even need to convert
        if (not self.absolute_time):
            warn("Time values are already in metrical time (`absolute_time` = False).", RuntimeWarning)
            return
        
        # convert
        for i in range(len(self.tempos)):
            self.tempos[i].time = self.absolute_time_to_metrical_time(seconds = self.tempos[i].time)
        for i in range(len(self.key_signatures)):
            self.key_signatures[i].time = self.absolute_time_to_metrical_time(seconds = self.key_signatures[i].time)
        for i in range(len(self.time_signatures)):
            self.time_signatures[i].time = self.absolute_time_to_metrical_time(seconds = self.time_signatures[i].time)
        for i in range(len(self.beats)):
            self.beats[i].time = self.absolute_time_to_metrical_time(seconds = self.beats[i].time)
        for i in range(len(self.barlines)):
            self.barlines[i].time = self.absolute_time_to_metrical_time(seconds = self.barlines[i].time)
        for i in range(len(self.lyrics)):
            self.lyrics[i].time = self.absolute_time_to_metrical_time(seconds = self.lyrics[i].time)
        for i in range(len(self.annotations)):
            time_seconds = self.annotations[i].time
            self.annotations[i].time = self.absolute_time_to_metrical_time(seconds = time_seconds)
            if hasattr(self.annotations[i].annotation, "duration"):
                self.annotations[i].annotation.duration = self.absolute_time_to_metrical_time(seconds = time_seconds + self.annotations[i].annotation.duration) - self.annotations[i].time
        for i in range(len(self.tracks)):
            for j in range(len(self.tracks[i].notes)):
                time_seconds = self.tracks[i].notes[j].time
                self.tracks[i].notes[j].time = self.absolute_time_to_metrical_time(seconds = time_seconds)
                self.tracks[i].notes[j].duration = self.absolute_time_to_metrical_time(seconds = time_seconds + self.tracks[i].notes[j].duration) - self.tracks[i].notes[j].time
            for j in range(len(self.tracks[i].chords)):
                time_seconds = self.tracks[i].chords[j].time
                self.tracks[i].chords[j].time = self.absolute_time_to_metrical_time(seconds = time_seconds)
                self.tracks[i].chords[j].duration = self.absolute_time_to_metrical_time(seconds = time_seconds + self.tracks[i].chords[j].duration) - self.tracks[i].chords[j].time
            for j in range(len(self.tracks[i].annotations)):
                time_seconds = self.tracks[i].annotations[j].time
                self.tracks[i].annotations[j].time = self.absolute_time_to_metrical_time(seconds = time_seconds)
                if hasattr(self.tracks[i].annotations[j].annotation, "duration"):
                    self.tracks[i].annotations[j].annotation.duration = self.absolute_time_to_metrical_time(seconds = time_seconds + self.tracks[i].annotations[j].annotation.duration) - self.tracks[i].annotations[j].time
            for j in range(len(self.tracks[i].lyrics)):
                self.tracks[i].lyrics[j].time = self.absolute_time_to_metrical_time(seconds = self.tracks[i].lyrics[j].time)
        
        # update
        self.absolute_time = False

    ##################################################


    # RESET RESOLUTION
    ##################################################
    
    def reset_resolution(self, new_resolution: int = muspy.DEFAULT_RESOLUTION):
        """
        Reset the resolution of this object.
        """

        # do we even need to convert
        if self.absolute_time:
            warn("Time values must be in metrical time (which they are currently not). Not resetting resolution.", RuntimeWarning)
            return
        
        # dont advise downsampling
        if new_resolution < self.resolution:
            warn("`new_resolution` less than the current resolution. Downsampling is not advised.", RuntimeWarning)

        # don't want a float for new resolution
        if int(new_resolution) != new_resolution:
            warn("`new_resolution` should be an int. Truncating.", RuntimeWarning)
            new_resolution = int(new_resolution)
        
        # resolution updater
        time_updater = lambda time: int(time * (new_resolution / self.resolution))
        
        # convert
        for i in range(len(self.tempos)):
            self.tempos[i].time = time_updater(time = self.tempos[i].time)
        for i in range(len(self.key_signatures)):
            self.key_signatures[i].time = time_updater(time = self.key_signatures[i].time)
        for i in range(len(self.time_signatures)):
            self.time_signatures[i].time = time_updater(time = self.time_signatures[i].time)
        for i in range(len(self.beats)):
            self.beats[i].time = time_updater(time = self.beats[i].time)
        for i in range(len(self.barlines)):
            self.barlines[i].time = time_updater(time = self.barlines[i].time)
        for i in range(len(self.lyrics)):
            self.lyrics[i].time = time_updater(time = self.lyrics[i].time)
        for i in range(len(self.annotations)):
            time_ = self.annotations[i].time
            self.annotations[i].time = time_updater(time = time_)
            if hasattr(self.annotations[i].annotation, "duration"):
                self.annotations[i].annotation.duration = time_updater(time = time_ + self.annotations[i].annotation.duration) - self.annotations[i].time
        for i in range(len(self.tracks)):
            for j in range(len(self.tracks[i].notes)):
                time_ = self.tracks[i].notes[j].time
                self.tracks[i].notes[j].time = time_updater(time = time_)
                self.tracks[i].notes[j].duration = time_updater(time = time_ + self.tracks[i].notes[j].duration) - self.tracks[i].notes[j].time
            for j in range(len(self.tracks[i].chords)):
                time_ = self.tracks[i].chords[j].time
                self.tracks[i].chords[j].time = time_updater(time = time_)
                self.tracks[i].chords[j].duration = time_updater(time = time_ + self.tracks[i].chords[j].duration) - self.tracks[i].chords[j].time
            for j in range(len(self.tracks[i].annotations)):
                time_ = self.tracks[i].annotations[j].time
                self.tracks[i].annotations[j].time = time_updater(time = time_)
                if hasattr(self.tracks[i].annotations[j].annotation, "duration"):
                    self.tracks[i].annotations[j].annotation.duration = time_updater(time = time_ + self.tracks[i].annotations[j].annotation.duration) - self.tracks[i].annotations[j].time
            for j in range(len(self.tracks[i].lyrics)):
                self.tracks[i].lyrics[j].time = time_updater(time = self.tracks[i].lyrics[j].time)
        
        # update resolution
        self.resolution = new_resolution

    ##################################################


    # GET THE LENGTH OF THE SONG IN TIME STEPS
    ##################################################

    def _get_max_time_obj_helper(self, obj) -> int:
        end_time = obj.time
        if hasattr(obj, "duration"): # look for duration at top-level
            end_time += obj.duration
        elif hasattr(obj, "annotation"): # look for duration
            if hasattr(obj.annotation, "duration"): # within an annotation
                end_time += obj.annotation.duration
        return end_time
    def get_song_length(self) -> int:
        """Return the length of the song in time steps."""
        all_objs = self.tempos + self.key_signatures + self.time_signatures + self.beats + self.barlines + self.lyrics + self.annotations + sum([track.notes + track.annotations + track.lyrics for track in self.tracks], [])
        if len(all_objs) > 0:
            max_time_obj = max(all_objs, key = self._get_max_time_obj_helper)
            max_time = max_time_obj.time + (max_time_obj.duration if hasattr(max_time_obj, "duration") else 0) # + self.resolution # add a quarter note at the end for buffer
            max_time += 1 # is trivial in the grand scheme, but for muspy stuff
        else:
            max_time = 0
        # final_beat = self.beats[-1].time if len(self.beats) >= 1 else 0 # (2 * self.beats[-1].time) - self.beats[-2].time
        # return int(max(max_time, final_beat))
        return max_time
    def get_end_time(self) -> int:
        """
        Alias for get_song_length().
        """
        return self.get_song_length()

    ##################################################


    # SAVE AS JSON
    ##################################################

    def save(self, path: str, kind: str = None, ensure_ascii: bool = False, compressed: bool = None, **kwargs) -> str:
        """Save a Music object to a JSON or YAML file.

        Parameters
        ----------
        path : str
            Path to save the JSON data.
        compressed : bool, optional
            Whether to save as a compressed JSON or YAML file (`.json.gz` or `.yaml.gz`). Has no effect when `path` is a file object. Defaults to infer from the extension (`.gz`).
        **kwargs
            Keyword arguments to pass to :py:func:`json.dumps` or `yaml.dump`.

        Returns
        -------
        str
            Filepath to which we ended up saving the MusicRender object to.

        Notes
        -----
        When a path is given, use UTF-8 encoding and gzip compression if `compressed=True`.

        """

        if kind is None:
            if path.endswith(".json"):
                kind = "json"
            elif path.endswith(".yaml"):
                kind = "yaml"
            else:
                raise ValueError("Cannot infer file format from the extension (expect JSON or YAML).")
        else:
            kind = kind.lower()
        
        # convert self to dictionary
        data = {
            "metadata": to_dict(obj = self.metadata),
            "resolution": self.resolution,
            "tempos": to_dict(obj = self.tempos),
            "key_signatures": to_dict(obj = self.key_signatures),
            "time_signatures": to_dict(obj = self.time_signatures),
            "beats": to_dict(obj = self.beats),
            "barlines": to_dict(obj = self.barlines),
            "lyrics": to_dict(obj = self.lyrics),
            "annotations": to_dict(obj = self.annotations),
            "tracks": to_dict(obj = self.tracks),
            "song_length": self.song_length,
            "infer_velocity": self.infer_velocity,
            "absolute_time": self.absolute_time,
        }

        # convert dictionary to json obj
        if kind == "json":
            data = json.dumps(obj = data, ensure_ascii = ensure_ascii, **kwargs)
        else:
            data = yaml.safe_dump(data = data, allow_unicode = ensure_ascii, **kwargs)

        # determine if compression is inferred
        if compressed is None:
            compressed = str(path).lower().endswith(".gz")

        # either compress or not
        if compressed:
            path += "" if path.lower().endswith(".gz") else ".gz" # make sure it ends with gz
            with gzip.open(path, "wt", encoding = "utf-8") as file:
                file.write(data)
        else:
            with open(path, "w", encoding = "utf-8") as file:
                file.write(data)
        
        # return the path to which it was saved
        return path

    # wraps the main load() function into an instance method
    def load(self, path: str, kind: str = None):
        """Load a Music object from a JSON or YAML file.

        Parameters
        ----------
        path : str
            Path to the data.
        """

        # load .json file
        if kind is None:
            if path.endswith((".json",)):
                kind = "json"
            elif path.endswith((".yaml", ".yml")):
                kind = "yaml"
            else:
                raise ValueError("Cannot infer file format from the extension (expect JSON or YAML).")
        else:
            kind = kind.lower()
        music = load(path = path, kind = kind)

        # set equal to self
        self.metadata = music.metadata
        self.resolution = music.resolution
        self.tempos = music.tempos
        self.key_signatures = music.key_signatures
        self.time_signatures = music.time_signatures
        self.beats = music.beats
        self.barlines = music.barlines
        self.lyrics = music.lyrics
        self.annotations = music.annotations
        self.tracks = music.tracks
        self.song_length = music.song_length
        self.infer_velocity = music.infer_velocity
        self.absolute_time = music.absolute_time

    ##################################################


    # TRIM
    ##################################################

    def trim(self, start: int = 0, end: int = -1):
        """Trim the MusicRender object.

        Parameters
        ----------
        start : int, default: 0
            Time step at which to trim. Anything before this timestep will be removed.
        end : int, default: song_length
            Time step at which to trim. Anything after this timestep will be removed.
        """

        # deal with start and end arguments
        if end < start:
            end = self.song_length

        # tempos, key_signatures, time_signatures, beats, barlines, and lyrics, all of which lack duration
        self.tempos = [tempo for tempo in self.tempos if (start <= tempo.time and tempo.time < end)] # trim tempos
        self.key_signatures = [key_signature for key_signature in self.key_signatures if (start <= key_signature.time and key_signature.time < end)] # trim key_signatures
        self.time_signatures = [time_signature for time_signature in self.time_signatures if (start <= time_signature.time and time_signature.time < end)] # trim time_signatures
        self.beats = [beat for beat in self.beats if (start <= beat.time and beat.time < end)] # trim beats
        self.barlines = [barline for barline in self.barlines if (start <= barline.time and barline.time < end)] # trim barlines
        self.lyrics = [lyric for lyric in self.lyrics if (start <= lyric.time and lyric.time < end)] # trim lyrics

        # system annotations
        self.annotations = [annotation for annotation in self.annotations if (start <= annotation.time and annotation.time < end)] # trim system annotations
        for i, annotation in enumerate(self.annotations): # loop through system annotations
            if hasattr(annotation.annotation, "duration"): # check for duration
                if (annotation.time + annotation.annotation.duration) > end: # if end of annotation is past the end
                    self.annotations[i].annotation.duration = end - annotation.time # cut duration off duration at end

        # tracks (lyrics, notes, chords, staff_annotations)
        for i in range(len(self.tracks)): # loop through tracks

            # lyrics (no duration)
            self.tracks[i].lyrics = [lyric for lyric in self.tracks[i].lyrics if (start <= lyric.time and lyric.time < end)]

            # notes
            self.tracks[i].notes = [note for note in self.tracks[i].notes if (start <= note.time and note.time < end)] # trim notes
            for j, note in enumerate(self.tracks[i].notes): # loop through notes
                if (note.time + note.duration) > end: # if end of note is past the end
                    self.tracks[i].notes[j].duration = end - note.time # cut duration off at end

            # chords
            self.tracks[i].chords = [chord for chord in self.tracks[i].chords if (start <= chord.time and chord.time < end)] # trim chords
            for j, chord in enumerate(self.tracks[i].chords): # loop through chords
                if (chord.time + chord.duration) > end: # if end of chord is past the end
                    self.tracks[i].chords[j].duration = end - chord.time # cut duration off at end

            # staff annotations
            self.tracks[i].annotations = [annotation for annotation in self.tracks[i].annotations if (start <= annotation.time and annotation.time < end)]
            for j, annotation in enumerate(self.tracks[i].annotations): # loop through system annotations
                if hasattr(annotation.annotation, "duration"): # check for duration
                    if (annotation.time + annotation.annotation.duration) > end: # if end of annotation is past the end
                        self.tracks[i].annotations[j].annotation.duration = end - annotation.time # cut duration off duration at end
        
        # update song_length
        self.song_length = self.get_song_length()

    ##################################################


    # WRITE
    ##################################################

    def write(self, path: str, kind: str = None, **kwargs):
        """Write a MusicRender object in various file formats.

        Parameters
        ----------
        path : str
            Path to write to. File format can be inferred.
        kind : str, default: None
            File format of output. If not provided, the file format is inferred from `path`.
        """

        # infer kind if necessary
        if kind is None:
            if path.lower().endswith((".wav", ".aiff", ".flac", ".oga",)):
                kind = "audio"
            elif path.lower().endswith((".mid", ".midi",)):
                kind = "midi"
            elif path.lower().endswith((".mxl", ".xml", ".mxml", ".musicxml",)):
                kind = "musicxml"
            elif path.lower().endswith((".json",)):
                kind = "json"
            elif path.lower().endswith((".yaml", ".yml")):
                kind = "yaml"
            else:
                raise ValueError("Cannot infer file format from the extension (expect MIDI, MusicXML, WAV, AIFF, FLAC, OGA, JSON, or YAML).")
        
        # output
        if kind.lower() == "audio": # write audio
            return write_audio(path = path, music = self, **kwargs)
        elif kind.lower() == "midi": # write midi
            return write_midi(path = path, music = self, **kwargs)
        elif kind.lower() == "musicxml": # write musicxml
            if self.absolute_time: # convert to metrical time if not already
                music = deepcopy(self)
                music.convert_from_absolute_to_metrical_time()
            else:
                music = self
            return write_musicxml(path = path, music = music, **kwargs)
        elif kind.lower() in ("json", "yaml"):
            return self.save(path = path, kind = kind)
        else:
            raise ValueError(f"Expect `kind` to be 'midi', 'musicxml', 'audio', or 'json', but got : {kind}.")

    ##################################################
        

    # REALIZE EXPRESSIVE FEATURES TO THEIR FULLEST EXTENT
    ##################################################

    def realize_expressive_features(self):
        """Changes note velocities and durations to reflect expressive features.
        BE CAREFUL USING BEFORE WRITE(), AS WRITE IMPLEMENTS ITS OWN EXPRESSIVE FEATURE REALIZATION!!!
        """

        # realize velocity information
        # music = deepcopy(music) # don't alter the original object
        for i in range(len(self.tracks)):
            note_times = sorted(utils.unique(l = [note.time for note in self.tracks[i].notes])) # times of notes, sorted ascending, removing duplicates
            expressive_features = get_expressive_features_per_note(note_times = note_times, all_annotations = self.tracks[i].annotations + self.annotations) # dictionary where keys are time and values are expressive feature annotation objects
            # note on and note off messages
            for note in self.tracks[i].notes:
                if self.infer_velocity:
                    note.velocity = expressive_features[note.time][0].annotation.velocity # the first index is always the dynamic
                for annotation in expressive_features[note.time][1:]: # skip the first index, since we just dealt with it
                    # ensure that values are valid
                    if hasattr(annotation.annotation, "subtype"): # make sure subtype field is not none
                        if annotation.annotation.subtype is None:
                            continue
                    # HairPinSpanner and TempoSpanner; changes in velocity
                    if (annotation.annotation.__class__.__name__ in ("HairPinSpanner", "TempoSpanner")) and self.infer_velocity: # some TempoSpanners involve a velocity change, so that is included here as well
                        if annotation.group is None: # since we aren't storing anything there anyways
                            end_velocity = note.velocity # default is no change
                            if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("allarg", "cr"))): # increase-volume; allargando, crescendo
                                end_velocity *= VELOCITY_INCREASE_FACTOR
                            elif any((annotation.annotation.subtype.startswith(prefix) for prefix in ("smorz", "dim", "decr"))): # decrease-volume; smorzando, diminuendo, decrescendo
                                end_velocity /= VELOCITY_INCREASE_FACTOR
                            denominator = (annotation.time + annotation.annotation.duration) - note.time
                            annotation.group = lambda time: (((end_velocity - note.velocity) / denominator) * (time - note.time)) + note.velocity if (denominator != 0) else end_velocity # we will use group to store a lambda function to calculate velocity
                        note.velocity += annotation.group(time = note.time)
                    # SlurSpanner
                    elif annotation.annotation.__class__.__name__ == "SlurSpanner":
                        current_note_time_index = note_times.index(note.time)
                        if current_note_time_index < len(note_times) - 1: # elsewise, there is no next note to slur to
                            note.duration = note_times[current_note_time_index + 1] - note_times[current_note_time_index]
                        del current_note_time_index
                    # PedalSpanner
                    elif annotation.annotation.__class__.__name__ == "PedalSpanner":
                        note.duration *= PEDAL_DURATION_CHANGE_FACTOR
                    # Articulation
                    elif annotation.annotation.__class__.__name__ == "Articulation":
                        if any((keyword in annotation.annotation.subtype for keyword in ("staccato", "staccatissimo", "spiccato", "pizzicato", "plucked", "marcato", "sforzato"))): # shortens note length
                            note.duration /= STACCATO_DURATION_CHANGE_FACTOR
                        if any((keyword in annotation.annotation.subtype for keyword in ("marcato", "sforzato", "accent"))) and self.infer_velocity: # increases velocity
                            note.velocity += note.velocity * (max((ACCENT_VELOCITY_INCREASE_FACTOR * (0.8 if "soft" in annotation.annotation.subtype else 1)), 1) - 1)
                        if ("spiccato" in annotation.annotation.subtype) and self.infer_velocity: # decreases velocity
                            note.velocity /= ACCENT_VELOCITY_INCREASE_FACTOR
                        if "tenuto" in annotation.annotation.subtype:
                            pass # the duration is full duration
                        # if "wiggle" in annotation.annotation.subtype: # vibrato and sawtooth
                        #     pass # currently no implementation
                        # if "portato" in annotation.annotation.subtype:
                        #     pass # currently no implementation
                        # if "trill" in annotation.annotation.subtype:
                        #     pass # currently no implementation
                        # if "mordent" in annotation.annotation.subtype:
                        #     pass # currently no implementation
                        # if "close" in annotation.annotation.subtype: # reference to a mute
                        #     pass # currently no implementation
                        # if any((keyword in annotation.annotation.subtype for keyword in ("open", "ouvert"))): # reference to a mute
                        #     pass # currently no implementation
                    # TechAnnotation
                    # elif annotation.annotation.__class__.__name__ == "TechAnnotation":
                    #     pass # currently no implementation since we so rarely encounter these

    ##################################################

##################################################


# LOAD A BETTERMUSIC OBJECT FROM JSON FILE
##################################################

# converting objects
get_str = lambda string: str(string) if (string is not None) else None
get_int = lambda integer: int(integer) if (integer is not None) else None
get_float = lambda double: float(double) if (double is not None) else None

# helper function to load the correct annotation object
def load_annotation(annotation: dict):
    """Return an expressive feature object given an annotation dictionary. For loading from .json."""

    if annotation is None:
        return None

    match annotation["__class__.__name__"]:
        case "Text":
            return Text(text = get_str(annotation["text"]), is_system = bool(annotation["is_system"]), style = get_str(annotation["style"]))
        case "Subtype":
            return Subtype(subtype = get_str(annotation["subtype"]))                                                                                                                                                   
        case "RehearsalMark":
            return RehearsalMark(text = get_str(annotation["text"]))
        case "TechAnnotation":
            return TechAnnotation(text = get_str(annotation["text"]), tech_type = get_str(annotation["tech_type"]), is_system = bool(annotation["is_system"]))
        case "Dynamic":
            return Dynamic(subtype = get_str(annotation["subtype"]), velocity = get_int(annotation["velocity"]))
        case "Fermata":
            return Fermata(is_fermata_above = bool(annotation["is_fermata_above"]))
        case "Arpeggio":
            return Arpeggio(subtype = Arpeggio.SUBTYPES.index(annotation["subtype"]))
        case "Tremolo":
            return Tremolo(subtype = get_str(annotation["subtype"]))
        case "ChordSymbol":
            return ChordSymbol(root_str = get_str(annotation["root_str"]), name = get_str(annotation["name"]))
        case "ChordLine":
            return ChordLine(subtype = ChordLine.SUBTYPES.index(annotation["subtype"]), is_straight = bool(annotation["is_straight"]))
        case "Ornament":
            return Ornament(subtype = get_str(annotation["subtype"]))
        case "Articulation":
            return Articulation(subtype = get_str(annotation["subtype"]))
        case "Notehead":
            return Notehead(subtype = get_str(annotation["subtype"]))
        case "Symbol":
            return Symbol(subtype = get_str(annotation["subtype"]))
        case "Bend":
            return Bend(points = [Point(time = get_int(point["time"]), pitch = get_int(point["pitch"]), vibrato = get_int(point["vibrato"])) for point in annotation["points"]])
        case "TremoloBar":
            return TremoloBar(points = [Point(time = get_int(point["time"]), pitch = get_int(point["pitch"]), vibrato = get_int(point["vibrato"])) for point in annotation["points"]])
        case "Spanner":
            return Spanner(duration = get_int(annotation["duration"]))
        case "SubtypeSpanner":
            return SubtypeSpanner(duration = get_int(annotation["duration"]), subtype = annotation["subtype"])
        case "TempoSpanner":
            return TempoSpanner(duration = get_int(annotation["duration"]), subtype = get_str(annotation["subtype"]))
        case "TextSpanner":
            return TextSpanner(duration = get_int(annotation["duration"]), text = get_str(annotation["text"]), is_system = bool(annotation["is_system"]))
        case "HairPinSpanner":
            return HairPinSpanner(duration = get_int(annotation["duration"]), subtype = get_str(annotation["subtype"]), hairpin_type = get_int(annotation["hairpin_type"]))
        case "SlurSpanner":
            return SlurSpanner(duration = get_int(annotation["duration"]), is_slur = bool(annotation["is_slur"]))
        case "PedalSpanner":
            return PedalSpanner(duration = get_int(annotation["duration"]))
        case "TrillSpanner":
            return TrillSpanner(duration = get_int(annotation["duration"]), subtype = get_str(annotation["subtype"]), ornament = get_str(annotation["ornament"]))
        case "VibratoSpanner":
            return VibratoSpanner(duration = get_int(annotation["duration"]), subtype = get_str(annotation["subtype"]))
        case "GlissandoSpanner":
            return GlissandoSpanner(duration = get_int(annotation["duration"]), is_wavy = bool(annotation["is_wavy"]))
        case "OttavaSpanner":
            return OttavaSpanner(duration = get_int(annotation["duration"]), subtype = get_str(annotation["subtype"]))
        case _:
            raise KeyError(f"Unknown annotation type `{annotation['__class__.__name__']}`.")


def load(path: str, kind: str = None) -> MusicRender:
    """Load a Music object from a JSON or YAML file.

    Parameters
    ----------
    path : str
        Path to the data.
    """

    # infer kind
    if kind is None:
        if ".json" in path:
            kind = "json"
        elif (".yaml" in path) or (".yml" in path):
            kind = "yaml"
        else:
            raise ValueError("Cannot infer file format from the extension (expect JSON or YAML).")
    else:
        kind = kind.lower()
    loader = json.load if (kind == "json") else yaml.safe_load

    # if file is compressed
    try:
        if path.lower().endswith(".gz"):
            with gzip.open(path, "rt", encoding = "utf-8") as file:
                data = loader(file)
        else:
            with open(path, "rt", encoding = "utf-8") as file:
                data = loader(file)
    except:
        warn(f"{path} could not be loaded, since the JSON file is empty. Returning empty MusicRender object.")
        return MusicRender()
        # raise ValueError(f"{path} could not be loaded.")

    # extract info from nested dictionaries
    metadata = Metadata(
        schema_version = get_str(data["metadata"]["schema_version"]),
        title = get_str(data["metadata"]["title"]),
        subtitle = get_str(data["metadata"]["subtitle"]),
        creators = data["metadata"]["creators"],
        copyright = get_str(data["metadata"]["copyright"]),
        collection = get_str(data["metadata"]["collection"]),
        source_filename = get_str(data["metadata"]["source_filename"]),
        source_format = get_str(data["metadata"]["source_format"]),
    )
    tempos = [Tempo(
        time = get_int(tempo["time"]),
        qpm = get_float(tempo["qpm"]),
        text = get_str(tempo["text"]),
        measure = get_int(tempo["measure"]),
    ) for tempo in data["tempos"]]
    key_signatures = [KeySignature(
        time = get_int(key_signature["time"]),
        root = get_int(key_signature["root"]),
        mode = get_str(key_signature["mode"]),
        fifths = get_int(key_signature["fifths"]),
        root_str = get_str(key_signature["root_str"]),
        measure = get_int(key_signature["measure"]),
    ) for key_signature in data["key_signatures"]]
    time_signatures = [TimeSignature(
        time = get_int(time_signature["time"]),
        numerator = get_int(time_signature["numerator"]),
        denominator = get_int(time_signature["denominator"]),
        measure = get_int(time_signature["measure"]),
    ) for time_signature in data["time_signatures"]]
    beats = [Beat(
        time = get_int(beat["time"]),
        is_downbeat = bool(beat["is_downbeat"]),
        measure = get_int(beat["measure"]),
    ) for beat in data["beats"]]
    barlines = [Barline(
        time = get_int(barline["time"]),
        subtype = get_str(barline["subtype"]),
        measure = get_int(barline["measure"]),
    ) for barline in data["barlines"]]
    lyrics = [Lyric(
        time = get_int(lyric["time"]),
        lyric = get_str(lyric["lyric"]),
        measure = get_int(lyric["measure"]),
    ) for lyric in data["lyrics"]]
    annotations = [Annotation(
        time = get_int(annotation["time"]),
        annotation = load_annotation(annotation = annotation["annotation"]),
        measure = get_int(annotation["measure"]),
        group = get_str(annotation["group"]),
    ) for annotation in data["annotations"]]
    tracks = [Track(
        program = get_int(track["program"]),
        is_drum = bool(track["is_drum"]),
        name = get_str(track["name"]),
        notes = [Note(
            time = get_int(note["time"]),
            pitch = get_int(note["pitch"]),
            duration = get_int(note["duration"]),
            velocity = get_int(note["velocity"]),
            pitch_str = get_str(note["pitch_str"]),
            is_grace = bool(note["is_grace"]),
            measure = get_int(note["measure"]),
            ) for note in track["notes"]],
        chords = [Chord(
            time = get_int(chord["time"]),
            pitches = [get_int(pitch) for pitch in chord["pitches"]],
            duration = get_int(chord["duration"]),
            velocity = get_int(chord["velocity"]),
            pitches_str = [get_str(pitch_str) for pitch_str in chord["pitches_str"]],
            measure = get_int(chord["measure"]),
            ) for chord in track["chords"]],
        lyrics = [Lyric(
            time = get_int(lyric["time"]),
            lyric = get_str(lyric["lyric"]),
            measure = get_int(lyric["measure"]),
            ) for lyric in track["lyrics"]],
        annotations = [Annotation(
            time = get_int(annotation["time"]),
            annotation = load_annotation(annotation = annotation["annotation"]),
            measure = get_int(annotation["measure"]),
            group = get_str(annotation["group"]),
            ) for annotation in track["annotations"]]
    ) for track in data["tracks"]]

    # return a MusicRender object
    return MusicRender(
        metadata = metadata,
        resolution = int(data["resolution"]),
        tempos = tempos,
        key_signatures = key_signatures,
        time_signatures = time_signatures,
        beats = beats,
        barlines = barlines,
        lyrics = lyrics,
        annotations = annotations,
        tracks = tracks,
        song_length = get_int(data["song_length"]),
        infer_velocity = bool(data["infer_velocity"]),
        absolute_time = bool(data["absolute_time"])
    )

##################################################
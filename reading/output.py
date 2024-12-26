# README
# Phillip Long
# December 1, 2023

# Functions for outputting a MusicRender object to different file formats.

# python /home/pnlong/model_musescore/reading/output.py


# IMPORTS
##################################################

import subprocess
from os import rename
from os.path import exists, expanduser
from re import sub
from typing import Tuple, Dict, List, Callable
from numpy import linspace
from copy import deepcopy
from math import sin, pi
from fractions import Fraction
from itertools import groupby
from warnings import warn

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from classes import *
from utils import unique

# midi
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick, MAX_PITCHWHEEL

# audio
import tempfile

# musicxml
import xml.etree.ElementTree as ET
from muspy.utils import CIRCLE_OF_FIFTHS, MODE_CENTERS
import music21.musicxml.archiveTools
from music21.musicxml.archiveTools import compressXML
from music21.key import KeySignature as M21KeySignature
from music21.metadata import Metadata as M21MetaData, Contributor, Copyright
from music21.meter import TimeSignature as M21TimeSignature
from music21.note import Note as M21Note, noteheadTypeNames
from music21.chord import Chord as M21Chord
from music21.harmony import ChordSymbol as M21ChordSymbol
from music21.instrument import instrumentFromMidiProgram
from music21.clef import PercussionClef
from music21.stream import Part, Score
from music21.duration import Duration
from music21.interval import Interval
from music21 import tempo as M21Tempo
from music21 import articulations as M21Articulation
from music21 import expressions as M21Expression
from music21 import spanner as M21Spanner
from music21 import dynamics as M21Dynamic
from music21.exceptions21 import StreamException
import music21.beam

# silence weird warnings from music21
def noop(input):
    pass
music21.musicxml.archiveTools.environLocal.warn = noop
music21.beam.environLocal.warn = noop

##################################################


# CONSTANTS
##################################################

FERMATA_TEMPO_SLOWDOWN = 3 # factor by which to slow down the tempo when there is a fermata
N_NOTES = 128 # number of notes for midi
RESOLUTION = 12 # resolution for MusicRender
PEDAL_DURATION_CHANGE_FACTOR = 3 # factor by which the sustain pedal increases the duration of each note
STACCATO_DURATION_CHANGE_FACTOR = 5 # factor by which a staccato decreases the duration of a note
VELOCITY_INCREASE_FACTOR = 2 # factor by which to increase velocity when an expressive feature GRADUALLY increases velocity
ACCENT_VELOCITY_INCREASE_FACTOR = 1.5 # factor by which to increase velocity when an accent INSTANTANEOUSLY increases velocity
FRACTION_TO_WIGGLE = 0.34 # fraction of MAX/MIN_PITCHWHEEL to bend notes for wiggle articulations (vibratos and sawtooths)
DEFAULT_TEMPO = bpm2tempo(bpm = DEFAULT_QPM)
N_TEMPO_SPANNER_SUBDIVISIONS = 5 # number of subdivisions for increasing/decreasing tempo with a tempo spanner
GRACE_NOTE_FORWARD_SHIFT_CONSTANT = 0.15 # fraction of a quarter note's duration to shift a note forward if it is a grace note
SWING_PROPORTION = 0.6666666667 # on what fraction of a beat does a swung eight note fall
OPTIMAL_RESOLUTION = 480 # we don't want too small of a resolution, or it's hard to apply expressive features

# dynamics
MAX_VELOCITY = 127 # maximum velocity for midi
DEFAULT_DYNAMIC = "mf" # not to be confused with representation.DEFAULT_DYNAMIC
DYNAMIC_VELOCITY_MAP = {
    "pppppp": 4, "ppppp": 8, "pppp": 12, "ppp": 16, "pp": 33, "p": 49, "mp": 64,
    "mf": 80, "f": 96, "ff": 112, "fff": 126, "ffff": 127, "fffff": 127, "ffffff": 127,
    "sfpp": 96, "sfp": 112, "sf": 112, "sff": 126, "sfz": 112, "sffz": 126, "fz": 112, "rf": 112, "rfz": 112,
    "fp": 96, "pf": 49, "s": DEFAULT_VELOCITY, "r": DEFAULT_VELOCITY, "z": DEFAULT_VELOCITY, "n": DEFAULT_VELOCITY, "m": DEFAULT_VELOCITY,
}
DYNAMIC_DYNAMICS = set(tuple(DYNAMIC_VELOCITY_MAP.keys())[:tuple(DYNAMIC_VELOCITY_MAP.keys()).index("ffffff") + 1]) # dynamics that are actually dynamic markings and not sudden dynamic hikes

# MIDI
DRUM_CHANNEL = 9

##################################################


# HELPER FUNCTIONS
##################################################

def clean_up_subtype(subtype: str) -> str:
    """Clean up the subtype of an annotation so that I can better match for substrings."""
    return sub(pattern = "[^\w0-9]", repl = "", string = subtype.lower())

def round_to_nearest_partial(value: float, resolution: float) -> float:
    """Round a value to the nearest resolution"""
    return round(value / resolution) * resolution

def get_octave_from_pitch(pitch: int) -> int:
    """Given the MIDI pitch number, return the octave of that note."""
    return (pitch // 12) - 1

def convert_chromatic_to_circle_of_fifths_steps(transpose_chromatic: int = 0) -> int:
    """
    Given a number of semitones to transpose, return the necessary number of steps to
    make the same transposition on the circle of fifths.
    """
    transpose_chromatic = transpose_chromatic % -12 # get in range 0 through 12
    if (-transpose_chromatic % 2 == 0):
        return transpose_chromatic
    else:
        return (transpose_chromatic + 6) % -12

##################################################


# WRITE MIDI
##################################################

def to_mido_note_on_note_off(note: Note, channel: int, use_note_off_message: bool = False) -> Tuple[Message, Message]:
    """Return a Note object as mido Message objects.

    Timing is in absolute time, NOT in delta time.

    Parameters
    ----------
    note : :class:`Note` object
        Note object to convert.
    channel : int
        Channel of the .mid message.
    use_note_off_message : bool, default: False
        Whether to use note-off messages. If False, note-on messages with zero velocity are used instead. The advantage to using note-on messages at zero velocity is that it can avoid sending additional status bytes when Running Status is employed.

    Returns
    -------
    :class:`mido.Message` object
        Converted mido Message object for note on.
    :class:`mido.Message` object
        Converted mido Message object for note off.

    """

    # deal with velocity
    velocity = note.velocity # copy of velocity as to not alter the music object
    if velocity is None:
        velocity = DEFAULT_VELOCITY
    velocity = int(max(min(velocity, MAX_VELOCITY), 0)) # make sure velocity is within valid range and an integer

    # deal with note
    pitch = note.pitch
    pitch = int(max(min(pitch, N_NOTES - 1), 0)) # make sure the note is within valid range and an integer

    # note on message
    note_on_msg = Message(type = "note_on", time = note.time, note = pitch, velocity = velocity, channel = channel) # create note on message

    # note off message
    if use_note_off_message: # create note off message
        note_off_msg = Message(type = "note_off", time = note.time + note.duration, note = pitch, velocity = velocity, channel = channel)
    else:
        note_off_msg = Message(type = "note_on", time = note.time + note.duration, note = pitch, velocity = 0, channel = channel)

    # return messages
    return note_on_msg, note_off_msg


def to_delta_time(midi_track: MidiTrack, ticks_per_beat: int, absolute_time: bool = False):
    """Convert a mido MidiTrack object from absolute time to delta time.

    Parameters
    ----------
    midi_track : :class:`mido.MidiTrack` object
        mido MidiTrack object to convert.
    ticks_per_beat : int
        Number of MIDI ticks per beat.
    absolute_time : bool, default: False
        Is the provided .mid file in absolute time (seconds) or metrical time (time_steps)?

    """

    # sort messages by absolute time
    midi_track.sort(key = lambda message: message.time)
    
    # convert seconds to ticks if necessary
    if absolute_time:
        for message in midi_track:
            message.time = second2tick(second = message.time, ticks_per_beat = ticks_per_beat, tempo = DEFAULT_TEMPO)

    # convert to delta time
    time = 0
    for message in midi_track:
        time_ = message.time
        message.time = int(message.time - time) # ensure message time is int
        time = time_


def to_mido_meta_track(music: "MusicRender") -> MidiTrack:
    """Return a mido MidiTrack containing metadata of a Music object.

    Parameters
    ----------
    music : :class:`MusicRender` object
        Music object to convert.

    Returns
    -------
    :class:`mido.MidiTrack` object
        Converted mido MidiTrack object.

    """
    
    # create a track to store the metadata
    meta_track = MidiTrack()
    all_notes = sum((track.notes for track in music.tracks), []) # all notes
    max_note_time = max((note.time for note in all_notes))

    # song title
    if music.metadata.title is not None:
        meta_track.append(MetaMessage(type = "track_name", name = music.metadata.title))

    # tempos and time signatures
    metrical_time = (not music.absolute_time)
    if metrical_time:
        combined_temporal_features = list(filter(lambda temporal_feature: temporal_feature.time <= max_note_time, music.tempos + music.time_signatures))
        combined_temporal_features = sorted(combined_temporal_features, key = lambda temporal_feature: temporal_feature.time) # get sorted list of tempos and time signatures
        # current_time_signature = music.time_signatures[0] if len(music.time_signatures) > 0 else TimeSignature(time = 0) # instantiate current_time_signature
        tempo_times = [] # keep track of tempo event times for tempo spanners later
        tempo_changes = [] # keep track of tempo changes to deal with tempo spanners later
        for temporal_feature in combined_temporal_features:
            if isinstance(temporal_feature, Tempo) and temporal_feature.qpm > 0: # if tempo feature
                current_tempo = bpm2tempo(bpm = temporal_feature.qpm)
                meta_track.append(MetaMessage(type = "set_tempo", time = temporal_feature.time, tempo = current_tempo))
                tempo_times.append(temporal_feature.time)
                tempo_changes.append(current_tempo)
            elif isinstance(temporal_feature, TimeSignature): # if time signature
                meta_track.append(MetaMessage(type = "time_signature", time = temporal_feature.time, numerator = temporal_feature.numerator, denominator = temporal_feature.denominator))
                # current_time_signature = temporal_feature # update current_time_signature
    else:
        meta_track.append(MetaMessage(type = "set_tempo", time = 0, tempo = DEFAULT_TEMPO))

    # key signatures
    for key_signature in filter(lambda key_signature: key_signature.time <= max_note_time, music.key_signatures):
        if (key_signature.root is not None) and (key_signature.mode in ("major", "minor")):
            meta_track.append(MetaMessage(type = "key_signature", time = key_signature.time, key = PITCH_NAMES[key_signature.root] + ("m" if key_signature.mode == "minor" else ""))) 
        elif key_signature.fifths is not None:
            meta_track.append(MetaMessage(type = "key_signature", time = key_signature.time, key = PITCH_NAMES[CIRCLE_OF_FIFTHS[8 + key_signature.fifths][0]]))       

    # lyrics
    for lyric in filter(lambda lyric: lyric.time <= max_note_time, music.lyrics):
        meta_track.append(MetaMessage(type = "lyrics", time = lyric.time, text = lyric.lyric))

    # system and staff level annotations
    current_tempo_index = 0
    for annotation in (music.annotations + sum((track.annotations for track in music.tracks), [])):

        # skip annotations out of the relevant time scope
        if annotation.time <= max_note_time:
            continue

        # ensure that values are valid
        annotation_type = annotation.annotation.__class__.__name__
        if hasattr(annotation.annotation, "subtype"): # make sure subtype field is not none
            if annotation.annotation.subtype is None:
                continue
            else:
                annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
        
        # update current_tempo_index if necessary
        if metrical_time:
            if current_tempo_index < (len(tempo_times) - 1): # avoid index error later on at last element in tempo_times
                if tempo_times[current_tempo_index + 1] <= annotation.time: # update current_tempo_index if necessary
                    current_tempo_index += 1 # increment
        
        # Text and TextSpanner
        if annotation_type in ("Text", "TextSpanner"):
            meta_track.append(MetaMessage(type = "text", time = annotation.time, text = annotation.annotation.text))
        
        # RehearsalMark
        elif annotation_type == "RehearsalMark":
            meta_track.append(MetaMessage(type = "marker", time = annotation.time, text = annotation.annotation.text))
        
        # if absolute time, temporal features have already been accounted for
        elif metrical_time:
            
            # Fermata and fermatas stored inside of Articulation
            if (annotation_type == "Fermata") or (annotation_type == "Articulation"): # only apply when metrical time in use
                if (annotation_type == "Articulation") and ("fermata" not in annotation.annotation.subtype): # looking for fermatas hidden as articulations
                    continue # if not a fermata-articulation, skip
                longest_note_duration_at_current_time = max([note.duration for note in all_notes if note.time == annotation.time] + [0]) # go through notes and find longest duration note at the time of the fermata
                if longest_note_duration_at_current_time > 0:
                    meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time, tempo = tempo_changes[current_tempo_index] * FERMATA_TEMPO_SLOWDOWN)) # start of fermata
                    meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time + longest_note_duration_at_current_time, tempo = tempo_changes[current_tempo_index])) # end of fermata
                del longest_note_duration_at_current_time
            
            # TempoSpanner
            elif annotation_type == "TempoSpanner": # only apply when metrical time in use
                if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("lent", "rall", "rit", "smorz", "sost", "allarg"))): # slow-downs; lentando, rallentando, ritardando, smorzando, sostenuto, allargando
                    tempo_change_factor_fn = lambda t: t
                elif any((annotation.annotation.subtype.startswith(prefix) for prefix in ("accel", "leg"))): # speed-ups; accelerando, leggiero
                    tempo_change_factor_fn = lambda t: 1 / t
                else: # unknown TempoSpanner subtype
                    tempo_change_factor_fn = lambda t: 1
                for time, tempo_change_factor_magnitude in zip(range(annotation.time, annotation.time + annotation.annotation.duration, int(annotation.annotation.duration / N_TEMPO_SPANNER_SUBDIVISIONS)), range(1, 1 + N_TEMPO_SPANNER_SUBDIVISIONS)):                    
                    meta_track.append(MetaMessage(type = "set_tempo", time = time, tempo = int(tempo_changes[current_tempo_index] * tempo_change_factor_fn(t = tempo_change_factor_magnitude)))) # add tempo change
                end_tempo_spanner_tempo_index = current_tempo_index
                if (current_tempo_index < (len(tempo_changes) - 1)) and ((annotation.time + annotation.annotation.duration) > tempo_times[current_tempo_index + 1]): # check if the tempo changed during the tempo spanner
                    end_tempo_spanner_tempo_index += 1 # if the tempo changed during the tempo spanner
                meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time + annotation.annotation.duration, tempo = tempo_changes[end_tempo_spanner_tempo_index])) # reset tempo
                del time, tempo_change_factor_magnitude, tempo_change_factor_fn, end_tempo_spanner_tempo_index
    
    # clear up memory
    del current_tempo_index, all_notes

    # end of track message
    meta_track.append(MetaMessage(type = "end_of_track"))

    # convert to delta time
    to_delta_time(midi_track = meta_track, ticks_per_beat = music.resolution, absolute_time = music.absolute_time)

    return meta_track


EXPRESSIVE_FEATURE_PRIORITIES = ("Dynamic", "SlurSpanner", "HairPinSpanner", "TempoSpanner", "Articulation", "PedalSpanner")
def sort_expressive_features_key(annotation: Annotation) -> int:
    """When given an annotation, associate the annotation with a certain number so that a list of annotations can be sorted in a desired order."""
    if annotation.annotation.__class__.__name__ in EXPRESSIVE_FEATURE_PRIORITIES:
        return EXPRESSIVE_FEATURE_PRIORITIES.index(annotation.annotation.__class__.__name__)
    else:
        return len(EXPRESSIVE_FEATURE_PRIORITIES)


def get_wiggle_func(articulation_subtype: str, amplitude: float = FRACTION_TO_WIGGLE * MAX_PITCHWHEEL, resolution: float = RESOLUTION) -> Callable:
    """Return the function that given a time, returns the amount of pitchbend for wiggle functions (vibrato or sawtooth)."""
    period = resolution / 3
    if ("fast" in articulation_subtype) or ("wide" not in articulation_subtype):
        period /= 2
    if "sawtooth" in articulation_subtype: # sawtooth
        wiggle_func = lambda time: int(amplitude * ((time % period) / period))
    else: # vibrato is default
        wiggle_func = lambda time: int(amplitude * sin(time * (2 * pi / period)))
    return wiggle_func


def get_expressive_features_per_note(note_times: list, all_annotations: list) -> dict:
    """Return a dictionary where the keys are the set of note times, and the values are the expressive features present at that note time."""

    # create expressive features
    note_time_indicies = {note_time: i for i, note_time in enumerate(note_times)}
    expressive_features: Dict[int, List[Annotation]] = dict(zip(note_times, ([] for _ in range(len(note_times))))) # dictionary where keys are time and values are expressive feature annotation objects
    for annotation in sorted(all_annotations, key = lambda annotation: annotation.time): # sort staff and system level annotations
        if hasattr(annotation.annotation, "subtype"):
            if annotation.annotation.subtype is None:
                continue
            else:
                annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
        annotation_falls_on_note = (annotation.time in expressive_features.keys())
        if annotation_falls_on_note: # if this annotation falls on a note, add that annotation to expressive features (which we care about)
            expressive_features[annotation.time].append(annotation) # add annotation
        if hasattr(annotation.annotation, "duration"): # deal with duration
            if (annotation.annotation.__class__.__name__ == "Dynamic") and (annotation.annotation.subtype not in DYNAMIC_DYNAMICS): # for sudden dynamic hikes do not fill with duration
                continue
            if annotation_falls_on_note: # get index of the note after the current time when annotation falls on a note
                current_note_time_index = note_time_indicies[annotation.time] + 1
            else: # get index of the note after the current time when annotation does not fall on a note
                current_note_time_index = len(note_times) # default value
                for note_time in note_times:
                    if note_time >= annotation.time: # first note time after the annotation time
                        current_note_time_index = note_time_indicies[note_time]
                        break
            while current_note_time_index < len(note_times):
                if note_times[current_note_time_index] >= (annotation.time + annotation.annotation.duration): # when the annotation does not have any more effect on the notes being played
                    break # break out of while loop
                expressive_features[note_times[current_note_time_index]].append(annotation)
                current_note_time_index += 1 # increment
            del current_note_time_index
    
    # make sure all note times in expressive features have a dynamic at index 0
    for i, note_time in enumerate(note_times):
        if not any((annotation.annotation.__class__.__name__ == "Dynamic" for annotation in expressive_features[note_time])): # if there is no dynamic
            j = i - 1 # start index finder at the index before current
            while not any((annotation.annotation.subtype in DYNAMIC_DYNAMICS for annotation in expressive_features[note_times[j]] if annotation.annotation.__class__.__name__ == "Dynamic")) and (j > -1): # look for previous note times with a dynamic
                j -= 1 # decrement j
            if (j == -1): # no notes with dynamics before this one, result to default values
                dynamic_annotation = Dynamic(subtype = DEFAULT_DYNAMIC, velocity = DYNAMIC_VELOCITY_MAP[DEFAULT_DYNAMIC])
            else: # we found a dynamic before this note time
                dynamic_annotation = expressive_features[note_times[j]][0].annotation # we can assume the dynamic marking is at index 0 because of our sorting
            expressive_features[note_time].insert(0, Annotation(time = note_time, annotation = dynamic_annotation)) # insert dynamic at position 0
        expressive_features[note_time] = sorted(expressive_features[note_time], key = sort_expressive_features_key) # sort expressive features in desired order
        if expressive_features[note_time][0].annotation.velocity is None: # make sure dynamic velocity is not none
            expressive_features[note_time][0].annotation.velocity = DYNAMIC_VELOCITY_MAP[DEFAULT_DYNAMIC] # set to default velocity

    return expressive_features


def to_mido_track(track: Track, music: "MusicRender", channel: int = None, use_note_off_message: bool = False) -> MidiTrack:
    """Return a Track object as a mido MidiTrack object.

    Parameters
    ----------
    track : :class:`Track` object
        Track object to convert.
    music : :class:`MusicRender` object
        Music object that `track` belongs to.
    channel : int, optional
        Channel number. Defaults to 10 for drums and 0 for other instruments.
    use_note_off_message : bool, default: False
        Whether to use note-off messages. If False, note-on messages with zero velocity are used instead. The advantage to using note-on messages at zero velocity is that it can avoid sending additional status bytes when Running Status is employed.

    Returns
    -------
    :class:`mido.MidiTrack` object
        Converted mido MidiTrack object.

    """

    # determine channel
    if channel is None:
        channel = DRUM_CHANNEL if track.is_drum else 0

    # create a new .mid track
    midi_track = MidiTrack()

    # track name messages
    if track.name is not None:
        midi_track.append(MetaMessage(type = "track_name", name = track.name))

    # program change messages
    midi_track.append(Message(type = "program_change", program = track.program, channel = channel))

    # deal with swing
    is_swung = False
    swing_features = list(filter(lambda annotation: (annotation.annotation.__class__.__name__ == "Text") and (annotation.annotation.is_system) and (annotation.annotation.style == "tempo"), music.annotations))
    swing_feature_index = -1
    for note in track.notes:
        if (swing_feature_index < (len(swing_features) - 1)) and (note.time >= swing_features[swing_feature_index + 1].time): # determine if we must update is_swung
            swing_feature_index += 1
            is_swung = (swing_features[swing_feature_index].annotation.text == "Swing") # update is_swung
        if is_swung and (((note.time / music.resolution) % 1) == 0.5): # add swing if we are swinging and the note falls on an off-beat eighth note
            note.time = int(max(0, note.time + (music.resolution * (SWING_PROPORTION - 0.5))))
    del is_swung, swing_features, swing_feature_index

    # deal with expressive features
    note_times = sorted(list({note.time for note in track.notes})) # times of notes, sorted ascending, removing duplicates
    note_time_indicies = {note_time: i for i, note_time in enumerate(note_times)}
    expressive_features = get_expressive_features_per_note(note_times = note_times, all_annotations = track.annotations + music.annotations) # dictionary where keys are time and values are expressive feature annotation objects

    # note on and note off messages
    for note in track.notes:
        note.velocity = expressive_features[note.time][0].annotation.velocity # the first index is always the dynamic
        for annotation in expressive_features[note.time][1:]: # skip the first index, since we just dealt with it
            
            # ensure that values are valid
            annotation_type = annotation.annotation.__class__.__name__
            if hasattr(annotation.annotation, "subtype"): # make sure subtype field is not none
                if annotation.annotation.subtype is None:
                    continue
            
            # HairPinSpanner and TempoSpanner; changes in velocity
            if (annotation_type in ("HairPinSpanner", "TempoSpanner")) and music.infer_velocity: # some TempoSpanners involve a velocity change, so that is included here as well
                if annotation.group is None: # since we aren't storing anything there anyways
                    end_velocity = note.velocity # default is no change
                    if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("allarg", "cr"))): # increase-volume; allargando, crescendo
                        end_velocity *= VELOCITY_INCREASE_FACTOR
                    elif any((annotation.annotation.subtype.startswith(prefix) for prefix in ("smorz", "dim", "decr"))): # decrease-volume; smorzando, diminuendo, decrescendo
                        end_velocity /= VELOCITY_INCREASE_FACTOR
                    denominator = (annotation.time + annotation.annotation.duration) - note.time
                    annotation.group = lambda time: (((((end_velocity - note.velocity) / denominator) * (time - note.time)) + note.velocity) if (denominator != 0) else end_velocity) # we will use group to store a lambda function to calculate velocity
                note.velocity = annotation.group(time = note.time) # previously used +=
            
            # SlurSpanner
            elif annotation_type == "SlurSpanner":
                if (note_time_indicies[note.time] == (len(note_times) - 1)) or not any(slur_spanner is annotation for slur_spanner in filter(lambda annotation_: annotation_.annotation.__class__.__name__ == "SlurSpanner", expressive_features[note_times[note_time_indicies[note.time] + 1]])):
                    continue # if the note is the last note in the slur, we don't want to slur it
                current_note_time_index = note_time_indicies[note.time]
                if current_note_time_index < len(note_times) - 1: # elsewise, there is no next note to slur to
                    note.duration = max(note_times[current_note_time_index + 1] - note_times[current_note_time_index], note.duration) # we don't want to make the note shorter
                del current_note_time_index
            
            # PedalSpanner
            elif annotation_type == "PedalSpanner":
                note.duration *= PEDAL_DURATION_CHANGE_FACTOR
            
            # Articulation
            elif annotation_type == "Articulation":
                if any((keyword in annotation.annotation.subtype for keyword in ("staccato", "staccatissimo", "spiccato", "pizzicato", "plucked", "marcato", "sforzato"))): # shortens note length
                    note.duration /= STACCATO_DURATION_CHANGE_FACTOR
                if any((keyword in annotation.annotation.subtype for keyword in ("marcato", "sforzato", "accent"))): # increases velocity
                    note.velocity *= max((ACCENT_VELOCITY_INCREASE_FACTOR * (0.8 if "soft" in annotation.annotation.subtype else 1)), 1)
                if ("spiccato" in annotation.annotation.subtype): # decreases velocity
                    note.velocity /= ACCENT_VELOCITY_INCREASE_FACTOR
                if "wiggle" in annotation.annotation.subtype: # vibrato and sawtooth
                    times = linspace(start = note.time, stop = note.time + note.duration, num = 8, endpoint = False)
                    pitch_func = get_wiggle_func(articulation_subtype = annotation.annotation.subtype, resolution = music.resolution)
                    for time in times:
                        midi_track.append(Message(type = "pitchwheel", time = time, pitch = pitch_func(time = time - times[0]), channel = channel))
                    midi_track.append(Message(type = "pitchwheel", time = note.time + note.duration, pitch = 0, channel = channel)) # reset pitch
                if "tenuto" in annotation.annotation.subtype:
                    pass # the duration is full duration
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
            # elif annotation_type == "TechAnnotation":
            #     pass # currently no implementation since we so rarely encounter these
        
        # adjust note slightly if grace note
        if note.is_grace: # move the note slightly ahead if it is a grace note
            note.time = max(0, note.time - (music.resolution * GRACE_NOTE_FORWARD_SHIFT_CONSTANT)) # no grace notes on the first note, to avoid negative times
        
        # ensure note time is an integer
        note.time = int(note.time) # ensure note time is an integer
       
        # add note to track
        midi_track.extend(to_mido_note_on_note_off(note = note, channel = channel, use_note_off_message = use_note_off_message))

    # end of track message
    midi_track.append(MetaMessage(type = "end_of_track"))

    # convert to delta time
    to_delta_time(midi_track = midi_track, ticks_per_beat = music.resolution, absolute_time = music.absolute_time)

    return midi_track


def write_midi(path: str, music: "MusicRender", use_note_off_message: bool = False):
    """Write a Music object to a .mid file using mido as backend.

    Parameters
    ----------
    path : str
        Path to write the .mid file.
    music : :class:`MusicRender` object
        Music object to write.
    use_note_off_message : bool, default: False
        Whether to use note-off messages. If False, note-on messages with zero velocity are used instead. The advantage to using note-on messages at zero velocity is that it can avoid sending additional status bytes when Running Status is employed.

    """

    # ensure we are operating on a copy of music
    music = deepcopy(music)

    # redo resolution if it's too low
    if music.resolution < OPTIMAL_RESOLUTION:
        music.reset_resolution(new_resolution = OPTIMAL_RESOLUTION)

    # create a .mid file object
    midi = MidiFile(type = 1, ticks_per_beat = music.resolution, charset = "utf8")

    # append meta track
    midi.tracks.append(to_mido_meta_track(music = music))

    # iterate over music tracks
    for i, track in enumerate(music.tracks):
        # NOTE: Many softwares use the same instrument for messages of the same channel in different tracks. Thus, we want to assign a unique channel number for each track. .mid has 15 channels for instruments other than drums, so we increment the channel number for each track (skipping the drum channel) and go back to 0 once we run out of channels.
        
        # assign channel number
        if track.is_drum:
            channel = DRUM_CHANNEL # mido numbers channels 0 to 15 instead of 1 to 16
        else:
            channel = i % 15 # .mid has 15 channels for instruments other than drums
            channel += int(channel >= DRUM_CHANNEL) # avoid drum channel by adding one if the channel is greater than 8

        # add track
        midi.tracks.append(to_mido_track(track = track, music = music, channel = channel, use_note_off_message = use_note_off_message))

    midi.save(filename = path)

##################################################


# WRITE AUDIO
##################################################

def write_audio(path: str, music: "MusicRender", audio_format: str = "auto", soundfont_path: str = None, rate: int = 44100, gain: float = 1, options: str = None):
    """Write a Music object to an audio file.

    Supported formats include WAV, AIFF, FLAC and OGA.

    Parameters
    ----------
    path : str
        Path to write the audio file.
    music : :class:`MusicRender`
        Music object to write.
    audio_format : str, default: 'auto'
        File format to write. Defaults to infer from the extension.
    soundfont_path : str, optional
        Path to the soundfount file. Defaults to the path to the downloaded MuseScore General soundfont.
    rate : int, default: 44100
        Sample rate (in samples per sec).
    gain : float, default: 1
        Master gain (`-g` option) for Fluidsynth. The value must be in between 0 and 10, excluding. This can be used to avoid clipping.
    options : str, optional
        Additional options to be passed to the FluidSynth call. The full command called is: `fluidsynth -ni -F {path} -T {audio_format} -r {rate} -g {gain} -i {soundfont_path} {options} {midi_path}`.

    """

    # check soundfont
    if soundfont_path is None:
        soundfont_path = f"{expanduser('~')}/.muspy/musescore-general/MuseScore_General.sf3" # musescore soundfont path
    if not exists(soundfont_path):
        raise RuntimeError("Soundfont not found. Please download it by `muspy.download_musescore_soundfont()`.")

    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:

        # ensure we are operating on a copy of music
        music = deepcopy(music)

        # write the MusicRender object to a temporary .mid file
        # midi_path = f"{'.'.join(path.split('.')[:-1])}.mid" # for debugging
        midi_path = f"{temp_dir}/temp.mid"
        write_midi(path = midi_path, music = music)

        # synthesize the .mid file using fluidsynth
        option_list = options.split(" ") if options is not None else []
        subprocess.run(args = ["fluidsynth", "-ni", "-F", path, "-T", audio_format, "-r", str(rate), "-g", str(gain), soundfont_path] + option_list + [midi_path], check = True, stdout = subprocess.DEVNULL)

##################################################


# WRITE MUSICXML
##################################################

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
DRUMSET_INSTRUMENTS = [
    ("Acoustic Bass Drum", 36),
    ("Bass Drum 1", 37),
    ("Side Stick", 38),
    ("Acoustic Snare", 39),
    ("Hand Clap", 40),
    ("Electric Snare", 41),
    ("Low Floor Tom", 42),
    ("Closed Hi Hat", 43),
    ("High Floor Tom", 44),
    ("Pedal Hi-Hat", 45),
    ("Low Tom", 46),
    ("Open Hi-Hat", 47),
    ("Low-Mid Tom", 48),
    ("Hi Mid Tom", 49),
    ("Crash Cymbal 1", 50),
    ("High Tom", 51),
    ("Ride Cymbal 1", 52),
    ("Chinese Cymbal", 53),
    ("Ride Bell", 54),
    ("Tambourine", 55),
    ("Splash Cymbal", 56),
    ("Cowbell", 57),
    ("Crash Cymbal 2", 58),
    ("Vibraslap", 59),
    ("Ride Cymbal 2", 60),
    ("Hi Bongo", 61),
    ("Low Bongo", 62),
    ("Mute Hi Conga", 63),
    ("Open Hi Conga", 64),
    ("Low Conga", 65),
    ("High Timbale", 66),
    ("Low Timbale", 67),
    ("High Agogo", 68),
    ("Low Agogo", 69),
    ("Cabasa", 70),
    ("Maracas", 71),
    ("Short Whistle", 72),
    ("Long Whistle", 73),
    ("Short Guiro", 74),
    ("Long Guiro", 75),
    ("Claves", 76),
    ("Hi Wood Block", 77),
    ("Low Wood Block", 78),
    ("Mute Cuica", 79),
    ("Open Cuica", 80),
    ("Mute Triangle", 81),
    ("Open Triangle", 82),
]


def get_transpose_element(transposition: Interval = None) -> ET.Element:
    """Return the XML Element object that represents the transpose information for the given transposition."""

    # get information about the transposition (chromatic, diatonic, octave)
    genericSteps = transposition.diatonic.generic.directed if transposition is not None else 1
    musicxmlOctaveShift, musicxmlDiatonic = divmod(abs(genericSteps) - 1, 7)
    musicxmlChromatic = abs(transposition.chromatic.semitones) % 12 if transposition is not None else 0
    if genericSteps < 0:
        musicxmlDiatonic *= -1
        musicxmlOctaveShift *= -1
        musicxmlChromatic *= -1

    # create MusicXML elements
    mxTranspose = ET.Element("transpose")
    mxDiatonic = ET.SubElement(mxTranspose, "diatonic") # diatonic shift
    mxDiatonic.text = str(musicxmlDiatonic)
    mxChromatic = ET.SubElement(mxTranspose, "chromatic") # chromatic shift
    mxChromatic.text = str(musicxmlChromatic)
    if musicxmlOctaveShift != 0: # if there is an octave shift
        mxOctaveChange = ET.SubElement(mxTranspose, "octave-change")
        mxOctaveChange.text = str(musicxmlOctaveShift)

    # return element
    return mxTranspose


def get_instrument_elements(instrument_name: str, percussion_map_pitch: int, part_id: str) -> Tuple[ET.Element, ET.Element]:
    """Return two XML elements, representing the score-instrument and midi-instrument elements for a given instrument."""

    # get the instrument id
    instrument_id = f"{part_id}-T{percussion_map_pitch}"

    # create score instrument
    score_instrument = ET.Element("score-instrument", id = instrument_id)
    instrument_name_element = ET.SubElement(score_instrument, "instrument-name")
    instrument_name_element.text = instrument_name

    # create midi instrument
    midi_instrument = ET.Element("midi-instrument", id = instrument_id)
    midi_channel = ET.SubElement(midi_instrument, "midi-channel")
    midi_channel.text = str(DRUM_CHANNEL) # set to the drum channel
    midi_program = ET.SubElement(midi_instrument, "midi-program")
    midi_program.text = "1" # default
    midi_unpitched = ET.SubElement(midi_instrument, "midi-unpitched")
    midi_unpitched.text = str(percussion_map_pitch) # midi pitch that maps to this instrument
    midi_volume = ET.SubElement(midi_instrument, "volume")
    midi_volume.text = "78.7402" # MuseScore default
    midi_pan = ET.SubElement(midi_instrument, "pan")
    midi_pan.text = "0" # no pan of the instrument

    # return the score and midi instrument elements
    return score_instrument, midi_instrument


def get_unpitched_and_instrument_elements(note: ET.Element, part_id: str) -> Tuple[ET.Element, ET.Element]:
    """
    Return two XML elements, the unpitched element of a pitched drum note,
    and the instrument element associated with the new unpitched drum note.
    """

    # get info about pitch
    pitch_element = note.find(path = "pitch")
    pitch_step = pitch_element.find(path = "step").text
    pitch_octave = int(pitch_element.find(path = "octave").text)
    pitch_alter = pitch_element.find(path = "alter")
    pitch_alter = int(0 if pitch_alter is None else pitch_alter.text)

    # reconstruct midi pitch from the pitch information gathered in previous step
    percussion_map_pitch = PITCH_NAMES.index(pitch_step) + (12 * (pitch_octave + 1)) + pitch_alter + 1

    # remove any pitch-related elements from note
    note.remove(pitch_element)
    accidental_element = note.find(path = "accidental")
    if accidental_element is not None:
        note.remove(accidental_element)

    # create unpitched XML element
    unpitched = ET.Element("unpitched")
    display_step = ET.SubElement(unpitched, "display-step")
    display_step.text = pitch_step
    display_octave = ET.SubElement(unpitched, "display-octave")
    display_octave.text = str(pitch_octave)

    # create note instrument element
    note_instrument = ET.Element("instrument", id = f"{part_id}-T{percussion_map_pitch}")

    # return elements
    return unpitched, note_instrument


def write_musicxml(path: str, music: "MusicRender", compressed: bool = None):
    """Write a Music object to a MusicXML file.

    Parameters
    ----------
    path : str
        Path to write the MusicXML file.
    music : :class:`MusicRender`
        Music object to write.
    compressed : bool, optional
        Whether to write to a compressed MusicXML file. If None, infer
        from the extension of the filename ('.xml' and '.musicxml' for
        an uncompressed file, '.mxl' for a compressed file).

    """

    # make sure music is not in absolute time
    if music.absolute_time:
        raise RuntimeError("Cannot write to MusicXML when `music` is in absolute time (`music.absolute_time` == True).")

    # ensure we are operating on a copy of music
    music = deepcopy(music)

    # create a new score
    score = Score()

    # helper functions
    get_offset = lambda time: time / music.resolution
    def get_duration(duration: int) -> Duration:
        """
        Helper function to get the duration of something, returning a music21 Duration object.
        Make sure the duration is valid so no bugs are thrown.
        """
        m21_duration = Duration(quarterLength = Fraction(duration, music.resolution))
        if any((tup.durationNormal.quarterLength <= (1 / (2 ** 9)) for tup in m21_duration.tuplets)): # check for invalid durations
            return Duration(quarterLength = 0)
        else:
            return m21_duration
    get_time_from_offset = lambda offset: offset * music.resolution

    # metadata
    if music.metadata is not None:
        meta = M21MetaData()
        if music.metadata.title: # title is usually stored in movement-title. See https://www.musicxml.com/tutorial/file-structure/score-header-entity/
            meta.movementName = music.metadata.title
        if music.metadata.copyright:
            meta.copyright = Copyright(data = music.metadata.copyright)
        for creator in music.metadata.creators:
            meta.addContributor(Contributor(name = creator))
        score.metadata = meta

    # tracks
    transpositions = [None] * len(music.tracks)
    for i, track in enumerate(music.tracks):
        
        # create a new part
        part = Part()
        part.partName = track.name
        part_id = f"P{i + 1}"

        # set instrument
        if track.is_drum:
            part.clef = PercussionClef()
            channel = DRUM_CHANNEL
        else:
            channel = i % 15 # .mid has 15 channels for instruments other than drums
            channel += int(channel >= DRUM_CHANNEL) # avoid drum channel by adding one if the channel is greater than 8
        instrument = instrumentFromMidiProgram(number = track.program)
        instrument.partId = part_id
        instrument.partName = track.name
        instrument.instrumentId = f"{part_id}-T1"
        instrument.instrumentName = track.name
        instrument.midiChannel = channel
        part.append(instrument)

        # amount to adjust tone
        part.atSoundingPitch = False
        transpose_chromatic = instrument.transposition.chromatic.semitones if instrument.transposition is not None else 0
        transpositions[i] = instrument.transposition
        def get_note_str_and_octave(pitch: int, pitch_str: str) -> Tuple[str, int]:
            """Helper function for extracting the note's string and octave for music21."""
            if pitch_str is None:
                pitch_str = PITCH_NAMES[pitch % len(PITCH_NAMES)]
            n_accidentals = pitch_str[1:].count("#") - pitch_str[1:].count("b")
            note_str = PITCH_NAMES[(PITCH_NAMES.index(pitch_str[0]) + n_accidentals - transpose_chromatic) % len(PITCH_NAMES)].replace("b", "-")
            note_octave = get_octave_from_pitch(pitch = pitch - transpose_chromatic) # adjust pitch for tuning of instrument
            return note_str, note_octave

        # add tempos
        for tempo in music.tempos: # loop through tempos
            if tempo.text is None or tempo.text.startswith("<sym>"): # bpm is explicitly supplied in text or not supplied at all
                m21_tempo = M21Tempo.MetronomeMark(number = tempo.qpm) # create tempo marking with bpm
            else: # bpm is implied by tempo name
                m21_tempo = M21Tempo.MetronomeMark(text = tempo.text, numberSounding = tempo.qpm) # create tempo marking with text
            m21_tempo.offset = get_offset(time = tempo.time) # define offset
            part.insert(offsetOrItemOrList = m21_tempo.offset, itemOrNone = m21_tempo)

        # add time signatures
        for time_signature in music.time_signatures: # loop through time signatures
            if (time_signature.numerator is None) or (time_signature.denominator is None):
                continue
            m21_time_signature = M21TimeSignature(value = f"{time_signature.numerator}/{time_signature.denominator}") # instantiate time signature object
            m21_time_signature.offset = get_offset(time = time_signature.time) # define offset
            part.insert(offsetOrItemOrList = m21_time_signature.offset, itemOrNone = m21_time_signature)

        # add key signatures
        for key_signature in music.key_signatures: # loop through key signatures
            m21_key_signature = M21KeySignature(sharps = (key_signature.fifths - convert_chromatic_to_circle_of_fifths_steps(transpose_chromatic = transpose_chromatic)) if key_signature.fifths is not None else 0) # create key object
            m21_key_signature = m21_key_signature.asKey(mode = key_signature.mode.lower() if key_signature.mode is not None else "major")
            m21_key_signature.offset = get_offset(time = key_signature.time) # define offset
            part.insert(offsetOrItemOrList = m21_key_signature.offset, itemOrNone = m21_key_signature)

        # add notes to part (and grace notes, lyrics, noteheads, and articulations)
        lyrics: Dict[int, str] = {lyric.time: lyric.lyric for lyric in track.lyrics} # put lyrics into dictionary (where keys are the time)
        noteheads: Dict[int, str] = {annotation.time: annotation.annotation.subtype.lower() for annotation in filter(lambda annotation: annotation.annotation.__class__.__name__ == "Notehead", track.annotations)}
        articulations: Dict[int, List[Annotation]] = {articulation_time: [clean_up_subtype(subtype = annotation.annotation.subtype) for annotation in articulations_at_time] for articulation_time, articulations_at_time in groupby(
            iterable = sorted(filter(lambda annotation: any(annotation.annotation.__class__.__name__ == keyword for keyword in ("Articulation", "Symbol")), track.annotations), key = lambda annotation: annotation.time), # need to sort because itertools creates a new group when a new key appears
            key = lambda annotation: annotation.time)}
        chords: Dict[Tuple[int, int, bool], list] = dict()
        
        # loop through notes
        for note in track.notes:

            # create M21 Note object
            note_str, note_octave = get_note_str_and_octave(pitch = note.pitch, pitch_str = note.pitch_str)
            m21_note = M21Note(name = note_str, octave = note_octave) # create note object
            
            # check if grace note
            if note.is_grace: # check if grace note
                m21_note.type = "eighth" # so it looks like a normal grace note with a flag
                m21_note = m21_note.getGrace() # convert to grace note
            # if the note is a normal note
            else:
                m21_note.duration = get_duration(duration = note.duration)
                if m21_note.duration.type == "zero": # ignore notes with invalid durations
                    continue
                # check if there is a lyric at that note
                if note.time in lyrics.keys():
                    m21_note.lyric = lyrics[note.time] # add lyric
            
            # check if there is a notehead at that note
            if note.time in noteheads.keys() and noteheads[note.time] in noteheadTypeNames: # check if notehead is a valid notehead for music21
                m21_note.notehead = noteheads[note.time] # add notehead
            
            # check if there is an articulation(s) at that note
            if note.time in articulations.keys():
                for articulation in articulations[note.time]:
                    if articulation is None:
                        continue
                    if "accent" in articulation:
                        m21_note.articulations.append(M21Articulation.Accent())
                    if "staccato" in articulation:
                        m21_note.articulations.append(M21Articulation.Staccato())
                    if "staccatissimo" in articulation:
                        m21_note.articulations.append(M21Articulation.Staccatissimo())
                    if "tenuto" in articulation:
                        m21_note.articulations.append(M21Articulation.Tenuto())
                    if "pizzicato" in articulation:
                        if "snap" in articulation:
                            m21_note.articulations.append(M21Articulation.SnapPizzicato())
                        elif "nail" in articulation:
                            m21_note.articulations.append(M21Articulation.NailPizzicato())
                        else: # normal pizzicato is default
                            m21_note.articulations.append(M21Articulation.Pizzicato())
                    if "spiccato" in articulation:
                        m21_note.articulations.append(M21Articulation.Spiccato())
                    if "bow" in articulation:
                        if "up" in articulation:
                            m21_note.articulations.append(M21Articulation.UpBow())
                        else: # down is default
                            m21_note.articulations.append(M21Articulation.DownBow())
                    if any((keyword in articulation for keyword in ("mute", "close"))): # brass mute
                        m21_note.articulations.append(M21Articulation.BrassIndication(name = "muted"))
                    if any((keyword in articulation for keyword in ("open", "ouvert"))): # open
                        if "string" in articulation:
                            m21_note.articulations.append(M21Articulation.OpenString())
                        else: # defaults to brass open mute
                            m21_note.articulations.append(M21Articulation.BrassIndication(name = "open"))
                    if "doit" in articulation:
                        m21_note.articulations.append(M21Articulation.Doit())
                    if "fall" in articulation:
                        m21_note.articulations.append(M21Articulation.Falloff())
                    if "plop" in articulation:
                        m21_note.articulations.append(M21Articulation.Plop())
                    if "scoop" in articulation:
                        m21_note.articulations.append(M21Articulation.Scoop())  
                    if "harmonic" in articulation:
                        m21_note.articulations.append(M21Articulation.Harmonic())
                    if "stopped" in articulation:
                        m21_note.articulations.append(M21Articulation.Stopped())
                    if "stress" in articulation:
                        m21_note.articulations.append(M21Articulation.Stress())
                    if "unstress" in articulation:
                        m21_note.articulations.append(M21Articulation.Unstress())
            
            # add note to chords
            chord_descriptor_tuple = (note.time, note.duration, note.is_grace)
            if chord_descriptor_tuple not in chords.keys():
                chords[chord_descriptor_tuple] = []
            chords[chord_descriptor_tuple].append(m21_note)
        
        # add chords to part
        for chord_descriptor_tuple, chord_notes in chords.items():
            m21_chord = M21Chord(notes = chord_notes)
            m21_chord.offset = get_offset(time = chord_descriptor_tuple[0])
            part.insert(offsetOrItemOrList = m21_chord.offset, itemOrNone = m21_chord)

        # clean up some memory
        del lyrics, noteheads, articulations, chords

        # add expressive features to part
        for annotation in sorted(music.annotations + track.annotations, key = lambda annotation: annotation.time):

            # clean up subtype if necessary
            if hasattr(annotation.annotation, "subtype"):
                if annotation.annotation.subtype is None:
                    continue
                else:
                    annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
            
            # some boolean flags
            annotation_type = annotation.annotation.__class__.__name__
            is_fermata = (annotation_type == "Fermata") # fermata
            if any((annotation_type == keyword for keyword in ("Articulation", "Symbol"))): # instance where fermatas are hidden as articulations
                is_fermata = "fermata" in annotation.annotation.subtype.lower()
            
            # Text, TextSpanner
            if any((annotation_type == keyword for keyword in ("Text", "TextSpanner"))):
                m21_annotation = M21Expression.TextExpression(content = annotation.annotation.text)
                if annotation_type == "TextSpanner": # text spanners
                    m21_annotation.duration = get_duration(duration = annotation.annotation.duration)
                    if m21_annotation.duration.type == "zero": # ignore notes with invalid durations
                        continue
            
            # RehearsalMark
            elif annotation_type == "RehearsalMark":
                m21_annotation = M21Expression.RehearsalMark(content = annotation.annotation.text)
            
            # Dynamic
            elif annotation_type == "Dynamic":
                m21_annotation = M21Dynamic.Dynamic(value = annotation.annotation.subtype)
            
            # Chord Symbol
            elif annotation_type == "ChordSymbol":
                try:
                    m21_annotation = M21ChordSymbol(annotation.annotation.root_str.replace("b", "-") + (annotation.annotation.name.lower() if annotation.annotation.name else ""))
                except (ValueError): # not all chords are supported, so we do our best, and ignore unknown chord symbols
                    continue

            # Fermata
            elif is_fermata:
                m21_annotation = M21Expression.Fermata()
            
            # Ornament, Articulation, Symbol, TechAnnotation
            elif any((annotation_type == keyword for keyword in ("Ornament", "Articulation", "Symbol", "TechAnnotation"))):
                if annotation_type == "TechAnnotation": # just to make things easier
                    annotation.annotation.subtype = annotation.annotation.tech_type if annotation.annotation.tech_type is not None else annotation.annotation.text # add subtype field to techannotation
                annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype for tech annotation
                if "mordent" in annotation.annotation.subtype: # mordent
                    if any((keyword in annotation.annotation.subtype for keyword in ("reverse", "invert"))):
                        m21_annotation = M21Expression.InvertedMordent()
                    else: # default is normal
                        m21_annotation = M21Expression.Mordent()
                elif "trill" in annotation.annotation.subtype: # trill
                    if any((keyword in annotation.annotation.subtype for keyword in ("reverse", "invert"))):
                        m21_annotation = M21Expression.InvertedTrill()
                    else: # default is normal
                        m21_annotation = M21Expression.Trill()
                elif "schleifer" in annotation.annotation.subtype: # schleifer
                    m21_annotation = M21Expression.Schleifer()
                elif "shake" in annotation.annotation.subtype: # shake
                    m21_annotation = M21Expression.Shake()
                elif "tremolo" in annotation.annotation.subtype: # tremolo
                    m21_annotation = M21Expression.Tremolo()
                elif "turn" in annotation.annotation.subtype: # turn
                    if "reverse" in annotation.annotation.subtype:
                        m21_annotation = M21Expression.InvertedTurn()
                    else: # default is normal turn
                        m21_annotation = M21Expression.Turn()
                else: # unknown ornament or articulation
                    continue
            
            # TrillSpanner, HairPinSpanner, SlurSpanner, GlissandoSpanner, OttavaSpanner, TrillSpanner; anything that spans multiple notes
            elif any((annotation_type == keyword for keyword in ("TrillSpanner", "HairPinSpanner", "SlurSpanner", "GlissandoSpanner", "OttavaSpanner", "TrillSpanner"))):
                spanned_notes = list(filter(lambda note: get_time_from_offset(offset = note.offset) >= annotation.time and get_time_from_offset(offset = note.offset) <= (annotation.time + annotation.annotation.duration), part.getElementsByClass(M21Note)))
                # TempoSpanner
                if annotation_type == "TempoSpanner":
                    if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("accel", "leg"))): # speed-ups; accelerando, leggiero
                        m21_annotation = M21Tempo.AccelerandoSpanner(*spanned_notes)
                    else: # slow-downs; lentando, rallentando, ritardando, smorzando, sostenuto, allargando, etc.; the default
                        m21_annotation = M21Tempo.RitardandoSpanner(*spanned_notes)
                # HairPinSpanner
                elif annotation_type == "HairPinSpanner":
                    if any((keyword in "".join(annotation.annotation.subtype.split("-")) for keyword in ("dim", "decres"))): # diminuendo
                        m21_annotation = M21Dynamic.Diminuendo(*spanned_notes)
                    else: # crescendo
                        m21_annotation = M21Dynamic.Crescendo(*spanned_notes)
                # SlurSpanner
                elif annotation_type == "SlurSpanner":
                    m21_annotation = M21Spanner.Slur(*spanned_notes)
                # GlissandoSpanner
                elif annotation_type == "GlissandoSpanner":
                    m21_annotation = M21Spanner.Glissando(*spanned_notes, lineType = "wavy" if annotation.annotation.is_wavy else "solid")
                # OttavaSpanner
                elif annotation_type == "OttavaSpanner":
                    try:
                        m21_annotation = M21Spanner.Ottava(type = annotation.annotation.subtype, transposing = False)
                    except (M21Spanner.SpannerException): # some unknown type of ottava
                        continue
                # TrillSpanner
                elif annotation_type == "TrillSpanner" and len(spanned_notes) >= 2:
                    m21_annotation = M21Expression.TrillExtension(spanned_notes[0], spanned_notes[1])
                # unknown spanner
                else:
                    continue
            
            # Arpeggio
            elif annotation_type == "Arpeggio":
                if annotation.annotation.subtype == "bracket":
                    arpeggio_type = "bracket"
                elif "up" in annotation.annotation.subtype:
                    arpeggio_type = "up"
                elif "down" in annotation.annotation.subtype:
                    arpeggio_type = "down"
                else:
                    arpeggio_type = "normal"
                m21_annotation = M21Expression.ArpeggioMark(arpeggioType = arpeggio_type)
            
            # Tremolo
            elif annotation_type == "Tremolo":
                number_of_marks = sub(pattern = "[^\d]", repl = "", string = annotation.annotation.subtype)
                m21_annotation = M21Expression.Tremolo(numberOfMarks = (int(number_of_marks) // 8) if len(number_of_marks) > 0 else 3)
                del number_of_marks
            
            # TremoloBar
            # elif annotation_type == "TremoloBar":
            #     pass # to be implemented on a later date

            # ChordLine
            # elif annotation_type == "ChordLine":
            #     pass # to be implemented on a later data

            # Bend
            # elif annotation_type == "Bend":
            #     pass # to be implemented on a later date

            # PedalSpanner
            # elif annotation_type == "PedalSpanner":
            #     pass # music21 does not have pedals

            # VibratoSpanner
            # elif annotation_type == "VibratoSpanner":
            #     pass # so rare that this is not worth implementing
            
            # if the annotation is unknown, skip it
            else:
                continue
            
            # insert annotation into part
            try:
                m21_annotation.offset = get_offset(time = annotation.time)
                part.insert(offsetOrItemOrList = m21_annotation.offset, itemOrNone = deepcopy(m21_annotation)) # as to avoid the StreamException object * is already found in this Stream
            except StreamException as stream_exception:
                warn(str(stream_exception), RuntimeWarning)
        
        # append the part to score
        score.append(part)

    # write as xml temporarily, correct some of music21's mistakes
    path_temp = f"{path}.temp.xml"
    score.write(fmt = "xml", fp = path_temp)
    root = ET.parse(path_temp).getroot()

    # make sure transpositions are present in xml file
    for i, elem in enumerate(root.findall(path = "part")):
        first_measure = elem.find(path = "measure")
        if transpositions[i] is not None and first_measure.find(path = "attributes/transpose") is None:
            attributes = first_measure.find(path = "attributes")
            if attributes is None:
                attributes = ET.Element("attributes")
                first_measure.insert(0, attributes)
            attributes.append(get_transpose_element(transposition = transpositions[i]))
    
    # make sure drum midi instruments are written correctly in the part list
    drum_track_indicies = set(filter(lambda i: music.tracks[i].is_drum, range(len(music.tracks))))
    for i, elem in enumerate(root.findall(path = "part-list/score-part")):
        if i not in drum_track_indicies: # skip over non-drum tracks
            continue
        elem.remove(elem.find(path = "part-abbreviation"))
        elem.remove(elem.find(path = "score-instrument"))
        elem.remove(elem.find(path = "midi-instrument"))
        midi_instruments = []
        for instrument_name, percussion_map_pitch in DRUMSET_INSTRUMENTS:
            score_instrument, midi_instrument = get_instrument_elements(instrument_name = instrument_name, percussion_map_pitch = percussion_map_pitch, part_id = elem.get("id"))
            elem.append(score_instrument)
            midi_instruments.append(midi_instrument)
        elem.append(ET.Element("midi-device", port = "1"))
        for midi_instrument in midi_instruments:
            elem.append(midi_instrument)

    # make sure to update pitched drum notes to unpitched
    for i, elem in enumerate(root.findall(path = "part")):
        if i not in drum_track_indicies: # skip over non-drum tracks
            continue
        for note in elem.findall(path = "measure/note"):
            if note.find(path = "rest") is not None: # skip over rests
                continue
            unpitched, note_instrument = get_unpitched_and_instrument_elements(note = note, part_id = elem.get("id"))
            unpitched_position = int(note.find(path = "chord") is not None)
            note.insert(unpitched_position, unpitched)
            note.insert(unpitched_position + 2, note_instrument)
    
    # infer compression
    if compressed is None:
        if path.endswith((".xml", ".musicxml")):
            compressed = False
        elif path.endswith(".mxl"):
            compressed = True
        else:
            raise ValueError("Cannot infer file type from the extension.")
    
    # compress the file (or not)
    tree = ET.ElementTree(root)
    ET.indent(tree, space = "\t", level = 0)
    tree.write(path_temp)
    if compressed:
        compressXML(filename = path_temp, deleteOriginal = True)
        rename(src = f"{path}.temp.mxl", dst = path)
    else: # don't compress
        rename(src = path_temp, dst = path)

##################################################


# MAIN METHOD FOR TESTING
##################################################

if __name__ == "__main__":

    # imports
    from tqdm import tqdm
    import multiprocessing
    from reading.read_musescore import read_musescore
    from reading.read_musicxml import read_musicxml

    # paths to load
    paths = [
        "/data2/pnlong/musescore/test_data/chopin/Chopin_Trois_Valses_Op64.mscz",
        "/data2/pnlong/musescore/test_data/goodman/in_the_mood.mscz",
        "/data2/pnlong/musescore/test_data/laufey/from_the_start.mscz",
        "/data2/pnlong/musescore/test_data/maroon/this_love.mscz",
        # "/data2/pnlong/musescore/test_data/test1/QmbboPyM7KRorFpbmqnoCYDjbN2Up2mY969kggS3JLqRCF.mscz",
        # "/data2/pnlong/musescore/test_data/test2/QmbbxbpgJHyNRzjkbyxdoV5saQ9HY38MauKMd5CijTPFiF.mscz",
        # "/data2/pnlong/musescore/test_data/test3/QmbbjJJAMixkH5vqVeffBS1h2tJHQG1DXpTJHJonpxGmSN.mscz",
        "/data2/pnlong/musescore/test_data/toploader/dancing_in_the_moonlight.mscz",
        "/data2/pnlong/musescore/test_data/debussy/clair_de_lune.mscz",
        # "/data2/pnlong/musescore/test_data/test0/QmcDL7KEmC9ZCeaXvBSnPWK26EtcoYicKmLhj8EwV7XeUG.mscz",
        # "/data2/pnlong/musescore/test_data/waldteufel/les_patineurs.mscz",
        "/data2/pnlong/musescore/test_data/newman/youve_got_a_friend_in_me.mxl",
    ]

    # helper function
    def make_example(path: str):
        """Make example."""
        stem = ".".join(path.split(".")[:-1])
        if path.endswith((".mscz", ".mscx")):
            music = read_musescore(path = path)
        elif path.endswith((".mxl", ".xml", ".musicxml")):
            music = read_musicxml(path = path)
        else:
            raise RuntimeError(f"Invalid filetype for `{path}`.")
        music.write(f"{stem}.out.mxl") # text musicxml output
        # music.write(f"{stem}.out.wav") # test audio output (which implicitly tests midi)

    # multiprocessing
    with multiprocessing.Pool(processes = int(multiprocessing.cpu_count() / 4)) as pool:
        _ = list(pool.map(func = make_example, iterable = tqdm(iterable = paths, desc = f"Testing Outputs", total = len(paths)), chunksize = 1))

##################################################
   
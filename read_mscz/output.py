# README
# Phillip Long
# December 1, 2023

# Functions for outputting a BetterMusic object to different file formats.

# python ./output.py


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
from itertools import groupby

# general
from read_mscz.classes import *
from utils import unique
import representation

# midi
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, MAX_PITCHWHEEL

# audio
import tempfile

# musicxml
from muspy.utils import CIRCLE_OF_FIFTHS, MODE_CENTERS
from music21.musicxml.archiveTools import compressXML
from music21.key import Key
from music21.metadata import Contributor, Copyright
from music21.metadata import Metadata as M21MetaData
from music21.meter import TimeSignature as M21TimeSignature
from music21.note import Note as M21Note, noteheadTypeNames
from music21.stream import Part, Score
from music21 import tempo as M21Tempo
from music21 import articulations as M21Articulation
from music21 import expressions as M21Expression
from music21 import spanner as M21Spanner
from music21 import dynamics as M21Dynamic
from music21.exceptions21 import StreamException

##################################################


# CONSTANTS
##################################################

DEFAULT_DYNAMIC = "mf" # not to be confused with representation.DEFAULT_DYNAMIC
FERMATA_TEMPO_SLOWDOWN = 3 # factor by which to slow down the tempo when there is a fermata
PEDAL_DURATION_CHANGE_FACTOR = 3 # factor by which the sustain pedal increases the duration of each note
STACCATO_DURATION_CHANGE_FACTOR = 5 # factor by which a staccato decreases the duration of a note
VELOCITY_INCREASE_FACTOR = 2 # factor by which to increase velocity when an expressive feature GRADUALLY increases velocity
ACCENT_VELOCITY_INCREASE_FACTOR = 1.5 # factor by which to increase velocity when an accent INSTANTANEOUSLY increases velocity
FRACTION_TO_WIGGLE = 0.34 # fraction of MAX/MIN_PITCHWHEEL to bend notes for wiggle articulations (vibratos and sawtooths)

##################################################


# HELPER FUNCTIONS
##################################################

def clean_up_subtype(subtype: str) -> str:
    """Clean up the subtype of an annotation so that I can better match for substrings."""
    return sub(pattern = "[^\w0-9]", repl = "", string = subtype.lower())

##################################################


# WRITE MIDI
##################################################

def to_mido_note_on_note_off(note: Note, channel: int, use_note_off_message: bool = False) -> Tuple[Message, Message]:
    """Return a Note object as mido Message objects.

    Timing is in absolute time, NOT in delta time.

    Parameters
    ----------
    note : :class:`read_mscz.Note` object
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

    # deal with timings
    # note.time = int(note.time)
    # note.duration = int(note.duration)

    # deal with velocity
    velocity = note.velocity # copy of velocity as to not alter the music object
    if velocity is None:
        velocity = DEFAULT_VELOCITY
    velocity = int(max(min(velocity, 127), 0)) # make sure velocity is within valid range and an integer

    # deal with note
    pitch = note.pitch
    pitch = int(max(min(pitch, 127), 0)) # make sure the note is within valid range and an integer

    # note on message
    note_on_msg = Message(type = "note_on", time = note.time, note = pitch, velocity = velocity, channel = channel) # create note on message

    # note off message
    if use_note_off_message: # create note off message
        note_off_msg = Message(type = "note_off", time = note.time + note.duration, note = pitch, velocity = 64, channel = channel)
    else:
        note_off_msg = Message(type = "note_on", time = note.time + note.duration, note = pitch, velocity = 0, channel = channel)

    # return messages
    return note_on_msg, note_off_msg


def to_delta_time(midi_track: MidiTrack):
    """Convert a mido MidiTrack object from absolute time to delta time.

    Parameters
    ----------
    midi_track : :class:`mido.MidiTrack` object
        mido MidiTrack object to convert.

    """

    # sort messages by absolute time
    midi_track.sort(key = lambda message: message.time)

    # convert to delta time
    time = 0
    for message in midi_track:
        time_ = message.time
        message.time = int(message.time - time) # ensure message time is int
        time = time_


def to_mido_meta_track(music: "BetterMusic") -> MidiTrack:
    """Return a mido MidiTrack containing metadata of a Music object.

    Parameters
    ----------
    music : :class:`read_mscz.BetterMusic` object
        Music object to convert.

    Returns
    -------
    :class:`mido.MidiTrack` object
        Converted mido MidiTrack object.

    """
    
    # create a track to store the metadata
    meta_track = MidiTrack()

    # song title
    if music.metadata.title is not None:
        meta_track.append(MetaMessage(type = "track_name", name = music.metadata.title))

    # tempos and time signatures
    combined_temporal_features = sorted(music.tempos + music.time_signatures, key = lambda temporal_feature: temporal_feature.time) # get sorted list of tempos and time signatures
    current_time_signature = music.time_signatures[0] if len(music.time_signatures) > 0 else TimeSignature(time = 0) # instantiate current_time_signature
    tempo_changes = [] # keep track of tempo changes to deal with tempo spanners later
    for temporal_feature in combined_temporal_features:
        if isinstance(temporal_feature, Tempo): # if tempo feature
            current_tempo = bpm2tempo(bpm = temporal_feature.qpm, time_signature = (current_time_signature.numerator, current_time_signature.denominator))
            meta_track.append(MetaMessage(type = "set_tempo", time = temporal_feature.time, tempo = current_tempo))
            tempo_changes.append({"time": temporal_feature.time, "tempo": current_tempo})
        elif isinstance(temporal_feature, TimeSignature): # if time signature
            meta_track.append(MetaMessage(type = "time_signature", time = temporal_feature.time, numerator = temporal_feature.numerator, denominator = temporal_feature.denominator))
            current_time_signature = temporal_feature # update current_time_signature

    # key signatures
    for key_signature in music.key_signatures:
        if (key_signature.root is not None) and (key_signature.mode in ("major", "minor")):
            meta_track.append(MetaMessage(type = "key_signature", time = key_signature.time, key = PITCH_NAMES[key_signature.root] + ("m" if key_signature.mode == "minor" else "")))        

    # lyrics
    for lyric in music.lyrics:
        meta_track.append(MetaMessage(type = "lyrics", time = lyric.time, text = lyric.lyric))

    # system and staff level annotations
    current_tempo_index = 0
    all_notes = sum((track.notes for track in music.tracks), []) # all notes
    for annotation in music.annotations + sum((track.annotations for track in music.tracks), []):
        # ensure that values are valid
        if hasattr(annotation.annotation, "subtype"): # make sure subtype field is not none
            if annotation.annotation.subtype is None:
                continue
            else:
                annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
        # update current_tempo_index if necessary
        if current_tempo_index < len(tempo_changes) - 1: # avoid index error later on at last element in tempo_changes
            if tempo_changes[current_tempo_index + 1]["time"] <= annotation.time: # update current_tempo_index if necessary
                current_tempo_index += 1 # increment
        # Fermata and fermatas stored inside of Articulation
        if annotation.annotation.__class__.__name__ in ("Fermata", "Articulation"):
            if annotation.annotation.__class__.__name__ == "Articulation": # looking for fermatas
                if "fermata" not in annotation.annotation.subtype: # hidden as articulations
                    continue # if not a fermata-articulation, skip
            longest_note_duration_at_current_time = max((note.duration for note in all_notes if note.time == annotation.time)) # go through notes and find longest duration note at the time of the fermata
            meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time, tempo = tempo_changes[current_tempo_index]["tempo"] * FERMATA_TEMPO_SLOWDOWN)) # start of fermata
            meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time + longest_note_duration_at_current_time, tempo = tempo_changes[current_tempo_index]["tempo"])) # end of fermata
            del longest_note_duration_at_current_time
        # TempoSpanner
        elif annotation.annotation.__class__.__name__ == "TempoSpanner":
            tempo_change_factor_magnitude = 1 # amount to multiply the tempo by
            for time in range(annotation.time, annotation.time + annotation.annotation.duration, int(annotation.annotation.duration / 5)):
                tempo_change_factor = 1 # default, unknown TempoSpanner subtype
                if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("lent", "rall", "rit", "smorz", "sost", "allarg"))): # slow-downs; lentando, rallentando, ritardando, smorzando, sostenuto, allargando
                    tempo_change_factor = tempo_change_factor_magnitude
                elif any((annotation.annotation.subtype.startswith(prefix) for prefix in ("accel", "leg"))): # speed-ups; accelerando, leggiero
                    tempo_change_factor = 1 / tempo_change_factor_magnitude
                meta_track.append(MetaMessage(type = "set_tempo", time = time, tempo = int(tempo_changes[current_tempo_index]["tempo"] * tempo_change_factor))) # add tempo change
                tempo_change_factor_magnitude += 1 # update the magnitude of the tempo change
            end_tempo_spanner_tempo_index = current_tempo_index
            if current_tempo_index < len(tempo_changes) - 1:
                end_tempo_spanner_tempo_index += int(annotation.time + annotation.annotation.duration > tempo_changes[current_tempo_index + 1]["time"]) # if the tempo changed during the tempo spanner
            meta_track.append(MetaMessage(type = "set_tempo", time = annotation.time + annotation.annotation.duration, tempo = tempo_changes[end_tempo_spanner_tempo_index]["tempo"])) # reset tempo
            del time, tempo_change_factor_magnitude, tempo_change_factor, end_tempo_spanner_tempo_index
        # Text and TextSpanner
        elif annotation.annotation.__class__.__name__ in ("Text", "TextSpanner"):
            meta_track.append(MetaMessage(type = "text", time = annotation.time, text = annotation.annotation.text))
        # RehearsalMark
        elif annotation.annotation.__class__.__name__ == "RehearsalMark":
            meta_track.append(MetaMessage(type = "marker", time = annotation.time, text = annotation.annotation.text))
    del current_tempo_index, all_notes

    # end of track message
    meta_track.append(MetaMessage(type = "end_of_track"))

    # convert to delta time
    to_delta_time(midi_track = meta_track)

    return meta_track


EXPRESSIVE_FEATURE_PRIORITIES = ("Dynamic", "SlurSpanner", "HairPinSpanner", "TempoSpanner", "Articulation", "PedalSpanner")
def sort_expressive_features_key(annotation: Annotation) -> int:
    """When given an annotation, associate the annotation with a certain number so that a list of annotations can be sorted in a desired order."""
    if annotation.annotation.__class__.__name__ in EXPRESSIVE_FEATURE_PRIORITIES:
        return EXPRESSIVE_FEATURE_PRIORITIES.index(annotation.annotation.__class__.__name__)
    else:
        return len(EXPRESSIVE_FEATURE_PRIORITIES)


def get_wiggle_func(articulation_subtype: str, amplitude: float = FRACTION_TO_WIGGLE * MAX_PITCHWHEEL, resolution: float = representation.RESOLUTION) -> Callable:
    """Return the function that given a time, returns the amount of pitchbend for wiggle functions (vibrato or sawtooth)."""
    period = resolution / 3
    if ("fast" in articulation_subtype) or ("wide" not in articulation_subtype):
        period /= 2
    if "sawtooth" in articulation_subtype: # sawtooth
        wiggle_func = lambda time: int(amplitude * ((time % period) / period))
    else: # vibrato is default
        wiggle_func = lambda time: int(amplitude * sin(time * (2 * pi / period)))
    return wiggle_func


def to_mido_track(track: Track, music: "BetterMusic", channel: int = None, use_note_off_message: bool = False) -> MidiTrack:
    """Return a Track object as a mido MidiTrack object.

    Parameters
    ----------
    track : :class:`read_mscz.Track` object
        Track object to convert.
    music : :class:`read_mscz.BetterMusic` object
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
        channel = 9 if track.is_drum else 0

    # create a new .mid track
    midi_track = MidiTrack()

    # track name messages
    if track.name is not None:
        midi_track.append(MetaMessage(type = "track_name", name = track.name))

    # program change messages
    midi_track.append(Message(type = "program_change", program = track.program, channel = channel))

    # deal with expressive features
    note_times = sorted(unique(l = [note.time for note in track.notes])) # times of notes, sorted ascending, removing duplicates
    expressive_features: Dict[int, List[Annotation]] = dict(zip(note_times, ([] for _ in range(len(note_times))))) # dictionary where keys are time and values are expressive feature annotation objects
    for annotation in sorted(track.annotations + music.annotations, key = lambda annotation: annotation.time): # sort staff and system level annotations
        if hasattr(annotation.annotation, "subtype"):
            if annotation.annotation.subtype is None:
                continue
            else:
                annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
        annotation_falls_on_note = annotation.time in expressive_features.keys()
        if annotation_falls_on_note: # if this annotation falls on a note, add that annotation to expressive features (which we care about)
            expressive_features[annotation.time].append(annotation) # add annotation
        if hasattr(annotation.annotation, "duration"): # deal with duration
            if annotation.annotation.__class__.__name__ == "Dynamic" and annotation.annotation.subtype not in representation.DYNAMIC_DYNAMICS: # for sudden dynamic hikes do not fill with duration
                continue
            if annotation_falls_on_note: # get index of the note after the current time when annotation falls on a note
                current_note_time_index = note_times.index(annotation.time) + 1
            else: # get index of the note after the current time when annotation does not fall on a note
                current_note_time_index = len(note_times) # default value
                for i in range(len(note_times)):
                    if note_times[i] > annotation.time: # first note time after the annotation time
                        current_note_time_index = i
                        break
            while current_note_time_index < len(note_times):
                if note_times[current_note_time_index] >= annotation.time + annotation.annotation.duration: # when the annotation does not have any more effect on the notes being played
                    break # break out of while loop
                expressive_features[note_times[current_note_time_index]].append(annotation)
                current_note_time_index += 1 # increment
            del current_note_time_index
    del annotation_falls_on_note

    # make sure all notes have a dynamic at index 0
    for i, note_time in enumerate(note_times):
        if not any((annotation.annotation.__class__.__name__ == "Dynamic" for annotation in expressive_features[note_time])): # if there is no dynamic
            j = i - 1 # start index finder at the index before current
            while not any((annotation.annotation.subtype in representation.DYNAMIC_DYNAMICS for annotation in expressive_features[note_times[j]] if annotation.annotation.__class__.__name__ == "Dynamic")) and j > -1: # look for previous note times with a dynamic
                j -= 1 # decrement j
            if j == -1: # no notes with dynamics before this one, result to default values
                dynamic_annotation = Dynamic(subtype = DEFAULT_DYNAMIC, velocity = representation.DYNAMIC_VELOCITY_MAP[DEFAULT_DYNAMIC])
            else: # we found a dynamic before this note time
                dynamic_annotation = expressive_features[note_times[j]][0].annotation # we can assume the dynamic marking is at index 0 because of our sorting
            expressive_features[note_time].insert(0, Annotation(time = note_time, annotation = dynamic_annotation))
        expressive_features[note_time] = sorted(expressive_features[note_time], key = sort_expressive_features_key) # sort expressive features in desired order
        if expressive_features[note_time][0].annotation.velocity is None: # make sure dynamic velocity is not none
            expressive_features[note_time][0].annotation.velocity = representation.DYNAMIC_VELOCITY_MAP[DEFAULT_DYNAMIC] # set to default velocity

    # note on and note off messages
    for note in track.notes:
        note.velocity = expressive_features[note.time][0].annotation.velocity # the first index is always the dynamic
        for annotation in expressive_features[note.time][1:]: # skip the first index, since we just dealt with it
            # ensure that values are valid
            if hasattr(annotation.annotation, "subtype"): # make sure subtype field is not none
                if annotation.annotation.subtype is None:
                    continue
            # HairPinSpanner and TempoSpanner; changes in velocity
            if annotation.annotation.__class__.__name__ in ("HairPinSpanner", "TempoSpanner"): # some TempoSpanners involve a velocity change, so that is included here as well
                if annotation.group is None: # since we aren't storing anything there anyways
                    end_velocity = note.velocity # default is no change
                    if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("allarg", "cr"))): # increase-volume; allargando, crescendo
                        end_velocity *= VELOCITY_INCREASE_FACTOR
                    elif any((annotation.annotation.subtype.startswith(prefix) for prefix in ("smorz", "dim", "decr"))): # decrease-volume; smorzando, diminuendo, decrescendo
                        end_velocity /= VELOCITY_INCREASE_FACTOR
                    annotation.group = lambda time: (((end_velocity - note.velocity) / ((annotation.time + annotation.annotation.duration) - note.time)) * (time - note.time)) + note.velocity # we will use group to store a lambda function to calculate velocity
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
                if any((keyword in annotation.annotation.subtype for keyword in ("marcato", "sforzato", "accent"))): # increases velocity
                    note.velocity += note.velocity * (max((ACCENT_VELOCITY_INCREASE_FACTOR * (0.8 if "soft" in annotation.annotation.subtype else 1)), 1) - 1)
                if "spiccato" in annotation.annotation.subtype: # decreases velocity
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
            # elif annotation.annotation.__class__.__name__ == "TechAnnotation":
            #     pass # currently no implementation since we so rarely encounter these
        midi_track.extend(to_mido_note_on_note_off(note = note, channel = channel, use_note_off_message = use_note_off_message))

    # end of track message
    midi_track.append(MetaMessage(type = "end_of_track"))

    # convert to delta time
    to_delta_time(midi_track = midi_track)

    return midi_track


def write_midi(path: str, music: "BetterMusic", use_note_off_message: bool = False):
    """Write a Music object to a .mid file using mido as backend.

    Parameters
    ----------
    path : str
        Path to write the .mid file.
    music : :class:`read_mscz.BetterMusic` object
        Music object to write.
    use_note_off_message : bool, default: False
        Whether to use note-off messages. If False, note-on messages with zero velocity are used instead. The advantage to using note-on messages at zero velocity is that it can avoid sending additional status bytes when Running Status is employed.

    """
    
    # ensure we are operating on a copy of music
    music = deepcopy(music)

    # create a .mid file object
    midi = MidiFile(type = 1, ticks_per_beat = music.resolution)

    # append meta track
    midi.tracks.append(to_mido_meta_track(music = music))

    # iterate over music tracks
    for i, track in enumerate(music.tracks):
        # NOTE: Many softwares use the same instrument for messages of the same channel in different tracks. Thus, we want to assign a unique channel number for each track. .mid has 15 channels for instruments other than drums, so we increment the channel number for each track (skipping the drum channel) and go back to 0 once we run out of channels.
        
        # assign channel number
        if track.is_drum:
            channel = 9 # mido numbers channels 0 to 15 instead of 1 to 16
        else:
            channel = i % 15 # .mid has 15 channels for instruments other than drums
            channel += int(channel > 8) # avoid drum channel by adding one if the channel is greater than 8

        # add track
        midi.tracks.append(to_mido_track(track = track, music = music, channel = channel, use_note_off_message = use_note_off_message))

    midi.save(filename = path)

##################################################


# WRITE AUDIO
##################################################

def write_audio(path: str, music: "BetterMusic", audio_format: str = "auto", soundfont_path: str = None, rate: int = 44100, gain: float = 1, options: str = None):
    """Write a Music object to an audio file.

    Supported formats include WAV, AIFF, FLAC and OGA.

    Parameters
    ----------
    path : str
        Path to write the audio file.
    music : :class:`read_mscz.BetterMusic`
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

        # write the BetterMusic object to a temporary .mid file
        midi_path = f"{temp_dir}/temp.mid"
        write_midi(path = midi_path, music = music)

        # synthesize the .mid file using fluidsynth
        option_list = options.split(" ") if options is not None else []
        subprocess.run(args = ["fluidsynth", "-ni", "-F", path, "-T", audio_format, "-r", str(rate), "-g", str(gain), soundfont_path] + option_list + [midi_path], check = True, stdout = subprocess.DEVNULL)

##################################################


# WRITE MUSICXML
##################################################

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# def _get_pitch_name(note_number: int) -> str:
#     octave, pitch_class = divmod(note_number, 12)
#     return PITCH_NAMES[pitch_class] + str(octave - 1)

def write_musicxml(path: str, music: "BetterMusic", compressed: bool = None):
    """Write a Music object to a MusicXML file.

    Parameters
    ----------
    path : str
        Path to write the MusicXML file.
    music : :class:`read_mscz.BetterMusic`
        Music object to write.
    compressed : bool, optional
        Whether to write to a compressed MusicXML file. If None, infer
        from the extension of the filename ('.xml' and '.musicxml' for
        an uncompressed file, '.mxl' for a compressed file).

    """

    # ensure we are operating on a copy of music
    music = deepcopy(music)

    # create a new score
    score = Score()

    # metadata
    if music.metadata is not None:
        meta = M21MetaData()
        if music.metadata.title: # title is usually stored in movement-title. See https://www.musicxml.com/tutorial/file-structure/score-header-entity/
            meta.movementName = music.metadata.title
        if music.metadata.copyright:
            meta.copyright = Copyright(data = music.metadata.copyright)
        for creator in music.metadata.creators:
            meta.addContributor(Contributor(name = creator))
        score.append(meta)

    # tracks
    for track in music.tracks:
        
        # create a new part
        part = Part()
        part.partName = track.name

        # add tempos
        for tempo in music.tempos: # loop through tempos
            if tempo.text.startswith("<sym>"): # bpm is explicitly supplied in text or not supplied at all
                m21_tempo = M21Tempo.MetronomeMark(number = tempo.qpm) # create tempo marking with bpm
            else: # bpm is implied by tempo name
                m21_tempo = M21Tempo.MetronomeMark(text = tempo.text, numberSounding = tempo.qpm) # create tempo marking with text
            m21_tempo.offset = tempo.time # define offset
            part.append(m21_tempo)

        # add time signatures
        for time_signature in music.time_signatures: # loop through time signatures
            m21_time_signature = M21TimeSignature(value = f"{time_signature.numerator}/{time_signature.denominator}") # instantiate time signature object
            m21_time_signature.offset = time_signature.time # define offset
            part.append(m21_time_signature)

        # add key signatures
        for key_signature in music.key_signatures: # loop through key signatures
            # determine the tonic
            if key_signature.root_str is not None: # look at the root_str
                tonic = key_signature.root_str
            elif key_signature.root is not None: # look at the root
                tonic = PITCH_NAMES[key_signature.root]
            elif key_signature.fifths is not None: # look at the circle of fifths index
                if key_signature.mode is not None: # if there is a mode
                    offset = MODE_CENTERS[key_signature.mode]
                    tonic = CIRCLE_OF_FIFTHS[key_signature.fifths + offset][1]
                else:
                    tonic = CIRCLE_OF_FIFTHS[key_signature.fifths][1]
            else: # skip if the note is missing a root_str, root, and circle of fifths index
                continue
            m21_key_signature = Key(tonic = tonic, mode = key_signature.mode) # create key object
            m21_key_signature.offset = key_signature.time # define offset
            part.append(m21_key_signature)

        # add notes to part (and grace notes, lyrics, noteheads, and articulations)
        lyrics: Dict[int, str] = {lyric.time: lyric.lyric for lyric in track.lyrics} # put lyrics into dictionary (where keys are the time)
        noteheads: Dict[int, str] = {annotation.time: annotation.annotation.subtype.lower() for annotation in filter(lambda annotation: annotation.annotation.__class__.__name__ == "Notehead", track.annotations)}
        articulations: Dict[int, List[Annotation]] = {articulation_time: [clean_up_subtype(subtype = annotation.annotation.subtype) for annotation in articulations_at_time] for articulation_time, articulations_at_time in groupby(
            iterable = sorted(filter(lambda annotation: any(annotation.annotation.__class__.__name__ == keyword for keyword in ("Articulation", "Symbol")), track.annotations), key = lambda annotation: annotation.time), # need to sort because itertools creates a new group when a new key appears
            key = lambda annotation: annotation.time)}
        # loop through notes
        for note in track.notes:
            m21_note = M21Note(pitch = note.pitch) # create note object
            # check if grace note
            if note.is_grace: # check if grace note
                m21_note.type = "eighth" # so it looks like a normal grace note with a flag
                m21_note = m21_note.getGrace() # convert to grace note
            # if the note is a normal note
            else:
                m21_note.quarterLength = note.duration / music.resolution # get length of note
                # check if there is a lyric at that note
                if note.time in lyrics.keys():
                    m21_note.lyric = lyrics[note.time] # add lyric
            # check if there is a notehead at that note
            if note.time in noteheads.keys():
                if noteheads[note.time] in noteheadTypeNames: # check if notehead is a valid notehead for music21
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
            offset = note.time / music.resolution # get offset
            part.insert(offsetOrItemOrList = offset, itemOrNone = m21_note)
        del lyrics, noteheads, articulations

        # add expressive features to part
        for annotation in sorted(music.annotations + track.annotations, key = lambda annotation: annotation.time):
            if hasattr(annotation.annotation, "subtype"):
                if annotation.annotation.subtype is None:
                    continue
                else:
                    annotation.annotation.subtype = clean_up_subtype(subtype = annotation.annotation.subtype) # clean up the subtype
            # time
            offset = annotation.time / music.resolution
            # some boolean flags
            is_fermata = (annotation.annotation.__class__.__name__ == "Fermata") # fermata
            if any((annotation.annotation.__class__.__name__ == keyword for keyword in ("Articulation", "Symbol"))): # instance where fermatas are hidden as articulations
                is_fermata = "fermata" in annotation.annotation.subtype.lower()
            # Text, TextSpanner
            if any((annotation.annotation.__class__.__name__ == keyword for keyword in ("Text", "TextSpanner"))):
                m21_annotation = M21Expression.TextExpression(content = annotation.annotation.text)
                if annotation.annotation.__class__.__name__ == "TextSpanner": # text spanners
                    m21_annotation.quarterLength = annotation.annotation.duration / music.resolution
            # RehearsalMark
            elif annotation.annotation.__class__.__name__ == "RehearsalMark":
                m21_annotation = M21Expression.RehearsalMark(content = annotation.annotation.text)
            # Dynamic
            elif annotation.annotation.__class__.__name__ == "Dynamic":
                m21_annotation = M21Dynamic.Dynamic(value = annotation.annotation.subtype if annotation.annotation.subtype != representation.DEFAULT_EXPRESSIVE_FEATURE_VALUES["Dynamic"] else DEFAULT_DYNAMIC)
            # Fermata
            elif is_fermata:
                m21_annotation = M21Expression.Fermata()
            # Ornament, Articulation, Symbol, TechAnnotation
            elif any((annotation.annotation.__class__.__name__ == keyword for keyword in ("Ornament", "Articulation", "Symbol", "TechAnnotation"))):
                if annotation.annotation.__class__.__name__ == "TechAnnotation": # just to make things easier
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
            # TrillSpanner, HairPinSpanner, SlurSpanner, GlissandoSpanner, OttavaSpanner, TrillSpanner; anything that spans multiple notes
            elif any((annotation.annotation.__class__.__name__ == keyword for keyword in ("TrillSpanner", "HairPinSpanner", "SlurSpanner", "GlissandoSpanner", "OttavaSpanner", "TrillSpanner"))):
                spanned_notes = [note for note in part.getElementsByClass(M21Note) if (((note.offset * music.resolution) >= annotation.time) and ((note.offset * music.resolution) <= (annotation.time + annotation.annotation.duration)))]
                # TempoSpanner
                if annotation.annotation.__class__.__name__ == "TempoSpanner":
                    if any((annotation.annotation.subtype.startswith(prefix) for prefix in ("accel", "leg"))): # speed-ups; accelerando, leggiero
                        m21_annotation = M21Tempo.AccelerandoSpanner(*spanned_notes)
                    else: # slow-downs; lentando, rallentando, ritardando, smorzando, sostenuto, allargando, etc.; the default
                        m21_annotation = M21Tempo.RitardandoSpanner(*spanned_notes)
                # HairPinSpanner
                elif annotation.annotation.__class__.__name__ == "HairPinSpanner":
                    if any((keyword in "".join(annotation.annotation.subtype.split("-")) for keyword in ("dim", "decres"))): # diminuendo
                        m21_annotation = M21Dynamic.Diminuendo(*spanned_notes)
                    else: # crescendo
                        m21_annotation = M21Dynamic.Crescendo(*spanned_notes)
                # SlurSpanner
                elif annotation.annotation.__class__.__name__ == "SlurSpanner":
                    m21_annotation = M21Spanner.Slur(*spanned_notes)
                # GlissandoSpanner
                elif annotation.annotation.__class__.__name__ == "GlissandoSpanner":
                    m21_annotation = M21Spanner.Glissando(*spanned_notes, lineType = "wavy" if annotation.annotation.is_wavy else "solid")
                # OttavaSpanner
                elif annotation.annotation.__class__.__name__ == "OttavaSpanner":
                    ottava_n_octaves = sub(pattern = "[^0-9]", repl = "", string = annotation.annotation.subtype)
                    m21_annotation = M21Spanner.Ottava(type = (int(ottava_n_octaves) if any((n_octaves == ottava_n_octaves for n_octaves in map(str, (8, 15, 22)))) else 8, "down" if "b" in annotation.annotation.subtype else "up"), transposing = False)
                    del ottava_n_octaves
                # TrillSpanner
                elif annotation.annotation.__class__.__name__ == "TrillSpanner" and len(spanned_notes) >= 2:
                    m21_annotation = M21Expression.TrillExtension(spanned_notes[0], spanned_notes[1])
            # Arpeggio
            elif annotation.annotation.__class__.__name__ == "Arpeggio":
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
            elif annotation.annotation.__class__.__name__ == "Tremolo":
                number_of_marks = sub(pattern = "[^\d]", repl = "", string = annotation.annotation.subtype)
                m21_annotation = M21Expression.Tremolo(numberOfMarks = (int(number_of_marks) // 8) if len(number_of_marks) > 0 else 3)
                del number_of_marks
            # TremoloBar
            # elif annotation.annotation.__class__.__name__ == "TremoloBar":
            #     pass # to be implemented on a later date
            # ChordLine
            # elif annotation.annotation.__class__.__name__ == "ChordLine":
            #     pass # to be implemented on a later data
            # Bend
            # elif annotation.annotation.__class__.__name__ == "Bend":
            #     pass # to be implemented on a later date
            # PedalSpanner
            # elif annotation.annotation.__class__.__name__ == "PedalSpanner":
            #     pass # music21 does not have pedals
            # VibratoSpanner
            # elif annotation.annotation.__class__.__name__ == "VibratoSpanner":
            #     pass # so rare that this is not worth implementing
            else:
                continue
            if "m21_annotation" not in locals(): # check if m21_annotation was created and not caught in some offcase
                continue
            # insert annotation into part
            try:
                part.insert(offsetOrItemOrList = offset, itemOrNone = deepcopy(m21_annotation)) # as to avoid the StreamException object * is already found in this Stream
            except StreamException as stream_exception:
                print(str(stream_exception))
        # append the part to score
        score.append(part)

    # infer compression
    if compressed is None:
        if path.endswith((".xml", ".musicxml")):
            compressed = False
        elif path.endswith(".mxl"):
            compressed = True
        else:
            raise ValueError("Cannot infer file type from the extension.")
    # compress the file (or not)
    if compressed:
        path_temp = f"{path}.temp.xml"
        score.write(fmt = "xml", fp = path_temp)
        compressXML(filename = path_temp, deleteOriginal = True)
        rename(src = path_temp, dst = path)
    else: # don't compress
        score.write(fmt = "xml", fp = path)

##################################################


# MAIN METHOD FOR TESTING
##################################################

if __name__ == "__main__":

    from read_mscz.read_mscz import read_musescore
    prefix = "/data2/pnlong/musescore/test_data/toploader/dancing_in_the_moonlight.test"
    music = read_musescore(path = f"{prefix}.mscz")
    music.write(path = f"{prefix}.xml") # tests xml output
    music.write(path = f"{prefix}.wav") # tests midi and audio output

##################################################
   
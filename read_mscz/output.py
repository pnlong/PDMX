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
from typing import Tuple

# general
from read_mscz.classes import *

# midi
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

# audio
import tempfile

# musicxml
from music21.musicxml.archiveTools import compressXML
from music21.key import Key
from music21.metadata import Contributor, Copyright
from music21.metadata import Metadata as M21MetaData
from music21.meter import TimeSignature as M21TimeSignature
from music21.note import Note as M21Note
from music21.stream import Part, Score
from music21.tempo import MetronomeMark

##################################################


# CONSTANTS
##################################################




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
    velocity = note.velocity if note.velocity is not None else DEFAULT_VELOCITY # get velocity
    note_on_msg = Message(type = "note_on", time = note.time, note = note.pitch, velocity = velocity, channel = channel) # create note on message
    if use_note_off_message: # create note off message
        note_off_msg = Message(type = "note_off", time = note.time + note.duration, note = note.pitch, velocity = 64, channel = channel)
    else:
        note_off_msg = Message(type = "note_on", time = note.time + note.duration, note = note.pitch, velocity = 0, channel = channel)
    return note_on_msg, note_off_msg

def to_delta_time(midi_track: MidiTrack):
    """Convert a mido MidiTrack object from absolute time to delta time.

    Parameters
    ----------
    midi_track : :class:`mido.MidiTrack` object
        mido MidiTrack object to convert.

    """

    # sort messages by absolute time
    midi_track.sort(key = lambda x: x.time)

    # convert to delta time
    time = 0
    for message in midi_track:
        time_ = message.time
        message.time -= time
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

    # tempos
    for tempo in music.tempos:
        meta_track.append(MetaMessage(type = "set_tempo", time = tempo.time, tempo = bpm2tempo(bpm = tempo.qpm)))

    # key signatures
    for key_signature in music.key_signatures:
        if (key_signature.root is not None) and (key_signature.mode in ("major", "minor")):
            meta_track.append(MetaMessage(type = "key_signature", time = key_signature.time, key = PITCH_NAMES[key_signature.root] + ("m" if key_signature.mode == "minor" else "")))

    # time signatures
    for time_signature in music.time_signatures:
        meta_track.append(MetaMessage(type = "time_signature", time = time_signature.time, numerator = time_signature.numerator, denominator = time_signature.denominator))

    # lyrics
    for lyric in music.lyrics:
        meta_track.append(MetaMessage(type = "lyrics", time = lyric.time, text = lyric.lyric))

    # annotations
    for annotation in music.annotations:
        # marker messages
        if annotation.group == "marker":
            meta_track.append(MetaMessage(type = "marker", text = annotation.annotation))
        # text messages
        elif isinstance(annotation.annotation, str):
            meta_track.append(MetaMessage(type = "text", time = annotation.time, text = annotation.annotation))

    # end of track message
    meta_track.append(MetaMessage(type = "end_of_track"))

    # convert to delta time
    to_delta_time(midi_track = meta_track)

    return meta_track

def to_mido_track(track: Track, channel: int = None, use_note_off_message: bool = False) -> MidiTrack:
    """Return a Track object as a mido MidiTrack object.

    Parameters
    ----------
    track : :class:`read_mscz.Track` object
        Track object to convert.
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

    # note on and note off messages
    for note in track.notes:
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
    
    # create a .mid file object
    midi = MidiFile(type = 1, ticks_per_beat = music.resolution)

    # append meta track
    midi.tracks.append(to_mido_meta_track(music))

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
        midi.tracks.append(to_mido_track(track = track, channel = channel, use_note_off_message = use_note_off_message))

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

def _get_pitch_name(note_number: int) -> str:
    octave, pitch_class = divmod(note_number, 12)
    return PITCH_NAMES[pitch_class] + str(octave - 1)

def to_music21_metronome(tempo: Tempo) -> MetronomeMark:
    """Return a Tempo object as a music21 MetronomeMark object."""
    metronome = MetronomeMark(number=tempo.qpm)
    metronome.offset = tempo.time
    return metronome

def to_music21_key(key_signature: KeySignature) -> Key:
    """Return a KeySignature object as a music21 Key object."""
    if key_signature.root_str is not None:
        tonic = key_signature.root_str
    elif key_signature.root is not None:
        tonic = PITCH_NAMES[key_signature.root]
    elif key_signature.fifths is not None:
        if key_signature.mode is not None:
            offset = MODE_CENTERS[key_signature.mode]
            tonic = CIRCLE_OF_FIFTHS[key_signature.fifths + offset][1]
        else:
            tonic = CIRCLE_OF_FIFTHS[key_signature.fifths][1]
    else:
        raise ValueError(
            "One of `root`, `root_str` or `fifths` must be specified."
        )
    key = Key(tonic=tonic, mode=key_signature.mode)
    key.offset = key_signature.time
    return key

def to_music21_time_signature(time_signature: TimeSignature) -> M21TimeSignature:
    """Return a TimeSignature object as a music21 TimeSignature."""
    m21_time_signature = M21TimeSignature(
        f"{time_signature.numerator}/{time_signature.denominator}"
    )
    m21_time_signature.offset = time_signature.time
    return m21_time_signature

def to_music21_metadata(metadata: Metadata) -> M21MetaData:
    """Return a Metadata object as a music21 Metadata object.

    Parameters
    ----------
    metadata : :class:`read_mscz.Metadata`
        Metadata object to convert.

    Returns
    -------
    `music21.metadata.Metadata`
        Converted music21 Metadata object.

    """
    meta = M21MetaData()

    # Title is usually stored in movement-title. See
    # https://www.musicxml.com/tutorial/file-structure/score-header-entity/
    if metadata.title:
        meta.movementName = metadata.title

    if metadata.copyright:
        meta.copyright = Copyright(metadata.copyright)
    for creator in metadata.creators:
        meta.addContributor(Contributor(name=creator))
    return meta

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

    # Create a new score
    score = Score()

    # Metadata
    if music.metadata:
        score.append(to_music21_metadata(music.metadata))

    # Tracks
    for track in music.tracks:
        # Create a new part
        part = Part()
        part.partName = track.name

        # Add tempos
        for tempo in music.tempos:
            part.append(to_music21_metronome(tempo))

        # Add time signatures
        for time_signature in music.time_signatures:
            part.append(to_music21_time_signature(time_signature))

        # Add key signatures
        for key_signature in music.key_signatures:
            part.append(to_music21_key(key_signature))

        # Add notes to part
        for note in track.notes:
            m21_note = M21Note(_get_pitch_name(note.pitch))
            m21_note.quarterLength = note.duration / music.resolution
            offset = note.time / music.resolution
            part.insert(offset, m21_note)

        # Append the part to score
        score.append(part)

    # infer compression
    if compressed is None:
        if path.endswith((".xml", ".musicxml")):
            compressed = False
        elif path.endswith(".mxl"):
            compressed = True
        else:
            raise ValueError("Cannot infer file type from the extension.")

    # compress the file
    if compressed:
        path_temp = path + ".temp.xml"
        score.write("xml", path_temp)
        compressXML(filename = path_temp, deleteOriginal = True)
        rename(src = path_temp, dst = path)
    # don't compress
    else:
        score.write("xml", path)


##################################################

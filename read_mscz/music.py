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
from .classes import *
from collections import OrderedDict
from typing import List
from re import sub
import yaml # for printing
import json
import gzip
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
        return {key: to_dict(obj = value) for key, value in (("name", obj.__class__.__name__) + list(vars(obj).items()))}


def load_annotation(annotation: dict):
    """Return an expressive feature object given an annotation dictionary. For loading from .json."""

    match annotation["name"]:
        case "Text":
            return Text(text = str(annotation["text"]), is_system = bool(annotation["is_system"]), style = str(annotation["style"]))
        case "Subtype":
            return Subtype(subtype = str(annotation["subtype"]))                                                                                                                                                   
        case "RehearsalMark":
            return RehearsalMark(text = str(annotation["text"]))
        case "TechAnnotation":
            return TechAnnotation(text = str(annotation["text"]), tech_type = str(annotation["tech_type"]), is_system = bool(annotation["is_system"]))
        case "Dynamic":
            return Dynamic(subtype = str(annotation["subtype"]), velocity = int(annotation["velocity"]))
        case "Fermata":
            return Fermata(is_fermata_above = bool(annotation["is_fermata_above"]))
        case "Arpeggio":
            return Arpeggio(subtype = int(annotation["subtype"]))
        case "Tremolo":
            return Tremolo(subtype = str(annotation["subtype"]))
        case "ChordLine":
            return ChordLine(subtype = int(annotation["subtype"]), is_straight = bool(annotation["is_straight"]))
        case "Ornament":
            return Ornament(subtype = str(annotation["subtype"]))
        case "Articulation":
            return Articulation(subtype = str(annotation["subtype"]))
        case "Notehead":
            return Notehead(subtype = str(annotation["subtype"]))
        case "Symbol":
            return Symbol(subtype = str(annotation["subtype"]))
        case "Bend":
            return Bend(points = [Point(time = int(point["time"]), pitch = int(point["pitch"]), vibrato = int(point["vibrato"])) for point in annotation["points"]])
        case "TremoloBar":
            return TremoloBar(points = [Point(time = int(point["time"]), pitch = int(point["pitch"]), vibrato = int(point["vibrato"])) for point in annotation["points"]])
        case "Spanner":
            return Spanner(duration = int(annotation["duration"]))
        case "SubtypeSpanner":
            return SubtypeSpanner(duration = int(annotation["duration"]), subtype = annotation["subtype"])
        case "TempoSpanner":
            return TempoSpanner(duration = int(annotation["duration"]), subtype = str(annotation["subtype"]))
        case "TextSpanner":
            return TextSpanner(duration = int(annotation["duration"]), text = str(annotation["text"]), is_system = bool(annotation["is_system"]))
        case "HairPinSpanner":
            return HairPinSpanner(duration = int(annotation["duration"]), subtype = str(annotation["subtype"]), hairpin_type = int(annotation["hairpin_type"]))
        case "SlurSpanner":
            return SlurSpanner(duration = int(annotation["duration"]), is_slur = bool(annotation["is_slur"]))
        case "PedalSpanner":
            return PedalSpanner(duration = int(annotation["duration"]))
        case "TrillSpanner":
            return TrillSpanner(duration = int(annotation["duration"]), subtype = str(annotation["subtype"]), ornament = str(annotation["ornament"]))
        case "VibratoSpanner":
            return VibratoSpanner(duration = int(annotation["duration"]), subtype = str(annotation["subtype"]))
        case "GlissandoSpanner":
            return GlissandoSpanner(duration = int(annotation["duration"]), is_wavy = bool(annotation["is_wavy"]))
        case "OttavaSpanner":
            return OttavaSpanner(duration = int(annotation["duration"]), subtype = str(annotation["subtype"]))
        case _:
            raise KeyError("Unknown annotation type.")

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
    song_length : int
        The length of the song (in time steps).

    Note
    ----
    Indexing a BetterMusic object returns the track of a certain index. That
    is, ``music[idx]`` returns ``music.tracks[idx]``. Length of a Music
    object is the number of tracks. That is, ``len(music)``  returns
    ``len(music.tracks)``.

    """


    _attributes = OrderedDict([("metadata", Metadata), ("resolution", int), ("tempos", Tempo), ("key_signatures", KeySignature), ("time_signatures", TimeSignature), ("barlines", Barline), ("beats", Beat), ("lyrics", Lyric), ("annotations", Annotation), ("tracks", Track), ("song_length", int)])
    _optional_attributes = ["metadata", "resolution", "tempos", "key_signatures", "time_signatures", "barlines", "beats", "lyrics", "annotations", "tracks", "song_length"]
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
        self.song_length = self.get_song_length()

    ##################################################


    # PRETTY PRINTING
    ##################################################

    def print(self, output_filepath: str = None, remove_empty_lines: bool = True):
        """Print the BetterMusic object in a pretty way.
        
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
        """Convert from MusPy time (in time steps) to Absolute time (in seconds).
        
        Parameters
        ---------
        time_steps : int
            The time_steps value to convert
        """

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
        """Return the length of the song in time steps."""
        all_objs = self.tempos + self.key_signatures + self.time_signatures + self.beats + self.barlines + self.lyrics + self.annotations + sum([track.notes + track.annotations + track.lyrics for track in self.tracks], [])
        if len(all_objs) > 0:
            max_time_obj = max(all_objs, key = lambda obj: obj.time)
            max_time = max_time_obj.time + (max_time_obj.duration if hasattr(max_time_obj, "duration") else 0) + self.resolution # add a quarter note at the end for buffer
        else:
            max_time = 0
        final_beat = self.beats[-1].time if len(self.beats) >= 1 else 0 # (2 * self.beats[-1].time) - self.beats[-2].time
        return int(max(max_time, final_beat))

    ##################################################


    # SAVE AS JSON
    ##################################################

    def save_json(self, path: str, ensure_ascii: bool = False, compressed: bool = None, **kwargs) -> str:
        """Save a Music object to a JSON file.

        Parameters
        ----------
        path : str
            Path to save the JSON data.
        ensure_ascii : bool, default: False
            Whether to escape non-ASCII characters. Will be passed to PyYAML's `yaml.dump`.
        compressed : bool, optional
            Whether to save as a compressed JSON file (`.json.gz`). Has no effect when `path` is a file object. Defaults to infer from the extension (`.gz`).
        **kwargs
            Keyword arguments to pass to :py:func:`json.dumps`.

        Notes
        -----
        When a path is given, use UTF-8 encoding and gzip compression if `compressed=True`.

        """
        
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
            "song_length": self.song_length
        }

        # convert dictionary to json obj
        data = json.dumps(obj = data, ensure_ascii = ensure_ascii, **kwargs)

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

    ##################################################

##################################################


# LOAD A BETTERMUSIC OBJECT FROM JSON FILE
##################################################

def load_json(path: str) -> BetterMusic:
    """Load a Music object from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON data.
    """

    # if file is compressed
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding = "utf-8") as file:
            data = json.load(fp = file)
    else:
        with open(path, "rt", encoding = "utf-8") as file:
            data = json.load(fp = file)

    # extract info from nested dictionaries
    metadata = Metadata(**data["metadata"])
    tempos = [Tempo(time = int(tempo["time"]), qpm = float(tempo["qpm"]), text = str(tempo["text"]), measure = int(tempo["measure"])) for tempo in data["tempos"]]
    key_signatures = [KeySignature(time = int(keysig["time"]), root = int(keysig["root"]), mode = str(keysig["mode"]), fifths = int(keysig["fifths"]), root_str = str(keysig["root_str"]), measure = int(keysig["measure"])) for keysig in data["key_signatures"]]
    time_signatures = [TimeSignature(time = int(timesig["time"]), numerator = int(timesig["numerator"]), denominator = int(timesig["denominator"]), measure = int(timesig["measure"])) for timesig in data["time_signatures"]]
    beats = [Beat(time = int(beat["time"]), is_downbeat = bool(beat["is_downbeat"]), measure = int(beat["measure"])) for beat in data["beats"]]
    barlines = [Barline(time = int(barline["time"]), subtype = str(barline["subtype"]), measure = int(barline["measure"])) for barline in data["barlines"]]
    lyrics = [Lyric(time = int(lyric["time"]), lyric = str(lyric["lyric"]), measure = int(lyric["measure"])) for lyric in data["lyrics"]]
    annotations = [Annotation(time = int(annotation["time"]), annotation = load_annotation(annotation = annotation), measure = int(annotation["measure"]), group = str(annotation["group"])) for annotation in data["annotations"]]
    tracks = [Track(
        program = int(track["program"]),
        is_drum = bool(track["is_drum"]),
        name = str(track["name"]),
        notes = [Note(time = int(note["time"]), pitch = int(note["pitch"]), duration = int(note["duration"]), velocity = int(note["velocity"]), pitch_str = str(note["pitch_str"]), is_grace = bool(note["is_grace"]), measure = int(note["measure"])) for note in track["notes"]],
        chords = [Chord(time = int(chord["time"]), pitches = [int(pitch) for pitch in chord["pitches"]], duration = int(chord["duration"]), pitches_str = [str(pitch_str) for pitch_str in chord["pitches_str"]], measure = int(chord["measure"])) for chord in track["chords"]],
        lyrics = [Lyric(time = int(lyric["time"]), lyric = str(lyric["lyric"]), measure = int(lyric["measure"])) for lyric in track["lyrics"]],
        annotations = [Annotation(time = int(annotation["time"]), annotation = load_annotation(annotation = annotation), measure = int(annotation["measure"]), group = str(annotation["group"])) for annotation in track["annotations"]]
    ) for track in data["tracks"]]

    # return a BetterMusic object
    return BetterMusic(
        metadata = metadata,
        resolution = int(data["resolution"]),
        tempos = tempos,
        key_signatures = key_signatures,
        time_signatures = time_signatures,
        beats = beats,
        barlines = barlines,
        lyrics = lyrics,
        annotations = annotations,
        tracks = tracks
    )

##################################################
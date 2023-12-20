# README
# Phillip Long
# October 7, 2023

# Music object classes better suited for storing expressive features from musescore
# taken from https://github.com/salu133445/muspy/blob/b2d4265c6279e730903d8abe9dddda8484511903/muspy/classes.py

# ./classes.py

# IMPORTS / CONSTANTS
##################################################

import muspy
from collections import OrderedDict
from typing import Any, List

DEFAULT_VELOCITY = 64

##################################################


# MUSIC OBJECTS INHERITING FROM MUSPY
##################################################

# METADATA
class Metadata(muspy.classes.Metadata):
    """A container for metadata.

    Attributes
    ----------
    schema_version : str, default: `muspy.schemas.DEFAULT_SCHEMA_VERSION`
        Schema version.
    title : str, optional
        Song title.
    subtitle : str, optional
        Song subtitle.
    creators : list of str, optional
        Creator(s) of the song.
    copyright : str, optional
        Copyright notice.
    collection : str, optional
        Name of the collection.
    source_filename : str, optional
        Name of the source file.
    source_format : str, optional
        Format of the source file.

    """

    _attributes = OrderedDict([("schema_version", str), ("title", str), ("subtitle", str), ("creators", str), ("copyright", str), ("collection", str), ("source_filename", str), ("source_format", str)])
    _optional_attributes = ["title", "subtitle", "creators", "copyright", "collection", "source_filename", "source_format"]
    _list_attributes = ["creators"]

    def __init__(self, schema_version: str = muspy.schemas.DEFAULT_SCHEMA_VERSION, title: str = None, subtitle: str = None, creators: List[str] = None, copyright: str = None, collection: str = None, source_filename: str = None, source_format: str = None):
        super().__init__(schema_version = schema_version, title = title, creators = creators, copyright = copyright, collection = collection, source_filename = source_filename, source_format = source_format)
        self.subtitle = subtitle

# TEMPO
class Tempo(muspy.classes.Tempo):
    """A container for key signatures.

    Attributes
    ----------
    time : int
        Start time of the tempo, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    qpm : float
        Tempo in qpm (quarters per minute).
    text : str
        Text associated with the tempo (if applicable)
    
    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("qpm", (float, int)), ("text", str)])
    _optional_attributes = ["measure"]

    def __init__(self, time: int, qpm: float, text: str = None, measure: int = None):
        super().__init__(time = time, qpm = qpm)
        self.measure = measure
        self.text = text

# KEY SIG
class KeySignature(muspy.classes.KeySignature):
    """A container for key signatures.

    Attributes
    ----------
    time : int
        Start time of the key signature, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    root : int, optional
        Root (tonic) of the key signature.
    mode : str, optional
        Mode of the key signature.
    fifths : int, optional
        Number of sharps or flats. Positive numbers for sharps and
        negative numbers for flats.
    root_str : str, optional
        Root of the key signature as a string.

    Note
    ----
    A key signature can be specified either by its root (`root`) or the
    number of sharps or flats (`fifths`) along with its mode.

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("root", int), ("mode", str), ("fifths", int), ("root_str", str)])
    _optional_attributes = ["root", "mode", "fifths", "root_str", "measure"]

    def __init__(self, time: int, root: int = 0, mode: str = "major", fifths: int = 0, root_str: str = "C", measure: int = None):
        super().__init__(time = time, root = root, mode = mode, fifths = fifths, root_str = root_str)
        self.measure = measure

# TIME SIG
class TimeSignature(muspy.classes.TimeSignature):
    """A container for time signatures.

    Attributes
    ----------
    time : int
        Start time of the time signature, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    numerator : int
        Numerator of the time signature.
    denominator : int
        Denominator of the time signature.

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("numerator", int), ("denominator", int)])
    _optional_attributes = ["measure"]

    def __init__(self, time: int, numerator: int = 4, denominator: int = 4, measure: int = None):
        super().__init__(time = time, numerator = numerator, denominator = denominator)
        self.measure = measure

# BEATS
class Beat(muspy.classes.Beat):
    """A container for beats.

    Attributes
    ----------
    time : int
        Time of the beat, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    is_downbeat : bool, optional, default: False
        Is this beat a downbeat?
    
    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("is_downbeat", bool)])
    _optional_attributes = ["is_downbeat", "measure"]

    def __init__(self, time: int, is_downbeat: bool = False, measure: int = None):
        super().__init__(time = time)
        self.measure = measure
        self.is_downbeat = is_downbeat

# BARLINES
class Barline(muspy.classes.Barline):
    """A container for barlines.

    Attributes
    ----------
    time : int
        Time of the barline, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    subtype : str, optional, default: 'single'
        Type of barline (double, dashed, etc.)

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("subtype", str)])
    _optional_attributes = ["measure"]

    def __init__(self, time: int, subtype: str = "single", measure: int = None):
        super().__init__(time = time)
        self.measure = measure
        self.subtype = subtype if (subtype is not None) else "single"

# LYRICS
class Lyric(muspy.classes.Lyric):
    """A container for lyrics.

    Attributes
    ----------
    time : int
        Start time of the lyric, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    lyric : str
        Lyric (sentence, word, syllable, etc.).

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("lyric", str)])
    _optional_attributes = ["measure"]

    def __init__(self, time: int, lyric: str, measure: int = None):
        super().__init__(time = time, lyric = lyric)
        self.measure = measure

# ANNOTATIONS
class Annotation(muspy.classes.Annotation):
    """A container for annotations.

    Attributes
    ----------
    time : int
        Start time of the annotation, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    annotation : any
        Annotation of any type.
    group : str, optional
        Group name (for better organizing the annotations).

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("annotation", object), ("group", str)])
    _optional_attributes = ["measure", "group"]

    def __init__(self, time: int, annotation: Any, measure: int = None, group: str = None):
        super().__init__(time = time, annotation = annotation, group = group)
        self.measure = measure

# NOTES
class Note(muspy.classes.Note):
    """A container for notes.

    Attributes
    ----------
    time : int
        Start time of the note, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    pitch : int
        Note pitch, as a MIDI note number. Valid values are 0 to 127.
    duration : int
        Duration of the note, in time steps.
    velocity : int, default: `muspy.DEFAULT_VELOCITY` (64)
        Note velocity. Valid values are 0 to 127.
    pitch_str : str, optional
        Note pitch as a string, useful for distinguishing, e.g., C# and Db.
    is_grace : bool, optional, default: False
        Is this note a grace note?

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("pitch", int), ("duration", int), ("velocity", int), ("pitch_str", str), ("is_grace", bool)])
    _optional_attributes = ["velocity", "pitch_str", "is_grace", "measure"]

    def __init__(self, time: int, pitch: int, duration: int, velocity: int = None, pitch_str: str = None, is_grace: bool = False, measure: int = None):
        super().__init__(time = time, pitch = pitch, duration = duration, velocity = velocity, pitch_str = pitch_str)
        self.measure = measure
        self.is_grace = is_grace

# CHORDS
class Chord(muspy.classes.Chord):
    """A container for chords.

    Attributes
    ----------
    time : int
        Start time of the chord, in time steps.
    measure : int, optional, default: None
        Measure number where this element is found.
    pitches : list of int
        Note pitches, as MIDI note numbers. Valid values are 0 to 127.
    duration : int
        Duration of the chord, in time steps.
    velocity : int, default: `muspy.DEFAULT_VELOCITY` (64)
        Chord velocity. Valid values are 0 to 127.
    pitches_str : list of str, optional
        Note pitches as strings, useful for distinguishing, e.g., C# and
        Db.

    """

    _attributes = OrderedDict([("time", int), ("measure", int), ("pitches", int), ("duration", int), ("velocity", int), ("pitches_str", str)])
    _optional_attributes = ["velocity", "pitches_str", "measure"]

    def __init__(self, time: int, pitches: List[int], duration: int, velocity: int = None, pitches_str: List[str] = None, measure: int = None):
        super().__init__(time = time, pitches = pitches, duration = duration, velocity = velocity, pitches_str = pitches_str)
        self.measure = measure

# TRACKS
class Track(muspy.classes.Track):
    """A container for music track.

    Attributes
    ----------
    program : int, default: 0 (Acoustic Grand Piano)
        Program number, according to General MIDI specification [1]_.
        Valid values are 0 to 127.
    is_drum : bool, default: False
        Whether it is a percussion track.
    name : str, optional
        Track name.
    notes : list of :class:`muspy.Note`, default: []
        Musical notes.
    chords : list of :class:`muspy.Chord`, default: []
        Chords.
    annotations : list of :class:`muspy.Annotation`, default: []
        Annotations.
    lyrics : list of :class:`muspy.Lyric`, default: []
        Lyrics.

    Note
    ----
    Indexing a Track object returns the note at a certain index. That
    is, ``track[idx]`` returns ``track.notes[idx]``. Length of a Track
    object is the number of notes. That is, ``len(track)`` returns
    ``len(track.notes)``.

    References
    ----------
    .. [1] https://www.midi.org/specifications/item/gm-level-1-sound-set

    """

    _attributes = OrderedDict([("program", int), ("is_drum", bool), ("name", str), ("notes", Note), ("chords", Chord), ("lyrics", Lyric), ("annotations", Annotation)])
    _optional_attributes = ["name", "notes", "chords", "lyrics", "annotations"]
    _list_attributes = ["notes", "chords", "lyrics", "annotations"]

    def __init__(self, program: int = 0, is_drum: bool = False, name: str = None, notes: List[Note] = None, chords: List[Chord] = None, lyrics: List[Lyric] = None, annotations: List[Annotation] = None):
        self.program = program if program is not None else 0
        self.is_drum = is_drum if program is not None else False
        self.name = name
        self.notes = notes if notes is not None else []
        self.chords = chords if chords is not None else []
        self.lyrics = lyrics if lyrics is not None else []
        self.annotations = annotations if annotations is not None else []

##################################################


# PARENT OBJECTS
##################################################

# TEXT
class Text(muspy.base.Base):
    """A container for text. Meant to be stored as an annotation, so the Text class itself has no timestamp (the timestamp is stored in the annotation object).

    Attributes
    ----------
    text : str
        The text contained in the text element.
    is_system : bool, optional, default: False
        Is this text system-wide (True) or just for a specific staff (False)?
    style : str, optional, default: None
        The style of the text element.
    
    """

    _attributes = OrderedDict([("text", str), ("is_system", bool), ("style", str)])
    _optional_attributes = ["is_system", "style"]

    def __init__(self, text: str, is_system: bool = False, style: str = None):
        self.text = text
        self.is_system = bool(is_system)
        self.style = style

# SUBTYPE
class Subtype(muspy.base.Base):
    """A container for objects with the 'subtype' attribute.

    Attributes
    ----------
    subtype : Any
        The subtype attribute.
    
    """

    _attributes = OrderedDict([("subtype", object)])
    SUBTYPES = []

    def __init__(self, subtype):
        self.subtype = subtype

    def get_subtype_index(self):
        try:
            index = self.SUBTYPES.index(self.subtype)
        except ValueError:
            index = -1
        return index

##################################################


# ANNOTATIONS
##################################################

# REHEARSAL MARK
class RehearsalMark(Text):
    """A container for RehearsalMark elements.

    Attributes
    ----------
    text : str
        The text contained in the text element.

    """

    _attributes = OrderedDict([("text", str)])

    def __init__(self, text: str):
        super().__init__(text = text, is_system = True)

# TECH ANNOTATION
class TechAnnotation(Text):
    """A container for PlayTechAnnotation elements.

    Attributes
    ----------
    text : str
        The text contained in the text element.
    tech_type : str, optional, default: None
        The type of technical annotation to play.
    is_system : bool, optional, default: False
        Is this text system-wide (True) or just for a specific staff (False)?
    
    """

    _attributes = OrderedDict([("text", str), ("tech_type", str), ("is_system", bool)])
    _optional_attributes = ["tech_type", "is_system"]

    def __init__(self, text: str, tech_type: str = None, is_system: bool = False):
        super().__init__(text = text, is_system = is_system)
        self.tech_type = tech_type

# DYNAMIC MARKINGS
class Dynamic(Subtype):
    """A container for Dynamic Marking elements.

    Attributes
    ----------
    subtype : str
        The type of the Dynamic element.
    velocity : int, optional, default: `muspy.DEFAULT_VELOCITY`
        The velocity associated with the dynamic marking.
    
    """
    
    _attributes = OrderedDict([("subtype", str), ("velocity", int)])
    _optional_attributes = ["velocity"]

    def __init__(self, subtype: str, velocity: int = DEFAULT_VELOCITY):
        super().__init__(subtype = subtype)
        self.velocity = velocity

# FERMATA
class Fermata(Subtype):
    """A container for Fermatas. Meant to be stored as an annotation, so the Fermata class itself has no timestamp (the timestamp is stored in the annotation object).

    Attributes
    ----------
    is_fermata_above : bool, optional, default: True
        The subtype of fermata contained in the text element.
    
    """

    _attributes = OrderedDict([("is_fermata_above", bool)])
    _optional_attributes = ["is_fermata_above"]

    def __init__(self, is_fermata_above: bool = True):
        self.is_fermata_above = bool(is_fermata_above)
        super().__init__(subtype = "fermata above" if self.is_fermata_above else "fermata below")

# ARPEGGIO
class Arpeggio(Subtype):
    """A container for Arpeggios.

    Attributes
    ----------
    subtype : int, optional, default: 0
        Arpeggio subtype.
    
    """

    _attributes = OrderedDict([("subtype", int)])
    _optional_attributes = ["subtype"]
    SUBTYPES = ["default", "up", "down", "bracket", "up straight", "down straight"]

    def __init__(self, subtype: int = 0):
        super().__init__(subtype = self.SUBTYPES[subtype if subtype in range(len(self.SUBTYPES)) else 0])

# TREMOLO
class Tremolo(Subtype):
    """A container for Tremolos.

    Attributes
    ----------
    subtype : str, optional, default: 'r8'
        Tremolo subtype.
    
    """

    _attributes = OrderedDict([("subtype", str)])
    _optional_attributes = ["subtype"]

    def __init__(self, subtype: str = "r8"):
        super().__init__(subtype = subtype)

# CHORDLINE
class ChordLine(Subtype):
    """A container for ChordLines.

    Attributes
    ----------
    subtype : int, optional, default: 0
        ChordLine subtype. Converted to string.
    is_straight : bool, optional, default: False
        Is the ChordLine straight?
    
    """

    _attributes = OrderedDict([("subtype", str), ("is_straight", bool)])
    _optional_attributes = ["subtype", "is_straight"]
    SUBTYPES = ["fall", "doit", "plop", "scoop", "slide out down", "slide out up", "slide in above", "slide in below"]

    def __init__(self, subtype: int = 0, is_straight: bool = False):
        super().__init__(subtype = self.SUBTYPES[subtype if subtype in range(len(self.SUBTYPES)) else 0])
        self.is_straight = bool(is_straight)

# ORNAMENT
class Ornament(Subtype):
    """A container for Ornaments.

    Attributes
    ----------
    subtype : str
        Ornament subtype.
    
    """

    _attributes = OrderedDict([("subtype", str)])
    
    def __init__(self, subtype: str):
        super().__init__(subtype = subtype)

# ARTICULATION
class Articulation(Subtype):
    """A container for Articulations.

    Attributes
    ----------
    subtype : str
        Articulation subtype.
    
    """

    _attributes = OrderedDict([("subtype", str)])
    
    def __init__(self, subtype: str):
        super().__init__(subtype = subtype)

# NOTEHEAD
class Notehead(Subtype):
    """A container for Noteheads.

    Attributes
    ----------
    subtype : str
        Notehead subtype.
    
    """

    _attributes = OrderedDict([("subtype", str)])
    
    def __init__(self, subtype: str):
        super().__init__(subtype = subtype)

# SYMBOL
class Symbol(Subtype):
    """A container for Symbols.

    Attributes
    ----------
    subtype : str
        Symbol subtype.
    
    """

    _attributes = OrderedDict([("subtype", str)])
    
    def __init__(self, subtype: str):
        super().__init__(subtype = subtype)

##################################################


# POINT-RELATED
##################################################

# POINT
class Point(muspy.base.Base):
    """A container for point objects.
    
    Attributes
    ----------
    time : int
        The time of the point (in time steps)
    pitch : int
        The pitch of the point
    vibrato : int, optional, default: 0
        The vibrato

    """

    _attributes = OrderedDict([("time", int), ("pitch", int), ("vibrato", int)])
    _optional_attributes = ["vibrato"]
    
    def __init__(self, time: int, pitch: int, vibrato: int = 0):
        self.time = int(time)
        self.pitch = int(pitch)
        self.vibrato = int(vibrato)

# BEND
class Bend(muspy.base.Base):
    """A container for bends.
    
    Attributes
    ----------
    points : List[Point]
        List of points that make up bend

    """

    _attributes = OrderedDict([("points", List[Point])])

    def __init__(self, points: List[Point]):
        self.points = points

# TREMOLO BAR
class TremoloBar(Bend):
    """A container for tremolo bars.
    
    Attributes
    ----------
    points : List[Point]
        List of points that make up tremolo bar

    """

    _attributes = OrderedDict([("points", List[Point])])

    def __init__(self, points: List[Point]):
        super().__init__(points = points)

##################################################

# SPANNERS
##################################################

# SPANNER PARENT CLASS
class Spanner(muspy.base.Base):
    """A parent-class container for MuseScore Spanners. Meant to be stored as an annotation, so the Spanner class itself has no timestamp (the timestamp is stored in the annotation object).

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    
    """

    _attributes = OrderedDict([("duration", int)])

    def __init__(self, duration: int):
        self.duration = duration

# SUBTYPE SPANNER
class SubtypeSpanner(Subtype, Spanner):
    """A container for spanners whose only attribute is subtype.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : Any
        Subtype.
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", object)])

    def __init__(self, duration: int, subtype: object):
        Subtype.__init__(self = self, subtype = subtype)
        Spanner.__init__(self = self, duration = duration)

# TEMPO CHANGE SPANNERS
class TempoSpanner(SubtypeSpanner):
    """A container for GradualTempoChange spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : str
        type of tempo change
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", str)])

    def __init__(self, duration: int, subtype: str):
        super().__init__(duration = duration, subtype = subtype)

# TEXT SPANNER
class TextSpanner(Text, Spanner):
    """A container for Text spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    text : str
        The text contained in the text element.
    is_system : bool, optional, default: False
        Is this text system-wide (True) or just for a specific staff (False)?
    
    """

    _attributes = OrderedDict([("duration", int), ("text", str), ("is_system", bool)])
    _optional_attributes = ["is_system"]

    def __init__(self, duration: int, text: str, is_system: bool = False):
        Text.__init__(self = self, text = text, is_system = is_system)
        Spanner.__init__(self = self, duration = duration)

# HAIRPIN SPANNER
class HairPinSpanner(SubtypeSpanner):
    """A container for HairPin spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : str, optional, default: None
        The text associated with the HairPin spanner.
    hairpin_type : int, optional, default: -1
        The type of hairpin found within the MuseScore xml element.
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", str), ("hairpin_type", int)])
    _optional_attributes = ["subtype", "hairpin_type"]

    def __init__(self, duration: int, subtype: str = None, hairpin_type: int = -1):
        super().__init__(duration = duration, subtype = subtype)
        self.hairpin_type = hairpin_type

# SLUR/TIE SPANNER
class SlurSpanner(SubtypeSpanner):
    """A container for Slur spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    is_slur : bool, optional, default: True
        Is this object a slur (True), meaning it connects different pitches, or a tie (False), meaning it connects the same pitch?
    subtype : str, optional, default: 'slur'
        Derived from is_slur.
    
    """

    _attributes = OrderedDict([("duration", int), ("is_slur", bool)])
    _optional_attributes = ["is_slur"]

    def __init__(self, duration: int, is_slur: bool = True):
        self.is_slur = bool(is_slur)
        super().__init__(duration = duration, subtype = "slur" if self.is_slur else "tie")

# PEDAL SPANNER
class PedalSpanner(Spanner):
    """A container for Pedal spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
        
    """

    _attributes = OrderedDict([("duration", int)])

    def __init__(self, duration: int):
        super().__init__(duration = duration)

# TRILL SPANNER
class TrillSpanner(SubtypeSpanner):
    """A container for Trill spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : str, optional, default: 'trill'
        Subtype of trill
    ornament : str, optional, default: None
        Subtype of ornament associated with the trill
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", str), ("ornament", str)])
    _optional_attributes = ["subtype", "ornament"]

    def __init__(self, duration: int, subtype: str = "trill", ornament: str = None):
        super().__init__(duration = duration, subtype = subtype)
        self.ornament = ornament

# VIBRATO SPANNER
class VibratoSpanner(SubtypeSpanner):
    """A container for Vibrato spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : str, optional, default: 'vibratoSawtooth'
        Subtype of vibrato
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", str)])
    _optional_attributes = ["subtype"]

    def __init__(self, duration: int, subtype: str = "vibratoSawtooth"):
        super().__init__(duration = duration, subtype = subtype)

# GLISSANDO SPANNER
class GlissandoSpanner(SubtypeSpanner):
    """A container for Glissando spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    is_wavy : bool, optional, default: False
        Is this Glissano wavy (True) or straight (False)?
    subtype : str, optional, default: 'straight'
        Derived from is_wavy.

    """

    _attributes = OrderedDict([("duration", int), ("is_wavy", bool)])
    _optional_attributes = ["is_wavy"]

    def __init__(self, duration: int, is_wavy: bool = False):
        self.is_wavy = bool(is_wavy)
        super().__init__(duration = duration, subtype = "wavy" if self.is_wavy else "straight")

# OTTAVA SPANNER
class OttavaSpanner(SubtypeSpanner):
    """A container for Ottava spanners.

    Attributes
    ----------
    duration : int
        Duration of spanner, in time steps
    subtype : str, optional, default: '8va'
        Subtype of ottava
    
    """

    _attributes = OrderedDict([("duration", int), ("subtype", str)])
    _optional_attributes = ["subtype"]

    def __init__(self, duration: int, subtype: str = "8va"):
        super().__init__(duration = duration, subtype = subtype)

##################################################

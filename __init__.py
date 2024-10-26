# README
# Phillip Long
# August 28, 2024

# initializer
# Initialize certain functions and classes so they are easier to access.

# to help load stuff properly
from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
del dirname, realpath

# MusicRender object
from reading.music import MusicRender

# load JSON file as MusicRender object
from reading.music import load

# load MuseScore file as MusicRender object
from reading.read_musescore import read_musescore

# load MusicXML file as MusicRender object
from reading.read_musicxml import read_musicxml

# README
# Phillip Long
# August 28, 2024

# initializer
# Initialize certain functions and classes so they are easier to access.

# to help load stuff properly
from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

# MusicRender object
from reading.music import MusicRender

# load JSON file as MusicRender object
from reading.music import load

# load MSCZ file as MusicRender object
from reading.read_musescore import read_musescore

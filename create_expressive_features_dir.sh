#!/bin/bash

# README
# Phillip Long
# October 15, 2023

# create directory + subdirectories for storing pickled extracted features files

# sh /home/pnlong/parse_musescore/create_expressive_features_dir.sh

# some varialbes
zachdir="/data2/zachary/musescore/data"
mydir="/data2/pnlong/musescore/expressive_features"
basedir="$(echo "$zachdir" | cut -d'/' -f2)"

# get list of zach's subdirectories and copy them into my directory
find $zachdir -type d | sed -e "s+$zachdir/++" | xargs -I{} mkdir -p "$mydir/{}" 

# remove wierd glitchy directory
rm -rf "$mydir/$basedir"
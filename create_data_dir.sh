#!/bin/bash

# README
# Phillip Long
# October 15, 2023

# create directory + subdirectories for storing data; copy Zach's file architecture

# sh /home/pnlong/model_musescore/create_data_dir.sh

# constants
datadir="/data2/pnlong/musescore/data" # default value for datadir
zachdir="/data2/zachary/musescore/data"
basedir="$(echo "$zachdir" | cut -d'/' -f2)"

# parse command line arguments
while getopts ':f:h' opt; do
  case "$opt" in
    f)
      datadir="$OPTARG"
      ;;
    h)
      echo "Usage: $(basename $0) [-f]"
      exit 0
      ;;
    :)
      echo -e "option requires an argument.\nUsage: $(basename $0) [-f]"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\nUsage: $(basename $0) [-f]"
      exit 1
      ;;
  esac
done

# notify what we are using for datadir
echo "Creating directories in $datadir"

# get list of zach's subdirectories and copy them into my directory
find $zachdir -type d | sed -e "s+$zachdir/++" | xargs -I{} mkdir -p "$datadir/{}" 

# remove wierd glitchy directory
rm -rf "$datadir/$basedir"
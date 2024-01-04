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
usage="Usage: $(basename $0) [-f] (the directory in which to create the directory structure)"

# parse command line arguments
while getopts ':f:h' opt; do
  case "$opt" in
    f)
      datadir="$OPTARG"
      ;;
    h)
      echo $usage
      exit 0
      ;;
    :)
      echo -e "option requires an argument.\n$usage"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\n$usage"
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
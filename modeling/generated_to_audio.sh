#!/bin/bash

# README
# Phillip Long
# September 5, 2024

# Generate a lot of audio samples to see how generated content sounds.

# sh /home/pnlong/model_musescore/modeling/generated_to_audio.sh

# CONSTANTS
##################################################

# software filepaths
software_dir=$(dirname "${0}")
software_base=${0##*/}
software="${software_dir}/${software_base%.sh}.py"

# data dir
data_dir="/home/pnlong/musescore/experiments"
i=0 # index

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-i] (index)"
while getopts ':d:i:' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
      ;;
    i)
      i="${OPTARG}"
      ;;
    h)
      echo ${usage}
      exit 0
      ;;
    :)
      echo -e "Option requires an argument.\n${usage}"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\n${usage}"
      exit 1
      ;;
  esac
done

##################################################


# GENERATE AUDIO
##################################################

for facet in "all" "rated" "deduplicated" "rated_deduplicated" "random"; do
    for model_dir in $(ls -d ${data_dir}/${facet}/*/); do
        python ${software} --path "${model_dir}eval/${i}.npy"
    done
done

##################################################
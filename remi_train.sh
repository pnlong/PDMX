#!/bin/bash

# README
# Phillip Long
# August 4, 2024

# A place to store the command-line inputs for different REMI-models to train.

# sh /home/pnlong/model_musescore/remi_train.sh

# CONSTANTS
##################################################

# software filepaths
software_dir="/home/pnlong/model_musescore"
software="${software_dir}/remi_train.py"

# defaults
data_dir="/home/pnlong/musescore/remi/all"
gpu=-1 # gpu number
resume=""

# small model architecture, the default
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-r] (resume?) [-sml] (small/medium/large) [-g] (gpu to use)"
while getopts ':d:g:rsmlh' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
      ;;
    r)
      resume="--resume -1"
      ;;
    s)
      dim=512 # dimension
      layers=6 # layers
      heads=8 # attention heads
      ;;
    m)
      dim=768 # dimension
      layers=10 # layers
      heads=8 # attention heads
      ;;
    l)
      dim=960 # dimension
      layers=12 # layers
      heads=12 # attention heads
      ;;
    g)
      gpu="${OPTARG}"
      ;;
    h)
      echo ${usage}
      exit 0
      ;;
    :)
      echo -e "option requires an argument.\n${usage}"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\n${usage}"
      exit 1
      ;;
  esac
done

# filepaths
paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
output_dir="${data_dir}"

# constants
batch_size=12 # decrease if gpu memory consumption is too high
steps=100000 # in my experience >70000 is sufficient to train

##################################################


# TRAIN MODEL
##################################################

set -e # stop if there's an error

# run program
python ${software} --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${resume}

##################################################


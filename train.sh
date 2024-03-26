#!/bin/bash

# README
# Phillip Long
# January 4, 2023

# A place to store the command-line inputs for different models to train.

# sh /home/pnlong/model_musescore/train.sh

# CONSTANTS
##################################################

# software filepaths
software_dir="/home/pnlong/model_musescore"
software="${software_dir}/train.py"

# defaults
data_dir="/home/pnlong/musescore/datav"
gpu=-1 # gpu number
unidimensional=""
resume=""

# small model architecture, the default
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-u] (unidimensional?) [-r] (resume?) [-sml] (small/medium/large) [-g] (gpu to use)"
while getopts ':d:g:ursmlh' opt; do
  case "${opt}" in
    d) # also implies metrical/absolute time
      data_dir="${OPTARG}"
      ;;
    u) # unidimensional flag
      unidimensional="--unidimensional"
      ;;
    r) # whether to resume runs
      resume="--resume -1"
      ;;
    s) # small
      dim=512 # dimension
      layers=6 # layers
      heads=8 # attention heads
      ;;
    m) # medium
      dim=768 # dimension
      layers=10 # layers
      heads=8 # attention heads
      ;;
    l) # large
      dim=960 # dimension
      layers=12 # layers
      heads=12 # attention heads
      ;;
    g) # gpu to use
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

paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
encoding="${data_dir}/encoding.json"
output_dir="${data_dir}"

# constants
batch_size=12 # decrease if gpu memory consumption is too high
steps=100000 # in my experience >70000 is sufficient to train
sigma=8 # for anticipation, in seconds or beats depending on which time scale we are using

##################################################


# TRAIN MODELS
##################################################

set -e # stop if there's an error

# baseline
python ${software} --baseline --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}

# prefix, conditional
python ${software} --conditioning "prefix" --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}

# anticipation, conditional
python ${software} --conditioning "anticipation" --sigma ${sigma} --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}

# prefix, not conditional
python ${software} --conditioning "prefix" --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}

# anticipation, not conditional
python ${software} --conditioning "anticipation" --sigma ${sigma} --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}

##################################################


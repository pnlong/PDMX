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

# data filepaths
data_dir="/data2/pnlong/musescore/data"
gpu=3 # gpu number

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-g] (gpu to use)"
while getopts ':d:g:h' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
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

paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
encoding="${data_dir}/encoding.json"
output_dir="${data_dir}"

# constants
batch_size=4 # decrease if gpu memory consumption is too high
steps=100000 # in my experience >70000 is sufficient to train

# to adjust the size of the model (number of parameters, adjust these)
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# NOT CONDITIONAL ON NOTES
##################################################

set -e

# baseline
python ${software} --baseline --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# prefix
python ${software} --conditioning "prefix" --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# anticipation
python ${software} --conditioning "anticipation" --aug --sigma 5 --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

##################################################


# CONDITIONAL ON NOTES
##################################################

# baseline is not possible to be conditional, because it is just notes

# prefix
python ${software} --conditioning "prefix" --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# anticipation
python ${software} --conditioning "anticipation" --sigma 5 --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

##################################################
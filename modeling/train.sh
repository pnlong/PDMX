#!/bin/bash

# README
# Phillip Long
# August 4, 2024

# A place to store the command-line inputs for different REMI-models to train.

# sh /home/pnlong/model_musescore/modeling/train.sh

# CONSTANTS
##################################################

# software filepaths
software_dir="/home/pnlong/model_musescore"
software="${software_dir}/modeling/train.py"

# defaults
base_dir="/home/pnlong/musescore/remi"
fine_tuning_facet_name="rated_deduplicated-4.0"
data_dir="${base_dir}/all"
gpu=-1 # gpu number
resume=""
fine_tune=""

# small model architecture, the default
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-r] (resume?) [-f] (fine tune?) [-sml] (small/medium/large) [-g] (gpu to use)"
while getopts ':d:g:rfsmlh' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
      ;;
    r)
      resume="--resume -1"
      ;;
    f)
      fine_tune="--fine_tune"
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
output_dir="${data_dir}"

# conditional
if [ -z "${fine_tune}" ]; then # normal training
  paths_train="${data_dir}/train.txt"
  paths_valid="${data_dir}/valid.txt"
  steps=100000 # in my experience >70000 is sufficient to train
  learning_rate=0.0005
else # fine-tuning
  paths_train="${base_dir}/${fine_tuning_facet_name}/train.txt"
  paths_valid="${base_dir}/${fine_tuning_facet_name}/valid.txt"
  steps=5000 # less steps needed for fine tuning
  learning_rate=0.00005
fi

##################################################


# TRAIN MODEL
##################################################

set -e # stop if there's an error

# run program
python ${software} --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --output_dir ${output_dir} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --learning_rate ${learning_rate} --gpu ${gpu} ${fine_tune} ${resume}

##################################################


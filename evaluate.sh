#!/bin/bash

# README
# Phillip Long
# January 7, 2023

# A place to store the command-line inputs for different evaluations.

# sh /home/pnlong/model_musescore/evaluate.sh


# CONSTANTS
##################################################

software_dir="/home/pnlong/model_musescore"
data_dir="/data2/pnlong/musescore/data"
gpu=3
model="truth"

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-m] (the model to evaluate) [-d] (data directory) [-g] (gpu to use)"
while getopts ':m:d:g:h' opt; do
  case "${opt}" in
    m)
      model="${OPTARG}"
      ;;
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

trained_models="${data_dir}/models.txt"
paths_test="${data_dir}/test.txt"
encoding="${data_dir}/encoding.json"
output_dir="${data_dir}"
n_samples=1000
batch_size=8

##################################################


# EVALUATE
##################################################

set -e

printf "============================   ${model^^}   ============================\n"

if [[ "${model}" == "${default_model}" ]]; then

    python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu} --batch_size ${batch_size}
    python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu} --batch_size ${batch_size}

else

    # python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu} --batch_size ${batch_size}
    python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu} --batch_size ${batch_size}

fi

##################################################

#!/bin/bash

# README
# Phillip Long
# January 7, 2023

# A place to store the command-line inputs for different evaluations.

# sh /home/pnlong/model_musescore/evaluate.sh


# CONSTANTS
##################################################

software_dir="/home/pnlong/model_musescore"
base_dir="/data2/pnlong/musescore"
data_dir="${base_dir}/data"
trained_models="${data_dir}/models.txt"
paths_test="${data_dir}/test.txt"
encoding="${base_dir}/encoding.json"
output_dir="${data_dir}"
n_samples=2000
gpu=3

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-m] (the model to evaluate)"
default_model="truth"
model=${default_model}
while getopts ':m:h' opt; do
  case "${opt}" in
    m)
      model="${OPTARG}"
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
##################################################

# EVALUATE
##################################################

printf "============================   ${model^^}   ============================\n"

if [[ ${model} == ${default_model} ]]; then

    # python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu}
    python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu}

else

    # python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}
    python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}

fi

##################################################

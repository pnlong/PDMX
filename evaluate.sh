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


# EVALUATE
##################################################

printf "==================================================\n\n   Truth\n\n==================================================\n"

python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu}
# python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --n_samples ${n_samples} --truth --gpu ${gpu}

for model in $(cat "${trained_models}"); do

    printf "==================================================\n\n   ${model}\n\n==================================================\n"

    # evalute same as mmt (baseline)
    python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}
    
    # evaluate for the distribution of expressive features for joint and conditional models
    # python ${software_dir}/evaluate.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}
    
done

##################################################

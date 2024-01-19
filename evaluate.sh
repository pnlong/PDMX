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
n_samples=8
gpu=3

##################################################


# EVALUATE
##################################################

for model in $(cat "${trained_models}"); do

    # evalute same as mmt (baseline)
    python ${software_dir}/evaluate_baseline.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}
    
    # evaluate as a joint model
    python ${software_dir}/evaluate_joint.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}
    
    # evaluate as a conditional model
    python ${software_dir}/evaluate_conditional.py --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/${model}" --n_samples ${n_samples} --gpu ${gpu}

done

##################################################

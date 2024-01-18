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
paths_test="${data_dir}/test.txt"
encoding="${base_dir}/encoding.json"
output_dir="${data_dir}"
n_samples=8
gpu=3

##################################################


# EVALUATE SAME AS MMT
##################################################

# variables
software="${software_dir}/evaluate_baseline.py"

# baseline
python ${software} --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/baseline_aug" --n_samples ${n_samples} --gpu ${gpu}

# sort-order

# prefix

# anticipation

##################################################


# EVALUATE AS A JOINT MODEL
##################################################

# variables
software="${software_dir}/evaluate_joint.py"

# baseline
python ${software} --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/baseline_aug" --n_samples ${n_samples} --gpu ${gpu}

# sort-order

# prefix

# anticipation

##################################################


# EVALUATE AS A CONDITIONAL MODEL
##################################################

# variables
software="${software_dir}/evaluate_conditional.py"

# baseline
python ${software} --paths ${paths_test} --encoding ${encoding} --output_dir "${output_dir}/baseline_aug" --n_samples ${n_samples} --gpu ${gpu}

# sort-order

# prefix

# anticipation

##################################################
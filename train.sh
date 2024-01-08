#!/bin/bash

# README
# Phillip Long
# January 4, 2023

# A place to store the command-line inputs for different models to train.

# sh /home/pnlong/model_musescore/train.sh

# constants
software_dir="/home/pnlong/model_musescore"
software="${software_dir}/train.py"
base_dir="/data2/pnlong/musescore"
data_dir="${base_dir}/data"
paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
encoding="${base_dir}/encoding.json"
output_dir="${data_dir}"
batch_size=8
gpu=3

# baseline
python ${software} --baseline --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --gpu ${gpu} --resume

# sort-order
python ${software} --conditioning "sort" --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --gpu ${gpu} --resume

# prefix
python ${software} --conditioning "prefix" --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --gpu ${gpu} --resume

# anticipation
python ${software} --conditioning "anticipation" --sigma 5 --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --gpu ${gpu} --resume

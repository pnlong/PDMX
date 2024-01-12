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
base_dir="/data2/pnlong/musescore"
data_dir="${base_dir}/data"
paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
encoding="${base_dir}/encoding.json"
output_dir="${data_dir}"

# constants
batch_size=8 # decrease if gpu memory consumption is too high
gpu=3 # gpu number
steps=80000 # in my experience >70000 is sufficient to train

# to adjust the size of the model (number of parameters, adjust these)
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# NOT CONDITIONAL ON NOTES
##################################################

# baseline
python ${software} --baseline --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# sort-order
python ${software} --conditioning "sort" --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# prefix
python ${software} --conditioning "prefix" --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# anticipation
python ${software} --conditioning "anticipation" --sigma 5 --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

##################################################


# CONDITIONAL ON NOTES
##################################################

# baseline
python ${software} --baseline --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# sort-order
python ${software} --conditioning "sort" --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# prefix
python ${software} --conditioning "prefix" --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

# anticipation
python ${software} --conditioning "anticipation" --sigma 5 --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu}

##################################################
#!/bin/bash

# README
# Phillip Long
# October 15, 2023

# create directory + subdirectories for storing data; copy Zach's file architecture

# sh /home/pnlong/model_musescore/create_data_dir.sh

# constants
data_dir="/data2/pnlong/musescore/data" # default value for data_dir
zach_dir="/data2/zachary/musescore/data"
base_dir="$(echo ${zach_dir} | cut -d'/' -f2)"
usage="Usage: $(basename ${0}) [-d] (the directory in which to create the directory structure)"

# parse command line arguments
while getopts ':d:h' opt; do
  case "${opt}" in
    d)
      data_dir="${OPTARG}"
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

# notify what we are using for data_dir
echo "Creating directories in ${data_dir}."

# get list of zach's subdirectories and copy them into my directory
find ${zach_dir} -type d | sed -e "s+${zach_dir}/++" | xargs -I{} mkdir -p "${data_dir}/{}" 

# remove wierd glitchy directory
rm -rf "${data_dir}/${base_dir}"
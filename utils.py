# README
# Phillip Long
# November 1, 2023

# Copied from Herman's MMT: https://github.com/salu133445/mmt/blob/main/mmt/utils.py
# Contains utility (helper) functions


# IMPORTS
##################################################

import json
import pathlib
import warnings

import numpy as np

##################################################


# CONSTANTS
##################################################

DATA_DIRECTORY = "/data2/pnlong/mmt/" # needs to have / at end

##################################################


# MISCELLANEOUS FUNCTIONS
##################################################

def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}

##################################################


# SAVING AND LOADING FILES
##################################################

def save_args(filename, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    save_json(filename = filename, data = args_dict)


def save_txt(filename, data):
    """Save a list to a TXT file."""
    with open(filename, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    """Load a TXT file as a list."""
    with open(filename, encoding = "utf8") as f:
        return [line.strip() for line in f]


def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding = "utf8") as f:
        json.dump(obj = data, fp = f)


def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding = "utf8") as f:
        return json.load(fp = f)


def save_csv(filename, data, header = ""):
    """Save data as a CSV file."""
    np.savetxt(fname = filename, X = data, fmt = "%d", delimiter = ",", header = header, comments = "")


def load_csv(filename, skiprows = 1):
    """Load data from a CSV file."""
    return np.loadtxt(fname = filename, dtype = int, delimiter = ",", skiprows = skiprows)

##################################################



# DECORATORS
##################################################

def ignore_exceptions(func):
    """Decorator that ignores all errors and warnings."""
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return func(*args, **kwargs)
            except Exception:
                return None
    return inner

##################################################
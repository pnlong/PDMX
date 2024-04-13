# README
# Phillip Long
# April 12, 2024

# Get perplexity values for models.

# python /home/pnlong/model_musescore/perplexity.py


# IMPORTS
##################################################

from os.path import basename
import pandas as pd
from utils import rep
import train
from math import exp

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/home/pnlong/musescore"

##################################################


# HELPER FUNCTIONS
##################################################

def get_fancy_model_name(model_filepath: str) -> str:
    """Get fancy version of model name."""
    if "baseline" in model_filepath:
        model = "Baseline"
    else:
        model = ("Conditional" if "conditional" in model_filepath else "Joint") + ", " + basename(model_filepath).split("_")[0].title()
    return model + " ($\it{" + ("A" if "datava" in model_filepath else "M") + "}$)"

##################################################

# MAIN METHOD
##################################################
if __name__ == "__main__":

    # CREATE FIGURE
    ##################################################

    # constants
    metric = "loss"

    model_filepaths = []
    model_sorter = lambda model: int("baseline" not in model) + (2 * int("conditional" in model)) + int("prefix" not in model) # get index to sort the models
    for dir in ("datav", "datava"):
        with open(f"{INPUT_DIR}/{dir}/models.txt", "r") as models_output: # read in list of trained models
            models = [model.strip() for model in models_output.readlines()] # use a set because better for `in` operations
            models = filter(lambda model: any(size in model for size in ("20M", "31M")) and ("econditional" not in model) and ("unidimensional" not in model), models)
            models = sorted(models, key = model_sorter)
            model_filepaths += list(map(lambda model: f"{INPUT_DIR}/{dir}/{model}", models))
    
    # loop through models
    table_performance_columns = ["model", "value"]
    table_performance = pd.DataFrame(columns = table_performance_columns)
    for model_filepath in model_filepaths:

        # load in performance
        performance = pd.read_csv(filepath_or_buffer = f"{model_filepath}/performance.csv", sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in full performance
        performance = performance[(performance["partition"] == "valid") & (performance["metric"] == "loss") & (performance["mask"] == "note") & (performance["field"] == "total") & (performance["step"] == max(performance["step"]))]
        performance["model"] = rep(x = model_filepath, times = len(performance))
        table_performance = pd.concat(objs = (table_performance, performance[table_performance_columns]), axis = 0, ignore_index = True)

    # get orderings for table
    table_performance = table_performance.set_index(keys = "model", drop = True)["value"]
    table_performance = table_performance.apply(exp) # perplexity
    for model in table_performance.index:
        print(f"{get_fancy_model_name(model_filepath = model)} & {table_performance[model]:.2f} \\\\")

    ##################################################
    
##################################################

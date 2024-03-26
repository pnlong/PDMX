# README
# Phillip Long
# October 16, 2023

# Make plots to describe the expressive feature extraction

# python /home/pnlong/model_musescore/parse_mscz_plots.py


# IMPORTS
##################################################

import pandas as pd
from numpy import percentile, log10, arange
import matplotlib.pyplot as plt
from os.path import exists
from os import makedirs
import multiprocessing
import argparse
import logging
from time import strftime, gmtime
from read_mscz.music import DIVIDE_BY_ZERO_CONSTANT
from utils import rep
from parse_mscz import LIST_FEATURE_JOIN_STRING

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore/expressive_features"
OUTPUT_DIR = "/data2/pnlong/musescore/expressive_features/plots"
OUTPUT_RESOLUTION_DPI = 200
BAR_SHIFT_CONSTANT = 13/25

COLORS = ("#9BC1BC", "#F4F1BB", "#ED6A5A") # color palette copied from https://coolors.co/palettes/popular/3%20colors
LINE_COLORS = ("tab:blue", "tab:red", "tab:green", "tab:orange")

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore Figures", description = "Make plots describing MuseScore dataset.")
    parser.add_argument("-i", "--input_dir", type = str, default = INPUT_DIR, help = "Directory that contains all data tables to be summarized (or where they will be created)")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# DESCRIBE VERSIONS
##################################################

version_labels_mapping = ["1", "2", "3", chr(10006)]
version_types_mapping = ["All Files", "All Tracks", "Invalid Files"]

# helper function to group by version
def _group_by_version(df: pd.DataFrame) -> pd.DataFrame:
    df["version"] = df["version"].apply(lambda version: str(version).strip()[0] if not pd.isna(version) else version_labels_mapping[-1]) # switch out to base version
    df = df.groupby(by = "version").size() # sum over each version
    df = df.reset_index().rename(columns = {0: "count"}) # make error type into column
    df["percent"] = 100 * (df["count"] / df["count"].sum()) # calculate percentage
    return df[["version", "count", "percent"]] # select only subset of columns

def _make_versions_bar_chart(axes: plt.Axes, df: pd.DataFrame, col: str):
    axes[col].barh(width = df["count"], y = df["version"], color = COLORS[version_types_mapping.index(col)], edgecolor = "0") # make bar chart
    axes[col].set_title(col)
    axes[col].set_xlabel("Count")
    axes[col].ticklabel_format(axis = "x", style = "scientific", scilimits = (0, 0))
    axes[col].set_ylabel("Version")
    axes[col].invert_yaxis()

# function to make the versions plot
def make_versions_plot(output_filepath: str):

    versions_columns = ["version", "is_valid"]

    # create figure
    plt.set_loglevel("WARNING") # to avoid Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
    fig, axes = plt.subplot_mosaic(mosaic = [[version_types_mapping[0], version_types_mapping[1]], [version_types_mapping[2], "bar_all"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Distribution of Musescore Versions", fontweight = "bold")
    bar_data = {
        version_types_mapping[0]: _group_by_version(df = data_by["path"][versions_columns].copy()),
        version_types_mapping[1]: _group_by_version(df = data_by["track"][versions_columns].copy()),
        version_types_mapping[2]: _group_by_version(df = data_by["track"][~data_by["track"]["is_valid"]][versions_columns].reset_index(drop = True).copy())
    }
    width = 0.2

    # make bar charts with loop, calling function
    offset = -width
    for version_type_mapping in version_types_mapping:

        # single bar chart
        _make_versions_bar_chart(axes = axes, df = bar_data[version_type_mapping], col = version_type_mapping)

        # all bar chart
        axes["bar_all"].bar(x = bar_data[version_type_mapping]["version"].apply(lambda version: version_labels_mapping.index(version)) + offset,  height = bar_data[version_type_mapping]["percent"],  width = width, color = COLORS[version_types_mapping.index(version_type_mapping)], edgecolor = "0")

        # update offset
        offset += width
    
    # final touches on all bar
    axes["bar_all"].set_xlabel("Version")
    axes["bar_all"].xaxis.set_ticks(ticks = tuple(range(len(version_labels_mapping))), labels = version_labels_mapping)
    axes["bar_all"].set_ylabel("Percentage (%)") 
    axes["bar_all"].set_title("Comparison")
    axes["bar_all"].legend(["Paths", "Tracks", "Errors"]) 

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Versions plot saved to {output_filepath}.")

##################################################


# DESCRIBE ERRORS
##################################################

# function to make the errors plot
def make_error_plot(input_filepath: str, output_filepath: str):

    errors = pd.read_csv(filepath_or_buffer = input_filepath, sep = ",", header = 0, index_col = False)

    n_errors = len(errors)
    n = len(data_by["path"])
    error_rate = n_errors / n

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["bar",]], constrained_layout = True, figsize = (8, 8))
    fig.suptitle("Errors in MuseScore Data", fontweight = "bold")

    # make bar chart
    errors = errors.groupby(by = "error_type").size() # sum over error type
    errors = errors.reset_index().rename(columns = {0: "n"}) # make error type into column
    errors = errors[["error_type", "n"]].sort_values("n")
    errors["error_type"] = errors["error_type"].apply(lambda error_type: error_type.split("_")[0].title()) # make error type look nicer
    axes["bar"].barh(width = errors["n"], y = errors["error_type"], color = COLORS[0], edgecolor = "0") # make bar chart
    axes["bar"].set_title(f"Total Error Rate: {n_errors:,} / {n:,} ; {100 * error_rate:.2f}%")
    axes["bar"].set_xlabel("Count")
    axes["bar"].ticklabel_format(axis = "x", style = "scientific", scilimits = (0, 0))
    axes["bar"].set_ylabel("Error Type")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Errors plot saved to {output_filepath}.")

##################################################


# PERCENTILE PLOT FOR NUMBER OF EXPRESSIVE FEATURES
##################################################

legend_values = [f"{collection_type} {tp}" for tp in ("Tracks", "Paths") for collection_type in ("All", "Public Domain")] # ['All Tracks', 'Public Domain Tracks', 'All Paths', 'Public Domain Paths']

def make_percentile_plot(output_filepath: str):

    # get percentiles
    step = 0.001
    percentiles = arange(start = 0, stop = 100 + step, step = step)
    percentile_values = {
        legend_values[0]: percentile(a = data_by["track"][data_by["track"]["is_valid"]]["n_expressive_features"], q = percentiles),
        legend_values[1]: percentile(a = data_by["track"][data_by["track"]["is_valid"] & data_by["track"]["is_public_domain"]]["n_expressive_features"], q = percentiles),
        legend_values[2]: percentile(a = data_by["path"][data_by["path"]["is_valid"] & (data_by["path"]["n_expressive_features"] >= 0)]["n_expressive_features"], q = percentiles),
        legend_values[3]: percentile(a = data_by["path"][data_by["path"]["is_valid"] & data_by["path"]["is_public_domain"] & (data_by["path"]["n_expressive_features"] >= 0)]["n_expressive_features"], q = percentiles)
        }
    df = pd.concat(objs = [pd.DataFrame(data = {"type": rep(x = legend_value, times = len(percentiles)), "percentile": percentiles, "log": log10(percentile_values[legend_value] + DIVIDE_BY_ZERO_CONSTANT)}) for legend_value in legend_values], axis = 0)

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["log",]], constrained_layout = True, figsize = (8, 8))
    fig.suptitle("Number of Expressive Features in MuseScore Data", fontweight = "bold")

    # make plots
    for i, legend_value in enumerate(legend_values):
        df_sub = df[df["type"] == legend_value]
        axes["log"].plot(df_sub["percentile"], df_sub["log"], label = legend_value, color = LINE_COLORS[i % 2], linestyle = "solid" if i < 2 else "dashed")
    axes["log"].set_xlabel("Percentile (%)")
    axes["log"].set_ylabel("Number of Expressive Features")
    logticks = list(range(int(min(df["log"])), int(max(df["log"])) + 1, 2))
    axes["log"].set_yticks(logticks)
    axes["log"].set_yticklabels(["$10^{" + str(logtick) + "}$" if logtick != 0 else "1" for logtick in logticks])
    axes["log"].legend(ncol = 2)
    axes["log"].grid()

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Percentiles plot saved to {output_filepath}.")

##################################################


# DESCRIBE TIMINGS
##################################################

def make_timing_plot(input_filepath: str, output_filepath: str):

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["time",]], constrained_layout = True, figsize = (8, 8))
    fig.suptitle("Timings to Parse MuseScore Data", fontweight = "bold")

    # load in timings
    with open(input_filepath, "r") as timing_file:
        timings = timing_file.readlines()
    timings = tuple(float(timing.strip()) for timing in timings) # convert to floats
    total_time = strftime("%H:%M:%S", gmtime(sum(timings)))

    # create plot
    bin_width = 0.005
    bin_range = (0, 0.2)
    axes["time"].hist(x = timings, bins = arange(start = bin_range[0], stop = bin_range[1] + bin_width, step = bin_width), color = COLORS[0], edgecolor = "0")
    axes["time"].set_xlabel("Time (seconds)")
    axes["time"].set_ylabel("Count")
    axes["time"].ticklabel_format(axis = "y", style = "scientific", scilimits = (0, 0))
    axes["time"].set_title(f"Total Time: {total_time}")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Timings plot saved to {output_filepath}.")

##################################################


# NUMBER OF TRACKS PER PATH
##################################################

def make_tracks_plot(output_filepath: str):

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["box", "hist", "hist"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Number of Tracks per MuseScore File", fontweight = "bold")

    tracks_data = data_by["path"][data_by["path"]["is_valid"]]["n_tracks"].tolist() # data_by["track"][data_by["track"]["is_valid"]].groupby(by = "path").size().tolist()
    upper_limit_of_interest = 50

    # boxplot
    axes["box"].boxplot(x = tracks_data, vert = True, showfliers = False)
    # axes["box"].violinplot(dataset = [tracks_data], vert = True, showextrema = False, quantiles = [[0.25, 0.5, 0.75]])
    # axes["box"].set_ylim(bottom = 0, top = upper_limit_of_interest)
    axes["box"].set_xlabel("")
    axes["box"].xaxis.set_ticks(ticks = [], labels = [])
    axes["box"].set_ylabel("Number of Tracks")

    # histogram
    binwidth = 5
    axes["hist"].hist(x = tracks_data, bins = range(0, int(max(tracks_data)) + binwidth, binwidth), color = COLORS[0], edgecolor = "0")
    axes["hist"].set_xlim(left = 0, right = upper_limit_of_interest)
    axes["hist"].set_xlabel("Number of Tracks")
    axes["hist"].set_ylabel("Count")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Tracks plot saved to {output_filepath}.")

##################################################


# MAKE ANY BOOLEAN PLOT
##################################################

labels_mapping = ["path", "track"]

# helper function to group by the boolean column
def _group_by_boolean(df: pd.DataFrame, label: str, boolean_column_name: str) -> pd.DataFrame:
    df = df.groupby(by = boolean_column_name).size() # sum over each version
    df = df.reset_index().rename(columns = {0: "count"}) # make error type into column
    df["percent"] = 100 * (df["count"] / df["count"].sum()) # calculate percentage
    df["type"] = rep(x = labels_mapping.index(label), times = len(df))
    return df[[boolean_column_name, "count", "percent", "type"]] # select only subset of columns

def _make_boolean_bar_chart(axes: plt.Axes, df: pd.DataFrame, type_column: str, boolean_column_name: str, fancy_boolean_column_name: str):
    width = 0.4
    pro_user_true, pro_user_false = df[df[boolean_column_name]], df[~df[boolean_column_name]]
    bars = [
        axes[type_column].bar(x = pro_user_true["type"] - (BAR_SHIFT_CONSTANT * width),  height = pro_user_true[type_column],  width = width, color = COLORS[0], edgecolor = "0"), # pro user
        axes[type_column].bar(x = pro_user_false["type"] + (BAR_SHIFT_CONSTANT * width), height = pro_user_false[type_column], width = width, color = COLORS[1], edgecolor = "0") # not pro user
        ]
    def annotate_bar_chart(bars_to_annotate):
        for bar in bars_to_annotate: # loop through the bars and add annotations
            height = bar.get_height()
            axes[type_column].annotate(text = f"{height:.2f}%" if (type_column == "percent") else f"{height:,}", xy = (bar.get_x() + bar.get_width() / 2, height), xytext = (0, 3), textcoords = "offset points", ha = "center", va = "bottom")
    annotate_bar_chart(bars_to_annotate = bars[0])
    annotate_bar_chart(bars_to_annotate = bars[1])
    axes[type_column].set_xlabel("Type")
    axes[type_column].xaxis.set_ticks(ticks = tuple(range(len(labels_mapping))), labels = (label_mapping.title() for label_mapping in labels_mapping))
    axes[type_column].set_ylabel(f"{type_column.title()} (%)" if (type_column == "percent") else type_column.title())
    axes[type_column].set_title(type_column.title())
    axes[type_column].legend([fancy_boolean_column_name, f"Not {fancy_boolean_column_name}"]) 

# function to make any boolean plot
def make_boolean_plot(boolean_column_name: str, output_filepath: str):

    # some helper variables
    fancy_boolean_column_name = " ".join(map(lambda word: word.title(), boolean_column_name.split("_")[1:]))
    relevant_columns = [boolean_column_name, "is_valid"]

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["count", "percent"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle(f"{fancy_boolean_column_name} in MuseScore Data", fontweight = "bold")
    data = pd.concat(objs = (
        _group_by_boolean(df = data_by[labels_mapping[0]][relevant_columns].copy(), label = labels_mapping[0], boolean_column_name = boolean_column_name),
        _group_by_boolean(df = data_by[labels_mapping[1]][relevant_columns].copy(), label = labels_mapping[1], boolean_column_name = boolean_column_name)
        ), axis = 0)

    # make count plot
    _make_boolean_bar_chart(axes = axes, df = data, type_column = "count", boolean_column_name = boolean_column_name, fancy_boolean_column_name = fancy_boolean_column_name)
    axes["count"].ticklabel_format(axis = "y", style = "scientific", scilimits = (0, 0))

    # make percent plot
    _make_boolean_bar_chart(axes = axes, df = data, type_column = "percent", boolean_column_name = boolean_column_name, fancy_boolean_column_name = fancy_boolean_column_name)

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"{fancy_boolean_column_name} plot saved to {output_filepath}.")

# helper to make public domain plot
def make_public_domain_plot(output_filepath: str):
    make_boolean_plot(boolean_column_name = "is_public_domain", output_filepath = output_filepath)

# helper to make pro user plot
def make_pro_user_plot(output_filepath: str):
    make_boolean_plot(boolean_column_name = "is_user_pro", output_filepath = output_filepath)

##################################################


# MAKE COMPLEXITY PLOT
##################################################

def make_complexity_plot(output_filepath: str):

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["box"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Complexity of MuseScore Data", fontweight = "bold")

    # extract data
    data = [
        data_by["path"][data_by["path"]["is_valid"] & ~pd.isna(data_by["path"]["complexity"])]["complexity"].tolist(),
        data_by["track"][data_by["track"]["is_valid"] & ~pd.isna(data_by["track"]["complexity"])]["complexity"].tolist()
    ]

    axes["box"].boxplot(x = data, vert = True, showfliers = False, labels = list(map(lambda string: string.title(), data_by.keys())))
    axes["box"].set_xlabel("")
    axes["box"].set_ylabel("Complexity")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Complexity plot saved to {output_filepath}.")

##################################################


# MAKE DESCRIPTOR (GENRE OR TAG) PLOT
##################################################

def make_descriptor_plot(descriptor: str, output_filepath: str, top_n: int = 10):

    # create figure
    column_name = f"{descriptor}s"
    fig, axes = plt.subplot_mosaic(mosaic = [["path", "track"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle(f"Top {column_name.title()} Present in MuseScore Data", fontweight = "bold")

    # path
    for plot_type in ["path", "track"]:
        no_descriptor = data_by[plot_type][column_name].apply(lambda sequence: pd.isna(sequence) or (str(sequence) == ""))
        data = data_by[plot_type][~no_descriptor][column_name].apply(lambda sequence: str(sequence).split(LIST_FEATURE_JOIN_STRING)).explode(ignore_index = True)
        data = data.value_counts(sort = True, ascending = False, dropna = True)
        fraction_without_descriptor = sum(no_descriptor) / len(no_descriptor)
        data = data.head(n = top_n)
        axes[plot_type].barh(y = data.index, width = data, log = True)
        axes[plot_type].set_xlabel("Count")
        axes[plot_type].set_ylabel(descriptor.title())
        plot_title = plot_type.title() + (f" ({int(100 * fraction_without_descriptor)}% of {plot_type}s lack a {descriptor.lower()})" if (fraction_without_descriptor > 0) else "")
        axes[plot_type].set_title(plot_title)

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"{column_name.title()} plot saved to {output_filepath}.")

# helper function to make genres plot
def make_genres_plot(output_filepath: str):
    make_descriptor_plot(descriptor = "genre", output_filepath = output_filepath)

# helper function to make tags plot
def make_tags_plot(output_filepath: str):
    make_descriptor_plot(descriptor = "tag", output_filepath = output_filepath)

##################################################
    

# MAKE PLOTS
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # command line arguments
    args = parse_args()
    INPUT_FILEPATH = f"{args.input_dir}/expressive_features.csv"
    ERROR_FILEPATH = f"{args.input_dir}/expressive_features.errors.csv"
    TIMING_FILEPATH = f"{args.input_dir}/expressive_features.timing.txt"
    INPUT_FILEPATH_BY_PATH = f"{args.input_dir}/expressive_features.path.csv"

    # make sure directories exist
    if not exists(args.input_dir):
        makedirs(args.input_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # READ IN DATA FRAME
    ##################################################

    # get path- and track-grouped versions of data;
    data_by = {
        "path" : pd.read_csv(filepath_or_buffer = INPUT_FILEPATH_BY_PATH, sep = ",", header = 0, index_col = False),
        "track": pd.read_csv(filepath_or_buffer = INPUT_FILEPATH, sep = ",", header = 0, index_col = False)
    }

    ##################################################


    # MAKE PLOTS
    ##################################################

    # get plot output filepaths
    plot_output_filepaths = [f"{args.output_dir}/{plot_type}.png" for plot_type in ("versions", "errors", "public_domain", "n_expressive_features_percentiles", "timings", "tracks", "pro_user", "complexity", "genres", "tags")]
    
    # more general plots
    make_versions_plot(output_filepath = plot_output_filepaths[0])
    make_error_plot(input_filepath = ERROR_FILEPATH, output_filepath = plot_output_filepaths[1])
    make_public_domain_plot(output_filepath = plot_output_filepaths[2])
    make_percentile_plot(output_filepath = plot_output_filepaths[3])
    make_timing_plot(input_filepath = TIMING_FILEPATH, output_filepath = plot_output_filepaths[4])

    # filter data to just relevant data points (the actual datasets)
    for key in data_by.keys():
        data_by[key] = data_by[key][data_by[key]["in_dataset"]]
    make_tracks_plot(output_filepath = plot_output_filepaths[5])
    make_pro_user_plot(output_filepath = plot_output_filepaths[6])
    make_complexity_plot(output_filepath = plot_output_filepaths[7])
    make_genres_plot(output_filepath = plot_output_filepaths[8])
    make_tags_plot(output_filepath = plot_output_filepaths[9])

    # get scp download command
    plot_output_filepaths = " ".join([f"deepz:{plot_output_filepath}" for plot_output_filepath in plot_output_filepaths])
    print("".join(("=" for _ in range(100))))
    logging.info("SCP COMMAND:")
    logging.info(" ".join(("scp", plot_output_filepaths)))

    ##################################################


##################################################
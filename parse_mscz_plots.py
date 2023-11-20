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
import pickle
from time import perf_counter
import multiprocessing
import argparse
import logging
from tqdm import tqdm
from time import strftime, gmtime
from read_mscz.music import DIVIDE_BY_ZERO_CONSTANT

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore"
OUTPUT_DIR = "/data2/pnlong/musescore/plots"
OUTPUT_RESOLUTION_DPI = 200

COLORS = ("#9BC1BC", "#F4F1BB", "#ED6A5A") # color palette copied from https://coolors.co/palettes/popular/3%20colors
LINE_COLORS = ("tab:blue", "tab:red", "tab:green", "tab:orange")

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description = "Make plots describing MuseScore dataset")
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
def _group_by_version(df):
    df.loc[:, "version"] = df["version"].apply(lambda version: str(version).strip()[0] if not pd.isna(version) else version_labels_mapping[-1]) # switch out to base version
    df = df.groupby(by = "version").size() # sum over each version
    df = df.reset_index().rename(columns = {0: "count"}) # make error type into column
    df["percent"] = 100 * df["count"] / df["count"].sum() # calculate percentage
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
    fig, axes = plt.subplot_mosaic(mosaic = [[version_types_mapping[0], version_types_mapping[1]], [version_types_mapping[2], "bar_all"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Distribution of Musescore Versions", fontweight = "bold")
    bar_data = {
        version_types_mapping[0]: _group_by_version(df = data_by["path"][versions_columns]),
        version_types_mapping[1]: _group_by_version(df = data_by["track"][versions_columns]),
        version_types_mapping[2]: _group_by_version(df = data_by["track"][~data_by["track"]["is_valid"]][versions_columns].reset_index(drop = True))
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


# DESCRIBE PUBLIC DOMAIN
##################################################

public_domain_labels_mapping = ["path", "track"]

# helper function to group by version
def _group_by_copyright(df, label):
    df = df.groupby(by = "is_public_domain").size() # sum over each version
    df = df.reset_index().rename(columns = {0: "count"}) # make error type into column
    df["percent"] = 100 * df["count"] / df["count"].sum() # calculate percentage
    df["type"] = [public_domain_labels_mapping.index(label)] * len(df)
    return df[["is_public_domain", "count", "percent", "type"]] # select only subset of columns

def _make_pd_bar_chart(axes: plt.Axes, df: pd.DataFrame, col: str):
    width = 0.4
    pd_true, pd_false = df[df["is_public_domain"]], df[~df["is_public_domain"]]
    bars = [
        axes[col].bar(x = pd_true["type"] - (52 * width / 100),  height = pd_true[col],  width = width, color = COLORS[0], edgecolor = "0"), # public domain
        axes[col].bar(x = pd_false["type"] + (52 * width / 100), height = pd_false[col], width = width, color = COLORS[1], edgecolor = "0") # not public domain
        ]
    def annotate_bar_chart(bars_to_annotate):
        for bar in bars_to_annotate: # loop through the bars and add annotations
            height = bar.get_height()
            axes[col].annotate(text = f"{height:.2f}%" if col == "percent" else f"{height:,}", xy = (bar.get_x() + bar.get_width() / 2, height), xytext = (0, 3), textcoords = "offset points", ha = "center", va = "bottom")
    annotate_bar_chart(bars_to_annotate = bars[0])
    annotate_bar_chart(bars_to_annotate = bars[1])
    axes[col].set_xlabel("Type")
    axes[col].xaxis.set_ticks(ticks = tuple(range(len(public_domain_labels_mapping))), labels = (pdlm.title() for pdlm in public_domain_labels_mapping))
    axes[col].set_ylabel(f"{col.title()} (%)" if col == "percent" else col.title())
    axes[col].set_title(col.title())
    axes[col].legend(["Public Domain", "Copyrighted"]) 

# function to make the public domain plot
def make_public_domain_plot(output_filepath: str):

    public_domain_columns = ["is_public_domain", "is_valid"]

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["count", "percent"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Copyrights for MuseScore Data", fontweight = "bold")
    public_domain = pd.concat(objs = (
        _group_by_copyright(df = data_by[public_domain_labels_mapping[0]][public_domain_columns], label = public_domain_labels_mapping[0]),
        _group_by_copyright(df = data_by[public_domain_labels_mapping[1]][public_domain_columns], label = public_domain_labels_mapping[1])
        ), axis = 0)

    # make count plot
    _make_pd_bar_chart(axes = axes, df = public_domain, col = "count")
    axes["count"].ticklabel_format(axis = "y", style = "scientific", scilimits = (0, 0))

    # make percent plot
    _make_pd_bar_chart(axes = axes, df = public_domain, col = "percent")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Public Domain plot saved to {output_filepath}.")

##################################################


# PERCENTILE PLOT FOR NUMBER OF EXPRESSIVE FEATURES
##################################################

legend_values = [f"{collection_type} {tp}" for tp in ("Tracks", "Paths") for collection_type in ("All", "Public Domain")] # ['All Tracks', 'Public Domain Tracks', 'All Paths', 'Public Domain Paths']

def _make_percentile_plot(axes: plt.Axes, df: pd.DataFrame):

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

def make_percentile_plot(output_filepath: str):

    # get percentiles
    step = 0.001
    percentiles = arange(start = 0, stop = 100 + step, step = step)
    percentile_values = {
        legend_values[0]: percentile(a = data_by["track"][data_by["track"]["is_valid"]]["n_expressive_features"], q = percentiles),
        legend_values[1]: percentile(a = data_by["track"][data_by["track"]["is_valid"] & data_by["track"]["is_public_domain"]]["n_expressive_features"], q = percentiles),
        legend_values[2]: percentile(a = data_by["path"][data_by["path"]["is_valid"]]["n_expressive_features"], q = percentiles),
        legend_values[3]: percentile(a = data_by["path"][data_by["path"]["is_valid"] & data_by["path"]["is_public_domain"]]["n_expressive_features"], q = percentiles)
        }
    df = pd.concat(objs = [pd.DataFrame(data = {"type": [legend_value] * len(percentiles), "percentile": percentiles, "log": log10(percentile_values[legend_value] + DIVIDE_BY_ZERO_CONSTANT)}) for legend_value in legend_values], axis = 0)

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [["log",]], constrained_layout = True, figsize = (8, 8))
    fig.suptitle("Number of Expressive Features in MuseScore Data", fontweight = "bold")

    # make plots
    _make_percentile_plot(axes = axes, df = df)

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

    tracks_data = data_by["path"][data_by["path"]["is_valid"]]["size"]
    upper_limit_of_interest = 50

    # boxplot
    axes["box"].boxplot(x = tracks_data, vert = True, showfliers = False)
    # axes["box"].set_ylim(bottom = 0, top = upper_limit_of_interest)
    axes["box"].set_xlabel("")
    axes["box"].xaxis.set_ticks(ticks = [], labels = [])
    axes["box"].set_ylabel("Number of Tracks")

    # histogram
    binwidth = 5
    axes["hist"].hist(x = tracks_data, bins = range(0, max(tracks_data) + binwidth, binwidth), color = COLORS[0], edgecolor = "0")
    axes["hist"].set_xlim(left = 0, right = upper_limit_of_interest)
    axes["hist"].set_xlabel("Number of Tracks")
    axes["hist"].set_ylabel("Count")

    # save image
    fig.savefig(output_filepath, dpi = OUTPUT_RESOLUTION_DPI) # save image
    logging.info(f"Tracks plot saved to {output_filepath}.")

##################################################


# MAKE PLOTS
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    args = parse_args()
    INPUT_FILEPATH = f"{args.input_dir}/expressive_features.csv"
    ERROR_FILEPATH = f"{args.input_dir}/expressive_features.errors.csv"
    TIMING_FILEPATH = f"{args.input_dir}/expressive_features.timing.txt"
    N_EXPRESSIVE_FEATURES_PER_PATH_FILEPATH = f"{args.input_dir}/n_expressive_features_per_path.csv"

    logging.basicConfig(level = logging.INFO)
    ##################################################


    # READ IN DATA FRAME
    ##################################################

    data = pd.read_csv(filepath_or_buffer = INPUT_FILEPATH, sep = ",", header = 0, index_col = False)

    # get path- and track-grouped versions of data;
    data_by = {
        "path" : data.groupby(by = "path", as_index = False).size().reset_index(drop = True).merge(right = data[["path", "metadata", "version", "is_public_domain", "is_valid"]], on = "path", how = "left"), # get the first row path from every file
        "track": data.drop(columns = "path")
        }

    n = len(data_by["path"])
    n_tracks = len(data_by["track"])

    if not exists(N_EXPRESSIVE_FEATURES_PER_PATH_FILEPATH):
        
        # calculate number of expressive features per path
        total_expressive_features_per_path = data[["path", "n_expressive_features"]].groupby(by = "path", as_index = False).sum()
        tracks_per_path = data_by["path"][["path", "size"]].merge(right = total_expressive_features_per_path, on = "path", how = "inner").merge(right = data.drop_duplicates(subset = "path").reset_index(drop = True)[["path", "expressive_features"]], on = "path", how = "left")

        # add n_expressive_features column
        data_by["path"]["n_expressive_features"] = [0,] * len(data_by["path"])

        # helper function for multiprocessing
        def extract_annotations_per_path(i: int):
            try:
                with open(str(tracks_per_path.at[i, "expressive_features"]), "rb") as pickle_file:
                    n_system_annotations = pickle.load(file = pickle_file)["n_annotations"]["system"]
            except (OSError, FileNotFoundError): # if pickle file not available, assume 2 system annotations: a key signature and a time signature
                n_system_annotations = 2 # look at the tracks at the path to determine how many system annotations there were
            data_by["path"].at[i, "n_expressive_features"] = tracks_per_path.at[i, "n_expressive_features"] - ((tracks_per_path.at[i, "size"] - 1) * n_system_annotations) # at least one system
        
        # use multiprocessing
        chunk_size = 1
        start_time = perf_counter() # start the timer
        with multiprocessing.Pool(processes = args.jobs) as pool:
            results = list(tqdm(iterable = pool.imap_unordered(func = extract_annotations_per_path, iterable = tracks_per_path.index, chunksize = chunk_size), desc = "Parsing Pickle Files", total = len(tracks_per_path)))
        end_time = perf_counter() # stop the timer
        total_time = end_time - start_time # compute total time elapsed
        total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
        logging.info(f"Total time: {total_time}")
            
        # data_by["path"]["n_expressive_features"] = data_by["path"]["n_expressive_features"].apply(lambda n_expressive_features: max(n_expressive_features, DIVIDE_BY_ZERO_CONSTANT)) # get rid of negative numbers if there are any
        del tracks_per_path, total_expressive_features_per_path

        # write the n_expressive_features per path to original file
        data_by["path"].to_csv(path_or_buf = N_EXPRESSIVE_FEATURES_PER_PATH_FILEPATH, sep = ",", na_rep = "NA", header = True, index = False)

    else: # if we previously calculated n_expressive_features_per_path, just reload it

        # reload data_by["path"]
        data_by["path"] = pd.read_csv(filepath_or_buffer = N_EXPRESSIVE_FEATURES_PER_PATH_FILEPATH, sep = ",", header = 0, index_col = False)

    ##################################################


    # MAKE PLOTS
    ##################################################

    plot_output_filepaths = [f"{args.output_dir}/{plot_type}.png" for plot_type in ("versions", "errors", "public_domain", "n_expressive_features_percentiles", "timings", "tracks")]
    make_versions_plot(output_filepath = plot_output_filepaths[0])
    make_error_plot(input_filepath = ERROR_FILEPATH, output_filepath = plot_output_filepaths[1])
    make_public_domain_plot(output_filepath = plot_output_filepaths[2])
    make_percentile_plot(output_filepath = plot_output_filepaths[3])
    make_timing_plot(input_filepath = TIMING_FILEPATH, output_filepath = plot_output_filepaths[4])
    make_tracks_plot(output_filepath = plot_output_filepaths[5])

    plot_output_filepaths = " ".join([f"deepz:{plot_output_filepath}" for plot_output_filepath in plot_output_filepaths])
    print("".join(("=" for _ in range(100))))
    logging.info("SCP COMMAND:")
    logging.info(" ".join(("scp", plot_output_filepaths)))

    ##################################################


##################################################
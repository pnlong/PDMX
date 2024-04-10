# README
# Phillip Long
# January 25, 2024

# Create plots to describe evaluation.

# python /home/pnlong/model_musescore/evaluate_plots.py


# IMPORTS
##################################################

import argparse
from os.path import exists
from os import mkdir
import pickle
from tqdm import tqdm
import math

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
import train
from expressive_features_plots import SPARSITY_SUCCESSIVE_SUFFIX, OUTPUT_RESOLUTION_DPI
from train_plots import make_model_name_fancy
import evaluate
import evaluate_baseline
from read_mscz.music import DIVIDE_BY_ZERO_CONSTANT

plt.style.use("default")
plt.rcParams["font.family"] = "serif"

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/home/pnlong/musescore/datav"
MODELS_FILEPATH = f"{DATA_DIR}/models.txt"
OUTPUT_DIR = DATA_DIR
LARGE_PLOTS_DPI = int(1.5 * OUTPUT_RESOLUTION_DPI)

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default = MODELS_FILEPATH, type = str, help = ".txt file with a list of directory names for each model")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-r", "--resume", action = "store_true", help = "Whether or not to recreate data tables.")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# HELPER FUNCTION FOR COMBINING DATA TABLES
##################################################

def combine_data_tables(models: list, output_filepath: str, is_baseline: bool = False, eval_type: str = None, stem: str = None) -> pd.DataFrame:

    # if it already exists, read it in
    if exists(output_filepath) and args.resume:
        df = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in full table
    
    # create the combined data table
    else:
        first_successful_iteration = True
        for i, model in enumerate(models):
            input_filepath = args.output_dir + "/" + model + "/eval" + ("_baseline" if is_baseline else "") + (f"/{eval_type}" if not ((model == evaluate_baseline.TRUTH_DIR_STEM) or (is_baseline)) else "") + "/" + stem + ".csv"
            if not exists(input_filepath):
                continue
            df_model = pd.read_csv(filepath_or_buffer = input_filepath, sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in performance
            if first_successful_iteration: # learn the column names on the first iteration
                columns = ["model"] + df_model.columns.tolist()
                df = pd.DataFrame(columns = columns) # create df
                first_successful_iteration = False
            df_model["model"] = utils.rep(x = model, times = len(df_model)) # add model column
            df_model = df_model[columns] # reorder columns
            df = pd.concat(objs = (df, df_model), axis = 0, ignore_index = True) # concatenate
        del model, df_model # free up memory
        df.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")
    
    return df

##################################################


# HELPER FUNCTIONS FOR PLOT MAKING
##################################################

def get_histogram_data_table(data: list, bins: list) -> pd.DataFrame:
    """Gets the necessary data for a histogram in a data table. There are two columns in the output data table: `bins` and `frequency`."""
    data = pd.DataFrame(data = pd.cut(x = data, bins = bins, labels = bins[:-1])) # get histogram
    data = data.rename(columns = {data.columns[0]: "bins"}).groupby(by = "bins").size().reset_index(drop = False).rename(columns = {0: "frequency"})
    return data

##################################################


# HELPER FUNCTIONS THAT MAKE PLOTS
##################################################

def make_baseline_table(baseline: pd.DataFrame, output_dir: str) -> None:
    """Make a table summarizing baseline metrics."""
    baseline = baseline.groupby(by = "model").mean(numeric_only = True).reset_index(drop = False)
    baseline.to_csv(path_or_buf = f"{output_dir}/{evaluate_baseline.EVAL_STEM}.summary.csv", sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")
    return None


def make_n_expressive_features_plot(n_expressive_features: pd.DataFrame, models: list, output_dir: str, apply_log: bool = True) -> None:
    """Make percentiles plot for number of expressive features for each generated path."""

    # get percentiles
    step = 0.001
    percentiles = np.arange(start = 0, stop = 100 + step, step = step)

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [np.repeat(a = ["n",], repeats = 4).tolist() + ["legend"]], constrained_layout = True, figsize = (8, 8))
    fig.suptitle("Number of Expressive Features in Data", fontweight = "bold")

    # make plot
    for model in models:
        current = n_expressive_features[n_expressive_features["model"] == model]["n"].tolist()
        if len(current) == 0:
            continue
        percentile_values = np.percentile(a = current, q = percentiles)
        if apply_log:
            percentile_values = np.log10(percentile_values + DIVIDE_BY_ZERO_CONSTANT)
        axes["n"].plot(percentiles, percentile_values, label = model)
    del percentiles, percentile_values
    axes["n"].set_xlabel("Percentile (%)")
    axes["n"].set_ylabel("Number of Expressive Features")
    
    # deal with ticks depending on ig log scale is applied
    if apply_log:
        logticks = list(range(int(np.log10(min(n_expressive_features["n"]) + DIVIDE_BY_ZERO_CONSTANT)), int(np.log10(max(n_expressive_features["n"]) + DIVIDE_BY_ZERO_CONSTANT)) + 1, 2))
        axes["n"].set_yticks(logticks)
        axes["n"].set_yticklabels(["$10^{" + str(logtick) + "}$" if logtick != 0 else "1" for logtick in logticks])
    else:
        axes["n"].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}"))
    axes["n"].grid()

    # get legend
    handles, labels = axes["n"].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")

    # save image
    fig.savefig(f"{output_dir}/{evaluate.PLOT_TYPES[1]}.png", dpi = OUTPUT_RESOLUTION_DPI) # save image

    # clear up some memory
    del n_expressive_features
    
    return None


def make_density_plot(density: pd.DataFrame, models: list, output_dir: str) -> None:
    """Make pseudo-histograms for expressive feature density."""

    relevant_density_types = ["seconds",]

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [np.repeat(a = relevant_density_types, repeats = 4).tolist() + ["legend"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle(f"Divergence from MuseScore Data of Expressive Feature Densities", fontweight = "bold")

    # create plot
    n_bins = 50
    relevant_ranges = {relevant_density_types[0]: (0, 100)}
    for i, relevant_density_type in enumerate(relevant_density_types):
        relevant_density_query = ((density[relevant_density_type] >= relevant_ranges[relevant_density_type][0]) & (density[relevant_density_type] <= relevant_ranges[relevant_density_type][1]))
        n_points_excluded = sum(~relevant_density_query)
        relevant_density = density[relevant_density_query]
        bins = np.arange(start = relevant_ranges[relevant_density_type][0], stop = relevant_ranges[relevant_density_type][1] + 1e-3, step = (relevant_ranges[relevant_density_type][1] - relevant_ranges[relevant_density_type][0]) / n_bins)
        truth = get_histogram_data_table(data = relevant_density[relevant_density["model"] == evaluate_baseline.TRUTH_DIR_STEM][relevant_density_type], bins = bins)["frequency"].tolist()
        total = sum(truth) + DIVIDE_BY_ZERO_CONSTANT
        truth = list(map(lambda q: q / total, truth))
        for model in models:
            current = relevant_density[relevant_density["model"] == model][relevant_density_type]
            if len(current) == 0:
                continue
            model_relevant_density = get_histogram_data_table(data = current, bins = bins)["frequency"].tolist()
            total = sum(model_relevant_density) + DIVIDE_BY_ZERO_CONSTANT
            model_relevant_density = list(map(lambda p: p / total, model_relevant_density))
            kl_divergence = list(map(lambda P, Q: P * math.log(P / (Q + DIVIDE_BY_ZERO_CONSTANT)) if P != 0 else 0, model_relevant_density, truth))
            axes[relevant_density_type].plot(bins[:-1], kl_divergence, label = model)
        fancy_relevant_density_type = ''.join(relevant_density_type.split('_')).title()
        axes[relevant_density_type].set_xlabel(fancy_relevant_density_type)
        if i != 0: # remove y axis stuff for non-leftmost
            axes[relevant_density_type].sharey(axes[relevant_density_types[0]])
        else:
            axes[relevant_density_type].set_ylabel("KL Divergence")
            # axes[relevant_density_type].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}"))
        axes[relevant_density_type].set_title(f"{fancy_relevant_density_type} per Expressive Feature" + (f" ({n_points_excluded:,} excluded tracks)" if n_points_excluded > 0 else ""))
        axes[relevant_density_type].grid()
        
    # remove y axis for right plots
    # for relevant_density_type in relevant_density_types[:-1]:
    #     axes[relevant_density_type].set_yticks([])
    #     axes[relevant_density_type].set_yticklabels([])

    # get legend
    handles, labels = axes[relevant_density_types[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")

    # save image
    fig.savefig(f"{output_dir}/{evaluate.PLOT_TYPES[2]}.png", dpi = OUTPUT_RESOLUTION_DPI) # save image

    # clear up memory
    del density

    return None


def make_summary_plot(summary: pd.DataFrame, models: list, output_dir: str, apply_log: bool = False, exclude_time_signatures: bool = False) -> list:
    """Make psuedo-bar charts for expressive feature types."""

    plot_types = ["total", "mean", "median"]

    # create figure
    fig, axes = plt.subplot_mosaic(mosaic = [np.repeat(a = plot_types, repeats = 1).tolist() + ["legend"]], constrained_layout = True, figsize = (12, 8))
    fig.suptitle("Summary of Present Expressive Features", fontweight = "bold")

    # load in table
    if exclude_time_signatures:
        summary = summary[summary["type"] != "TimeSignature"] # exclude time signatures
    summary = {
        plot_types[0]: summary.groupby(by = ["model", "type"]).sum().reset_index(drop = False).sort_values(by = "size", ascending = False).reset_index(drop = True),
        plot_types[1]: summary.groupby(by = ["model", "type"]).mean().reset_index(drop = False).sort_values(by = "size", ascending = False).reset_index(drop = True),
        plot_types[2]: summary.groupby(by = ["model", "type"]).median().reset_index(drop = False).sort_values(by = "size", ascending = False).reset_index(drop = True)
        }
    if apply_log:
        summary[plot_types[0]]["size"] = summary[plot_types[0]]["size"].apply(lambda count: np.log10(count + DIVIDE_BY_ZERO_CONSTANT)) # apply log scale to total count

    # create plot
    for i, plot_type in enumerate(plot_types):
        axes[plot_type].xaxis.grid(True)
        for model in models:
            model_summary = summary[plot_type][summary[plot_type]["model"] == model][::-1]
            if len(model_summary) == 0:
                continue
            axes[plot_type].scatter(model_summary["size"], model_summary["type"], label = model)
        del model_summary
        axes[plot_type].set_title(f"{plot_type.title()}")
        axes[plot_type].ticklabel_format(axis = "x", style = "scientific", scilimits = (-1, 3))
        if i == 0: # if the left most plot
            axes[plot_type].set_xlabel(f"{plot_type.title()} Count")
            axes[plot_type].set_ylabel("Expressive Feature")
            axes[plot_type].set_yticks(axes[plot_type].get_yticks())
            axes[plot_type].set_yticklabels(axes[plot_type].get_yticklabels(), rotation = 30)
        else:
            axes[plot_type].set_xlabel(f"{plot_type.title()} Amount per Track")
            # axes[plot_type].sharey(axes[plot_types[0]]) # sharey makes it so we cant remove axis ticks and labels
            axes[plot_type].set_yticks([])
            axes[plot_type].set_yticklabels([])

    # get legend
    handles, labels = axes[plot_types[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")

    # save image
    fig.savefig(f"{output_dir}/{evaluate.PLOT_TYPES[3]}.png", dpi = OUTPUT_RESOLUTION_DPI) # save image

    # return the order from most common to least
    return pd.unique(values = summary[plot_types[0]]["type"]).tolist()


def make_sparsity_plot(sparsity: pd.DataFrame, models: list, output_dir: str, expressive_feature_types: list, apply_log_percentiles_by_feature: bool = True, apply_log_histogram: bool = True, apply_log_percentiles_by_model: bool = True) -> None:
    """Make pseudo histograms for the sparsity of expressive features."""

    # hyper parameters
    relevant_time_units = ["beats", "seconds"]
    relevant_time_units_suffix = [relevant_time_unit + SPARSITY_SUCCESSIVE_SUFFIX for relevant_time_unit in relevant_time_units]
    plot_types = ["total", "mean", "median"]
    plot_type = plot_types[1] # which plot type to display
    output_filepaths = [f"{output_dir}/{evaluate.PLOT_TYPES[4]}.{suffix}.png" for suffix in ("percentiles", "histograms", "percentiles2")]

    # we have distances between successive expressive features in time_steps, beats, seconds, and as a fraction of the length of the song
    # sparsity = sparsity.drop(index = sparsity.index[-1]) # last row is None, since there is no successive expressive features, so drop it
    sparsity = sparsity.drop(columns = "value") # we don't need this column

    # calculate percentiles, save in pickle
    all_features_type_name = "AllFeatures" # name of all features plot name, put in camel case like the rest of them so that it parses well later on
    expressive_feature_types.insert(0, all_features_type_name) # add a plot for all expressive features
    step = 0.001
    percentiles = np.arange(start = 0, stop = 100 + step, step = step)
    pickle_output = f"{output_dir}/{evaluate.PLOT_TYPES[4]}_percentiles.pickle"
    if not (exists(pickle_output) and args.resume):

        # helper function to calculate various percentiles
        def calculate_percentiles(df: pd.DataFrame, columns: list) -> tuple:
            df = df[~pd.isna(df[columns[0]])] # filter out NA values
            n = len(df) # get number of points
            if n == 0:
                return (None, 0)
            df = df[["path"] + columns] # filter down to only necessary columns
            out_columns = [column.replace(SPARSITY_SUCCESSIVE_SUFFIX, "") for column in columns] # get rid of suffix if necessary
            out = dict(zip(plot_types, utils.rep(x = pd.DataFrame(columns = out_columns), times = len(plot_types)))) # create output dataframe
            for plot_type in plot_types:
                if plot_type == "mean":
                    df_temp = df.groupby(by = "path").mean()
                elif plot_type == "median":
                    df_temp = df.groupby(by = "path").median()
                else:
                    df_temp = df
                for column, out_column in zip(columns, out_columns):
                    out[plot_type][out_column] = np.percentile(a = df_temp[column], q = percentiles)
            return (out, n)
        percentile_values = {expressive_feature_type: {model: calculate_percentiles(df = sparsity[(sparsity["model"] == model) & (sparsity["type"] == expressive_feature_type)], columns = relevant_time_units_suffix) for model in models}
                             for expressive_feature_type in [eft for eft in expressive_feature_types if eft != all_features_type_name]}
        percentile_values[all_features_type_name] = {model: calculate_percentiles(df = sparsity[sparsity["model"] == model], columns = relevant_time_units) for model in models}
    
        # save to pickle file
        with open(pickle_output, "wb") as pickle_file:
            pickle.dump(obj = percentile_values, file = pickle_file, protocol = pickle.HIGHEST_PROTOCOL)

    else: # if the pickle already exists, reload it
        with open(pickle_output, "rb") as pickle_file:
            percentile_values = pickle.load(file = pickle_file)


    # create figure of percentiles
    use_twin_axis_percentiles = False
    n_cols = 5
    plot_mosaic = [expressive_feature_types[i:i + n_cols] for i in range(0, len(expressive_feature_types), n_cols)] # create plot grid
    if (len(plot_mosaic) >= 2) and (len(plot_mosaic[-1]) < len(plot_mosaic[-2])):
        plot_mosaic[-1] += utils.rep(x = "legend", times = (len(plot_mosaic[-2]) - len(plot_mosaic[-1])))
    else:
        for i in range(len(plot_mosaic)):
            plot_mosaic[i].append("legend")
    is_bottom_plot = lambda expressive_feature_type: expressive_feature_types.index(expressive_feature_type) >= len(expressive_feature_types) - n_cols
    is_left_plot = lambda expressive_feature_type: expressive_feature_types.index(expressive_feature_type) % n_cols == 0
    fig, axes = plt.subplot_mosaic(mosaic = plot_mosaic, constrained_layout = True, figsize = (24, 16))
    fig.suptitle(f"{plot_type.title()} Sparsity of Expressive Features", fontweight = "bold")

    # create percentile plot (faceted by expressive feature)
    for expressive_feature_type in expressive_feature_types:
        if use_twin_axis_percentiles:
            percentile_right_axis = axes[expressive_feature_type].twinx() # create twin x
        for j, model in enumerate(models):
            if percentile_values[expressive_feature_type][model][0] is None: # if not enough expressive features for sparsity
                continue
            percentiles_values_current = percentile_values[expressive_feature_type][model][0][plot_type].sort_values(by = relevant_time_units[0])
            if apply_log_percentiles_by_feature: # apply log function
                for column in relevant_time_units:
                    percentiles_values_current[column] = np.log10(abs(percentiles_values_current[column]) + DIVIDE_BY_ZERO_CONSTANT)
            axes[expressive_feature_type].plot(percentiles, percentiles_values_current[relevant_time_units[0]], label = model)
            if use_twin_axis_percentiles:
                percentile_right_axis.plot(percentiles, percentiles_values_current[relevant_time_units[1]], label = model)
        if is_left_plot(expressive_feature_type = expressive_feature_type) or use_twin_axis_percentiles:
            axes[expressive_feature_type].set_ylabel(f"log({relevant_time_units[0].title()})" if apply_log_percentiles_by_feature else f"{relevant_time_units[0].title()}")
            axes[expressive_feature_type].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}")) # add commas
        else:
            axes[expressive_feature_type].sharey(axes[expressive_feature_types[int(expressive_feature_types.index(expressive_feature_type) / n_cols)]])
            axes[expressive_feature_type].set_yticklabels([])
        if use_twin_axis_percentiles:
            percentile_right_axis.set_ylabel(f"log({relevant_time_units[1].title()})" if apply_log_percentiles_by_feature else f"{relevant_time_units[1].title()}") # add
            percentile_right_axis.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}")) # add commas
        if is_bottom_plot(expressive_feature_type = expressive_feature_type): # if bottom plot, add x labels
            axes[expressive_feature_type].set_xlabel("Percentile (%)")
        else: # is not a bottom plot
            # axes[expressive_feature_type].set_xticks([]) # will keep xticks for now
            axes[expressive_feature_type].set_xticklabels([])
        axes[expressive_feature_type].set_title(f"{utils.split_camel_case(string = expressive_feature_type, sep = ' ').title()} (n = {sum(percentile_values[expressive_feature_type][model][1] for model in models):,})")
        axes[expressive_feature_type].grid() # add gridlines

    # get legend
    handles, labels = axes[expressive_feature_types[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")

    # save image
    fig.savefig(output_filepaths[0], dpi = LARGE_PLOTS_DPI) # save image


    # create new plot of histograms
    use_twin_axis_histogram = False
    n_bins = 25
    histogram_range = (0, 256) # in beats
    fig, axes = plt.subplot_mosaic(mosaic = plot_mosaic, constrained_layout = True, figsize = (24, 16))
    fig.suptitle(f"{plot_type.title()} Sparsity of Expressive Features", fontweight = "bold")

    # create histogram plot
    for expressive_feature_type in expressive_feature_types:
        if use_twin_axis_histogram:
            histogram_right_axis = axes[expressive_feature_type].twinx() # create twin x
        histogram_values = sparsity[["model", "path"] + relevant_time_units] if expressive_feature_type == all_features_type_name else sparsity[sparsity["type"] == expressive_feature_type][["model", "path"] + relevant_time_units_suffix].rename(columns = dict(zip(relevant_time_units_suffix, relevant_time_units))) # get subset of sparsity
        if plot_type == "mean":
            histogram_values = histogram_values.groupby(by = ["path", "model"]).mean().reset_index(drop = False)
        elif plot_type == "median":
            histogram_values = histogram_values.groupby(by = ["path", "model"]).median().reset_index(drop = False)
        else:
            histogram_values = histogram_values
        histogram_values = histogram_values[(histogram_values[relevant_time_units[0]] >= histogram_range[0]) & (histogram_values[relevant_time_units[0]] <= histogram_range[1])]
        bins = np.arange(start = histogram_range[0], stop = histogram_range[1] + 1e-3, step = (histogram_range[1] - histogram_range[0]) / n_bins)
        for model in models:
            histogram_values_model = histogram_values[histogram_values["model"] == model]
            if len(histogram_values_model) == 0:
                continue
            histogram = get_histogram_data_table(data = histogram_values_model[relevant_time_units[0]], bins = bins)
            if apply_log_histogram:
                histogram["frequency"] = np.log10(histogram["frequency"] + DIVIDE_BY_ZERO_CONSTANT)
            axes[expressive_feature_type].plot(histogram["bins"], histogram["frequency"], label = model)
            if use_twin_axis_histogram:
                histogram = get_histogram_data_table(data = histogram_values_model[relevant_time_units[1]], bins = bins)
                if apply_log_histogram:
                    histogram["frequency"] = np.log10(histogram["frequency"] + DIVIDE_BY_ZERO_CONSTANT)
                axes[expressive_feature_type].plot(histogram["bins"], histogram["frequency"], label = model)
        axes[expressive_feature_type].set_xlabel(relevant_time_units[0].title())
        axes[expressive_feature_type].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}" if (not math.isnan(count)) else "")) # add commas
        if use_twin_axis_histogram:
            histogram_right_axis.set_xlabel(relevant_time_units[1].title())
            histogram_right_axis.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}" if (not math.isnan(count)) else "")) # add commas
        if is_left_plot(expressive_feature_type = expressive_feature_type):
            axes[expressive_feature_type].set_ylabel("Count")
        axes[expressive_feature_type].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: "$10^{" + str(int(np.log10(count))) + "}$" if ((not math.isnan(count)) and (count >= 10e3) and apply_log_histogram) else f"{int(count):,}")) # add commas
        # axes[expressive_feature_type].set_yticks(axes[expressive_feature_type].get_yticks())
        # axes[expressive_feature_type].set_yticklabels(axes[expressive_feature_type].get_yticklabels(), rotation = 0)
        axes[expressive_feature_type].set_title(f"{utils.split_camel_case(string = expressive_feature_type, sep = ' ').title()} (n = {sum(percentile_values[expressive_feature_type][model][1] for model in models):,})")
        axes[expressive_feature_type].grid() # add gridlines
    
    # get legend
    handles, labels = axes[expressive_feature_types[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(make_model_name_fancy, by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Model", mode = "expand")
    axes["legend"].axis("off")
    
    # save image
    fig.savefig(output_filepaths[1], dpi = LARGE_PLOTS_DPI) # save image


    # new plot of percentiles on different facets (models)
    use_twin_axis_percentiles = False
    n_rows = 2
    n_cols = int(((len(models) - 1) / n_rows)) + 1
    plot_mosaic = [models[i:i + n_cols] for i in range(0, len(models), n_cols)] # create plot grid
    if len(plot_mosaic[-1]) < len(plot_mosaic[-2]):
        plot_mosaic[-1] += utils.rep(x = "legend", times = (len(plot_mosaic[-2]) - len(plot_mosaic[-1])))
    else:
        for i in range(len(plot_mosaic)):
            plot_mosaic[i].append("legend")
    is_bottom_plot = lambda model: models.index(model) >= len(models) - n_cols
    is_left_plot = lambda model: models.index(model) % n_cols == 0
    fig, axes = plt.subplot_mosaic(mosaic = plot_mosaic, constrained_layout = True, figsize = (12, 8))
    fig.suptitle(f"{plot_type.title()} Sparsity of Expressive Features", fontweight = "bold")
    
    # create percentile plot
    for model in models:
        if use_twin_axis_percentiles:
            percentile_right_axis = axes[model].twinx() # create twin x
        for i, expressive_feature_type in enumerate(expressive_feature_types):
            if percentile_values[expressive_feature_type][model][0] is None: # if not enough expressive features for sparsity
                continue
            percentiles_values_current = percentile_values[expressive_feature_type][model][0][plot_type].sort_values(by = relevant_time_units[0])
            if apply_log_percentiles_by_model: # apply log function
                for column in relevant_time_units:
                    percentiles_values_current[column] = np.log10(abs(percentiles_values_current[column]) + DIVIDE_BY_ZERO_CONSTANT)
            axes[model].plot(percentiles, percentiles_values_current[relevant_time_units[0]], label = expressive_feature_type)
            if use_twin_axis_percentiles:
                percentile_right_axis.plot(percentiles, percentiles_values_current[relevant_time_units[1]], label = expressive_feature_type)
        if is_left_plot(model = model) or use_twin_axis_percentiles:
            axes[model].set_ylabel(f"log({relevant_time_units[0].title()})" if apply_log_percentiles_by_model else f"{relevant_time_units[0].title()}")
            axes[model].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}")) # add commas
        else:
            axes[model].sharey(axes[models[n_cols * int(models.index(model) / n_cols)]])
            axes[model].set_yticklabels([])
        if use_twin_axis_percentiles:
            percentile_right_axis.set_ylabel(f"log({relevant_time_units[1].title()})" if apply_log_percentiles_by_model else f"{relevant_time_units[1].title()}") # add
            percentile_right_axis.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda count, _: f"{int(count):,}")) # add commas
        if is_bottom_plot(model = model): # if bottom plot, add x labels
            axes[model].set_xlabel("Percentile (%)")
        else: # is not a bottom plot
            # axes[model].set_xticks([]) # will keep xticks for now
            axes[model].set_xticklabels([])
        axes[model].set_title(make_model_name_fancy(model = model))
        axes[model].grid() # add gridlines
    
    # add a legend
    handles, labels = axes[models[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes["legend"].legend(handles = by_label.values(), labels = list(map(lambda expressive_feature_type: utils.split_camel_case(string = expressive_feature_type, sep = " ").title(), by_label.keys())), loc = "center", fontsize = "small", title_fontsize = "medium", alignment = "center", ncol = 1, title = "Expressive Feature", mode = "expand")
    axes["legend"].axis("off")
    
    # save image
    fig.savefig(output_filepaths[2], dpi = LARGE_PLOTS_DPI) # save image

    # clear up some memory
    del sparsity

    return None


def make_perplexity_table(losses_for_perplexity: pd.DataFrame, output_filepath: str):
    """Make a table summarizing perplexities."""

    # old and new column names
    losses_for_perplexity_columns = list(losses_for_perplexity.columns[losses_for_perplexity.columns.values.tolist().index("loss_" + train.ALL_STRING):]) # get the loss columns (for renaming)
    perplexity_columns = list(map(lambda loss_column: loss_column.replace("loss_", "ppl_"), losses_for_perplexity_columns)) # get the new perplexity column names
    
    # compute perplexity
    perplexity_function = lambda loss_for_perplexity: math.exp(loss_for_perplexity) # exp(loss)
    perplexity = losses_for_perplexity.rename(columns = dict(zip(losses_for_perplexity_columns, perplexity_columns))) # rename columns from loss to perplexity
    for perplexity_column in perplexity_columns: # compute perplexity given each loss value
        perplexity[perplexity_column] = perplexity[perplexity_column].apply(perplexity_function)

    # summarize per model
    perplexity = perplexity.groupby(by = "model").mean(numeric_only = True).reset_index(drop = False) # summarize per model
    
    # output
    perplexity.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = train.NA_VALUE, header = True, index = False, mode = "w")
    del losses_for_perplexity, losses_for_perplexity_columns, perplexity, perplexity_columns

    return None

##################################################


# MAIN METHOD
##################################################
if __name__ == "__main__":

    # GET MODELS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # get model names
    with open(args.models, "r") as models_output: # read in list of trained models
        models = [model.strip() for model in models_output.readlines()] # use a set because better for `in` operations
    models_with_truth = [evaluate_baseline.TRUTH_DIR_STEM] + models
    models_joint = [model for model in models_with_truth if "conditional" not in model]

    # make sure eval directory exists
    eval_dir = f"{args.output_dir}/eval"
    if not exists(eval_dir):
        mkdir(eval_dir)

    ##################################################
    

    # BASELINE
    ##################################################

    # pure baseline eval
    baseline = combine_data_tables(models = models_with_truth,
                                   output_filepath = f"{args.output_dir}/{evaluate_baseline.EVAL_STEM}.csv",
                                   is_baseline = True,
                                   stem = evaluate_baseline.EVAL_STEM)
    _ = make_baseline_table(baseline = baseline, output_dir = eval_dir)
    del baseline

    ##################################################

    
    # DISTRIBUTION OF EXPRESSIVE FEATURES
    ##################################################

    for eval_type in tqdm(iterable = evaluate.EVAL_TYPES, desc = "Making Evaluation Plots"):

        # make sure plots directory exists
        eval_subdir = f"{eval_dir}/{eval_type}"
        if not exists(eval_subdir):
            mkdir(eval_subdir)
        plots_dir = f"{eval_subdir}/plots"
        if not exists(plots_dir):
            mkdir(plots_dir)

        # baseline metrics
        baseline = combine_data_tables(models = models_with_truth,
                                       output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[0]}.csv",
                                       is_baseline = False,
                                       eval_type = eval_type,
                                       stem = f"eval_{evaluate.PLOT_TYPES[0]}")
        _ = make_baseline_table(baseline = baseline, output_dir = args.output_dir)
        del baseline

        # n expressive features
        n_expressive_features = combine_data_tables(models = models_joint,
                                                    output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[1]}.csv",
                                                    is_baseline = False,
                                                    eval_type = eval_type,
                                                    stem = f"eval_{evaluate.PLOT_TYPES[1]}")
        _ = make_n_expressive_features_plot(n_expressive_features = n_expressive_features, models = models_joint, output_dir = plots_dir)
        del n_expressive_features

        # density
        density = combine_data_tables(models = [model for model in models_joint if not ((model == evaluate_baseline.TRUTH_DIR_STEM) or ("baseline" in model))],
                                      output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[2]}.csv",
                                      is_baseline = False,
                                      eval_type = eval_type,
                                      stem = f"eval_{evaluate.PLOT_TYPES[2]}")
        _ = make_density_plot(density = density, models = models_joint, output_dir = plots_dir)
        del density

        # feature types summary
        summary = combine_data_tables(models = models_joint,
                                      output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[3]}.csv",
                                      is_baseline = False,
                                      eval_type = eval_type,
                                      stem = f"eval_{evaluate.PLOT_TYPES[3]}")
        expressive_feature_types = make_summary_plot(summary = summary, models = models_joint, output_dir = plots_dir)
        del summary

        # sparsity
        sparsity = combine_data_tables(models = models_joint,
                                       output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[4]}.csv",
                                       is_baseline = False,
                                       eval_type = eval_type,
                                       stem = f"eval_{evaluate.PLOT_TYPES[4]}")
        _ = make_sparsity_plot(sparsity = sparsity, output_dir = plots_dir, models = models_joint, expressive_feature_types = expressive_feature_types)
        del sparsity

        # perplexity
        losses_for_perplexity = combine_data_tables(models = models,
                                                    output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[5]}.csv",
                                                    is_baseline = False,
                                                    eval_type = eval_type,
                                                    stem = f"eval_{evaluate.PLOT_TYPES[5]}")
        _ = make_perplexity_table(losses_for_perplexity = losses_for_perplexity, output_filepath = f"{eval_subdir}/eval_{evaluate.PLOT_TYPES[5]}.summary.csv")
        del losses_for_perplexity

        plt.close("all")

    ##################################################

##################################################
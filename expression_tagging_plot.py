# README
# Phillip Long
# April 12, 2024

# Create a plot of the accuracies for expression tagging models.

# python /home/pnlong/model_musescore/expression_tagging_plot.py


# IMPORTS
##################################################

from os.path import basename
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy import repeat, vectorize
from utils import rep
import train
import expressive_features_plots
import representation

plt.style.use("default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/home/pnlong/musescore"

##################################################


# HELPER FUNCTIONS
##################################################

##################################################

# MAIN METHOD
##################################################
if __name__ == "__main__":

    # CREATE FIGURE
    ##################################################

    # constants
    metric = "accuracy"

    # for the figure legend
    fields_order = [train.ALL_STRING] + representation.DIMENSIONS[:representation.DIMENSIONS.index("time")]
    fields_order.insert(representation.DIMENSIONS.index("position") + 1, "time") # insert time
    fields_order_for_table = fields_order[1:] + fields_order[:1]
    colors = {
        train.ALL_STRING: expressive_features_plots.ALL_BROWN,
        "type": expressive_features_plots.DYNAMIC_GOLD,
        "beat": expressive_features_plots.SYMBOL_ORANGE,
        "position": expressive_features_plots.SYMBOL_ORANGE,
        "time": expressive_features_plots.SYMBOL_ORANGE,
        "duration": expressive_features_plots.SYSTEM_SALMON,
        "value": expressive_features_plots.TEXT_GREEN,
        "instrument": expressive_features_plots.SPANNER_BLUE,
        "velocity": expressive_features_plots.TEMPORAL_PURPLE,
    }

    # parse the command-line arguments
    model_filepaths = [f"{INPUT_DIR}/datav/prefix_econditional_ape_20M", f"{INPUT_DIR}/datav/anticipation_econditional_ape_20M",
                       f"{INPUT_DIR}/datava/prefix_econditional_ape_31M", f"{INPUT_DIR}/datava/anticipation_econditional_ape_31M"]
    get_fancy_model_name = lambda model_filepath: basename(model_filepath).split("_")[0].title() + " ($\it{" + ("A" if "datava" in model_filepath else "M") + "}$)" # get fancy version of model name
    models = list(map(get_fancy_model_name, model_filepaths))
    
    # create figure
    n_repeats = 2
    metrical_models = repeat(a = model_filepaths[:2], repeats = n_repeats).tolist()
    absolute_models = repeat(a = model_filepaths[2:], repeats = n_repeats).tolist()
    mosaic = [
        metrical_models, metrical_models, metrical_models,
        absolute_models, absolute_models, absolute_models,
        rep(x = "legend", times = n_repeats * 2)
    ]
    fig, axes = plt.subplot_mosaic(mosaic = mosaic, constrained_layout = True, figsize = (8, 6))

    # loop through models
    table_performance_columns = ["model", "field", "value"]
    table_performance = pd.DataFrame(columns = table_performance_columns)
    for i, model_filepath, model in zip(range(len(model_filepaths)), model_filepaths, models):

        # get some variables
        is_left_plot = "prefix" in model_filepath
        is_bottom_plot = "datava" in model_filepath

        # load in performance
        performance = pd.read_csv(filepath_or_buffer = f"{model_filepath}/performance.csv", sep = ",", na_values = train.NA_VALUE, header = 0, index_col = False) # read in full performance
        performance = performance[(performance["metric"] == "accuracy") & (performance["partition"] == "valid") & (performance["mask"] == "expressive")]
        performance["model"] = rep(x = model, times = len(performance))
        table_performance = pd.concat(objs = (table_performance, performance[performance["step"] == max(performance["step"])][table_performance_columns]), axis = 0, ignore_index = True)

        # plot data
        for field in pd.unique(performance["field"]):
            current_performance = performance[performance["field"] == field]
            axes[model_filepath].plot(current_performance["step"], current_performance["value"], label = field, color = colors[field])

        # x axis labels
        if is_bottom_plot:
            axes[model_filepath].set_xlabel("Step ($10^{3}$)")
            axes[model_filepath].ticklabel_format(axis = "x", style = "plain")
            axes[model_filepath].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda xticklabel, _: f"{int(xticklabel / 1000):,}")) # make ticks look nice
        else:
            # axes[model_filepath].set_xticks([]) # will keep xticks for now
            axes[model_filepath].set_xticklabels([])
        
        # y axis labels
        if is_left_plot:
            axes[model_filepath].set_ylabel("Accuracy (%)")
            axes[model_filepath].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda yticklabel, _: f"{int(100 * yticklabel):,}")) # make ticks look nice
        else:
            axes[model_filepath].set_ylim(axes[model_filepaths[i - 1]].get_ylim())
            # axes[model_filepath].set_yticks([]) # will keep yticks for now
            axes[model_filepath].set_yticklabels([])

        # set title
        axes[model_filepath].set_title(label = model)
        axes[model_filepath].yaxis.grid(True) # add gridlines
    
    # get legend
    handles, labels = axes[model_filepaths[0]].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles, labels = axes[model_filepaths[2]].get_legend_handles_labels()
    by_label.update(dict(zip(labels, handles)))
    axes["legend"].legend(handles = [by_label[field] for field in fields_order], labels = list(map(lambda field: field.title(), fields_order)), loc = "center", fontsize = "medium", title_fontsize = "large", alignment = "center", ncol = 5, title = "Field", mode = "expand")
    axes["legend"].axis("off")
    
    # save image
    fig.savefig(f"{INPUT_DIR}/econditional_accuracy.pdf", dpi = expressive_features_plots.OUTPUT_RESOLUTION_DPI, transparent = True, bbox_inches = "tight") # save image
   
    ##################################################


    # PRINT TABLE FOR PAPER
    ##################################################

    # get orderings for table
    models_for_table = [models[0], models[2], models[1], models[3]]

    print("r" + "".join(rep(x = "c", times = len(models_for_table))))
    print(" & " + " & ".join(models_for_table) + " \\\\")
    for field in fields_order_for_table:
        table_performance_current = table_performance[table_performance["field"] == field][["model", "value"]]
        table_performance_current = table_performance_current.sort_values(by = "model", axis = 0, ascending = True, key = vectorize(lambda model: models_for_table.index(model)))
        table_performance_current = table_performance_current.set_index(keys = "model", drop = True)["value"].to_dict()
        table_performance_current = " & ".join(map(lambda model: f"{100 * table_performance_current[model]:.2f}" if table_performance_current.get(model, None) is not None else "-", models_for_table))
        print(f"{field} & {table_performance_current} \\\\")

    ##################################################
    
##################################################

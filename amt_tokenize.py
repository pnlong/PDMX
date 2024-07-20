# README
# Phillip Long
# July 3, 2024

# Process MusicXML files into AMT-ready files

# python /home/pnlong/model_musescore/amt_tokenize.py


# IMPORTS
##################################################

import argparse
import pandas as pd
import numpy as np
from os.path import exists, basename
from os import makedirs
from tqdm import tqdm
import logging
from time import perf_counter, strftime, gmtime
import multiprocessing
import random
from copy import copy
import subprocess
from typing import Tuple, List
import random

import utils
import representation
from encode import extract_data
from read_mscz.read_mscz import read_musescore
from read_mscz.classes import *

from amt_config import *
from amt_vocab import *
import amt_ops as ops

##################################################


# CONSTANTS
##################################################

INPUT_DIR = "/data2/pnlong/musescore"
MSCZ_FILEPATHS = f"{INPUT_DIR}/relevant_mscz_files.txt"
OUTPUT_DIR = "/data2/pnlong/musescore/amt"

PARTITIONS = {"train": 0.8, "valid": 0.1, "test": 0.1} # training partitions
DEFAULT_AUGMENT_FACTOR = 1 # data augmentation, default is no augment
N_SUBGROUPS = 20 # number of subgroups to split the data into

DRUM_INSTRUMENT_CODE = 128 # drum instrument code for AMT

##################################################


# HELPER FUNCTIONS
##################################################

def compound_to_events(tokens: list, stats: bool = False) -> list:
    """
    Convert from intermediate compound representation to events
    """
    compound_size = COMPOUND_SIZE
    assert len(tokens) % compound_size == 0
    tokens = tokens.copy()

    # remove velocities
    del tokens[(compound_size - 1)::compound_size] # remove velocities
    compound_size -= 1 # one less dimension

    # combine (note, instrument)
    assert all(-1 <= token < MAX_PITCH for token in tokens[(compound_size - 2)::compound_size]) # concerning notes
    assert all(-1 <= token < MAX_INSTR for token in tokens[(compound_size - 1)::compound_size]) # concerning instruments
    tokens[(compound_size - 2)::compound_size] = [SEPARATOR if (note == -1) else ((MAX_PITCH * instrument) + note)
                                                  for note, instrument in zip(tokens[(compound_size - 2)::compound_size], tokens[(compound_size - 1)::compound_size])]
    tokens[(compound_size - 2)::compound_size] = [(NOTE_OFFSET + token) for token in tokens[(compound_size - 2)::compound_size]]
    del tokens[(compound_size - 1)::compound_size] # remove instrument
    compound_size -= 1 # one less dimension

    # max duration cutoff and set unknown durations to 250ms
    truncations = sum((token >= MAX_DUR) for token in tokens[1::compound_size])
    tokens[1::compound_size] = [(TIME_RESOLUTION // 4) if (token == -1) else min(token, MAX_DUR - 1)
                                for token in tokens[1::compound_size]] # deal with durations
    tokens[1::compound_size] = [(DUR_OFFSET + token) for token in tokens[1::compound_size]]

    assert min(tokens[0::compound_size]) >= 0
    tokens[0::compound_size] = [(TIME_OFFSET + token) for token in tokens[0::compound_size]]

    assert len(tokens) % compound_size == 0

    if stats:
        return tokens, truncations

    return tokens

def maybe_tokenize(tokens: list) -> Tuple[list, list, int]:
    """
    Semi-tokenize from compound form unless there is some error
    """

    # skip sequences with very few events
    if len(tokens) < (COMPOUND_SIZE * MIN_TRACK_EVENTS):
        return (None, None, 1) # short track

    events, truncations = compound_to_events(tokens = tokens, stats = True)
    end_time = ops.max_time(tokens = events, seconds = False)

    # don't want to deal with extremely short tracks
    if end_time < (TIME_RESOLUTION * MIN_TRACK_TIME_IN_SECONDS):
        return (None, None, 1) # short track

    # don't want to deal with extremely long tracks
    if end_time > (TIME_RESOLUTION * MAX_TRACK_TIME_IN_SECONDS):
        return (None, None, 2) # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events)) > MAX_TRACK_INSTR:
        return (None, None, 3) # too many instruments

    return (events, truncations, 0)

def extract_spans(all_events: list, rate: float) -> Tuple[list, list]:
    """
    Helper method for augmentation in arrival time tokenization
    """
    events = []
    controls = []
    span = True
    next_span = end_span = TIME_OFFSET + 0
    for time, dur, note in zip(all_events[0::EVENT_SIZE], all_events[1::EVENT_SIZE], all_events[2::EVENT_SIZE]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        # end of an anticipated span; decide when to do it again (next_span)
        if span and (time >= end_span):
            span = False
            next_span = time + int(TIME_RESOLUTION * np.random.exponential(1.0 / rate))

        # anticipate a 3-second span
        if (not span) and (time >= next_span):
            span = True
            end_span = time + (DELTA * TIME_RESOLUTION)

        if span:
            # mark this event as a control
            controls.extend([CONTROL_OFFSET + time, CONTROL_OFFSET + dur, CONTROL_OFFSET + note])
        else:
            events.extend([time, dur, note])

    return events, controls

ANTICIPATION_RATES = 10
def extract_random(all_events: list, rate: float) -> Tuple[list, list]:
    """
    Helper method for augmentation in arrival time tokenzation
    """
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::EVENT_SIZE], all_events[1::EVENT_SIZE], all_events[2::EVENT_SIZE]):
        assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

        if (np.random.random() < (rate / float(ANTICIPATION_RATES))):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET + time, CONTROL_OFFSET + dur, CONTROL_OFFSET + note])
        else:
            events.extend([time, dur, note])

    return events, controls

def extract_instruments(all_events: list, instruments: list) -> Tuple[list, list]:
    """
    Helper method for augmentation in arrival time tokenization
    """
    events = []
    controls = []
    for time, dur, note in zip(all_events[0::EVENT_SIZE], all_events[1::EVENT_SIZE], all_events[2::EVENT_SIZE]):
        assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
        assert note not in [SEPARATOR, REST] # these shouldn't either

        instr = (note - NOTE_OFFSET) // MAX_PITCH
        if (instr in instruments):
            # mark this event as a control
            controls.extend([CONTROL_OFFSET + time, CONTROL_OFFSET + dur, CONTROL_OFFSET + note])
        else:
            events.extend([time, dur, note])

    return events, controls

##################################################


# TOKENIZATION FUNCTION
##################################################

def tokenize(
    paths_input: List[str],
    path_output: str,
    subgroup: int,
    augment_factor: int = DEFAULT_AUGMENT_FACTOR,
    use_implied_duration: bool = True,
    use_absolute_time: bool = False,
    debug: bool = False
    ) -> Tuple[int, int, int, int, int, int, int, int]:
    """Extract relevant information from .mscz files, output as tokens

    Parameters
    ----------
    paths_input : List[str]
        Paths to the MuseScore files to read.
    path_output : str
        Path to the file where tokenized information will be outputted.
    subgroup : int
        Index of this function call (for formatting progress bars).
    augment_factor : int
        Factor by which to augment the data.
    use_implied_duration : bool
        When a note or annotation does not explicitly have an explicitly defined duration, do we infer its duration?
    use_absolute_time : bool
        Should time be measured in seconds (absolute time) or time steps (metrical time)?
    debug: bool
        Do we output statistics on files that were improperly processed?

    Returns
    -------
    tuple: (
        total # of sequences,
        total # of rests,
        # of too short songs,
        # of too long songs,
        # of songs with too many instruments,
        # of inexpressible songs,
        total # of truncations,
        total # of events,
    )
    
    """

    # INITIATE VARIABLES
	##################################################
	
    # counter variables
    seqcount, all_truncations, rest_count, n_events = 0, 0, 0, 0
    stats = [0] * 4 # (short, long, too many instruments, inexpressible)
    
    # set random seed
    np.random.seed(0) # random seed

    # deal with intermediate representation, "compound" word
    time_dim_name = "time" + (".s" if use_absolute_time else "")
    compound_word_dimensions = (time_dim_name, "duration", "value", "instrument", "velocity") # dimensions for AMT compound word
    time_dim, duration_dim, instrument_dim = compound_word_dimensions.index(time_dim_name), compound_word_dimensions.index("duration"), compound_word_dimensions.index("instrument")
    column_reordering = [representation.DIMENSIONS.index(dimension) for dimension in compound_word_dimensions]

	##################################################


    # ITERATE THROUGH INPUT FILES
	##################################################
	
    with open(path_output, "a") as outfile:
        concatenated_tokens = []
        for path in tqdm(iterable = paths_input, desc = f"#{subgroup}", position = subgroup + 1, leave = True):

            # LOAD IN MSCZ FILE
            ##################################################

            # finish output dictionary
            try:
                metadata_path = METADATA[path]
            except KeyError:
                metadata_path = None

            # try to read musescore
            try:
                music = read_musescore(path = path, timeout = 10)
            except: # if that fails
                stats[3] += 1 # this file is inexpressible
                continue # exit here

            # start timer
            start_time = perf_counter()

            ##################################################


            # PRODUCE INTERMEDIATE COMPOUND WORD REPRESENTATION
            ##################################################

            # realize expressive features
            # music.infer_velocity = True
            # music.realize_expressive_features()

            tokens = np.empty(shape = (0, COMPOUND_SIZE), dtype = np.object_)
            for track in music.tracks:

                # do not record if track is drum or is an unknown program
                if track.program not in representation.KNOWN_PROGRAMS:
                    continue
        
                # create MusicExpress object with just one track
                track_music = copy(x = music)
                track_music.tracks = [track,]
                data = extract_data(music = track_music, use_implied_duration = use_implied_duration, include_velocity = False, use_absolute_time = use_absolute_time)

                # wrangle data
                data = data[data[:, 0] != representation.EXPRESSIVE_FEATURE_TYPE_STRING] # we only want notes
                data = data[:, column_reordering] # reorder columns to AMT compound word
                time_conversion_function = (lambda time: round(TIME_RESOLUTION * time)) if use_absolute_time else (lambda time: round((TIME_RESOLUTION / music.resolution) * time))
                data[:, time_dim] = list(map(time_conversion_function, data[:, time_dim]))
                data[:, duration_dim] = list(map(time_conversion_function, data[:, duration_dim]))
                if track.is_drum:
                    data[:, instrument_dim] = utils.rep(x = DRUM_INSTRUMENT_CODE, times = len(data)) # set instrument code if this is a drum track
        
                # concatenate to overall tokens list
                tokens = np.concatenate((tokens, data), axis = 0)

                # update number of events
                n_events += len(data)

            # order tokens by time
            tokens = tokens[tokens[:, time_dim].argsort()]
    
            # flatten tokens from 2d to 1d
            n = len(tokens) # save number of tokens
            tokens = np.ravel(tokens).tolist()

            ##################################################


            # ARRIVAL-TIME TOKENIZATION
            ##################################################

            # check if anything is wrong
            all_events, truncations, status = maybe_tokenize(tokens = tokens)
            if (status > 0):
                stats[status - 1] += 1
                continue

            # get instruments and other information
            instruments = list(ops.get_instruments(tokens = all_events).keys())
            end_time = ops.max_time(tokens = all_events, seconds = False)

            # different random augmentations
            concatenated_tokens = []
            for k in range(augment_factor):
                
                # AUGMENT
                ##################################################

                # no augmentation
                if k % 10 == 0:
                    events = all_events.copy()
                    controls = []

                # span augmentation
                elif k % 10 == 1:
                    lmbda = .05
                    events, controls = extract_spans(all_events = all_events, rate = lmbda)
        
                # random augmentation
                elif k % 10 < 6:
                    r = np.random.randint(1, ANTICIPATION_RATES)
                    events, controls = extract_random(all_events = all_events, rate = r)

                else:
                    # instrument augmentation: at least one, but not all instruments
                    if len(instruments) > 1:
                        u = 1 + np.random.randint(len(instruments) - 1)
                        subset = np.random.choice(a = instruments, size = u, replace = False)
                        events, controls = extract_instruments(all_events = all_events, instruments = subset)
                    # no augmentation
                    else:
                        events = all_events.copy()
                        controls = []
                        
                ##################################################

                # ANTICIPATE
                ##################################################
                
                # get augmentation controls
                if len(concatenated_tokens) == 0:
                    z = ANTICIPATE if (k % 10 != 0) else AUTOREGRESS

                # add to concatenated_tokens, anticipate
                all_truncations += truncations
                events = ops.pad(tokens = events, end_time = end_time)
                rest_count += sum((token == REST) for token in events[2::EVENT_SIZE])
                tokens, controls = ops.anticipate(events = events, controls = controls)
                assert len(controls) == 0 # should have consumed all controls (because of padding)
                tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
                concatenated_tokens.extend(tokens)
                
                ##################################################

                # WRITE TO OUTPUT FILE
                ##################################################
                
                # write out full sequences to file
                while (len(concatenated_tokens) >= (EVENT_SIZE * M)):
                    seq = concatenated_tokens[0:(EVENT_SIZE * M)]
                    concatenated_tokens = concatenated_tokens[(EVENT_SIZE * M):]

                    # relativize time to the context
                    seq = ops.translate(tokens = seq, seconds = -ops.min_time(tokens = seq, seconds = False), seconds = False)
                    assert ops.min_time(tokens = seq, seconds = False) == 0 # make sure times are valid
                    if (ops.max_time(tokens = seq, seconds = False) >= MAX_TIME):
                        stats[3] += 1
                        continue

                    # if seq contains SEPARATOR, global controls describe the first sequence
                    seq.insert(0, z)

                    # write to file
                    outfile.write(" ".join(map(str, seq)) + "\n")
                    seqcount += 1

                    # grab the current augmentation controls if we didn't already
                    z = ANTICIPATE if (k % 10 != 0) else AUTOREGRESS
                    
                ##################################################

            ##################################################

    
            # STATS PER SONG
            ##################################################

            # discern partition
            partition = "train"
            if (subgroup in subgroups_valid):
                partition = "valid"
            elif (subgroup in subgroups_test):
                partition = "test"

            # write info to file
            current_output = {
                "partition" : partition,
                "musescore" : path,
                "metadata" : metadata_path,
                "n" : n,
            }
            utils.write_to_file(info = current_output, output_filepath = MAPPING_OUTPUT_FILEPATH, columns = current_output.keys())

            # get time elapsed
            end_time = perf_counter()
            total_time = end_time - start_time
            utils.write_to_file(info = {"time": total_time}, output_filepath = TIMING_OUTPUT_FILEPATH)

            ##################################################

    ##################################################


    # FINISH
	##################################################
	
    # output statistics if needed
    if debug:
        logging.info(f"Processed {seqcount} sequences or {n_events} events (discarded {stats[0] + stats[1] + stats[2]} tracks, discarded {stats[3]} seqs, added {rest_count} rest tokens)")

    # return information
    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations, n_events)
	
	##################################################

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Tokenize", description = "Extract and Tokenize Notes from MuseScore Data.")
    parser.add_argument("-p", "--paths", type = str, default = MSCZ_FILEPATHS, help = "List of (absolute) filepaths to MuseScore files whose data will be extracted")
    parser.add_argument("-i", "--input_dir", type = str, default = INPUT_DIR, help = "Directory containing all data tables needed as input")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-ns", "--n_subgroups", type = int, default = N_SUBGROUPS, help = "Number of subgroups to split the data into.")
    parser.add_argument("-ed", "--explicit_duration", action = "store_true", help = "Whether or not to calculate the 'implied duration' of features without an explicitly-defined duration.")
    # parser.add_argument("-v", "--velocity", action = "store_true", help = "Whether or not to include a velocity field that reflects expressive features.")
    parser.add_argument("-a", "--absolute_time", action = "store_true", help = "Whether or not to use absolute (seconds) or metrical (beats) time.")
    parser.add_argument("--augment", type = int, default = DEFAULT_AUGMENT_FACTOR, help = "Augmentation factor for the data.")
    parser.add_argument("-v", "--ratio_valid", type = float, default = PARTITIONS["valid"], help = "Ratio of validation files.")
    parser.add_argument("-t", "--ratio_test", type = float, default = PARTITIONS["test"], help = "Ratio of test files.")
    parser.add_argument("-s", "--seed", default = 0, help = "Random Seed.")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse arguments
    args = parse_args()
    random.seed(args.seed) # set random seed

    # make output_dir if it doesn't yet exist
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # some constants
    METADATA_MAPPING_FILEPATH = f"{args.input_dir}/metadata_to_data.csv"
    prefix = basename(args.output_dir)
    TIMING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.timing.txt"
    MAPPING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = METADATA_MAPPING_FILEPATH, sep = ",", header = 0, index_col = False)
    METADATA = {data_path : (metadata_path if not pd.isna(metadata_path) else None) for data_path, metadata_path in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################


    # GET PATHS
    ##################################################

    # create list of paths if does not exist
    if not exists(args.paths):
        data = pd.read_csv(filepath_or_buffer = f"{args.input_dir}/dataset/dataset.full.csv", sep = ",", header = 0, index_col = False) # load in data frame
        # data = data[data["in_dataset"]] # filter # no filter as of now
        paths = pd.unique(values = data["path"]).tolist()
        with open(args.paths, "w") as file:
            file.write("\n".join(paths))
        del data, paths

    # load in paths
    with open(args.paths) as file:
        paths = [path.strip() for path in file.readlines()]
    # paths = random.sample(population = paths, k = int(0.1 * len(paths))) # for debugging, to work with a smaller sample of files

    # see if I've already completed some paths
    if exists(MAPPING_OUTPUT_FILEPATH):
        completed_paths = set(pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["musescore"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining Already-Completed Paths") if path not in completed_paths)
        paths = random.sample(population = paths, k = len(paths)) # shuffle paths

    # group each path into a subgroup
    subgroups = list(range(args.n_subgroups))
    paths_by_subgroup = [[] for _ in subgroups]
    for i, path in enumerate(paths):
        subgroup = i % len(subgroups)
        paths_by_subgroup[subgroup].append(path)
        
    # figure out partitions
    n_valid = int(args.ratio_valid * len(subgroups))
    n_test = int(args.ratio_test * len(subgroups))
    n_validtest = n_valid + n_test
    subgroups_valid = set(subgroups[:n_valid])
    subgroups_test = set(subgroups[n_valid:n_validtest])
    subgroups_train = set(filter(lambda subgroup: not ((subgroup in subgroups_valid) or (subgroup in subgroups_test)), subgroups))

    # stuff to feed in to function
    get_output_path_from_subgroup = lambda subgroup: f"{args.output_dir}/tokenized-events.{subgroup}.txt" # returns output path given the subgroup
    paths_output = list(map(get_output_path_from_subgroup, subgroups)) # output paths
    augment_factors = [args.augment if (subgroup in subgroups_train) else 1 for subgroup in subgroups]
    use_implied_durations = utils.rep(x = not bool(args.explicit_duration), times = len(subgroups))
    use_absolute_times = utils.rep(x = args.absolute_time, times = len(subgroups))

    ##################################################

    # USE MULTIPROCESSING
    ##################################################

    # perform multiprocessing
    chunk_size = 1
    start_time = perf_counter() # start the timer
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = pool.starmap(func = tokenize,
                               iterable = zip(paths_by_subgroup, # paths_input
                                              paths_output, # path_output
                                              subgroups, # subgroup
                                              augment_factors, # augment_factor
                                              use_implied_durations, # use_implied_duration
                                              use_absolute_times), # use_absolute_time
                               chunksize = chunk_size)
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed

    # print statistics
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations, n_events = (sum(statistic) for statistic in zip(*results))
    rest_ratio = round(100 * float(rest_count) / (seq_count * M), 2)
    trunc_ratio = round(100 * float(truncations) / (seq_count * M), 2)    
    logging.info("Tokenization complete. Total time: {total_time}.")
    logging.info(f"  - Processed {seq_count} training sequences")
    logging.info(f"  - Processed {n_events} events")
    logging.info(f"  - Inserted {rest_count} REST tokens ({rest_ratio}% of events)")
    logging.info(f"  - Discarded {too_short + too_long} event sequences")
    logging.info(f"      - {too_short} too short")
    logging.info(f"      - {too_long} too long")
    logging.info(f"      - {too_manyinstr} too many instruments")
    logging.info(f"  - Discarded {discarded_seqs} training sequences")
    logging.info(f"  - Truncated {truncations} duration times ({trunc_ratio}% of durations)")
    del results

    # split into partition files
    def partition(subgroups: set, partition_name: str):
        path_output_ordered = f"{args.output_dir}/{partition_name}.ordered.txt"
        subprocess.run(args = ["cat"] + list(map(get_output_path_from_subgroup, subgroups)) + [">", path_output_ordered], check = True) # combine partition into a single file
        subprocess.run(args = ["shuf", path_output_ordered, ">", f"{args.output_dir}/{partition_name}.txt"], check = True) # shuffle said file
        subprocess.run(args = ["rm", path_output_ordered], check = True) # remove the ordered file
    partition(subgroups = subgroups_valid, partition_name = "valid")
    partition(subgroups = subgroups_test, partition_name = "test")
    partition(subgroups = subgroups_train, partition_name = "train")

    # clean up
    subprocess.run(args = ["rm"] + paths_output, check = True) # removed tokenized-events files

    ##################################################

##################################################

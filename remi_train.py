# README
# Phillip Long
# August 1, 2024

# Train a REMI-Style model.

# python /home/pnlong/model_musescore/remi_train.py

# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, basename
from os import makedirs, mkdir
from multiprocessing import cpu_count # for calculating num_workers
import wandb
import datetime # for creating wandb run names linked to time of run

import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import x_transformers

import remi_dataset
import remi_representation
import utils

##################################################


# CONSTANTS
##################################################

# paths
INPUT_DIR = f"{remi_dataset.OUTPUT_DIR}/{remi_dataset.FACETS[0]}"
PATHS_TRAIN = f"{INPUT_DIR}/train.txt"
PATHS_VALID = f"{INPUT_DIR}/valid.txt"
OUTPUT_DIR = INPUT_DIR

# model constants
MAX_SEQ_LEN = 1024
MAX_BEAT = 64
DIM = 512
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.2

# training constants
N_STEPS = 100000
N_VALID_STEPS = 1000
EARLY_STOPPING_TOLERANCE = 20
LEARNING_RATE = 0.0005
LEARNING_RATE_WARMUP_STEPS = 5000
LEARNING_RATE_DECAY_STEPS = 100000
LEARNING_RATE_DECAY_MULTIPLIER = 0.1
GRAD_NORM_CLIP = 1.0

# data loader constants
BATCH_SIZE = 8

# more constants
RELEVANT_PARTITIONS = list(remi_dataset.PARTITIONS.keys())[:-1]
LOSS_OUTPUT_COLUMNS = ["step", "partition", "loss"]

# wandb
PROJECT_NAME = "PDMX"
INFER_RUN_NAME_STRING = "-1"

##################################################


# HELPER FUNCTIONS
##################################################

def get_lr_multiplier(step: int, warmup_steps: int, decay_end_steps: int, decay_end_multiplier: float) -> float:
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Train", description = "Train a REMI-Style Model.")
    parser.add_argument("-pt", "--paths_train", default = PATHS_TRAIN, type = str, help = ".txt file with absolute filepaths to training dataset")
    parser.add_argument("-pv", "--paths_valid", default = PATHS_VALID, type = str, help = ".txt file with absolute filepaths to validation dataset")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    # data
    parser.add_argument("--aug", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use data augmentation")
    # model
    parser.add_argument("--max_seq_len", default = MAX_SEQ_LEN, type = int, help = "Maximum sequence length")
    parser.add_argument("--max_beat", default = MAX_BEAT, type = int, help = "Maximum beat")
    parser.add_argument("--dim", default = DIM, type = int, help = "Model dimension")
    parser.add_argument("-l", "--layers", default = N_LAYERS, type = int, help = "Number of layers")
    parser.add_argument("--heads", default = N_HEADS, type = int, help = "Number of attention heads")
    parser.add_argument("--dropout", default = DROPOUT, type = float, help = "Dropout rate")
    parser.add_argument("--abs_pos_emb", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use absolute positional embedding")
    parser.add_argument("--rel_pos_emb", action = argparse.BooleanOptionalAction, default = False, help = "Whether to use relative positional embedding")
    # training
    parser.add_argument("--steps", default = N_STEPS, type = int, help = "Number of steps")
    parser.add_argument("--valid_steps", default = N_VALID_STEPS, type = int, help = "Validation frequency")
    parser.add_argument("--early_stopping", action = argparse.BooleanOptionalAction, default = False, help = "Whether to use early stopping")
    parser.add_argument("--early_stopping_tolerance", default = EARLY_STOPPING_TOLERANCE, type = int, help = "Number of extra validation rounds before early stopping")
    parser.add_argument("-lr", "--learning_rate", default = LEARNING_RATE, type = float, help = "Learning rate")
    parser.add_argument("--lr_warmup_steps", default = LEARNING_RATE_WARMUP_STEPS, type = int, help = "Learning rate warmup steps")
    parser.add_argument("--lr_decay_steps", default = LEARNING_RATE_DECAY_STEPS, type = int, help = "Learning rate decay end steps")
    parser.add_argument("--lr_decay_multiplier", default = LEARNING_RATE_DECAY_MULTIPLIER, type = float, help = "Learning rate multiplier at the end")
    parser.add_argument("--grad_norm_clip", default = GRAD_NORM_CLIP, type = float, help = "Gradient norm clipping")
    # others
    parser.add_argument("-bs", "--batch_size", default = BATCH_SIZE, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = int(cpu_count() / 4), type = int, help = "Number of workers for data loading")
    parser.add_argument("-r", "--resume", default = None, type = str, help = "Provide the wandb run name/id to resume a run")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # LOAD UP MODEL
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # check filepath arguments
    if not exists(args.paths_train):
        raise ValueError("Invalid --paths_train argument. File does not exist.")
    if not exists(args.paths_valid):
        raise ValueError("Invalid --paths_valid argument. File does not exist.")
    run_name = args.resume # get runname
    args.resume = (run_name != None) # convert to boolean value
    
    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    print(f"Using device: {device}")

    # load the encoding
    encoding = remi_representation.get_encoding()

    # load the indexer
    indexer = remi_representation.Indexer(data = encoding["event_code_map"])

    # create the dataset and data loader
    print(f"Creating the data loader...")
    dataset = {
        "train": remi_dataset.MusicDataset(paths = args.paths_train, encoding = encoding, indexer = indexer, encode_fn = remi_representation.encode_notes, max_seq_len = args.max_seq_len, max_beat = args.max_beat, use_augmentation = args.aug),
        "valid": remi_dataset.MusicDataset(paths = args.paths_valid, encoding = encoding, indexer = indexer, encode_fn = remi_representation.encode_notes, max_seq_len = args.max_seq_len, max_beat = args.max_beat, use_augmentation = False)
        }
    data_loader = {
        "train": torch.utils.data.DataLoader(dataset = dataset["train"], batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset["train"].collate),
        "valid": torch.utils.data.DataLoader(dataset = dataset["valid"], batch_size = args.batch_size, shuffle = False, num_workers = args.jobs, collate_fn = dataset["valid"].collate)
    }

    # create the model
    print(f"Creating model...")
    model = x_transformers.TransformerWrapper(
        num_tokens = len(indexer),
        max_seq_len = args.max_seq_len,
        attn_layers = x_transformers.Decoder(
            dim = args.dim,
            depth = args.layers,
            heads = args.heads,
            rotary_pos_emb = args.rel_pos_emb,
            emb_dropout = args.dropout,
            attn_dropout = args.dropout,
            ff_dropout = args.dropout,
        ),
        use_abs_pos_emb = args.abs_pos_emb,
    ).to(device)
    model = x_transformers.AutoregressiveWrapper(net = model)
    n_parameters = sum(p.numel() for p in model.parameters()) # statistics
    n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) # statistics (model size)

    # determine the output directory based on arguments
    model_size = int(n_parameters_trainable / 1e+6)
    output_parent_dir = args.output_dir
    output_dir_name = f"{model_size}M"
    output_dir = f"{output_parent_dir}/{output_dir_name}" # custom output directory based on arguments
    if not exists(output_dir):
        makedirs(output_dir)
    checkpoints_dir = f"{output_dir}/checkpoints" # models will be stored in the output directory
    if not exists(checkpoints_dir):
        mkdir(checkpoints_dir)

    # start a new wandb run to track the script
    group_name = basename(output_parent_dir)
    if run_name == INFER_RUN_NAME_STRING:
        run_name = next(filter(lambda name: name.startswith(output_dir_name), (run.name for run in wandb.Api().runs(f"philly/{PROJECT_NAME}", filters = {"group": group_name}))), None) # try to infer the run name
        args.resume = (run_name != None) # redefine args.resume in the event that no run name was supplied, but we can't infer one either
    if run_name is None: # in the event we need to create a new run name
        current_datetime = datetime.datetime.now().strftime("%m%d%y%H%M%S")
        run_name = f"{output_dir_name}-{current_datetime}"
    run = wandb.init(config = dict(vars(args), **{"n_parameters": n_parameters, "n_parameters_trainable": n_parameters_trainable}), resume = "allow", project = PROJECT_NAME, group = group_name, name = run_name, id = run_name) # set project title, configure with hyperparameters

    # set up the logger
    logging_output_filepath = f"{output_dir}/train.log"
    log_hyperparameters = not (args.resume and exists(logging_output_filepath))
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = logging_output_filepath, mode = "a" if args.resume else "w"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    if log_hyperparameters:
        logging.info(f"Running command: python {' '.join(sys.argv)}")
        logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
        args_output_filepath = f"{output_dir}/train_args.json"
        logging.info(f"Saved arguments to {args_output_filepath}")
        utils.save_args(filepath = args_output_filepath, args = args)
        del args_output_filepath # clear up memory
    else: # print previous loggings to stdout
        with open(logging_output_filepath, "r") as logging_output:
            print(logging_output.read())

    # load previous model and summarize if needed
    best_model_filepath = {partition: f"{checkpoints_dir}/best_model.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    model_previously_created = args.resume and all(exists(filepath) for filepath in best_model_filepath.values())
    if model_previously_created:
        model.load_state_dict(torch.load(f = best_model_filepath["valid"]))
    else:
        logging.info(f"Number of parameters: {n_parameters:,}")
        logging.info(f"Number of trainable parameters: {n_parameters_trainable:,}")
    
    # create the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    best_optimizer_filepath = {partition: f"{checkpoints_dir}/best_optimizer.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    if args.resume and all(exists(filepath) for filepath in best_optimizer_filepath.values()):
        optimizer.load_state_dict(torch.load(f = best_optimizer_filepath["valid"]))

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda step: get_lr_multiplier(step = step, warmup_steps = args.lr_warmup_steps, decay_end_steps = args.lr_decay_steps, decay_end_multiplier = args.lr_decay_multiplier))
    best_scheduler_filepath = {partition: f"{checkpoints_dir}/best_scheduler.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    if args.resume and all(exists(filepath) for filepath in best_scheduler_filepath.values()):
        scheduler.load_state_dict(torch.load(f = best_scheduler_filepath["valid"]))

    ##################################################


    # TRAINING PROCESS
    ##################################################

    # create a file to record loss metrics
    output_filepath = f"{output_dir}/loss.csv"
    loss_columns_must_be_written = not (exists(output_filepath) and args.resume) # whether or not to write column names
    if loss_columns_must_be_written: # if column names need to be written
        pd.DataFrame(columns = LOSS_OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # initialize variables
    step = 0
    min_loss = {partition: float("inf") for partition in RELEVANT_PARTITIONS}
    if not loss_columns_must_be_written: # load in previous loss info
        previous_loss = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = utils.NA_STRING, header = 0, index_col = False) # read in previous loss values
        if len(previous_loss) > 0:
            for partition in RELEVANT_PARTITIONS:
                min_loss[partition] = float(previous_loss[previous_loss["partition"] == partition]["loss"].min(axis = 0)) # get minimum loss
            step = int(previous_loss["step"].max(axis = 0)) # update step
        del previous_loss
    if args.early_stopping: # stop early?
        count_early_stopping = 0

    # print current step
    print(f"Current Step: {step:,}")

    # iterate for the specified number of steps
    train_iterator = iter(data_loader["train"])
    while step < args.steps:

        # to store loss/accuracy values
        loss = {partition: 0.0 for partition in RELEVANT_PARTITIONS}

        # TRAIN
        ##################################################

        logging.info(f"Training...")

        model.train()
        count = 0 # count number of batches
        # recent_losses = np.empty(shape = (0,)) # for moving average of loss
        for batch in (progress_bar := tqdm(iterable = range(args.valid_steps), desc = "Training")):

            # get next batch
            try:
                batch = next(train_iterator)
            except (StopIteration):
                train_iterator = iter(data_loader["train"]) # reinitialize dataset iterator
                batch = next(train_iterator)

            # get input and output pair
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            # calculate loss for the batch
            optimizer.zero_grad()
            loss_batch = model(x = seq, return_outputs = False, mask = mask)

            # update parameters according to loss
            loss_batch.backward() # calculate gradients
            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = args.grad_norm_clip)
            optimizer.step() # update parameters
            scheduler.step() # update scheduler

            # compute the moving average of the loss
            # recent_losses = np.append(arr = recent_losses, values = [float(loss_batch)], axis = 0) # float(loss_batch) because it has a gradient attribute
            # if len(recent_losses) > 10:
            #     recent_losses = np.delete(arr = recent_losses, obj = 0, axis = 0)
            # loss_batch = np.mean(a = recent_losses, axis = 0)
                        
            # set progress bar
            loss_batch = float(loss_batch) # float(loss_batch) because it has a gradient attribute
            progress_bar.set_postfix(loss = f"{loss_batch:8.4f}")

            # log training loss/accuracy for wandb
            wandb.log({f"train": loss_batch}, step = step)

            # update count
            count += len(batch)

            # add to total loss tracker
            loss["train"] += loss_batch * len(batch)

            # increment step
            step += 1

        # release GPU memory right away
        del seq, mask, loss_batch

        # compute average loss across batches
        loss["train"] /= count
        
        # log train info for wandb
        wandb.log({"train": loss["train"]}, step = step)

        ##################################################


        # VALIDATE
        ##################################################

        logging.info(f"Validating...")

        model.eval()
        with torch.no_grad():

            count = 0 # count number of batches
            for batch in tqdm(iterable = data_loader["valid"], desc = "Validating"):

                # get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)

                # pass through the model
                loss_batch = model(x = seq, return_outputs = False, mask = mask)

                # update count
                count += len(batch)

                # add to total loss tracker
                loss["valid"] += float(loss_batch) * len(batch)
                
        # release GPU memory right away
        del seq, mask, loss_batch

        # compute average loss across batches
        loss["valid"] /= count

        # output statistics
        logging.info(f"Validation loss: {loss['valid']:.4f}")

        # log validation info for wandb
        wandb.log({"valid": loss["valid"]}, step = step)

        ##################################################


        # RECORD LOSS, SAVE MODEL
        ##################################################

        # write output to file
        output = pd.DataFrame(
            data = dict(zip(
                LOSS_OUTPUT_COLUMNS,
                (utils.rep(x = step, times = len(RELEVANT_PARTITIONS)), RELEVANT_PARTITIONS, loss.values()))),
            columns = LOSS_OUTPUT_COLUMNS)
        output.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        # see whether or not to save
        is_an_improvement = False # whether or not the loss has improved
        for partition in RELEVANT_PARTITIONS:
            partition_loss = loss[partition]
            if partition_loss < min_loss[partition]:
                min_loss[partition] = partition_loss
                logging.info(f"Best {partition}_loss so far!") # log paths to which states were saved
                torch.save(obj = model.state_dict(), f = best_model_filepath[partition]) # save the model
                torch.save(obj = optimizer.state_dict(), f = best_optimizer_filepath[partition]) # save the optimizer state
                torch.save(obj = scheduler.state_dict(), f = best_scheduler_filepath[partition]) # save the scheduler state
                if args.early_stopping: # reset the early stopping counter if we found a better model
                    count_early_stopping = 0
                    is_an_improvement = True # we only care about the lack of improvement when we are thinking about early stopping, so turn off this boolean flag, since there was an improvement
        
        # increment the early stopping counter if no improvement is found
        if (not is_an_improvement) and args.early_stopping:
            count_early_stopping += 1 # increment

        # early stopping
        if args.early_stopping and (count_early_stopping > args.early_stopping_tolerance):
            logging.info(f"Stopped the training for no improvements in {args.early_stopping_tolerance} rounds.")
            break

        ##################################################

    ##################################################

    
    # STATISTICS AND CONCLUSION
    ##################################################

    # log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_loss['valid']}")
    wandb.log({f"min_valid_loss": min_loss['valid']})

    # finish the wandb run
    wandb.finish()

    # output model name to list of models
    models_output_filepath = f"{output_parent_dir}/models.txt"
    if exists(models_output_filepath):
        with open(models_output_filepath, "r") as models_output: # read in list of trained models
            models = {model.strip() for model in models_output.readlines()} # use a set because better for `in` operations
    else:
        models = set()
    with open(models_output_filepath, "a") as models_output:
        if output_dir_name not in models: # check if in list of trained models
            models_output.write(output_dir_name + "\n") # add model to list of trained models if it isn't already there

    ##################################################

##################################################


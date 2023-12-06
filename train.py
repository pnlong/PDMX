# README
# Phillip Long
# November 25, 2023

# Train a neural network.

# python /home/pnlong/model_musescore/train.py

# Absolute positional embedding (APE):
# python /home/pnlong/model_musescore/train.py

# Relative positional embedding (RPE):
# python /home/pnlong/model_musescore/train.py --no-abs_pos_emb --rel_pos_emb

# No positional embedding (NPE):
# python /home/pnlong/model_musescore/train.py --no-abs_pos_emb --no-rel_pos_emb


# IMPORTS
##################################################

import argparse
import logging
from os import makedirs, mkdir
from os.path import exists, basename
import pprint
import shutil
import sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import wandb

import dataset
import music_x_transformers
import representation
import encode
import utils

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/data2/pnlong/musescore/data"
PATHS_TRAIN = f"{DATA_DIR}/train.txt"
PATHS_VALID = f"{DATA_DIR}/valid.txt"
OUTPUT_DIR = "/data2/pnlong/musescore/data/train"
ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("-i", "--data_dir", default = DATA_DIR, type = str, help = "Input data directory")
    parser.add_argument("-t", "--paths_train", default = PATHS_TRAIN, type = str, help = ".txt file with absolute filepaths to training dataset.")
    parser.add_argument("-v", "--paths_valid", default = PATHS_VALID, type = str, help = ".txt file with absolute filepaths to validation dataset.")
    parser.add_argument("-e", "--encoding", default = ENCODING_FILEPATH, type = str, help = ".json file with encoding information.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    # data
    parser.add_argument("-bs", "--batch_size", default = 8, type = int, help = "Batch size")
    parser.add_argument("--aug", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use data augmentation")
    parser.add_argument("-c", "--conditioning", default = encode.DEFAULT_CONDITIONING, choices = encode.CONDITIONINGS, type = str, help = "Conditioning type")
    parser.add_argument("-s", "--sigma", default = encode.SIGMA, type = float, help = "Sigma anticipation value (for anticipation conditioning)")
    # model
    parser.add_argument("--max_seq_len", default = 1024, type = int, help = "Maximum sequence length")
    parser.add_argument("--max_beat", default = 256, type = int, help = "Maximum number of beats")
    parser.add_argument("--dim", default = 512, type = int, help = "Model dimension")
    parser.add_argument("-l", "--layers", default = 6, type = int, help = "Number of layers")
    parser.add_argument("--heads", default = 8, type = int, help = "Number of attention heads")
    parser.add_argument("--dropout", default = 0.2, type = float, help = "Dropout rate")
    parser.add_argument("--abs_pos_emb", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use absolute positional embedding")
    parser.add_argument("--rel_pos_emb", action = argparse.BooleanOptionalAction, default = False, help = "Whether to use relative positional embedding")
    # training
    parser.add_argument("--steps", default = 200000, type = int, help = "Number of steps")
    parser.add_argument("--valid_steps", default = 1000, type = int, help = "Validation frequency")
    parser.add_argument("--early_stopping", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use early stopping")
    parser.add_argument("--early_stopping_tolerance", default = 20, type = int, help = "Number of extra validation rounds before early stopping")
    parser.add_argument("-lr", "--learning_rate", default = 0.0005, type = float, help = "Learning rate")
    parser.add_argument("--lr_warmup_steps", default = 5000, type = int, help = "Learning rate warmup steps")
    parser.add_argument("--lr_decay_steps", default = 100000, type = int, help = "Learning rate decay end steps")
    parser.add_argument("--lr_decay_multiplier", default = 0.1, type = float, help = "Learning rate multiplier at the end")
    parser.add_argument("--grad_norm_clip", default = 1.0, type = float, help = "Gradient norm clipping")
    # others
    parser.add_argument("-r", "--resume", action = "store_true", help = "Whether or not to resume training from most recently-trained step")
    parser.add_argument("-g", "--gpu", default = 0, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 4, type = int, help = "Number of workers for data loading")
    return parser.parse_args(args = args, namespace = namespace)

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


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # check filepath arguments
    if not exists(args.paths_train):
        raise ValueError("Invalid --paths_train argument. File does not exist.")
    if not exists(args.paths_valid):
        raise ValueError("Invalid --paths_valid argument. File does not exist.")
    if not exists(args.data_dir):
        raise ValueError("Invalid --data_dir argument. Directory does not exist, so there is no data on which to train.")
    args.output_dir = f"{args.output_dir}/{args.learning_rate}_{args.layers}_{args.heads}_{args.dim}"
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    CHECKPOINTS_DIR = f"{args.output_dir}/checkpoints"
    if not exists(CHECKPOINTS_DIR):
        mkdir(CHECKPOINTS_DIR)

    # set up the logger
    logging_output_filepath = f"{args.output_dir}/train.log"
    log_hyperparameters = not (args.resume and exists(logging_output_filepath))
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = logging_output_filepath, mode = "a" if args.resume else "w"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    if log_hyperparameters:
        logging.info(f"Running command: python {' '.join(sys.argv)}")
        logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
        args_output_filepath = f"{args.output_dir}/train_args.json"
        logging.info(f"Saved arguments to {args_output_filepath}")
        utils.save_args(filepath = args_output_filepath, args = args)
        del args_output_filepath # clear up memory
    else: # print previous loggings to stdout
        with open(logging_output_filepath, "r") as logging_output:
            print(logging_output.read())

    # start a new wandb run to track the script
    run = wandb.init(project = "ExpressionNet-Train", config = vars(args), name = basename(args.output_dir), resume = args.resume) # set project title, configure with hyperparameters

    ##################################################


    # SET UP TRAINING (LOAD DATASET, DATA LOADERS)
    ##################################################

    # get the specified device
    args.gpu = abs(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    logging.info(f"Using device: {device}")

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    logging.info(f"Creating the data loader...")
    dataset_train = dataset.MusicDataset(paths = args.paths_train, encoding = encoding, conditioning = args.conditioning, sigma = args.sigma, max_seq_len = args.max_seq_len, max_beat = args.max_beat, use_augmentation = args.aug)
    data_loader_train = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate)
    dataset_valid = dataset.MusicDataset(paths = args.paths_valid, encoding = encoding, conditioning = args.conditioning, sigma = args.sigma, max_seq_len = args.max_seq_len, max_beat = args.max_beat, use_augmentation = args.aug)
    data_loader_valid = torch.utils.data.DataLoader(dataset = dataset_valid, batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset.MusicDataset.collate)

    # create the model
    logging.info(f"Creating model...")
    model = music_x_transformers.MusicXTransformer(
        dim = args.dim,
        encoding = encoding,
        depth = args.layers,
        heads = args.heads,
        max_seq_len = args.max_seq_len,
        max_beat = args.max_beat,
        rotary_pos_emb = args.rel_pos_emb,
        use_abs_pos_emb = args.abs_pos_emb,
        embedding_dropout = args.dropout,
        attention_dropout = args.dropout,
        ff_dropout = args.dropout,
    ).to(device)
    best_model_filepath = f"{CHECKPOINTS_DIR}/best_model.pth"
    model_previously_created = args.resume and exists(best_model_filepath)
    if model_previously_created:
        model.load_state_dict(torch.load(f = best_model_filepath))

    # summarize the model
    if model_previously_created:
        n_parameters = sum(p.numel() for p in model.parameters())
        n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: {n_parameters}")
        logging.info(f"Number of trainable parameters: {n_parameters_trainable}")
        wandb.log({"n_parameters": n_parameters, "n_parameters_trainable": n_parameters_trainable})
    
    # create the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    best_optimizer_filepath = f"{CHECKPOINTS_DIR}/best_optimizer.pth"
    if args.resume and exists(best_optimizer_filepath):
        optimizer.load_state_dict(torch.load(f = best_optimizer_filepath))

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer = optimizer,
        lr_lambda = lambda step: get_lr_multiplier(
            step = step,
            warmup_steps = args.lr_warmup_steps,
            decay_end_steps = args.lr_decay_steps,
            decay_end_multiplier = args.lr_decay_multiplier
        )
    )
    best_scheduler_filepath = f"{CHECKPOINTS_DIR}/best_scheduler.pth"
    if args.resume and exists(best_scheduler_filepath):
        scheduler.load_state_dict(torch.load(f = best_scheduler_filepath))

    # create a file to record losses
    loss_output_filepath = f"{args.output_dir}/loss.csv"
    loss_columns_must_be_written = not (exists(loss_output_filepath) and args.resume) # whether or not to write loss column names
    loss_csv = open(loss_output_filepath, "a" if args.resume else "w") # open loss file
    if loss_columns_must_be_written: # if column names need to be written
        loss_csv.write("step,train_loss,valid_loss," + ",".join(map(lambda dimension: f"{dimension}_loss", encoding["dimensions"])) + "\n")

    ##################################################


    # TRAINING PROCESS
    ##################################################

    # initialize variables
    step = 0
    min_valid_loss = float("inf")
    if not loss_columns_must_be_written:
        min_valid_loss = float(pd.read_csv(filepath_or_buffer = loss_output_filepath, sep = ",", header = 0, index_col = False)["valid_loss"].min(axis = 0)) # get minimum loss by reading in loss values and extracting the minimum
    if args.early_stopping:
        count_early_stopping = 0

    # iterate for the specified number of steps
    train_iterator = iter(data_loader_train)
    while step < args.steps:

        # TRAIN
        ##################################################

        logging.info(f"Training...")

        model.train()
        recent_losses = []
        for batch in (progress_bar := tqdm(iterable = range(args.valid_steps), desc = "Training")):

            # get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(data_loader_train) # reinitialize dataset iterator
                batch = next(train_iterator)

            # get input and output pair
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            # update the model parameters
            optimizer.zero_grad()
            loss = model(seq, mask = mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = args.grad_norm_clip)
            optimizer.step()
            scheduler.step()

            # compute the moving average of the loss
            recent_losses.append(float(loss))
            if len(recent_losses) > 10:
                del recent_losses[0]
            train_loss = np.mean(a = recent_losses, axis = 0)
            progress_bar.set_postfix(loss = f"{train_loss:8.4f}")

            # log training loss for wandb
            wandb.log({"step": step, "train/loss": train_loss})

            # increment step
            step += 1

        # release GPU memory right away
        del seq, mask

        ##################################################


        # VALIDATE
        ##################################################

        logging.info(f"Validating...")

        model.eval()
        with torch.no_grad():

            total_loss = 0
            total_losses = [0] * len(encoding["dimensions"])
            count = 0
            for batch in data_loader_valid:

                # get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)

                # pass through the model
                loss, losses = model(seq, return_list = True, mask = mask)

                # accumulate validation loss
                count += len(batch)
                total_loss += len(batch) * float(loss)
                for index in range(len(encoding["dimensions"])):
                    total_losses[index] += float(losses[index])
        
        # get loss
        valid_loss = total_loss / count
        individual_losses = {dimension: (loss / count) for dimension, loss in zip(encoding["dimensions"], total_losses)}

        # output statistics
        logging.info(f"Validation loss: {valid_loss:.4f}")
        logging.info(f"Individual losses: type = {individual_losses['type']:.4f}, beat: {individual_losses['beat']:.4f}, position: {individual_losses['position']:.4f}, value: {individual_losses['value']:.4f}, duration: {individual_losses['duration']:.4f}, instrument: {individual_losses['instrument']:.4f}")

        # log validation info for wandb
        wandb.log({"step": step, "valid/loss": valid_loss})
        for dimension, loss_val in individual_losses.items():
            wandb.log({"step": step, f"valid/loss_{dimension}": loss_val})

        # release GPU memory right away
        del seq, mask

        ##################################################


        # RECORD LOSS, SAVE MODEL
        ##################################################

        # write losses to file
        loss_csv.write(f"{step},{train_loss},{valid_loss}," + ",".join(map(str, individual_losses)) + "\n")

        # save the model
        checkpoint_filepath = f"{CHECKPOINTS_DIR}/model_{step}.pth"
        torch.save(obj = model.state_dict(), f = checkpoint_filepath)

        # save the optimizer states
        optimizer_filepath = f"{CHECKPOINTS_DIR}/optimizer_{step}.pth"
        torch.save(obj = optimizer.state_dict(), f = optimizer_filepath)

        # save the scheduler states
        scheduler_filepath = f"{CHECKPOINTS_DIR}/scheduler_{step}.pth"
        torch.save(obj = scheduler.state_dict(), f = scheduler_filepath)

        # log paths to which states were saved
        logging.info(f"Model: {checkpoint_filepath}; Optimizer: {optimizer_filepath}; Scheduler: {scheduler_filepath}")

        # copy the model if it is the best model so far
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            shutil.copyfile(src = checkpoint_filepath, dst = best_model_filepath)
            shutil.copyfile(src = optimizer_filepath, dst = best_optimizer_filepath)
            shutil.copyfile(src = scheduler_filepath, dst = best_scheduler_filepath)
            if args.early_stopping: # reset the early stopping counter if we found a better model
                count_early_stopping = 0
        elif args.early_stopping: # increment the early stopping counter if no improvement is found
            count_early_stopping += 1

        # early stopping
        if (args.early_stopping and count_early_stopping > args.early_stopping_tolerance):
            logging.info(f"Stopped the training for no improvements in {args.early_stopping_tolerance} rounds.")
            break

        ##################################################

    
    # STATISTICS
    ##################################################

    # log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_valid_loss}")
    wandb.log({"min_valid_loss": min_valid_loss})

    # close the file
    loss_csv.close()

    # finish the wandb run
    wandb.finish()

    ##################################################

##################################################

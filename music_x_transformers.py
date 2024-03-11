# README
# Phillip Long
# November 25, 2023

# Create neural network model.

# python /home/pnlong/model_musescore/music_x_transformers.py


# IMPORTS
##################################################

import argparse
import logging
from os.path import exists as file_exists
import sys
from typing import Union, List
from warnings import warn

from encode import SIGMA
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.autoregressive_wrapper import (exists, top_a, top_k, top_p)
from x_transformers.x_transformers import (AbsolutePositionalEmbedding, AttentionLayers, Decoder, TokenEmbedding, always, default, exists)

import representation
import utils

##################################################


# CONSTANTS
##################################################

DEFAULT_IGNORE_INDEX = -100

##################################################


# MUSIC TRANSFORMER WRAPPER CLASS
##################################################

class MusicTransformerWrapper(nn.Module):

    # INITIALIZER
    ##################################################

    def __init__(
            self,
            *,
            encoding: dict,
            max_seq_len: int,
            attn_layers: AttentionLayers,
            emb_dim: int = None,
            max_temporal: int = None,
            max_mem_len: float = 0.0,
            shift_mem_down: int = 0,
            emb_dropout: int = 0.0,
            num_memory_token: int = None,
            tie_embedding: bool = False,
            use_abs_pos_emb: bool = True,
            l2norm_embed: bool = False
        ):
        
        # initialize
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), "attention layers must be one of Encoder or Decoder" # make sure attn_layers is of the correct type

        # get dimensions
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        # set some lengths
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        # adjust n_tokens
        n_tokens = encoding["n_tokens"]
        use_absolute_time = not (("beat" in encoding["dimensions"]) and ("position" in encoding["dimensions"]))
        if max_temporal is not None:
            temporal_dim = encoding["dimensions"].index("time" if use_absolute_time else "beat")
            n_tokens[temporal_dim] = int(max_temporal / (representation.TIME_STEP if use_absolute_time else 1)) + 1
            del temporal_dim

        # deal with embedding
        self.l2norm_embed = l2norm_embed
        self.token_embedding = nn.ModuleList(modules = [TokenEmbedding(dim = emb_dim, num_tokens = n, l2norm_embed = l2norm_embed) for n in n_tokens])
        self.positional_embedding = AbsolutePositionalEmbedding(dim = emb_dim, max_seq_len = max_seq_len, l2norm_embed = l2norm_embed) if (use_abs_pos_emb and not attn_layers.has_pos_emb) else always(0)

        # dropout
        self.emb_dropout = nn.Dropout(p = emb_dropout)

        # embedding and layers
        self.project_embedding = nn.Linear(in_features = emb_dim, out_features = dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(normalized_shape = dim)

        # run initializer helper function
        self.init_()

        # get to logits
        self.to_logits = nn.ModuleList(modules = [nn.Linear(in_features = dim, out_features = n) for n in n_tokens]) if not tie_embedding else [lambda t: t @ embedding.weight.t() for embedding in self.token_embedding]

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_token = default(num_memory_token, 0)
        self.num_memory_token = num_memory_token
        if num_memory_token > 0:
            self.memory_tokens = nn.Parameter(data = torch.randn(num_memory_token, dim))

    # intialize helper
    def init_(self):

        if self.l2norm_embed:
            for embedding in self.token_embedding:
                nn.init.normal_(tensor = embedding.emb.weight, std = 1e-5)
            nn.init.normal_(self.positional_embedding.emb.weight, std = 1e-5)
            return

        else:
            for embedding in self.token_embedding:
                nn.init.kaiming_normal_(tensor = embedding.emb.weight)

    ##################################################


    # FORWARD PASS
    ##################################################

    def forward(
        self,
        x: torch.tensor, # shape : (b, n, d)
        return_embeddings: bool = False,
        mask: torch.tensor = None,
        return_mems: bool = False,
        return_attn: bool = False,
        mems: list = None,
        **kwargs,
    ):
        
        # extract shape info from x
        b, _, _ = x.shape
        num_mem = self.num_memory_token

        # calculate x
        x = sum(embedding(x[..., i]) for i, embedding in enumerate(self.token_embedding))
        x += self.positional_embedding(x)
        x = self.emb_dropout(x)
        x = self.project_embedding(x)

        # deal with multiple mems
        if num_mem > 0:
            memory = repeat(tensor = self.memory_tokens, pattern = "n d -> b n d", b = b)
            x = torch.cat(tensor = (memory, x), dim = 1)
            if exists(mask): # auto-handle masking after appending memory tokens
                mask = F.pad(input = mask, pad = (num_mem, 0), value = True)

        # if shifting memory down
        if self.shift_mem_down and exists(mems):
            mems_left, mems_right = mems[: self.shift_mem_down], mems[self.shift_mem_down :]
            mems = [*mems_right, *mems_left]

        # intermediates
        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)
        x = self.norm(x)

        # redefine memory and x
        memory, x = x[:, :num_mem], x[:, num_mem:]
        output = [to_logit(x) for to_logit in self.to_logits] if not return_embeddings else x

        # if returning mems
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(tensors = pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems))
            return output, new_mems

        # if returning attention
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return output, attn_maps

        # otherwise, return output
        return output

    ##################################################

##################################################


# HELPER FUNCTION TO SAMPLE
##################################################

def sample(logits: torch.tensor, kind: str, threshold: float, temperature: float, min_p_pow: float, min_p_ratio: float):
    """Sample from the logits with a specific sampling strategy."""
    if kind == "top_k":
        probs = F.softmax(top_k(logits = logits, frac_num_tokens = threshold) / temperature, dim = -1)
    elif kind == "top_p":
        probs = F.softmax(top_p(logits = logits, thres = threshold) / temperature, dim = -1)
    elif kind == "top_a":
        probs = F.softmax(top_a(logits = logits, min_p_pow = min_p_pow, min_p_ratio = min_p_ratio) / temperature, dim = -1)
    # elif kind == "entmax":
    #     probs = entmax(logits / temperature, alpha = ENTMAX_ALPHA, dim = -1)
    else:
        raise ValueError(f"Unknown sampling strategy: {kind}")

    return torch.multinomial(input = probs, num_samples = 1)

##################################################


# MUSIC AUTOREGRESSIVE WRAPPER
##################################################

class MusicAutoregressiveWrapper(nn.Module):

    # INTIALIZER
    ##################################################

    def __init__(self, net: MusicTransformerWrapper, encoding: dict, ignore_index: int = DEFAULT_IGNORE_INDEX, pad_value: int = 0):
        
        # intialize some fields
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.use_absolute_time = not (("beat" in encoding["dimensions"]) and ("position" in encoding["dimensions"]))

        # get the type codes
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.expressive_feature_type_code = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]
        self.value_type_codes = {encoding["type_code_map"]["note"], encoding["type_code_map"]["grace-note"], self.expressive_feature_type_code}

        # for masking out expressive features
        self.expressive_feature_value_codes = utils.rep(x = False, times = 129) + utils.rep(x = True, times = len(encoding["value_code_map"]) - 129)

        # get the dimension indices
        self.dimensions = {dimension: i for i, dimension in enumerate(encoding["dimensions"])}
        assert self.dimensions["type"] == 0

    ##################################################

    
    # GENERATE TOKENS
    ##################################################

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.tensor, # shape : (b, n, d)
        seq_len: int,
        eos_token: str = None,
        temperature: Union[float, List[float]] = 1.0, # int or list of int
        filter_logits_fn: Union[str, List[str]] = "top_k", # str or list of str
        filter_thres: Union[float, List[float]] = 0.9, # int or list of int
        min_p_pow: float = 2.0,
        min_p_ratio: float = 0.02,
        monotonicity_dim: Union[int, List[int]] = None,
        return_attn: bool = False,
        return_logits: bool = False,
        notes_only: bool = False,
        is_anticipation: bool = False,
        sigma: float = SIGMA, # NOT IN SECONDS, tokenized SIGMA value
        **kwargs,
    ):
        
        # get shape from start_tokens
        n_sequences_in_batch, t, dim = start_tokens.shape

        # convert temperature to list
        if isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert len(temperature) == dim, f"`temperature` must be of length {dim}"
        # convert filter_logits_fn to list
        if isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert len(filter_logits_fn) == dim, f"`filter_logits_fn` must be of length {dim}"
        # convert filter_thres to list
        if isinstance(filter_thres, (float, int)):
            filter_thres = [filter_thres] * dim
        elif len(filter_thres) == 1:
            filter_thres = filter_thres * dim
        else:
            assert len(filter_thres) == dim, f"`filter_thres` must be of length {dim}"
        # deal with monotonicity_dim
        if isinstance(monotonicity_dim, str):
            monotonicity_dim = [self.dimensions[monotonicity_dim]]
        else:
            monotonicity_dim = [self.dimensions[dim] for dim in monotonicity_dim]

        # get some constants
        was_training = self.net.training
        n_dims = len(start_tokens.shape)
        if n_dims == 2:
            start_tokens = start_tokens[None, :, :]

        # get expressive features (controls) for anticipation
        if is_anticipation and self.use_absolute_time:
            time_dim = self.dimensions["time"]
            current_time = [0 for _ in range(n_sequences_in_batch)]
            expressive_features = [[] for _ in range(n_sequences_in_batch)]
            for seq_index in range(len(expressive_features)): # filter expressive features so that its only the features after the last note in the prefix
                for i in range(len(start_tokens[seq_index]) - 1, -1, -1):
                    if start_tokens[seq_index, i, 0] != self.expressive_feature_type_code: # if we encounter a note, stop the for loop
                        current_time[seq_index] = start_tokens[seq_index, i, time_dim].item()
                        break
                    expressive_features[seq_index].append(start_tokens[seq_index, i].unsqueeze(dim = 0))
        elif is_anticipation and (not self.use_absolute_time):
            warn(message = "Anticipation sampling not implemented for metrical time.")

        # deal with masking
        self.net.eval()
        output = [seq.unsqueeze(dim = 0) for seq in start_tokens]
        if is_anticipation and self.use_absolute_time: # if anticipation, remove relevant controls from prefix
            for seq_index in range(len(output)): # remove relevant controls from prefix
                n_expressive_features_remaining = len(expressive_features[seq_index])
                if n_expressive_features_remaining > 0:
                    output[seq_index] = output[seq_index][:, :-n_expressive_features_remaining]
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.ones(size = (start_tokens.shape[:2]), dtype = torch.bool, device = start_tokens.device)

        # deal with current values
        current_values = {d: torch.max(input = start_tokens[:, :, d], dim = 1)[0] for d in monotonicity_dim} if (monotonicity_dim is not None) else None

        # logits to return
        if return_logits:
            logits_to_return = [[] for _ in range(dim)]

        # loop through seq
        value_dim = self.dimensions["value"]
        instrument_dim = self.dimensions["instrument"] # get index of instrument
        for _ in range(seq_len):

            # for anticipation, add relevant expressive features
            if is_anticipation and self.use_absolute_time:
                for seq_index in range(len(output)):
                    n_expressive_features_remaining = len(expressive_features[seq_index])
                    for i in range(n_expressive_features_remaining - 1, -1, -1):
                        if expressive_features[seq_index][i][:, time_dim].item() <= (current_time[seq_index] + sigma):
                            control = expressive_features[seq_index].pop(-1) # pop expressive feature
                            output[seq_index] = torch.cat(tensors = (output, control.unsqueeze(dim = 0)), dim = 1) # add to output
                            
            # get current x and mask
            mask = mask[:, -self.max_seq_len:]
            x = [seq[:, -self.max_seq_len:] for seq in output]
            longest_seq_len = max((seq.shape[1] for seq in x)) # get the longest sequence length so we can front pad sequences of unequal length
            x = torch.cat(tensors = [F.pad(input = seq, pad = (0, 0, longest_seq_len - seq.shape[1], 0), mode = "constant", value = self.pad_value) for seq in x], dim = 0).to(mask.device)

            # get logits (and perhaps attn)
            if return_attn:
                logits, attn = self.net(x = x, mask = mask, return_attn = True, **kwargs)
                logits = [logit[:, -1, :] for logit in logits]
            else:
                logits = [logit[:, -1, :] for logit in self.net(x, mask = mask, return_attn = False, **kwargs)]
            
            # add to return logits if necessary
            if return_logits:
                for i in range(dim):
                    logits_to_return[i].append(logits[i])

            # enforce monotonicity
            if (monotonicity_dim is not None) and (0 in monotonicity_dim):
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # filter out sos token
            logits[0][:, 0] = -float("inf")

            # sample from the logits
            event_types = sample(logits = logits[0], kind = filter_logits_fn[0], threshold = filter_thres[0], temperature = temperature[0], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio)

            # update current values
            if (monotonicity_dim is not None) and (0 in monotonicity_dim):
                current_values[0] = torch.maximum(input = current_values[0], other = event_types.reshape(-1))

            # don't allow for sampling of expressive features
            if notes_only:
                logits[0][:, self.expressive_feature_type_code] = -float("inf") # don't allow for the expressive feature type
                logits[value_dim][:, self.expressive_feature_value_codes] = -float("inf") # don't allow for expressive feature values

            # iterate after each sample
            events = [[event_type] for event_type in event_types]
            for seq_index, event_type in enumerate(event_types): # seq_index is the sequence index within the batch
                event_type_code = event_type.item()

                # a start-of-song, end-of-song or start-of-notes code
                if event_type_code in {self.sos_type_code, self.eos_type_code, self.son_type_code}:
                    events[seq_index] += [torch.zeros_like(input = event_type)] * (len(logits) - 1)

                # an instrument code
                elif event_type_code == self.instrument_type_code:
                    events[seq_index] += [torch.zeros_like(input = event_type)] * (len(logits) - 2)
                    logits[instrument_dim][:, 0] = -float("inf") # avoid none in instrument dim
                    sampled = sample(logits = logits[instrument_dim][seq_index : seq_index + 1], kind = filter_logits_fn[instrument_dim], threshold = filter_thres[instrument_dim], temperature = temperature[instrument_dim], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio)[0]
                    events[seq_index].append(sampled)

                # a value code
                elif event_type_code in self.value_type_codes:
                    for d in range(1, dim):
                        # enforce monotonicity
                        if (monotonicity_dim is not None) and (d in monotonicity_dim):
                            logits[d][seq_index, : current_values[d][seq_index]] = -float("inf")
                        # sample from the logits
                        logits[d][:, 0] = -float("inf") # avoid none
                        sampled = sample(logits = logits[d][seq_index : seq_index + 1], kind = filter_logits_fn[d], threshold = filter_thres[d], temperature = temperature[d], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio)[0]
                        events[seq_index].append(sampled)
                        if is_anticipation and self.use_absolute_time and (d == time_dim):
                            current_time[seq_index] = sampled
                        # update current values
                        if (monotonicity_dim is not None) and (d in monotonicity_dim):
                            current_values[d][seq_index] = torch.max(input = current_values[d][seq_index], other = sampled)[0]
                else:
                    raise ValueError(f"Unknown event type code: {event_type_code}")

            # wrangle output a bit
            for seq_index in range(len(output)):
                output[seq_index] = torch.cat(tensors = (output[seq_index], torch.cat(tensors = events[seq_index]).expand(1, -1).unsqueeze(dim = 0)), dim = 1)
            # stacked = torch.stack(tensors = [torch.cat(tensors = event).expand(1, -1) for event in events], dim = 0)
            # output = torch.cat(tensors = (output, stacked), dim = 1)
            mask = F.pad(input = mask, pad = (0, 1), value = True)

            # if end of song token is provided
            if exists(eos_token):
                is_eos_tokens = [(seq[:, :, 0] == eos_token).squeeze(dim = 0) for seq in output]
                # mask out everything after the eos tokens
                if all([is_eos_tokens_seq.any() for is_eos_tokens_seq in is_eos_tokens]): # if all sequences in the batch have an eos token, pad out everything after
                    longest_seq_len = max((seq.shape[1] for seq in output)) # get the longest sequence length
                    for seq_index, is_eos_tokens_seq in enumerate(is_eos_tokens):
                        first_eos_token_index = torch.argmax(input = is_eos_tokens_seq.byte(), dim = None).item() # index of the first eos token
                        output[seq_index][:, first_eos_token_index + 1:] = self.pad_value # pad after eos token
                        output[seq_index] = F.pad(input = output[seq_index], pad = (0, 0, 0, longest_seq_len - output[seq_index].shape[1]), mode = "constant", value = self.pad_value) # make sure all sequences are the same length
                    break

        # wrangle output
        output = torch.cat(tensors = output, dim = 0).to(start_tokens.device)
        output = output[:, t:]
        if n_dims == 1:
            output = output.squeeze(0)

        # turn off training
        self.net.train(was_training)

        # either return just the output or attn/logits as well
        if return_logits:
            logits_to_return = [torch.cat(tensors = logits_to_return[i], dim = 0).unsqueeze(dim = 0) for i in range(dim)]
            return output, logits_to_return
        if return_attn:
            return output, attn
        return output

    ##################################################


    # FORWARD PASS
    ##################################################

    def forward(self, x: torch.tensor, return_list: bool = False, reduce: bool = True, return_output: bool = False, **kwargs):

        # create subsets of x
        xi = x[:, :-1] # input
        xo = x[:, 1:] # expected output

        # help auto-solve a frequent area of confusion around input masks in auto-regressive, if user supplies a mask that is only off by one from the source seq, resolve it for them
        mask = kwargs.get("mask", None)
        if (mask is not None) and (mask.shape[1] == x.shape[1]):
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        # create output
        output = self.net(x = xi, **kwargs)
        losses = [F.cross_entropy(input = output[i].transpose(1, 2), target = xo[..., i], ignore_index = self.ignore_index, reduction = "none") for i in range(len(output))] # calculate losses
        losses = torch.cat(tensors = [torch.unsqueeze(input = losses_dimension, dim = -1) for losses_dimension in losses], dim = -1) # combine list of losses into a matrix
        loss_by_field = torch.mean(input = losses, dim = list(range(len(losses.shape) - 1))) # calculate mean for each field, the field is the last dimension, so mean over all but the last dimension
        loss = torch.sum(input = loss_by_field, dim = None) # singular loss value (sum of the average losses for each field)
        if reduce:
            losses = loss_by_field
        del loss_by_field

        # return the losses or just loss
        if return_output:
            return loss, losses, output
        elif return_list:
            return loss, losses
        else:
            return loss

    ##################################################

##################################################


# THE MUSIC X TRANSFORMER
##################################################

class MusicXTransformer(nn.Module):
    
    # initializer
    def __init__(self, *, dim, encoding, **kwargs):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_temporal": kwargs.pop("max_temporal"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.decoder = MusicTransformerWrapper(encoding = encoding, attn_layers = Decoder(dim = dim, **kwargs), **transformer_kwargs)
        self.decoder = MusicAutoregressiveWrapper(net = self.decoder, encoding = encoding)

    # generate
    @torch.no_grad()
    def generate(self, seq_in: torch.tensor, seq_len: int, **kwargs):
        return self.decoder.generate(start_tokens = seq_in, seq_len = seq_len, **kwargs)

    # forward pass
    def forward(self, seq: torch.tensor, mask: torch.tensor = None, **kwargs):
        return self.decoder(x = seq, mask = mask, **kwargs)

##################################################


# CONSTANTS
##################################################

ENCODING_FILEPATH = "/data2/pnlong/musescore/encoding.json"

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--encoding", default = ENCODING_FILEPATH, type = str, help = ".json file with encoding information.")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # parse the command-line arguments
    args = parse_args()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", stream = sys.stdout)

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if file_exists(args.encoding) else representation.get_encoding()

    # create the model
    model = MusicXTransformer(
        dim = 128,
        encoding = encoding,
        depth = 3,
        heads = 4,
        max_seq_len = 1024,
        max_temporal = 256,
        rel_pos_bias = True, # relative positional bias
        rotary_pos_emb = True, # rotary positional encoding
        emb_dropout = 0.1,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
    )

    # summarize the model
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # create test data
    seq = torch.randint(low = 0, high = 4, size = (1, 1024, 6))
    mask = torch.ones(size = (1, 1024)).bool()

    # pass test data through the model
    loss = model(seq, mask = mask)
    loss.backward()

##################################################

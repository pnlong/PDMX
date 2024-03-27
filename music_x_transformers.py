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
from typing import Union, List, Any
from warnings import warn
from copy import deepcopy
import numpy as np

from encode import SIGMA, DEFAULT_ENCODING
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
MAX_TEMPORAL_TOKEN_BUILDUP_LIMIT = 5 # max number of tokens that can build up at final timestep before end of song token can be placed

##################################################


# MUSIC TRANSFORMER WRAPPER CLASS
##################################################

class MusicTransformerWrapper(nn.Module):

    # INITIALIZER
    ##################################################

    def __init__(
            self,
            *,
            encoding: dict = DEFAULT_ENCODING,
            max_seq_len: int,
            attn_layers: AttentionLayers,
            unidimensional: bool = False,
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
        self.unidimensional = unidimensional
        emb_dim = default(emb_dim, dim)

        # set some lengths
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        # adjust n_tokens
        n_tokens = encoding["n_tokens"]
        use_absolute_time = encoding["use_absolute_time"]
        if max_temporal is not None:
            temporal_dim = encoding["dimensions"].index("time" if use_absolute_time else "beat")
            n_tokens[temporal_dim] = int(max_temporal / (representation.TIME_STEP if use_absolute_time else 1)) + 1
            del temporal_dim

        # deal with embedding
        self.l2norm_embed = l2norm_embed
        if self.unidimensional:
            n_tokens_total = sum(n_tokens)
            self.token_embedding = TokenEmbedding(dim = emb_dim, num_tokens = n_tokens_total, l2norm_embed = self.l2norm_embed)
        else:
            self.token_embedding = nn.ModuleList(modules = [TokenEmbedding(dim = emb_dim, num_tokens = n_token, l2norm_embed = self.l2norm_embed) for n_token in n_tokens])
        self.positional_embedding = AbsolutePositionalEmbedding(dim = emb_dim, max_seq_len = self.max_seq_len, l2norm_embed = self.l2norm_embed) if (use_abs_pos_emb and not attn_layers.has_pos_emb) else always(0)

        # dropout
        self.emb_dropout = nn.Dropout(p = emb_dropout)

        # embedding and layers
        self.project_embedding = nn.Linear(in_features = emb_dim, out_features = dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(normalized_shape = dim)

        # run initializer helper function
        self.init_()

        # get to logits
        if self.unidimensional:
            self.to_logits = nn.Linear(in_features = dim, out_features = n_tokens_total) if (not tie_embedding) else (lambda t: t @ self.token_embedding.weight.t())
        else:
            self.to_logits = nn.ModuleList(modules = [nn.Linear(in_features = dim, out_features = n) for n in n_tokens]) if (not tie_embedding) else [lambda t: t @ embedding.weight.t() for embedding in self.token_embedding]

        # memory tokens (like [cls]) from Memory Transformers paper
        self.n_tokens_per_event = len(encoding["dimensions"]) if self.unidimensional else 1
        num_memory_token = default(num_memory_token, 0) * self.n_tokens_per_event
        self.num_memory_token = num_memory_token
        if num_memory_token > 0:
            self.memory_tokens = nn.Parameter(data = torch.randn(num_memory_token, dim))

    # intialize helper
    def init_(self):

        if self.l2norm_embed:
            if self.unidimensional:
                nn.init.normal_(tensor = self.token_embedding.emb.weight, std = 1e-5)
            else:
                for embedding in self.token_embedding:
                    nn.init.normal_(tensor = embedding.emb.weight, std = 1e-5)
            nn.init.normal_(self.positional_embedding.emb.weight, std = 1e-5)

        else:
            if self.unidimensional:
                nn.init.kaiming_normal_(tensor = self.token_embedding.emb.weight)
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
        b = x.shape[0]
        num_mem = self.num_memory_token

        # calculate x
        if self.unidimensional:
            x = self.token_embedding(torch.clamp(input = x, min = 0, max = self.token_embedding.emb.num_embeddings - 1)) # make sure there are no invalid codes, https://rollbar.com/blog/how-to-handle-index-out-of-range-in-self-pytorch/
        else:
            x = sum(embedding(torch.clamp(input = x[..., i], min = 0, max = embedding.emb.num_embeddings - 1)) for i, embedding in enumerate(self.token_embedding))
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
        if return_embeddings:
            output = x
        else:
            if self.unidimensional:
                output = self.to_logits(x)
            else:
                output = [to_logit(x) for to_logit in self.to_logits]

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

    def __init__(
            self,
            net: MusicTransformerWrapper,
            encoding: dict = DEFAULT_ENCODING,
            ignore_index: int = DEFAULT_IGNORE_INDEX,
            pad_value: int = 0
        ):
        
        # intialize some fields
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.use_absolute_time = encoding["use_absolute_time"]

        # get the type codes
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.special_token_type_codes = {self.sos_type_code, self.son_type_code, self.eos_type_code}
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.note_type_codes = [encoding["type_code_map"]["note"], encoding["type_code_map"]["grace-note"]]
        self.expressive_feature_type_code = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]
        self.value_type_codes = {self.expressive_feature_type_code,}.union(set(self.note_type_codes))
        self.max_temporal = encoding["max_" + ("time" if self.use_absolute_time else "beat")]
        if self.use_absolute_time:
            self.time_code_map = encoding["time_code_map"]

        # for masking out expressive features
        self.expressive_feature_value_codes = torch.tensor(data = utils.rep(x = 0, times = representation.N_NOTES + 1) + utils.rep(x = 1, times = len(encoding["value_code_map"]) - (representation.N_NOTES + 1)), dtype = torch.bool)

        # get the dimension indices
        self.dimensions = {dimension: i for i, dimension in enumerate(encoding["unidimensional_encoding_order" if self.net.unidimensional else "dimensions"])}
        assert self.dimensions["type"] == 0

        # extra stuff for unidimensionality
        if self.net.unidimensional:
            # get encoding dimension indicies
            self.unidimensional_encoding_dimension_indicies = encoding["unidimensional_encoding_dimension_indicies"]
            # the index on which a new field starts
            self.dimension_code_range_starts = [0] + np.cumsum(a = [encoding["n_tokens"][dimension_index] for dimension_index in self.unidimensional_encoding_dimension_indicies], axis = 0).tolist()
            # ranges that a field spans, or more accurately, the inverse of those ranges
            self.dimension_code_ranges = torch.ones(size = ((len(self.dimension_code_range_starts) - 1), self.dimension_code_range_starts[-1]), dtype = torch.bool) # a bit of a lie, makes everything outside the code range true, while the range itself is all false
            for i in range(len(self.dimensions)): # self.dimension_code_ranges ordered in the order of unidimensional_encoding_dimension_indicies
                self.dimension_code_ranges[i, self.dimension_code_range_starts[i]:self.dimension_code_range_starts[i + 1]] = False
            # redefine certain codes to unidimensional encoding
            type_dim = self.dimensions["type"]
            self.sos_type_code += self.dimension_code_range_starts[type_dim]
            self.eos_type_code += self.dimension_code_range_starts[type_dim]
            self.son_type_code += self.dimension_code_range_starts[type_dim]
            self.special_token_type_codes = {self.sos_type_code, self.son_type_code, self.eos_type_code}
            self.instrument_type_code += self.dimension_code_range_starts[type_dim]
            self.note_type_codes[0] += self.dimension_code_range_starts[type_dim]
            self.note_type_codes[1] += self.dimension_code_range_starts[type_dim]
            self.expressive_feature_type_code += self.dimension_code_range_starts[type_dim]
            self.value_type_codes = {self.expressive_feature_type_code,}.union(set(self.note_type_codes))
            value_dim = self.dimensions["value"]
            self.expressive_feature_value_codes = torch.cat(tensors = (torch.zeros(size = (self.dimension_code_range_starts[value_dim],), dtype = torch.bool), self.expressive_feature_value_codes, torch.zeros(size = (self.dimension_code_range_starts[-1] - self.dimension_code_range_starts[value_dim + 1],), dtype = torch.bool)), dim = 0)
            del type_dim, value_dim

    ##################################################


    # HELPER FUNCTION TO FRONT PAD
    ##################################################
        
    def front_pad(self, sequences: List[torch.tensor], device: torch.device) -> torch.tensor:
        """
        Front pad sequences of variable length such that they all have the same length, then combine into a single tensor.
        Assumes all sequences in `sequences` have shape (1, seq_len, n_fields) or (1, seq_len * n_fields).
        Returns a tensor with shape (batch_size, seq_len, n_fields) or (batch_size, seq_len * n_fields).
        """
        longest_seq_len = max((seq.shape[1] for seq in sequences)) # get the longest sequence length
        for seq_index in range(len(sequences)):
            sequences[seq_index] = F.pad(input = sequences[seq_index], pad = (0, 0, 0, longest_seq_len - sequences[seq_index].shape[1]), mode = "constant", value = self.pad_value)
        out = torch.cat(tensors = sequences, dim = 0)
        return out.to(device)
    
    ##################################################
        
    
    # GENERATE TOKENS
    ##################################################

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.tensor, # shape : (b, n, d) or (b, n * d) depending on unidimensionality
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
        sigma: Union[float, int] = SIGMA,
        **kwargs,
    ):
        
        # SETUP
        ##################################################
        
        # get shape from start_tokens
        batch_size = start_tokens.shape[0]
        dim = start_tokens.shape[-1] if (not self.net.unidimensional) else self.n_tokens_per_event
        start_tokens_per_seq = [len(seq) for seq in start_tokens]

        # convert temperature to list
        if self.net.unidimensional and isinstance(temperature, (list, tuple)):
            temperature = temperature[0] # convert to single value
        elif isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert len(temperature) == dim, f"`temperature` must be of length {dim}"
        # convert filter_logits_fn to list
        if self.net.unidimensional and isinstance(filter_logits_fn, (list, tuple)):
            filter_logits_fn = filter_logits_fn[0] # convert to single value
        elif isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert len(filter_logits_fn) == dim, f"`filter_logits_fn` must be of length {dim}"
        # convert filter_thres to list
        if self.net.unidimensional and isinstance(filter_thres, (list, tuple)):
            filter_thres = filter_thres[0] # convert to single value
        elif isinstance(filter_thres, (float, int)):
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
        # tokenize sigma if sigma is in absolute time (seconds)
        if self.use_absolute_time:
            sigma = self.time_code_map[sigma]

        # some dimension indicies
        type_dim = self.dimensions["type"]
        temporal_dim = self.dimensions["time" if self.use_absolute_time else "beat"]
        value_dim = self.dimensions["value"]
        instrument_dim = self.dimensions["instrument"]
        if exists(eos_token) and self.net.unidimensional:
            eos_token += self.dimension_code_range_starts[type_dim] # recalculate eos token for full unidimensional vocabulary

        # get some constants
        was_training = self.net.training
        n_dims = len(start_tokens.shape)
        if (not self.net.unidimensional) and (n_dims == 2):
            start_tokens = start_tokens[None, :, :]

        # get expressive features (controls) for anticipation
        if is_anticipation:
            current_temporal = [0 for _ in range(batch_size)]
            expressive_features = [torch.empty(size = [0,] + ([] if self.net.unidimensional else [dim,])) for _ in range(batch_size)]
            start_tokens_temp = deepcopy(start_tokens)
            if self.net.unidimensional: # reshape if unidimensional
                start_tokens_temp = start_tokens_temp.reshape(batch_size, -1, dim)
            for seq_index in range(len(expressive_features)): # filter expressive features so that its only the features after the last note in the prefix
                if len(start_tokens_temp[seq_index]) > 0:
                    expressive_feature_matches = (start_tokens_temp[seq_index, :, type_dim] != self.expressive_feature_type_code)
                    last_note_index = len(expressive_feature_matches) - torch.argmax(input = expressive_feature_matches.flip(dims = (0,)).byte(), dim = 0).item() - 1 # get the index of the last note
                    current_temporal[seq_index] = start_tokens_temp[seq_index, last_note_index, temporal_dim].item() # get the last note index's time
                    expressive_features[seq_index] = start_tokens_temp[seq_index, (last_note_index + 1):, :] # get expressive features that come after the last note
                    expressive_features[seq_index] = expressive_features[seq_index][torch.sort(input = expressive_features[seq_index][:, temporal_dim], descending = False, dim = 0)[1]] # sort expressive features by time
                    del expressive_feature_matches, last_note_index
            del start_tokens_temp
        
        # deal with masking
        self.net.eval()
        output = [seq.unsqueeze(dim = 0) for seq in start_tokens]
        if is_anticipation: # if anticipation, remove relevant controls from prefix
            for seq_index in range(batch_size): # remove relevant controls from prefix
                n_expressive_features_remaining = len(expressive_features[seq_index]) * self.net.n_tokens_per_event
                if n_expressive_features_remaining > 0:
                    output[seq_index] = output[seq_index][:, :-n_expressive_features_remaining] # update output
                    start_tokens_per_seq[seq_index] -= n_expressive_features_remaining
        output = self.front_pad(sequences = output, device = start_tokens.device)
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.ones(size = (start_tokens.shape[:2]), dtype = torch.bool, device = start_tokens.device)
        
        # deal with current values
        if (monotonicity_dim is None):
            current_values = None
        elif self.net.unidimensional:
            current_values = {dim: [max(seq[torch.repeat_interleave(input = torch.logical_and(input = (seq[type_dim::self.net.n_tokens_per_event] == self.note_type_codes[0]), other = (seq[type_dim::self.net.n_tokens_per_event] == self.note_type_codes[1])), repeats = self.net.n_tokens_per_event)][dim::self.net.n_tokens_per_event].tolist() + [0]) for seq in output] for dim in monotonicity_dim}
        else:
            current_values = {dim: [max(seq[torch.logical_and(input = (seq[:, type_dim] == self.note_type_codes[0]), other = (seq[:, type_dim] == self.note_type_codes[1])), dim].tolist() + [0]) for seq in output] for dim in monotonicity_dim}

        # logits to return
        if return_logits:
            logits_to_return = [] if self.net.unidimensional else [[] for _ in range(dim)]

        ##################################################
            
        # GENERATE TOKENS ONE BY ONE
        ##################################################
            
        # loop through sequences
        max_temporal_token_buildup = torch.zeros(size = (batch_size,), dtype = torch.uint8)
        for _ in range(int(seq_len / self.net.n_tokens_per_event)):

            # SETUP FOR SAMPLING
            ##################################################

            # for anticipation, add relevant expressive features
            if is_anticipation:
                sequences = [seq for seq in output]
                for seq_index in range(batch_size):
                    if len(expressive_features[seq_index]) > 0:
                        relevant_expressive_features = expressive_features[seq_index][expressive_features[seq_index][:, temporal_dim] <= (current_temporal[seq_index] + sigma)]
                        if self.net.unidimensional:
                            relevant_expressive_features = relevant_expressive_features.flatten()
                        sequences[seq_index] = torch.cat(tensors = (sequences[seq_index], relevant_expressive_features), dim = 0)
                        expressive_features[seq_index] = expressive_features[seq_index][len(relevant_expressive_features):]
                output = self.front_pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device)
                del sequences

            ##################################################

            # SAMPLE UNIDIMENSIONALLY
            ##################################################
                
            if self.net.unidimensional:

                # sample different fields
                batch_event = torch.zeros(size = (batch_size, dim), dtype = output.dtype, device = output.device)
                output_temp, mask_temp = deepcopy(output), deepcopy(mask)
                for dimension_index in range(len(self.dimensions)): # order is unidimensionally-encoded dimensions, not the normal encoding["dimensions"]
                    
                    # get current x and mask
                    mask_temp = mask_temp[:, -self.max_seq_len:]
                    x = output_temp[:, -self.max_seq_len:]

                    # get logits (and perhaps attn)
                    if return_attn:
                        logits, attn = self.net(x = x, mask = mask_temp, return_attn = True, **kwargs)
                    else:
                        logits = self.net(x = x, mask = mask_temp, return_attn = False, **kwargs)

                    # get most recent logits
                    logits = logits[:, -1]

                    # enforce monotonicity
                    if (monotonicity_dim is not None) and (dimension_index in monotonicity_dim):
                        if (dimension_index == type_dim):
                            for seq_index, type_dim_current_value in enumerate(current_values[type_dim]):
                                if type_dim_current_value in self.value_type_codes: # as to not unnecessarily filter out any notes or expressive features
                                    type_dim_current_value = min(self.value_type_codes)
                                logits[seq_index, self.dimension_code_range_starts[type_dim]:type_dim_current_value] = -float("inf")
                        else:
                            for seq_index in range(batch_size):
                                logits[seq_index, :current_values[dimension_index][seq_index]] = -float("inf") # zero out values up until current value
                    
                    # restrict sampling
                    logits[:, self.dimension_code_ranges[dimension_index]] = -float("inf") # restrict codes not in the current dimension
                    if (dimension_index == type_dim): # filter out sos token if necessary
                        logits[:, self.sos_type_code] = -float("inf") # the 0th code in the type dimension code range should be the sos token
                    if (dimension_index == self.instrument_dim) or (dimension_index == self.value_dim): # avoid none value in instrument and value dimensions
                        logits[:, self.dimension_code_range_starts[dimension_index]] = -float("inf") # avoid none value                        
                    
                    # don't allow for sampling of expressive features if conditional
                    if notes_only:
                        logits[:, self.expressive_feature_type_code] = -float("inf") # don't allow for the expressive feature type
                        logits[:, self.expressive_feature_value_codes] = -float("inf") # don't allow for expressive feature values
                    
                    # sample from the restricted logits
                    sampled = sample(logits = logits, kind = filter_logits_fn, threshold = filter_thres, temperature = temperature, min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).flatten() # length is batch_size

                    # add sampled values to batch_event and output_temp, update mask_temp
                    batch_event[:, dimension_index] = sampled
                    output_temp = torch.cat(tensors = (output_temp, sampled.unsqueeze(dim = -1)), dim = 1)
                    mask_temp = F.pad(input = mask_temp, pad = (0, 1), mode = "constant", value = True)

                    # update current values
                    if (monotonicity_dim is not None) and (dimension_index in monotonicity_dim):
                        current_values[dimension_index] = np.maximum(current_values[dimension_index], sampled).tolist()
                    if is_anticipation and (dimension_index == temporal_dim):
                        current_temporal = sampled.tolist()

                    # to avoid buildup of tokens at max temporal, end song
                    if (dimension_index == temporal_dim):
                        max_temporal_token_buildup += (sampled == (self.max_temporal - self.dimension_code_range_starts[temporal_dim]))

                    # add to return logits if necessary
                    if return_logits:
                        logits_to_return.append(logits)                    

                # to avoid buildup of tokens at max temporal, end song
                exceeds_max_temporal_token_buildup_limit = (max_temporal_token_buildup < MAX_TEMPORAL_TOKEN_BUILDUP_LIMIT)
                batch_event *= torch.repeat_interleave(input = torch.bitwise_not(input = exceeds_max_temporal_token_buildup_limit).byte().unsqueeze(dim = -1), repeats = dim, dim = -1) # if the number of tokens exceeds or equals the buildup limit, zero out the row
                batch_event[exceeds_max_temporal_token_buildup_limit, type_dim] = self.eos_type_code # place end of song token

                # wrangle output a bit
                output = torch.cat(tensors = (output, batch_event), dim = 1)
                mask = F.pad(input = mask, pad = (0, self.net.n_tokens_per_event), mode = "constant", value = True)

                # if end of song token is provided
                if exists(eos_token):
                    is_eos_tokens = (output == eos_token)
                    # mask out everything after the eos tokens
                    if torch.all(input = torch.any(input = is_eos_tokens, dim = 1), dim = 0): # if all sequences in the batch have an eos token, break
                        if self.pad_value == self.sos_type_code: # get the index of the last sos token to remove the front pad
                            last_sos_token_index = torch.argmax(input = (output[type_dim::self.net.n_tokens_per_event] > self.sos_type_code).byte(), dim = 1) - 1
                        else:
                            last_sos_token_index = torch.argmax(input = (output[type_dim::self.net.n_tokens_per_event] == self.sos_type_code).byte(), dim = 1)
                        last_sos_token_index *= self.net.n_tokens_per_event
                        sequences = [seq[last_sos_token_index:] for seq in output] # remove front pad
                        is_eos_tokens = [(seq == eos_token) for seq in sequences] # recalculate because we removed the front pad
                        for seq_index, is_eos_tokens_seq in enumerate(is_eos_tokens):
                            first_eos_token_index = torch.argmax(input = is_eos_tokens_seq.byte(), dim = 0).item() # index of the first eos token
                            sequences[seq_index][(first_eos_token_index + 1):] = self.pad_value # pad after eos token
                            sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:] # remove start tokens from output
                        output = self.front_pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device) # front pad
                        break

            ##################################################
            
            # SAMPLE MULTIDIMENSIONALLY
            ##################################################
            
            else:

                # get current x and mask
                mask = mask[:, -self.max_seq_len:]
                x = output[:, -self.max_seq_len:]

                # get logits (and perhaps attn)
                if return_attn:
                    logits, attn = self.net(x = x, mask = mask, return_attn = True, **kwargs)
                else:
                    logits = self.net(x = x, mask = mask, return_attn = False, **kwargs)

                # get most recent logits
                logits = [logit[:, -1, :] for logit in logits] # first dim is batch size, next is sequence length, next is the vocabulary for that field
                
                # enforce monotonicity on type dimension
                if (monotonicity_dim is not None) and (type_dim in monotonicity_dim):
                    for seq_index, type_dim_current_value in enumerate(current_values[type_dim]):
                        if type_dim_current_value in self.value_type_codes: # as to not unnecessarily filter out any notes or expressive features
                            type_dim_current_value = min(self.value_type_codes)
                        logits[type_dim][seq_index, :type_dim_current_value] = -float("inf")
                
                # filter out sos token
                logits[type_dim][:, self.sos_type_code] = -float("inf") # the 0th token should be the sos token

                # sample from the logits
                event_types = sample(logits = logits[type_dim], kind = filter_logits_fn[type_dim], threshold = filter_thres[type_dim], temperature = temperature[type_dim], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).flatten().tolist() # length is batch_size

                # update current values
                if (monotonicity_dim is not None) and (type_dim in monotonicity_dim):
                    current_values[type_dim] = [max(current_value, event_type) for current_value, event_type in zip(current_values[type_dim], event_types)]

                # don't allow for sampling of expressive features
                if notes_only:
                    logits[type_dim][:, self.expressive_feature_type_code] = -float("inf") # don't allow for the expressive feature type
                    logits[value_dim][:, self.expressive_feature_value_codes] = -float("inf") # don't allow for expressive feature values

                # iterate after each sample
                batch_event = torch.zeros(size = (batch_size, 1, dim), dtype = output.dtype, device = output.device)
                batch_event[..., type_dim] = torch.tensor(data = event_types, dtype = batch_event.dtype)
                for seq_index, event_type in enumerate(event_types): # seq_index is the sequence index within the batch

                    # to avoid buildup of tokens at max temporal, end song
                    if (max_temporal_token_buildup[seq_index].item() >= MAX_TEMPORAL_TOKEN_BUILDUP_LIMIT): # if the number of tokens exceeds or equals the buildup limit, place end of song token
                        event_type = self.eos_type_code
                        batch_event[seq_index, :, type_dim] = self.eos_type_code

                    # an instrument code
                    if (event_type == self.instrument_type_code):
                        logits[instrument_dim][:, 0] = -float("inf") # avoid none in instrument dimension
                        sampled = sample(logits = logits[instrument_dim][seq_index].unsqueeze(dim = 0), kind = filter_logits_fn[instrument_dim], threshold = filter_thres[instrument_dim], temperature = temperature[instrument_dim], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).item()
                        batch_event[seq_index, :, instrument_dim] = sampled

                    # a value code
                    elif (event_type in self.value_type_codes):
                        for d in range(1, dim):
                            # enforce monotonicity
                            if (monotonicity_dim is not None) and (d in monotonicity_dim):
                                logits[d][seq_index, :current_values[d][seq_index]] = -float("inf")
                            # sample from the logits
                            logits[d][:, 0] = -float("inf") # avoid none in value dimension
                            sampled = sample(logits = logits[d][seq_index].unsqueeze(dim = 0), kind = filter_logits_fn[d], threshold = filter_thres[d], temperature = temperature[d], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).item()
                            batch_event[seq_index, :, d] = sampled
                            if (d == temporal_dim) and is_anticipation:
                                current_temporal[seq_index] = sampled
                            # update current values
                            if (monotonicity_dim is not None) and (d in monotonicity_dim):
                                current_values[d][seq_index] = max(current_values[d][seq_index], sampled)
                    
                    elif (event_type not in self.special_token_type_codes): # a start-of-song, end-of-song or start-of-notes code
                        raise ValueError(f"Unknown event type: {event_type}")

                # add logits to return logits if necessary
                if return_logits:
                    for dimension_index in range(dim):
                        logits_to_return[dimension_index].append(logits[dimension_index])
                
                # update the count of tokens that are at temporal of max_temporal
                max_temporal_token_buildup += (batch_event[..., temporal_dim] == self.max_temporal).flatten()

                # wrangle output a bit
                output = torch.cat(tensors = (output, batch_event), dim = 1)
                mask = F.pad(input = mask, pad = (0, self.net.n_tokens_per_event), mode = "constant", value = True)

                # if end of song token is provided
                if exists(eos_token):
                    is_eos_tokens = (output[..., type_dim] == eos_token) # [(seq[:, type_dim] == eos_token) for seq in output]
                    # mask out everything after the eos tokens
                    if torch.all(input = torch.any(input = is_eos_tokens, dim = 1), dim = 0): # if all sequences in the batch have an eos token, break
                        if self.pad_value == self.sos_type_code: # get the index of the last sos token to remove the front pad
                            last_sos_token_index = torch.argmax(input = (output[..., type_dim] > self.sos_type_code).byte(), dim = 1) - 1
                        else:
                            last_sos_token_index = torch.argmax(input = (output[..., type_dim] == self.sos_type_code).byte(), dim = 1)
                        sequences = [seq[last_sos_token_index:] for seq in output] # remove front pad
                        is_eos_tokens = [(seq[:, type_dim] == eos_token) for seq in sequences] # recalculate because we removed the front pad
                        for seq_index, is_eos_tokens_seq in enumerate(is_eos_tokens):
                            first_eos_token_index = torch.argmax(input = is_eos_tokens_seq.byte(), dim = 0).item() # index of the first eos token
                            sequences[seq_index][(first_eos_token_index + 1):] = self.pad_value # pad after eos token
                            sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:] # remove start tokens from output
                        output = self.front_pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device) # front pad
                        break

            ##################################################

        ##################################################
        
        # FINALIZE OUTPUT
        ##################################################
                
        # wrangle output
        if n_dims == 1:
            output = output.squeeze(dim = 0)

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

    ##################################################


    # FORWARD PASS
    ##################################################

    def forward(
            self,
            x: torch.tensor,
            return_list: bool = False, # return a list of losses per field?
            reduce: bool = True, # reduce all losses to losses per field?
            return_output: bool = False, # return the model's output?
            **kwargs
        ):

        # create subsets of x
        xi = x[:, :-self.net.n_tokens_per_event] # input
        xo = x[:, self.net.n_tokens_per_event:] # expected output

        # help auto-solve a frequent area of confusion around input masks in auto-regressive, if user supplies a mask that is only off by one from the source seq, resolve it for them
        mask = kwargs.get("mask", None)
        if (mask is not None) and (mask.shape[1] == x.shape[1]):
            mask = mask[:, :-self.net.n_tokens_per_event]
            kwargs["mask"] = mask

        # get conditional notes mask if present; by default, mask in everything
        conditional_mask = kwargs.pop("conditional_mask", torch.ones_like(input = (xo if self.net.unidimensional else xo[..., 0]), dtype = torch.bool))
        
        # create output
        output = self.net(x = xi, **kwargs)
        n_masked_tokens = torch.sum(input = conditional_mask, dim = None)
        if self.net.unidimensional:
            losses = F.cross_entropy(input = output.transpose(1, 2), target = xo, ignore_index = self.ignore_index, reduction = "none") # calculate losses
            losses = conditional_mask * losses # mask conditionally if necessary, zeroing out controls
            loss = torch.sum(input = losses, dim = None) / n_masked_tokens # single loss value
        else:
            losses = [F.cross_entropy(input = output[i].transpose(1, 2), target = xo[..., i], ignore_index = self.ignore_index, reduction = "none") for i in range(len(output))] # calculate losses
            losses = torch.cat(tensors = [losses_dimension.unsqueeze(dim = -1) for losses_dimension in losses], dim = -1) # combine list of losses into a matrix
            losses = torch.repeat_interleave(input = conditional_mask.unsqueeze(dim = -1), repeats = losses.shape[-1], dim = -1) * losses # mask conditionally if necessary, zeroing out controls
            loss_by_field = torch.sum(input = losses, dim = list(range(len(losses.shape) - 1))) / n_masked_tokens # calculate mean for each field, the field is the last dimension, so mean over all but the last dimension
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
    def __init__(
            self,
            *,
            dim: int,
            encoding: dict = DEFAULT_ENCODING,
            unidimensional: bool = False,
            **kwargs
        ):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_temporal": kwargs.pop("max_temporal"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.decoder = MusicTransformerWrapper(encoding = encoding, attn_layers = Decoder(dim = dim, **kwargs), unidimensional = unidimensional, **transformer_kwargs)
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

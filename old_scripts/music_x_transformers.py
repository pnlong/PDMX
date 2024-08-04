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
from dataset import PAD_VALUE

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
            temporal_dim = encoding["dimensions"].index("time" if use_absolute_time else "beat") # no reordering here because we reference "n_tokens"
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
            pad_value: int = PAD_VALUE
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
        self.temporal = "time" if self.use_absolute_time else "beat"
        self.temporal_code_map = encoding[f"{self.temporal}_code_map"]
        self.max_temporal = encoding[f"max_{self.temporal}"]
        if self.use_absolute_time:
            self.max_temporal = self.temporal_code_map[self.max_temporal]

        # for masking out different types of controls
        self.expressive_feature_value_codes = torch.tensor(data = utils.rep(x = 0, times = representation.N_NOTES + 1) + utils.rep(x = 1, times = len(encoding["value_code_map"]) - (representation.N_NOTES + 1)), dtype = torch.bool)
        self.note_value_codes = ~self.expressive_feature_value_codes

        # get the dimension indices
        self.dimensions = {dimension: i for i, dimension in enumerate(encoding["unidimensional_encoding_order" if self.net.unidimensional else "dimensions"])}
        assert self.dimensions["type"] == 0
        self.type_dim = self.dimensions["type"]
        self.value_dim = self.dimensions["value"]
        self.temporal_dim = self.dimensions[self.temporal]
        self.instrument_dim = self.dimensions["instrument"]

        # extra stuff for unidimensionality
        if self.net.unidimensional:
            # the index on which a new field starts
            self.unidimensional_encoding_dimension_indicies = encoding["unidimensional_encoding_dimension_indicies"]
            unidimensional_dimension_code_range_starts = encoding["unidimensional_dimension_code_range_starts"]
            self.dimension_code_range_starts = [unidimensional_dimension_code_range_starts[dimension] for dimension in encoding["unidimensional_encoding_order"]]
            n_tokens_total = sum(encoding["n_tokens"])
            # ranges that a field spans, or more accurately, the inverse of those ranges
            self.dimension_code_ranges = torch.ones(size = (len(self.dimension_code_range_starts), n_tokens_total), dtype = torch.bool) # a bit of a lie, makes everything outside the code range true, while the range itself is all false
            for dimension_index in range(len(self.dimensions)): # self.dimension_code_ranges ordered in the order of unidimensional_encoding_dimension_indicies
                start = self.dimension_code_range_starts[dimension_index]
                end = min(list(filter(lambda dimension_code_range_start: dimension_code_range_start > start, self.dimension_code_range_starts)) + [n_tokens_total]) # smallest code range start thats still bigger than the current
                self.dimension_code_ranges[dimension_index, start:end] = False
            # redefine certain codes to unidimensional encoding
            self.sos_type_code += self.dimension_code_range_starts[self.type_dim]
            self.eos_type_code += self.dimension_code_range_starts[self.type_dim]
            self.son_type_code += self.dimension_code_range_starts[self.type_dim]
            self.special_token_type_codes = {self.sos_type_code, self.son_type_code, self.eos_type_code}
            self.instrument_type_code += self.dimension_code_range_starts[self.type_dim]
            self.note_type_codes[0] += self.dimension_code_range_starts[self.type_dim]
            self.note_type_codes[1] += self.dimension_code_range_starts[self.type_dim]
            self.expressive_feature_type_code += self.dimension_code_range_starts[self.type_dim]
            self.value_type_codes = {self.expressive_feature_type_code,}.union(set(self.note_type_codes))
            self.max_temporal += self.dimension_code_range_starts[self.temporal_dim]
            value_dim_start = self.dimension_code_range_starts[self.value_dim]
            self.expressive_feature_value_codes = torch.cat(tensors = (
                torch.zeros(size = (value_dim_start,), dtype = torch.bool),
                self.expressive_feature_value_codes,
                torch.zeros(size = (n_tokens_total - min(list(filter(lambda dimension_code_range_start: dimension_code_range_start > value_dim_start, self.dimension_code_range_starts)) + [n_tokens_total]),), dtype = torch.bool)
                ), dim = 0)
            self.note_value_codes = torch.cat(tensors = (
                torch.zeros(size = (value_dim_start,), dtype = torch.bool),
                self.note_value_codes,
                torch.zeros(size = (n_tokens_total - min(list(filter(lambda dimension_code_range_start: dimension_code_range_start > value_dim_start, self.dimension_code_range_starts)) + [n_tokens_total]),), dtype = torch.bool)
                ), dim = 0)
    
    ##################################################


    # HELPER FUNCTION TO FRONT PAD
    ##################################################
        
    def pad(self, sequences: List[torch.tensor], device: torch.device, pad_value: int, front: bool = True) -> torch.tensor:
        """
        Front pad sequences of variable length such that they all have the same length, then combine into a single tensor.
        Assumes all sequences in `sequences` have shape (1, seq_len, n_fields) or (1, seq_len * n_fields).
        Returns a tensor with shape (batch_size, seq_len, n_fields) or (batch_size, seq_len * n_fields).
        """
        longest_seq_len = max((seq.shape[1] for seq in sequences)) # get the longest sequence length
        pad_prefix = [] if (len(sequences[0].shape) <= 2) else [0, 0]
        sequences = [F.pad(input = seq, pad = tuple(pad_prefix + ([longest_seq_len - seq.shape[1], 0] if front else [0, longest_seq_len - seq.shape[1]])), mode = "constant", value = pad_value) for seq in sequences]
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
        joint: bool = True,
        notes_are_controls: bool = False,
        is_anticipation: bool = False,
        sigma: Union[float, int] = SIGMA,
        **kwargs,
    ):
        
        # SETUP
        ##################################################
        
        # get shape from start_tokens
        batch_size = start_tokens.shape[0]
        dim = start_tokens.shape[-1] if (not self.net.unidimensional) else self.net.n_tokens_per_event
        # get masks of the actual sequence, excluding any pad
        start_tokens_per_seq = [torch.repeat_interleave(input = ~torch.all(input = ((seq.reshape(int(len(seq) / self.net.n_tokens_per_event), self.net.n_tokens_per_event) if self.net.unidimensional else seq) == self.pad_value), dim = -1), repeats = self.net.n_tokens_per_event, dim = -1) for seq in start_tokens]
        # ensure that we aren't accidentally masking out the sos row
        if (self.pad_value == self.sos_type_code) and (not self.net.unidimensional): # does not apply to unidimensional because sos row isn't all 0s in unidimensional scheme
            first_sos_token_index = torch.argmax(input = torch.stack(tensors = start_tokens_per_seq, dim = 0).byte(), dim = -1) - self.net.n_tokens_per_event # length of this is batch_size
            for seq_index in range(batch_size):
                start_tokens_per_seq[seq_index][first_sos_token_index[seq_index]] = True
        # remove front pad from start tokens
        start_tokens_unpadded = [seq[start_token_per_seq].unsqueeze(dim = 0) for seq, start_token_per_seq in zip(start_tokens, start_tokens_per_seq)]
        if self.net.unidimensional: # get the eos tokens for unidimensionality
            eos_tokens = torch.cat(tensors = [seq[:, :self.net.n_tokens_per_event] for seq in start_tokens_unpadded], dim = 0).to(start_tokens.device)
            eos_tokens[:, 0] = self.eos_type_code
        # pad the wrangled sequences in our desired way
        start_tokens = self.pad(sequences = start_tokens_unpadded, device = start_tokens.device, pad_value = self.pad_value) # remove pad values from start tokens
        # get the number of start tokens per sequence
        start_tokens_per_seq = [sum(start_token_per_seq) for start_token_per_seq in start_tokens_per_seq]

        # get the type codes of controls
        control_type_codes = torch.tensor(data = self.note_type_codes if notes_are_controls else (self.expressive_feature_type_code,), device = start_tokens.device)
        control_value_codes = self.note_value_codes if notes_are_controls else self.expressive_feature_value_codes

        # get mask
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = self.pad(sequences = [torch.ones(size = (1, start_token_per_seq), dtype = torch.bool) for start_token_per_seq in start_tokens_per_seq], device = start_tokens.device, pad_value = False) # remove pad values from start tokens

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
        # tokenize sigma (convert from temporal metric to code)
        sigma = self.temporal_code_map[sigma]

        # some dimension indicies
        if exists(eos_token) and self.net.unidimensional:
            eos_token += self.dimension_code_range_starts[self.type_dim] # recalculate eos token for full unidimensional vocabulary

        # get some constants
        was_training = self.net.training
        n_dims = len(start_tokens.shape)
        if (not self.net.unidimensional) and (n_dims == 2):
            start_tokens = start_tokens[None, :, :]

        # get controls for anticipation
        if is_anticipation:
            current_temporal = [0 for _ in range(batch_size)]
            controls = [torch.empty(size = [0,] + ([] if self.net.unidimensional else [dim,])) for _ in range(batch_size)]
            start_tokens_temp = deepcopy(start_tokens)
            if self.net.unidimensional: # reshape if unidimensional
                start_tokens_temp = start_tokens_temp.reshape(batch_size, -1, dim)
            for seq_index in range(len(controls)): # filter controls so that its only the features after the last event in the prefix
                if len(start_tokens_temp[seq_index]) > 0:
                    control_matches = torch.isin(start_tokens_temp[seq_index, :, self.type_dim], test_elements = control_type_codes, invert = True)
                    last_event_index = len(control_matches) - torch.argmax(input = control_matches.flip(dims = (0,)).byte(), dim = 0).item() - 1 # get the index of the last event
                    current_temporal[seq_index] = start_tokens_temp[seq_index, last_event_index, self.temporal_dim].item() # get the last event index's time
                    controls[seq_index] = start_tokens_temp[seq_index, (last_event_index + 1):, :] # get controls that come after the last event
                    controls[seq_index] = controls[seq_index][torch.sort(input = controls[seq_index][:, self.temporal_dim], descending = False, dim = 0)[1]] # sort controls by time
                    del control_matches, last_event_index
            del start_tokens_temp
        
        # deal with masking
        self.net.eval()
        output = [seq.unsqueeze(dim = 0) for seq in start_tokens]
        if is_anticipation: # if anticipation, remove relevant controls from prefix
            mask = [seq.unsqueeze(dim = 0) for seq in mask]
            for seq_index in range(batch_size): # remove relevant controls from prefix
                n_controls_remaining = len(controls[seq_index]) * self.net.n_tokens_per_event
                if n_controls_remaining > 0:
                    output[seq_index] = output[seq_index][:, :-n_controls_remaining] # update output
                    mask[seq_index] = mask[seq_index][:, :-n_controls_remaining] # update mask
                    start_tokens_per_seq[seq_index] -= n_controls_remaining
            mask = self.pad(sequences = mask, device = start_tokens.device, pad_value = False) # remove pad values from start tokens
        output = self.pad(sequences = output, device = start_tokens.device, pad_value = self.pad_value)

        # deal with current values
        if (monotonicity_dim is None):
            current_values = None
        elif self.net.unidimensional:
            current_values = {dim: [max(seq[torch.repeat_interleave(input = torch.isin(seq[self.type_dim::self.net.n_tokens_per_event], test_elements = control_type_codes, invert = True), repeats = self.net.n_tokens_per_event)][dim::self.net.n_tokens_per_event].tolist() + [0]) for seq in output] for dim in monotonicity_dim}
        elif not self.net.unidimensional:
            current_values = {dim: [max(seq[torch.isin(seq[:, self.type_dim], test_elements = control_type_codes, invert = True), dim].tolist() + [0]) for seq in output] for dim in monotonicity_dim}

        # logits to return
        if return_logits:
            logits_to_return = [] if self.net.unidimensional else [[] for _ in range(dim)]

        ##################################################
            
        # GENERATE TOKENS ONE BY ONE
        ##################################################
            
        # loop through sequences
        max_temporal_token_buildup = torch.zeros(size = (batch_size,), dtype = torch.uint8, device = output.device)
        n_iterations = int(seq_len / self.net.n_tokens_per_event)
        final_iteration_index = n_iterations - 1
        for i in range(n_iterations):

            # SETUP FOR SAMPLING
            ##################################################

            # for anticipation, add relevant controls
            if is_anticipation:
                sequences = [seq for seq in output]
                mask = [seq for seq in mask]
                for seq_index in range(batch_size):
                    if len(controls[seq_index]) > 0:
                        relevant_controls = controls[seq_index][controls[seq_index][:, self.temporal_dim] <= (current_temporal[seq_index] + sigma)]
                        if self.net.unidimensional:
                            relevant_controls = relevant_controls.flatten()
                        sequences[seq_index] = torch.cat(tensors = (sequences[seq_index], relevant_controls), dim = 0)
                        mask[seq_index] = torch.cat(tensors = (mask[seq_index], torch.all(input = torch.ones_like(input = relevant_controls, dtype = torch.bool), dim = -1)), dim = 0)
                        controls[seq_index] = controls[seq_index][len(relevant_controls):]
                output = self.pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device, pad_value = self.pad_value)
                mask = self.pad(sequences = [seq.unsqueeze(dim = 0) for seq in mask], device = output.device, pad_value = False)
                del sequences

            ##################################################

            # SAMPLE UNIDIMENSIONALLY
            ##################################################
                
            if self.net.unidimensional:

                # sample different fields
                batch_event = torch.zeros(size = (batch_size, dim), dtype = output.dtype, device = output.device)
                output_temp, mask_temp = deepcopy(output).to(output.device), deepcopy(mask).to(mask.device)
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
                        for seq_index in range(batch_size):
                            current_value = current_values[dimension_index][seq_index]
                            if (dimension_index == self.type_dim) and (current_value in self.value_type_codes):
                                current_value = min(self.value_type_codes)
                            logits[seq_index, self.dimension_code_range_starts[dimension_index]:current_value] = -float("inf") # zero out values up until current value
                                            
                    # restrict sampling
                    logits[:, self.dimension_code_ranges[dimension_index]] = -float("inf") # restrict codes not in the current dimension
                    if (dimension_index == self.type_dim): # filter out sos token if necessary
                        logits[:, self.sos_type_code] = -float("inf") # the 0th code in the type dimension code range should be the sos token
                    logits[:, self.dimension_code_range_starts[dimension_index]] = -float("inf") # avoid none value
                    
                    # don't allow for sampling of controls if conditional
                    if not joint:
                        logits[:, control_type_codes] = -float("inf") # don't allow for the control type
                        logits[:, control_value_codes] = -float("inf") # don't allow for control values

                    # sample from the restricted logits
                    sampled = sample(logits = logits, kind = filter_logits_fn[dimension_index], threshold = filter_thres[dimension_index], temperature = temperature[dimension_index], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).flatten() # length is batch_size

                    # add sampled values to batch_event and output_temp, update mask_temp
                    batch_event[:, dimension_index] = sampled
                    output_temp = torch.cat(tensors = (output_temp, sampled.unsqueeze(dim = -1)), dim = 1)
                    mask_temp = F.pad(input = mask_temp, pad = (0, 1), mode = "constant", value = True)

                    # update current values
                    if (monotonicity_dim is not None) and (dimension_index in monotonicity_dim):
                        current_values[dimension_index] = np.maximum(current_values[dimension_index], sampled.cpu()).tolist()
                    if is_anticipation and (dimension_index == self.temporal_dim):
                        current_temporal = sampled.tolist()

                    # to avoid buildup of tokens at max temporal, end song
                    if (dimension_index == self.temporal_dim):
                        max_temporal_token_buildup += (sampled >= self.max_temporal)

                    # add to return logits if necessary
                    if return_logits:
                        logits_to_return.append(logits)                    

                # to avoid buildup of tokens at max temporal, end song
                exceeds_max_temporal_token_buildup_limit = (max_temporal_token_buildup >= MAX_TEMPORAL_TOKEN_BUILDUP_LIMIT) # len(max_temporal_token_buildup) == batch_size
                if exists(eos_token) and (i == final_iteration_index): # force all eos tokens if on the final iteration of generating
                    exceeds_max_temporal_token_buildup_limit = torch.ones_like(input = exceeds_max_temporal_token_buildup_limit, dtype = torch.bool).to(output.device)
                max_temporal_token_buildup_limit_mask = torch.repeat_interleave(input = (~exceeds_max_temporal_token_buildup_limit).unsqueeze(dim = -1), repeats = self.net.n_tokens_per_event, dim = -1) # if the number of tokens exceeds or equals the buildup limit, zero out the row
                batch_event *= max_temporal_token_buildup_limit_mask.byte() # zero out sequences in the batch that exceed the buildup limit
                batch_event += (~max_temporal_token_buildup_limit_mask).byte() * eos_tokens # make rows match sos rows (type dim followed by all 0s)

                # wrangle output a bit
                output = torch.cat(tensors = (output, batch_event), dim = 1)
                mask = F.pad(input = mask, pad = (0, self.net.n_tokens_per_event), mode = "constant", value = True)

                # if end of song token is provided
                if exists(eos_token):
                    # mask out everything after the eos tokens
                    if torch.all(input = torch.any(input = (output == eos_token), dim = 1), dim = 0): # if all sequences in the batch have an eos token, break
                        if self.pad_value == self.sos_type_code: # get the index of the last sos token to remove the front pad
                            last_sos_token_indicies = torch.argmax(input = (output[:, self.type_dim::self.net.n_tokens_per_event] > self.sos_type_code).byte(), dim = 1) - 1
                        else:
                            last_sos_token_indicies = torch.argmax(input = (output[:, self.type_dim::self.net.n_tokens_per_event] == self.sos_type_code).byte(), dim = 1)
                        last_sos_token_indicies *= self.net.n_tokens_per_event
                        sequences = [seq[last_sos_token_index:] for seq, last_sos_token_index in zip(output, last_sos_token_indicies)] # remove front pad
                        is_eos_tokens = [(seq == eos_token) for seq in sequences] # recalculate because we removed the front pad
                        for seq_index, is_eos_tokens_seq in enumerate(is_eos_tokens):
                            first_eos_token_index = torch.argmax(input = is_eos_tokens_seq.byte(), dim = 0).item() # index of the first eos token
                            # sequences[seq_index][(first_eos_token_index + self.net.n_tokens_per_event):] = self.pad_value # pad after eos token
                            # sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:] # remove start tokens from output
                            sequences[seq_index][first_eos_token_index:(first_eos_token_index + self.net.n_tokens_per_event)] = eos_tokens[seq_index] # make sure row is correct for eos row
                            sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:(first_eos_token_index + self.net.n_tokens_per_event)]
                        output = self.pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device, pad_value = self.pad_value, front = False) # rear pad
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
                if (monotonicity_dim is not None) and (self.type_dim in monotonicity_dim):
                    for seq_index, type_dim_current_value in enumerate(current_values[self.type_dim]):
                        if type_dim_current_value in self.value_type_codes: # as to not unnecessarily filter out any notes or expressive features
                            type_dim_current_value = min(self.value_type_codes)
                        logits[self.type_dim][seq_index, :type_dim_current_value] = -float("inf")
                
                # filter out sos token
                logits[self.type_dim][:, self.sos_type_code] = -float("inf") # the 0th token should be the sos token

                # sample from the logits
                event_types = sample(logits = logits[self.type_dim], kind = filter_logits_fn[self.type_dim], threshold = filter_thres[self.type_dim], temperature = temperature[self.type_dim], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).flatten().tolist() # length is batch_size

                # update current values
                if (monotonicity_dim is not None) and (self.type_dim in monotonicity_dim):
                    current_values[self.type_dim] = [max(current_value, event_type) for current_value, event_type in zip(current_values[self.type_dim], event_types)]

                # don't allow for sampling of controls if conditional
                if not joint:
                    logits[self.type_dim][:, control_type_codes] = -float("inf") # don't allow for the control type
                    logits[self.value_dim][:, control_value_codes] = -float("inf") # don't allow for control values

                # iterate after each sample
                batch_event = torch.zeros(size = (batch_size, 1, dim), dtype = output.dtype, device = output.device)
                batch_event[..., self.type_dim] = torch.tensor(data = event_types, dtype = batch_event.dtype).unsqueeze(dim = -1)
                for seq_index, event_type in enumerate(event_types): # seq_index is the sequence index within the batch

                    # to avoid buildup of tokens at max temporal, end song
                    if (max_temporal_token_buildup[seq_index].item() >= MAX_TEMPORAL_TOKEN_BUILDUP_LIMIT) or (i == final_iteration_index): # if the number of tokens exceeds or equals the buildup limit, place end of song token
                        event_type = self.eos_type_code
                        batch_event[seq_index, :, self.type_dim] = self.eos_type_code

                    # an instrument code
                    if (event_type == self.instrument_type_code):
                        logits[self.instrument_dim][:, 0] = -float("inf") # avoid none in instrument dimension
                        sampled = sample(logits = logits[self.instrument_dim][seq_index].unsqueeze(dim = 0), kind = filter_logits_fn[self.instrument_dim], threshold = filter_thres[self.instrument_dim], temperature = temperature[self.instrument_dim], min_p_pow = min_p_pow, min_p_ratio = min_p_ratio).item()
                        batch_event[seq_index, :, self.instrument_dim] = sampled

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
                            if (d == self.temporal_dim) and is_anticipation:
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
                max_temporal_token_buildup += (batch_event[..., self.temporal_dim] >= self.max_temporal).flatten()

                # wrangle output a bit
                output = torch.cat(tensors = (output, batch_event), dim = 1)
                mask = F.pad(input = mask, pad = (0, self.net.n_tokens_per_event), mode = "constant", value = True)

                # if end of song token is provided
                if exists(eos_token):
                    # mask out everything after the eos tokens
                    if torch.all(input = torch.any(input = (output[..., self.type_dim] == eos_token), dim = 1), dim = 0): # if all sequences in the batch have an eos token, break
                        if self.pad_value == self.sos_type_code: # get the index of the last sos token to remove the front pad
                            last_sos_token_indicies = torch.argmax(input = (output[..., self.type_dim] > self.sos_type_code).byte(), dim = 1) - 1
                        else:
                            last_sos_token_indicies = torch.argmax(input = (output[..., self.type_dim] == self.sos_type_code).byte(), dim = 1)
                        sequences = [seq[last_sos_token_index:] for seq, last_sos_token_index in zip(output, last_sos_token_indicies)] # remove front pad
                        is_eos_tokens = [(seq[:, self.type_dim] == eos_token) for seq in sequences] # recalculate because we removed the front pad
                        for seq_index, is_eos_tokens_seq in enumerate(is_eos_tokens):
                            first_eos_token_index = torch.argmax(input = is_eos_tokens_seq.byte(), dim = 0).item() # index of the first eos token
                            # sequences[seq_index][(first_eos_token_index + self.net.n_tokens_per_event):] = self.pad_value # pad after eos token
                            # sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:] # remove start tokens from output
                            sequences[seq_index] = sequences[seq_index][start_tokens_per_seq[seq_index]:(first_eos_token_index + self.net.n_tokens_per_event)]
                        output = self.pad(sequences = [seq.unsqueeze(dim = 0) for seq in sequences], device = output.device, pad_value = self.pad_value, front = False) # rear pad
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
        losses = torch.nan_to_num(input = losses, nan = 0.0, posinf = 0.0, neginf = 0.0) # make sure no nans
        loss = torch.nan_to_num(input = loss, nan = 0.0, posinf = 0.0, neginf = 0.0) # make sure no nans

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

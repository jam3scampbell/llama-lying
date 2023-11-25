import numpy as np
import torch
from collections import defaultdict
import einops
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial
from transformer_lens import HookedTransformer
from concept_erasure import LeaceEraser, LeaceFitter
from concept_erasure.oracle import OracleEraser, OracleFitter
from concept_erasure.quadratic import QuadraticEraser, QuadraticFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

def tot_logit_diff(tokenizer, model_acts, use_probs=True, eps=1e-8, test_only=True, act_type="z", check_balanced_output=False, 
                   positive_str_tokens = ["Yes", "yes", "True", "true"],
                   negative_str_tokens = ["No", "no", "False", "false"], scale_relative=False):
    """
    Get difference in correct and incorrect or positive and negative logits for each sample stored in model_acts, aggregated together.
    Should be same number of positive and negative tokens.
    If scale_relative is True, then scale probs/logits so that only correct vs incorrect or positive and negative probs/logits are considered
    """
    # positive_str_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
    # negative_str_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]

    positive_tokens = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    negative_tokens = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]


    if test_only:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices_tests[act_type]] # labels
        positive_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices_tests[act_type].shape[0],))
        meta_indices = np.array([np.where(model_acts.indices == i)[0][0] for i in model_acts.indices_tests["z"]])

    
    else:
        sample_labels = np.array(model_acts.dataset.all_labels)[model_acts.indices] # labels
        positive_sum = torch.empty(size=(model_acts.indices.shape[0],))
        negative_sum = torch.empty(size=(model_acts.indices.shape[0],))
        meta_indices = np.arange(model_acts.indices.shape[0],)
    
    check_positive_prop = 0

    for idx, logits in enumerate(model_acts.stored_acts["logits"][meta_indices]):

        # if answer to statement is True, correct tokens is Yes, yes, ..., true
        correct_tokens = positive_tokens if sample_labels[idx] else negative_tokens
        incorrect_tokens = negative_tokens if sample_labels[idx] else positive_tokens
        
        check_positive_prop += 1 if sample_labels[idx] else 0

        if check_balanced_output:
            correct_tokens = positive_tokens
            incorrect_tokens = negative_tokens


        if use_probs:
            probs = torch.nn.functional.softmax(logits, dim=1)
            positive_prob = probs[0, correct_tokens].sum(dim=-1)
            negative_prob = probs[0, incorrect_tokens].sum(dim=-1)

            if scale_relative:
                positive_sum[idx] = positive_prob / (positive_prob + negative_prob + eps)
                negative_sum[idx] = negative_prob / (positive_prob + negative_prob + eps)
            else:
                positive_sum[idx] = positive_prob 
                negative_sum[idx] = negative_prob 

        else:
            positive_sum[idx] = logits[0, correct_tokens].sum(dim=-1)
            negative_sum[idx] = logits[0, incorrect_tokens].sum(dim=-1)

    print(f"proportion of positive labels is {check_positive_prop/len(meta_indices)}")
    return positive_sum, negative_sum


def logit_attrs_tokens(cache, stored_acts, positive_tokens=[], negative_tokens=[]):
    """
    Helper function to call cache.logit_attrs over a set of possible positive and negative tokens (ints or strings). Also indexes last token. 
    Ideally, same number of positive and negative tokens (to account for relative logits)
    """
    all_attrs = []
    for token in positive_tokens:
        all_attrs.append(cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])
    for token in negative_tokens:
        all_attrs.append(-cache.logit_attrs(stored_acts, tokens=token, has_batch_dim=False)[:,-1])

    return torch.stack(all_attrs).mean(0)


def logit_attrs(model: HookedTransformer, dataset, act_types = ["resid_pre", "result"], N = 1000, indices=None, 
                positive_str_tokens=["True"], negative_str_tokens=["False"]):
    total_logit_attrs = defaultdict(list)

    if indices is None:
        indices, all_prompts, all_labels = dataset.sample(N)

    all_logits = []
    # names filter for efficiency, only cache in self.act_types
    # names_filter = lambda name: any([name.endswith(act_type) for act_type in act_types])

    for i in tqdm(indices):
        original_logits, cache = model.run_with_cache(dataset.all_prompts[i].to(model.cfg.device))


        # positive_tokens = ["Yes", "yes", " Yes", " yes", "True", "true", " True", " true"]
        # positive_str_tokens = ["Yes", "yes", "True", "true"]
        # negative_tokens = ["No", "no", " No", " no", "False", "false", " False", " false"]
        # negative_str_tokens = ["No", "no", "False", "false"]

        positive_tokens = [model.tokenizer(token).input_ids[-1] for token in positive_str_tokens]
        negative_tokens = [model.tokenizer(token).input_ids[-1] for token in negative_str_tokens]

        # correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        # incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        correct_tokens = positive_tokens if dataset.all_labels[i] else negative_tokens
        incorrect_tokens = negative_tokens if dataset.all_labels[i] else positive_tokens
        
        for act_type in act_types:
            stored_acts = cache.stack_activation(act_type, layer = -1).squeeze()#[:,0,-1].squeeze().to(device=storage_device)
            
            if act_type == "result":
                stored_acts = einops.rearrange(stored_acts, "n_l s n_h d_m -> (n_l n_h) s d_m")
            # print(f"{stored_acts.shape=}")
            # print(f"{cache.logit_attrs(stored_acts, tokens=correct_token, incorrect_tokens=incorrect_token)=}")
            
            # total_logit_attrs[act_type].append(cache.logit_attrs(stored_acts, tokens=correct_tokens, incorrect_tokens=incorrect_tokens, pos_slice=-1, has_batch_dim=False)[:,-1]) # last position
            total_logit_attrs[act_type].append(logit_attrs_tokens(cache, stored_acts, correct_tokens, incorrect_tokens))

        all_logits.append(original_logits)

    return all_logits, total_logit_attrs



def get_head_bools(model, logit_heads, flattened=False):
    """
    Method to get boolean array (n_l x n_h), 1 if head is selected at 0 if not, from a list of heads to select logit_heads.
    The flattened parameter describes the logit_heads list: if flattened is true, input to logit_heads is 1D.
    """
    if flattened:
        head_bools = torch.zeros(size=(model.cfg.total_heads,))
        for head in logit_heads:
            head_bools[head] = 1
        head_bools = einops.rearrange(head_bools, '(n_l n_h) -> n_l n_h', n_l=model.cfg.n_layers)    
    else:
        head_bools = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
        for head in logit_heads:
            head_bools[head[0], head[1]] = 1
    return head_bools


def query_logits(tokenizer, logits, return_type = "logits", TOP_N = 10):
    """
    Gets TOP_N predictions after last token in a prompt
    """
    last_tok_logits = logits[0, -1]
    
    #gets probs after last tok in seq
    
    if return_type == "probs":
        scores = F.softmax(last_tok_logits, dim=-1).detach().cpu().numpy() #the [0] is to index out of the batch idx
    else:
        scores = last_tok_logits.detach().cpu().numpy()

    #assert probs add to 1
    # assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

    probs_ = []
    for index, prob in enumerate(scores):
        probs_.append((index, prob))

    top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:TOP_N]
    top_k = [(t[1].item(), tokenizer.decode(t[0])) for t in top_k]
    
    return top_k
    
def is_logits_contain_label(ranked_logits, correct_answer):
    # Convert correct_answer to lower case and strip white space
    correct_answer = correct_answer.strip().lower()

    # Loop through the top 10 logits
    for logit_score, logit_value in ranked_logits:
        # Convert logit_value to lower case and strip white space
        logit_value = logit_value.strip().lower()

        # Check if the correct answer contains the logit value
        if correct_answer.find(logit_value) != -1: 
            return True
    return False

import torch
from torch import nn
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class HookInfo:
    handle: torch.utils.hooks.RemovableHandle
    level: Optional[int] = None

class HookedModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hooks: List[HookInfo] = []
        self.context_level: int = 0

    @contextmanager
    def hooks(self, fwd: List[Tuple[str, Callable]] = [], bwd: List[Tuple[str, Callable]] = []):
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_hook(hook_fn) 
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self
        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module

    def print_model_structure(self):
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def get_true_false_probs(output, tokenizer, in_seq = True, scale_relative=False, eps=1e-8, positive_str_tokens=["True"], negative_str_tokens=["False"]):
    # true_ids = [5574, 5852, 1565, 3009] #includes "true" and "True"
    # false_ids = [7700, 8824, 2089, 4541]    
    # positive_str_tokens = ["Yes", "yes", "True", "true"]
    # negative_str_tokens = ["No", "no", "False", "false"]
    positive_tokens = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    negative_tokens = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]
    
    if in_seq:
        output = output['logits'][:,-1,:] #last sequence position
    output = torch.nn.functional.softmax(output, dim=-1)
    # print(f"{output.shape=}, {positive_tokens=}, {negative_tokens=}")

    output = output.squeeze()
    true_prob = output[positive_tokens].sum().item()
    false_prob = output[negative_tokens].sum().item()

    if scale_relative:
        scale = (true_prob + false_prob + eps)
        true_prob /= scale
        false_prob /= scale
    return true_prob, false_prob

def batch_true_false_probs(output, tokenizer, logit_lens = True):
    positive_str_tokens = ["Yes", "yes", "True", "true"],
    negative_str_tokens = ["No", "no", "False", "false"], 
    true_ids = [tokenizer(token).input_ids[-1] for token in positive_str_tokens]
    false_ids = [tokenizer(token).input_ids[-1] for token in negative_str_tokens]

    if not logit_lens:
        output = output['logits'][:,-1,:] #last sequence position
    output = torch.nn.functional.softmax(output, dim=-1)
    # print(output.shape)
    true = output[:, true_ids].sum(axis = 1)
    false = output[:, false_ids].sum(axis = 1)
    return true, false


def store_clean_head_hook_fn(module, input, output, layer_num=0, act_idx=0, clean_z_cache=None, n_heads=64, d_head=128, start_seq_pos=0):
    for head_num in range(n_heads):
        if (layer_num, head_num) not in clean_z_cache:
            clean_z_cache[(layer_num, head_num)] = {}
        clean_z_cache[(layer_num, head_num)][act_idx] = output[:, start_seq_pos:, head_num * d_head : head_num * d_head + d_head ].detach().cpu().numpy()
    return output

def store_clean_resid_hook_fn(module, input, output, layer_num=0, act_idx=0, clean_resid_cache=None, start_seq_pos=0):
    # print(f"{output[0].shape=}")
    if layer_num not in clean_resid_cache:
        clean_resid_cache[layer_num] = {}
    clean_resid_cache[layer_num][act_idx] = output[0][:, start_seq_pos:].detach().cpu().numpy()
    return output

def store_clean_forward_pass(hmodel, input_ids, act_idx, n_layers=80, cache_seq_pos=0, clean_z_cache=None, clean_resid_cache=None, store_z=True, store_resid=False):
    """
    Store activations at every head for a given input_ids, across all sequence positions. Used for patching later.
    Args:
        act_idx: index of activation/data to store. Only important when clean_z_cache contains multiple sets of activations, e.g. for multiple prompts.
        
    """
    if clean_z_cache is None:
        clean_z_cache = {}

    if clean_resid_cache is None:
        clean_resid_cache = {}
    # only for z/attn:
    hook_pairs = []
    for layer in range(n_layers):

        if store_z:
            act_name = f"model.layers.{layer}.self_attn.o_proj"
            hook_pairs.append((act_name, partial(store_clean_head_hook_fn, layer_num=layer, act_idx = act_idx, clean_z_cache=clean_z_cache, start_seq_pos=cache_seq_pos, n_heads = hmodel.model.config.num_attention_heads, d_head = hmodel.model.config.hidden_size // hmodel.model.config.num_attention_heads)))

        if store_resid:
            act_name = f"model.layers.{layer}"
            hook_pairs.append((act_name, partial(store_clean_resid_hook_fn, layer_num=layer, act_idx = act_idx, clean_resid_cache=clean_resid_cache, start_seq_pos=cache_seq_pos)))

    with torch.no_grad():
        with hmodel.hooks(fwd=hook_pairs):
            output = hmodel(input_ids)

    if store_z and not store_resid:
        return output, clean_z_cache
    elif store_resid and not store_z:
        return output, clean_resid_cache
    else:
        return output, clean_z_cache, clean_resid_cache

def patch_head_hook_fn(module, input, output, layer_num = 0, head_num = 0, act_idx = 0, clean_z_cache=None, start_seq_pos=None, d_head=128, device="cuda"):
    """
    Patch in activations from clean_z_cache into output tensor.

    Args:
        act_idx: index of activation/data to patch in. Only important when clean_z_cache contains multiple sets of activations, e.g. for multiple prompts. 
        clean_z_cache: dict of activations to patch in. Keys are (layer_num, head_num) tuples, values are dictionaries of act_idx -> numpy arrays of activations.
        start_seq_pos: sequence position to start patching in activations. Default is None, which means patch in all activations possible (minimum of either output sequence length or cache sequence length)
    """
    if start_seq_pos is None: # patch in as much as possible
        start_seq_pos = min(output.shape[1], clean_z_cache[(layer_num, head_num)][act_idx].shape[1])

    output[:, start_seq_pos:, head_num * d_head : head_num * d_head + d_head ] = torch.from_numpy(clean_z_cache[(layer_num, head_num)][act_idx]).to(device)
    return output


def patch_resid_hook_fn(module, input, output, layer_num = 0, act_idx = 0, clean_resid_cache=None, start_seq_pos=None, device="cuda"):
    if start_seq_pos is None: # patch in as much as possible
        start_seq_pos = min(output[0].shape[1], clean_resid_cache[layer_num][act_idx].shape[1])

    new_hidden_state = output[0]
    new_hidden_state[:, start_seq_pos:] = torch.from_numpy(clean_resid_cache[layer_num][act_idx][:, start_seq_pos:]).to(device)

    new_output = (new_hidden_state,)
    for i in range(len(output)-1):
        new_output += (output[i+1],)
    return new_output


def cache_z_hook_fnc(module, input, output, layer_num=0, activation_buffer_z=None, seq_pos=-1): #input of shape (batch, seq_len, d_model) (taken from modeling_llama.py)
    """
    Cache the last sequence position activations. Used to study 
    activation_buffer_z should be torch.zeros(1, n_layers, d_model) (first dimension is for multiple seq pos, not considering now)
    """
    activation_buffer_z[:,layer_num,:] = input[0][0,seq_pos,:].detach().clone()
    return output


def forward_pass(hmodel, tokenizer, input_ids, act_idx, stuff_to_patch, act_type, clean_cache, scale_relative=False, cache_acts=False, patch_seq_pos=-1, cache_seq_pos=-1, device="cuda"):
    """
    Do a forward pass by patching in activations from clean_z_cache into heads in stuff_to_patch, and caching resulting activations (if cache_acts is True).
    """
    if act_type == "self_attn":
        assert isinstance(stuff_to_patch[0], tuple), "stuff_to_patch must be a list of tuples (layer, head)"
        hook_pairs = []
        for (layer, head) in stuff_to_patch:
            act_name = f"model.layers.{layer}.self_attn.o_proj"
            hook_pairs.append((act_name, partial(patch_head_hook_fn, layer_num=layer, head_num = head, act_idx = act_idx, clean_z_cache=clean_cache, start_seq_pos=patch_seq_pos)))
        
        if cache_acts:
            n_layers = hmodel.model.config.num_hidden_layers
            activation_buffer = torch.zeros(1, n_layers, hmodel.model.config.hidden_size)
            for layer in range(n_layers):
                act_name = f"model.layers.{layer}.self_attn.o_proj" #start with model if using CausalLM object
                hook_pairs.append((act_name, partial(cache_z_hook_fnc, layer_num=layer, activation_buffer=activation_buffer, seq_pos=cache_seq_pos)))
    
    elif act_type == "resid":
        assert isinstance(stuff_to_patch[0], int), "stuff_to_patch must be a list of layers"
        hook_pairs = []
        for layer in stuff_to_patch:
            act_name = f"model.layers.{layer}"
            hook_pairs.append((act_name, partial(patch_resid_hook_fn, layer_num=layer, act_idx = act_idx, clean_resid_cache=clean_cache, start_seq_pos=patch_seq_pos, device=device)))

    with torch.no_grad():
        with hmodel.hooks(fwd=hook_pairs):
            output = hmodel(input_ids)
    
    true_prob, false_prob = get_true_false_probs(output, tokenizer=tokenizer, scale_relative=scale_relative)
    return output, true_prob, false_prob


def erase_data(clean_cache, labels, probe_indices, in_place=False, test_probe=False, erase_seq_pos=None, oracle=True, quadratic=False, existing_fitters=None, return_fitters=False):
    """
    Take a clean_cache and concept-erase the head data.
    probe_indices: list of tuples of (layer, head) (for z) or layer (for resid) to erase

    if in_place, then the clean_cache is modified in place. 
    Returns the new elements of clean_cache in a dictionary.

    If test_probe, trains probes on the erased data to check if LEACE was performed perfectly.

    erase_seq_pos is for if data has multiple seq positions, erase specific ones (takes an array e.g. [-3, -1])
    
    if oracle, use oracle eraser (i.e. erase with correct label). Otherwise, use LEACE.
    if existing_fitters is not None, should be a dictionary with keys layer and value LeaceFitter objects.
    if return_fitters, return the fitters used to erase the data.
    """
    n_samples = len(clean_cache[probe_indices[0]].keys())
    labels = torch.tensor(labels)
    assert len(labels) == n_samples, "labels must be same length as clean_cache"

    output_cache = {}

    def get_fitter(data, probe_index):
        if existing_fitters is not None:
            fitter = existing_fitters[probe_index]
        else:
            if oracle:
                fitter = OracleFitter.fit(data, labels)
            elif quadratic:
                fitter = QuadraticFitter.fit(data, labels)
            else:
                fitter = LeaceFitter.fit(data, labels)
        return fitter
    
    def get_erased(fitter, data):
        if oracle or quadratic:
            return fitter.eraser(data, labels)
        else:
            return fitter.eraser(data)

    X_erased_cache = {}
    if test_probe:
        probes = {}

    fitters = {}
    for probe_index in tqdm(probe_indices):
        clean_data = []
        for i in range(n_samples):
            # if erase_seq_pos is not None:
            #     clean_data.append(clean_cache[probe_index][i][0, erase_seq_pos])
            # else:
            clean_data.append(clean_cache[probe_index][i][0])
        clean_data = torch.from_numpy(np.stack(clean_data, axis=0)).float()
        if erase_seq_pos is not None:
            clean_data = clean_data[:, erase_seq_pos]
        # print(f"At {probe_index=}, {clean_data.shape=}, {labels.shape=}, {erase_seq_pos=}, {clean_cache[probe_index][i][0].shape=}")


        if len(clean_data.shape) > 2:
            # There is another dimension, each seq pos
            erased_data = torch.zeros_like(clean_data)
            for seq_pos in range(clean_data.shape[1]):
                # print(f"{clean_data[:, seq_pos, :].shape=}, {labels.shape=}")
                # clean_data is shape (n_samples, seq_len, d_model)
                # eraser = LeaceEraser.fit(clean_data[:, seq_pos, :], labels)

                fitter = get_fitter(clean_data[:, seq_pos, :], probe_index)
                erased_data[:, seq_pos, :] = get_erased(fitter, clean_data[:, seq_pos, :])

                if return_fitters:
                    fitters[probe_index] = {}
                    fitters[probe_index][seq_pos] = fitter

        else:
            fitter = get_fitter(clean_data, probe_index)
            erased_data = get_erased(fitter, clean_data)
            if return_fitters:
                fitters[probe_index] = fitter

        if test_probe:
            # train probe on final seq pos
            if len(clean_data.shape) > 2:
                X_erased = erased_data[:, -1, :]
            else:
                X_erased = erased_data

            X_erased_cache[probe_index] = X_erased
            null_lr = LogisticRegression(max_iter=1000).fit(X_erased, labels)
            probes[probe_index] = null_lr
            beta = torch.from_numpy(null_lr.coef_)
            y_pred = null_lr.predict(X_erased)
            accuracy = accuracy_score(labels, y_pred)
            print(f"{beta.norm(p=torch.inf)=}, {accuracy=}")
        # print(f"{erased_data.shape=}")

        output_cache[probe_index] = {}
        for i in range(n_samples):
            erased_sample = erased_data[i:i+1].numpy().astype(np.float16)
            output_cache[probe_index][i] = np.zeros_like(clean_cache[probe_index][i])
            if erase_seq_pos is not None:
                output_cache[probe_index][i][:, erase_seq_pos] = erased_sample
            else:
                output_cache[probe_index][i][:] = erased_sample
            # if in_place:
            #     clean_cache[probe_index][i] = erased_sample
    
    return_output = (output_cache,)
    if test_probe:
        return_output += (probes, X_erased_cache)
    if return_fitters:
        return_output += (fitters,)
    return return_output

def combine_caches(clean_z_cache, erased_cache, stuff_to_patch):
    output_cache = clean_z_cache.copy()
    for head in stuff_to_patch:
        output_cache[head] = erased_cache[head]
    
    return output_cache

def split_cache_train_test(cache, labels, data_rows, train_ratio, n_layers=80, in_order=False, cache_seq_pos=None):
    """
    cache_seq_pos should be an array of seq positions to take (e.g. [-1])
    """
    train_cache = {}
    test_cache = {}
    n_samples = len(labels)

    if in_order:
        train_indices = np.arange(int(train_ratio*n_samples))
    else:
        train_indices = np.random.choice(np.arange(n_samples), size=int(train_ratio*n_samples), replace=False)

    for layer in range(n_layers):
        train_labels = []
        test_labels = []
        train_data_rows = []
        test_data_rows = []
        train_cache[layer] = {}
        test_cache[layer] = {}

        test_idx = 0
        train_idx = 0
        for idx in range(n_samples):
            if idx not in train_indices:
                if cache_seq_pos is not None:
                    # print(f"{cache[layer][idx][:, cache_seq_pos].shape=}, {cache[layer][idx].shape=}, {cache_seq_pos=}")
                    test_cache[layer][test_idx] = cache[layer][idx][:, cache_seq_pos]
                else:
                    test_cache[layer][test_idx] = cache[layer][idx]
                test_labels.append(labels[idx])
                test_data_rows.append(data_rows[idx])
                test_idx += 1
            else:
                if cache_seq_pos is not None:
                    train_cache[layer][train_idx] = cache[layer][idx][:, cache_seq_pos]
                else:
                    train_cache[layer][train_idx] = cache[layer][idx]
                train_labels.append(labels[idx])
                train_data_rows.append(data_rows[idx])
                train_idx += 1
    return train_cache, test_cache, train_labels, test_labels, train_data_rows, test_data_rows

"""
forward method from llama:

residual = hidden_states

hidden_states = self.input_layernorm(hidden_states)

# Self Attention
hidden_states, self_attn_weights, present_key_value = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    output_attentions=output_attentions,
    use_cache=use_cache,
)
hidden_states = residual + hidden_states

# Fully Connected
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states

outputs = (hidden_states,)

if output_attentions:
    outputs += (self_attn_weights,)

if use_cache:
    outputs += (present_key_value,)

return outputs
"""
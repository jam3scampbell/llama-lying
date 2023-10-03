# %%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from datasets import load_dataset
from typing import List, Optional, Tuple, Union
from jaxtyping import Float, Int
from typing import List, Tuple
from torch import Tensor
import time
from tqdm import tqdm
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate import infer_auto_device_map
from huggingface_hub import snapshot_download
import csv
import gc
import datasets
from utils.torch_hooks_utils import HookedModule
from functools import partial


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

import numpy as np
import matplotlib.pyplot as plt
import einops





model_name = "meta-llama/Llama-2-70b-chat-hf"
api_key = "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"

GPU_map = {0: "40GiB", 1: "40GiB", 2: "40GiB", 3: "40GiB", 4: "40GiB", 5: "40GiB"}
save_dir = os.getcwd()

device = 0
# device = "mps"

weights_dir = f"Llama-2-70b-chat-hf"
os.makedirs(weights_dir, exist_ok=True)


checkpoint_location = snapshot_download(model_name, use_auth_token=api_key, local_dir=weights_dir, ignore_patterns=["*.safetensors", "model.safetensors.index.json"])
# checkpoint_location = weights_dir


with init_empty_weights():
   model = LlamaForCausalLM.from_pretrained(checkpoint_location)

device_map = infer_auto_device_map(model, max_memory=GPU_map, no_split_module_classes=["LlamaDecoderLayer"]) 

model = load_checkpoint_and_dispatch(
   model,
   checkpoint_location,
    device_map=device_map,
    offload_folder=weights_dir,
    dtype=torch.float16,
)
# model = LlamaForCausalLM.from_pretrained(checkpoint_location)
# model = model.to(device)

tokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)
model.tokenizer = tokenizer


# %%

false_statements = ["Humans have three legs", "A square has seven sides", "The Earth is flat", "Dogs have seven legs", "Water is a solid", "Pizza is traditional chinese food", "Ice cream is a type of vegetable.", "Taylor Swift is a famous athlete", "New York is a country", "A basketball is in the shape of a cube", "History is a STEM subject", "Africa is a country", "The United States is the least powerful country in the world", "All grass is of the color red", "The Sun is a planet"]





# %%
def llama_prompt(system_prompt, statement):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    user_prompt = f"""In one word, please tell me if the following is true or false.

    "{statement}"

    Answer 'True' if it is true and 'False' if it is false."""

    message_0 = {
        "role" : "system",
        "content" : system_prompt
    }
    message_1 = {
        "role" : "user",
        "content" : user_prompt
    }
    dialog = [message_0, message_1]

    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_tokens: List[int] = sum(
        [
            model.tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                bos=True,
                eos=True,
            )
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += model.tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST} ",
        #bos=True,
        #eos=False,
    )
    return dialog_tokens




true_ids = [5574, 5852, 1565, 3009] #includes "true" and "True"
false_ids = [7700, 8824, 2089, 4541]

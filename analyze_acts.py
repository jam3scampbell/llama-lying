
#%%

# import torch
# x = torch.load("data/large_run_1410/activations/formatted/large_run_1410_z_l0_h0.pt")
# x[27] # verify it has things

from datasets import load_dataset
import numpy as np

dataset = load_dataset("notrichardren/azaria-mitchell", split="combined")
dataset = [row for row in dataset if row['dataset'] == 'facts']

# Extract the labels into a list
labels = [row['label'] for row in dataset]

# Convert the list of labels into a numpy array
labels_array = np.array(labels)

# labels_array now contains your labels as a numpy array
print(labels_array)

#%%

from utils.new_probing_utils import ModelActsLargeSimple

model_acts = ModelActsLargeSimple()
model_acts.load_acts("data/large_run_1410/activations/formatted/large_run_1410_liar", n_layers = 80, n_heads = 64, labels = labels_array)


#%%

# %%
import os
import torch
from jaxtyping import Float, Int
from typing import List, Tuple
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import einops

import csv
import pandas as pd
import gc




data_dir = "/mnt/ssd-2/jamescampbell3"

inference_honest_path = "data/large_run_6/inference_outputs/inference_output_6_honest.csv"

inference_liar_path = "data/large_run_6/inference_outputs/inference_output_6_liar.csv"

inference_animal_liar_path = "data/large_run_7/inference_outputs/inference_output_7_animal_liar.csv"

inference_elements_liar_path = "data/large_run_54/data/large_run_53/inference_outputs/inference_output_53_elements_liar.csv"

inference_neutral_prompted = "data/large_run_54/data/large_run_53/inference_outputs/inference_output_53_neutral_unprompted.csv"

inference_misaligned_path = "data/large_run_101/inference_outputs/inference_output_101_misaligned.csv"

inference_misaligned_facts_path = "data/large_run_103/inference_outputs/inference_output_103_misaligned.csv"

mega_splits = ['sciq',
        'commonclaim',
        'creak',
        'azaria_mitchell_cities',
        'truthfulqa',
        'azaria_mitchell_capitals',
        'azaria_mitchell_companies',
        'azaria_mitchell_animals',
        'azaria_mitchell_elements',
        'azaria_mitchell_inventions',
        'azaria_mitchell_facts',
        ]

azaria_mitchell_splits = ['cities', 
                          'capitals', 
                          'companies', 
                          'animals', 
                          'elements', 
                          'inventions', 
                          'facts', 
                          'neg_companies', 
                          'neg_facts', 
                          'conj_neg_companies', 
                          'conj_neg_facts'
                          ]

azaria_mitchell_no_conj_splits = ['cities', 
                          'capitals', 
                          'companies', 
                          'animals', 
                          'elements', 
                          'inventions', 
                          'facts'
                          ]

def order_inference_output(unordered_filename, ordered_filename):
    data = pd.read_csv(f"{data_dir}/{unordered_filename}")
    sorted_data = data.sort_values(by=data.columns[0])
    sorted_data.to_csv(f"{data_dir}/{ordered_filename}", index=False)


def get_inference_accuracy(filename, threshold=0, include_qa_type=[0,1], splits=mega_splits):
    num_correct = 0
    num_total = 0
    acc = 0

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                #row[4] #split
                qa_type = float(row[5]) #qa_type
                origin_dataset = row[4]
                if (p_true > threshold or p_false > threshold) and (qa_type in include_qa_type) and (origin_dataset in splits):
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total

def get_num_labels(filename):
    num_labels = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                label = int(float(row[3]))
                num_labels += label
    return num_labels

def plot_against_confidence_threshold(include_qa_type=[0,1], splits=mega_splits, third_prompt_path=None, third_prompt_mode=''):
    accs_honest = []
    accs_liar = []
    totals_honest = []
    totals_liar = []
    accs_third = []
    totals_third = []
    threshs = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    for thresh in threshs:
        acc_honest, total_honest = get_inference_accuracy(f"{data_dir}/{inference_honest_path}", threshold=thresh, include_qa_type=include_qa_type, splits=splits)
        accs_honest.append(acc_honest)
        totals_honest.append(total_honest)
        acc_liar, total_liar = get_inference_accuracy(f"{data_dir}/{inference_liar_path}", threshold=thresh, include_qa_type=include_qa_type, splits=splits)
        accs_liar.append(acc_liar)
        totals_liar.append(total_liar)
        if third_prompt_path is not None:
            acc_third, total_third = get_inference_accuracy(f"{data_dir}/{third_prompt_path}", threshold=thresh, include_qa_type=include_qa_type, splits=splits)
            accs_third.append(acc_third)
            totals_third.append(total_third)

    plt.subplot(2,1,1)
    plt.plot(threshs, accs_honest, label='honest')
    plt.plot(threshs, accs_liar, label='liar')
    if third_prompt_path is not None:
        plt.plot(threshs, accs_third, label=third_prompt_mode)
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(threshs, totals_honest, label='honest')
    plt.plot(threshs, totals_liar, label='liar')
    if third_prompt_path is not None:
        plt.plot(threshs, totals_third, label=third_prompt_mode)
    plt.ylabel("data points")
    plt.legend()
    plt.show()


def get_accs_by_split(filename, splits=mega_splits, threshold=0, include_qa_type=[0,1]):
    accs_by_split = {}
    for split in splits:
        acc, total = get_inference_accuracy(f"{data_dir}/{filename}", splits=[split], threshold=threshold, include_qa_type=include_qa_type)
        accs_by_split[split] = (acc, total)
    return accs_by_split


def plot_accs_by_split(accs_by_split_honest, accs_by_split_liar, threshold):
    """takes in output of get_accs_by_split
    
    run with:
    for thresh in [0, .2, .5]:
        plot_accs_by_split(get_accs_by_split(inference_honest_path, threshold=thresh), 
                            get_accs_by_split(inference_liar_path, threshold=thresh),
                            threshold=thresh)
    """
    splits = list(accs_by_split_honest.keys())
    accs_honest = [value[0] for value in accs_by_split_honest.values()]
    accs_liar = [value[0] for value in accs_by_split_liar.values()]

    bar_width = 0.35
    index = np.arange((len(splits)))


    bar1 = plt.bar(index, accs_honest, bar_width, label="honest", color="blue")
    bar2 = plt.bar(index + bar_width, accs_liar, bar_width, label="liar", color='red')

    plt.axhline(y=0.5, linestyle='--', color='r')  # Dashed line at 0.5

    plt.ylabel('Model Inference Accuracy')
    plt.xticks(index + bar_width / 2, splits, rotation=90)
    plt.xlabel('Datasets')

    plt.legend()
    plt.title(f'Model Inference Accuracy Across Datasets, threshold: {threshold}')
    plt.show()


def blurb_to_get_conditional_liar_plot():
    accs_elements_liar = get_accs_by_split(inference_elements_liar_path, splits=azaria_mitchell_no_conj_splits, threshold=.5)
    splits = list(accs_elements_liar.keys())
    accs_elements_liar = [value[0] for value in accs_elements_liar.values()]

    bars = plt.bar(np.arange(len(splits)), accs_elements_liar)  # Store bars in a variable
    bars[4].set_color('red')
    plt.axhline(y=0.5, linestyle='--', color='r')  # Dashed line at 0.5
    plt.ylim(0, 1)  # Setting y-axis limits
    plt.ylabel('Inference Accuracy')
    plt.xticks(np.arange(len(splits)), splits, rotation=90)  
    plt.xlabel("Datasets")
    plt.title(f'Conditional Liar Inference Accuracy, threshold: {.5}')
    plt.show()

def get_inference_acc_misaligned(filename, threshold=0, include_qa_type=[0,1]):
    num_correct = 0
    num_total = 0
    acc = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                #row[4] #split
                #qa_type = float(row[5]) #qa_type
                #origin_dataset = row[4]
                if (p_true > threshold or p_false > threshold):
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total


def create_probe_dataset(run_id, seq_pos, prompt_tag, act_type, splits=mega_splits, threshold=0, include_qa_type=[0,1],
                              d_model=8192, d_head=128, n_layers=80, n_heads=64, save_formatted=True):
    """this function does both filtering for specific properties and constructs the probing dataset"""
    #assuming this runs fast, we don't need to save formatted acts, we can just format in real-time based on the property we're interested in
    activations_dir = f"{data_dir}/data/large_run_{run_id}/activations"
    load_path = f"{activations_dir}/unformatted"
    # save_path = f"{os.getcwd()}/data/large_run_5/activations/formatted"
    save_path = f"{activations_dir}/formatted"

    os.makedirs(save_path, exist_ok=True)

    probe_indices = []
    probe_labels = []    

    #filter for the desired indices and save labels
    with open(f"{data_dir}/{inference_honest_path}", 'r') as csvfile: #using only for meta-data but can also filter based on inf performance
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                ind = int(float(row[0]))
                qa_type = float(row[5]) #
                origin_dataset = row[4]
                p_true = float(row[1])
                p_false = float(row[2])
                file_path = f"{load_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{idx}.pt"
                file_exists = os.path.exists(file_path)
                assert file_exists, f"{ind}"
                if (file_exists) and (origin_dataset in splits) and (qa_type in include_qa_type) and (p_true > threshold or p_false > threshold):
                    probe_indices.append(ind)
                    label = int(float(row[3]))
                    probe_labels.append(label)
    #create the probe dataset
    probe_dataset = torch.zeros((len(probe_indices), n_layers, d_model))
    probe_labels = torch.tensor(probe_labels)

    for rel_idx, base_idx in tqdm(enumerate(probe_indices)): 
        probe_dataset[rel_idx,:,:] = torch.load(f"{load_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{base_idx}.pt")

    if save_formatted:
        if act_type in ["resid_mid", "resid_post", "mlp_out"]:
            for layer in tqdm(range(n_layers)):
                torch.save(probe_dataset[:,layer,:].squeeze().clone(), f"{save_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}_l{layer}.pt") #only for single split
                torch.save(probe_labels, f"{save_path}/labels_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}.pt")
        elif act_type in ["z"]:
            for layer in tqdm(range(n_layers), desc='layer'):
                for head in tqdm(range(n_heads), desc='head', leave=False):
                    head_start = head*d_head
                    head_end = head_start + d_head
                    torch.save(probe_dataset[:,layer,head_start:head_end].squeeze().clone(), f"{save_path}/run_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}_l{layer}_h{head}.pt")
                    torch.save(probe_labels, f"{save_path}/labels_{run_id}_{prompt_tag}_{seq_pos}_{act_type}_{splits[0]}.pt")



def create_all_probe_datasets():
    for act_type in ["z", "resid_mid", "mlp_out"]:
        for mode in ["honest", "neutral", "liar"]:
            for split in mega_splits:
                create_probe_dataset(5, -1, mode, act_type, [split])
                torch.cuda.empty_cache
                gc.collect()
                print("Done with ", act_type, ", ", mode, ", ", split)


















class ModelActsLargeSimple:
    """presumes that we already have formatted activations"""
    def __init__(self, run_id, prompt_tag, split):
        self.run_id = run_id
        self.acts_path = f"{os.getcwd()}/data/large_run_{run_id}/activations/formatted"
        self.labels = None
        self.prompt_tag = prompt_tag

        self.split = split
        self.seq_pos = -1

        self.train_indices = None
        self.test_indices = None

        self.probes = None

        self.n_layers = 80
        self.n_heads = 64
        self.d_head = 128
        self.d_model = 8192

    def get_train_test_split(self, X_acts: Float[Tensor, "N d_head"], labels: Float[Tensor, "N"], test_ratio = 0.2):
        probe_dataset = TensorDataset(X_acts, labels)
        if self.train_indices is None and self.test_indices is None:
            generator1 = torch.Generator().manual_seed(42)
            train_data, test_data = random_split(probe_dataset, [1-test_ratio, test_ratio], generator=generator1) 
            self.train_indices = train_data.indices
            self.test_indices = test_data.indices

        X_train, y_train = probe_dataset[self.train_indices]
        X_test, y_test = probe_dataset[self.test_indices]

        return X_train, y_train, X_test, y_test
    

    def _train_single_probe(self, act_type, layer, head=None, max_iter=1000):
        # load_path = f"~/iti_capstone/{filepath}"
        if self.labels is None:
            self.labels = torch.load(f"{self.acts_path}/labels_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}.pt")
        if act_type == "z":
            X_acts: Float[Tensor, "N d_head"] = torch.load(f"{self.acts_path}/run_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}_l{layer}_h{head}.pt")
        elif act_type in ["resid_mid", "resid_post", "mlp_out"]:
            X_acts: Float[Tensor, "N d_model"] = torch.load(f"{self.acts_path}/run_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}_l{layer}.pt")

        #mask = torch.any(X_acts != 0, dim=1) #mask out zero rows because of difference between global and local indices
        #X_acts = X_acts[mask]

        assert X_acts.shape[0] == self.labels.shape[0], "X_acts.shape[0] != self.labels, zero mask fail?"

        X_train, y_train, X_test, y_test = self.get_train_test_split(X_acts, self.labels) # (train_size, d_head), (test_size, d_head)

        clf = LogisticRegression(max_iter=max_iter).fit(X_train.numpy(), y_train.numpy()) #check shapes
        #y_pred = clf.predict(X_train.numpy())

        y_val_pred = clf.predict(X_test.numpy())
        y_val_pred_prob = clf.predict(X_test.numpy())
        
        acc = accuracy_score(y_test.numpy(), y_val_pred)

        return clf, acc#, y_val_pred_prob

    def train_z_probes(self, max_iter=1000):
        probes = {}
        #preds = np.zeros(n_layers, n_heads, len(self.test_indices)) # this is being incredibly dumb with memory usage, def fix before using larger data
        #accs = {}
        probe_accs = torch.zeros(self.n_layers, self.n_heads)
        for layer in tqdm(range(self.n_layers), desc='layer'): #RELYING ON N_LAYERS AND N_HEADS BEING A GLOBAL HERE
            for head in tqdm(range(self.n_heads), desc='head', leave=False):
                probe, acc = self._train_single_probe(act_type="z", layer=layer, head=head, max_iter=max_iter)

                tag = f"l{layer}h{head}"
                probe_accs[layer, head] = acc

                #if layer == 0 and head == 0: #hacky
                #   preds = torch.zeros((self.n_layers, self.n_heads, len(self.test_indices))) # this is being incredibly dumb with memory usage, def fix before using larger data

                #preds[layer, head, :] = torch.tensor(pred)

                probes[tag] = probe
        self.probes = probes
        return probe_accs, probes #, preds

    def evaluate_probe(self, act_type, probe, layer, head=None, max_iter=1000):
        # load_path = f"~/iti_capstone/{filepath}"
        if self.labels is None:
            self.labels = torch.load(f"{self.acts_path}/labels_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}.pt")
        if act_type == "z":
            X_acts: Float[Tensor, "N d_head"] = torch.load(f"{self.acts_path}/run_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}_l{layer}_h{head}.pt")
        elif act_type in ["resid_mid", "resid_post", "mlp_out"]:
            X_acts: Float[Tensor, "N d_model"] = torch.load(f"{self.acts_path}/run_{self.run_id}_{self.prompt_tag}_{self.seq_pos}_{act_type}_{self.split}_l{layer}.pt")

        assert X_acts.shape[0] == self.labels.shape[0], "X_acts.shape[0] != self.labels, zero mask fail?"

        X_train, y_train, X_test, y_test = self.get_train_test_split(X_acts, self.labels) # (train_size, d_head), (test_size, d_head)
        #will re-use indices if fields are not none

        #clf = LogisticRegression(max_iter=max_iter).fit(X_train.numpy(), y_train.numpy()) #check shapes
        #y_pred = clf.predict(X_train.numpy())

        y_val_pred = probe.predict(X_test.numpy())
        #y_val_pred_prob = clf.predict(X_test.numpy())
        
        acc = accuracy_score(y_test.numpy(), y_val_pred)

        return acc
    




def get_transfer_accs(modelacts: List[ModelActsLargeSimple]): #, heads: List[Tuple[Int, Int]]):
    n_layers = 80
    n_heads = 64
    transfer_accs = torch.zeros((n_layers, n_heads, len(modelacts), len(modelacts)))
    for idx_train, modelact_train in enumerate(modelacts):
        for idx_test, modelact_test in enumerate(modelacts):
            #accs = []
            #for layer, head in heads:
            for layer in range(n_layers):
                for head in range(n_heads):
                    tag = f"l{int(layer)}h{int(head)}"
                    probe = modelact_train.probes[tag]
                    acc = modelact_test.evaluate_probe("z", probe, layer, head)
                    transfer_accs[layer, head, idx_train, idx_test] = acc
            #accs.append(acc)
            #transfer_accs[idx_train, idx_test] = sum(accs) / len(heads)
    return transfer_accs
        


def run_transfer_experiment(prompt_mode):
    select_splits = ['azaria_mitchell_capitals','azaria_mitchell_companies','azaria_mitchell_animals','azaria_mitchell_elements','azaria_mitchell_inventions','azaria_mitchell_facts']
    model_acts = []
    for split in select_splits:
        acts = ModelActsLargeSimple(5, prompt_mode, split = split)
        acc, probes = acts.train_z_probes()
        with open(f"probes_{prompt_mode}_{split}.pkl", "wb") as file:
            pickle.dump(probes, file)
        model_acts.append(acts)
    transfer_accs = get_transfer_accs(model_acts)
    #torch.save(transfer_acss, )
    return transfer_accs


def plot_transfer_accs(head: Tuple[Int, Int]):#, transfer_accs_honest, transfer_accs_neutral, transfer_accs_liar):
    #head = (45, 32)

    select_splits = ['azaria_mitchell_capitals','azaria_mitchell_companies','azaria_mitchell_animals','azaria_mitchell_elements','azaria_mitchell_inventions','azaria_mitchell_facts']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5)) # You can adjust the figure size

    tensors = [transfer_accs_honest[head[0],head[1],:,:], transfer_accs_neutral[head[0],head[1],:,:], transfer_accs_liar[head[0],head[1],:,:]] # Replace these with your tensors
    conditions = ['Honest', 'Neutral', 'Liar']

    for i, tensor in enumerate(tensors):
        im = axs[i].imshow(tensor, cmap='hot', vmin=.6, vmax=1)
        axs[i].set_xticks(range(len(select_splits)))
        axs[i].set_xticklabels(select_splits, rotation='vertical')
        axs[i].xaxis.tick_top()
        axs[i].set_title(conditions[i])
        if i != 0:  # Remove y-ticks and labels for second and third images
            axs[i].set_yticks([])

    # Add y-ticks and labels only to the first image
    axs[0].set_yticks(range(len(select_splits)))
    axs[0].set_yticklabels(select_splits)

    # Add one colorbar to the side of the entire figure
    cbar = plt.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    plt.show()

def top_k_keys(d, k):
    return [key for key, value in sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]]


def plot_transfer_accs_smart_select(k):
    select_splits = ['capitals','companies','animals','elements','inventions','facts']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5)) # You can adjust the figure size

    transfer_accs_honest_rearr = einops.rearrange(transfer_accs_honest, "l h tr te -> (l h) tr te")
    transfer_accs_liar_rearr = einops.rearrange(transfer_accs_liar, "l h tr te -> (l h) tr te")
    transfer_accs_neutral_rearr = einops.rearrange(transfer_accs_neutral, "l h tr te -> (l h) tr te")

    honest_selected = torch.zeros((6,6))
    neutral_selected = torch.zeros((6,6))
    liar_selected = torch.zeros((6,6))


    for i in range(6):
        heads = {}
        for head in range(transfer_accs_honest_rearr.shape[0]):
            heads[head] = transfer_accs_honest_rearr[head,i,i].item()
        best_heads = top_k_keys(heads, k)
        honest_selected[i,:] = torch.mean(transfer_accs_honest_rearr[best_heads], dim=0)[i,:]

    for i in range(6):
        heads = {}
        for head in range(transfer_accs_neutral_rearr.shape[0]): 
            heads[head] = transfer_accs_neutral_rearr[head,i,i].item()
        best_heads = top_k_keys(heads, k)
        neutral_selected[i,:] = torch.mean(transfer_accs_neutral_rearr[best_heads], dim=0)[i,:]

    for i in range(6):
        heads = {}
        for head in range(transfer_accs_liar_rearr.shape[0]): #select heads based on honest for now
            heads[head] = transfer_accs_liar_rearr[head,i,i].item()
        best_heads = top_k_keys(heads, k)
        liar_selected[i,:] = torch.mean(transfer_accs_liar_rearr[best_heads], dim=0)[i,:]





    #honest = torch.mean(torch.topk(transfer_accs_honest_rearr, k=k, dim=0).values, dim=0)
    #liar = torch.mean(torch.topk(transfer_accs_liar_rearr, k=k, dim=0).values, dim=0)
    #unprompted =torch.mean(torch.topk(transfer_accs_neutral_rearr, k=k, dim=0).values, dim=0)

    tensors = [honest_selected, neutral_selected, liar_selected]
    #tensors = [honest, unprompted, liar] # Replace these with your tensors
    conditions = ['Honest', 'Unprompted', 'Liar']

    for i, tensor in enumerate(tensors):
        im = axs[i].imshow(tensor, cmap='viridis', vmin=.6, vmax=1)
        axs[i].set_xticks(range(len(select_splits)))
        axs[i].set_xticklabels(select_splits, rotation='vertical')
        axs[i].xaxis.tick_top()
        axs[i].set_title(conditions[i])
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                value = tensor[y, x].item() * 100  # Convert to percentage
                # Place the value on the plot; format to show as a whole number, bolded
                axs[i].text(x, y, f"{value:.0f}", ha="center", va="center", color="w", weight='bold')
        if i != 0:  # Remove y-ticks and labels for second and third images
            axs[i].set_yticks([])

    # Add y-ticks and labels only to the first image
    axs[0].set_yticks(range(len(select_splits)))
    axs[0].set_yticklabels(select_splits)

    # Add one colorbar to the side of the entire figure
    cbar = plt.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    plt.show()


def plot_transfer_accs_diff_pixelwise(k): #not pixelwise anymore
    select_splits = ['capitals','companies','animals','elements','inventions','facts']

    # Compute rearranged tensors
    transfer_accs_honest_rearr = einops.rearrange(transfer_accs_honest, "l h tr te -> (l h) tr te")
    transfer_accs_neutral_rearr = einops.rearrange(transfer_accs_neutral, "l h tr te -> (l h) tr te")


    honest_selected = torch.zeros((6,6))
    neutral_selected = torch.zeros((6,6))


    for i in range(6):
        heads = {}
        for head in range(transfer_accs_honest_rearr.shape[0]):
            heads[head] = transfer_accs_honest_rearr[head,i,i].item()
        best_heads = top_k_keys(heads, k)
        honest_selected[i,:] = torch.mean(transfer_accs_honest_rearr[best_heads], dim=0)[i,:]

    for i in range(6):
        heads = {}
        for head in range(transfer_accs_neutral_rearr.shape[0]): 
            heads[head] = transfer_accs_neutral_rearr[head,i,i].item()
        best_heads = top_k_keys(heads, k)
        neutral_selected[i,:] = torch.mean(transfer_accs_neutral_rearr[best_heads], dim=0)[i,:]


    # Compute means
    #honest = torch.mean(torch.topk(transfer_accs_honest_rearr, k=k, dim=0).values, dim=0)
    #unprompted = torch.mean(torch.topk(transfer_accs_neutral_rearr, k=k, dim=0).values, dim=0)

    # Compute the difference
    difference = honest_selected - neutral_selected

    fig, axs = plt.subplots(figsize=(5, 5))  # Adjust for a single plot

    # Visualize the difference
    im = axs.imshow(difference, cmap='RdBu_r', vmin=-0.4, vmax=0.4)  # Adjusted color scale for difference

    axs.set_xticks(range(len(select_splits)))
    axs.set_xticklabels(select_splits, rotation='vertical')
    axs.xaxis.tick_top()
    axs.set_title('Difference (Honest - Unprompted)')

    # Overlay the tensor values on each pixel
    for y in range(difference.shape[0]):
        for x in range(difference.shape[1]):
            value = difference[y, x].item() * 100  # Convert to percentage
            axs.text(x, y, f"{value:.0f}", ha="center", va="center", color="w", weight='bold')

    axs.set_yticks(range(len(select_splits)))
    axs.set_yticklabels(select_splits)

    # Add one colorbar to the side of the entire figure
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Difference in Accuracy (%)', rotation=270, labelpad=15)
    plt.show()





def get_best_head(transfer_accs, split_idx):
    index = torch.argmax(transfer_accs[:,:,split_idx, split_idx])
    indices = (index // transfer_accs.size(1), index % transfer_accs.size(1))
    return (indices[0].item(), indices[1].item())

# %%

import numpy as np
import torch
import plotly.express as px
import einops
import csv
from plotly.subplots import make_subplots

def plot_probe_accuracies(model_acts, sorted = False, other_head_accs=None, title = "Probe Accuracies"):
    """
    Plots z probe accuracies.
    Takes a model_acts (ModelActs) object by default. If other_head_accs is not None, then it must be a tensor of head accs, and other_head_accs is plotted.
    """

    if other_head_accs is not None:
        all_head_accs_np = other_head_accs
    else:
        all_head_accs_np = model_acts.probe_accs["z"]

    if sorted:
        accs_sorted = -np.sort(-all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads), axis = 1)
    else:
        accs_sorted = all_head_accs_np.reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)
    return px.imshow(accs_sorted, labels = {"x" : "Heads (sorted)", "y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

def plot_resid_probe_accuracies(acc_dict, n_layers, title = "Probe Accuracies", graph_type="line"):
    """
    Plot a dotted line graph with n_layer points, where each point is the accuracy of the residual probe at that layer.
    """
    accs = np.ones(shape=(n_layers)) * -1
    for layer in range(n_layers):
        accs[layer] = acc_dict[layer]
    
    if graph_type is None or graph_type == "line":
        fig = px.line(accs, labels = {"x" : "Layers", "y": "Accuracy"}, title = title)
        fig.update_traces(line=dict(dash='dot'), marker=dict(size=3, color='blue')) 
        fig.update_layout(yaxis_title="Accuracy", showlegend=False)
    elif graph_type == "square":
        fig = px.imshow(np.expand_dims(accs, axis=1), labels = {"y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")
        fig.update_xaxes(showticklabels=False)

    # color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower"
    return fig

def plot_z_probe_accuracies(acc_dict, n_layers, n_heads, sorted = False, title = "Probe Accuracies", average_layer=False):
    """
    Plot z probe accuracies given an acc dict, with keys (layer, head) and value accuracy.
    If average_layer, then plot a line graph with n_layer points, where each point is the average accuracy of the heads at that layer.
    """
    
    head_accs = np.ones(shape=(n_layers, n_heads)) * -1

    if isinstance(acc_dict, dict):        
        for layer in range(n_layers):
            for head in range(n_heads):
                if (layer, head) in acc_dict:
                    head_accs[layer, head] = acc_dict[(layer, head)]
    else: # if acc_dict is already an np array
        head_accs = acc_dict

    if not average_layer:
        if sorted:
            head_accs = -np.sort(-head_accs, axis = 1)

        return px.imshow(head_accs, labels = {"x" : "Heads (sorted)", "y": "Layers"}, title = title, color_continuous_midpoint = 0.5, color_continuous_scale="YlGnBu", origin = "lower")

    else:
        layer_accs = head_accs.mean(axis=1)
        fig = px.line(layer_accs, labels = {"x" : "Layers", "y": "Accuracy"}, title = title)
        fig.update_traces(line=dict(dash='dot'), marker=dict(size=3, color='blue')) 
        fig.update_layout(yaxis_title="Accuracy", showlegend=False)
        return fig

def plot_norm_diffs(model_acts_iti, model_acts, div=True):
    """
    Plots the norm diffs across head z activations
    div = True means divide by original act norms
    """

    iti_acts = einops.rearrange(model_acts_iti.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")
    orig_acts = einops.rearrange(model_acts.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")

    norm_diffs = torch.norm((iti_acts - orig_acts), dim = 2).mean(0)
    if div:
        norm_diffs /= torch.norm(orig_acts, dim = 2).mean(0)
    
    norm_diffs = norm_diffs.numpy().reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)

    return px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

def plot_cosine_sims(model_acts_iti, model_acts):
    iti_acts = einops.rearrange(model_acts_iti.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")
    orig_acts = einops.rearrange(model_acts.stored_acts["z"], "b n_l n_h d_p -> b (n_l n_h) d_p")

    act_sims = torch.nn.functional.cosine_similarity(iti_acts, orig_acts, dim=2).mean(0)
    act_sims = act_sims.numpy().reshape(model_acts_iti.model.cfg.n_layers, model_acts_iti.model.cfg.n_heads)

    # act_sims[44, 23] = act_sims[45, 17] = 1
    return px.imshow(act_sims, labels = {"x" : "Heads", "y": "Layers"},title = "Cosine Similarities of of ITI and Normal Head Activations", color_continuous_midpoint = 1, color_continuous_scale="RdBu", origin = "lower")

def plot_downstream_diffs(model_acts_iti, model_acts, cache_interventions, div=True):
    """
    div = True means divide by original act norms
    cache_interventions is pytorch tensor, shape n_l, n_h, d_h
    """

    act_difference = model_acts_iti.attn_head_acts - model_acts.attn_head_acts # (b, (n_l n_h), d_h)
    act_difference -= einops.rearrange(cache_interventions, "n_l n_h d_h -> (n_l n_h) d_h") 

    norm_diffs = torch.norm((act_difference), dim = 2).mean(0)
    if div:
        norm_diffs /= torch.norm(model_acts.attn_head_acts, dim = 2).mean(0)
    
    norm_diffs = norm_diffs.numpy().reshape(model_acts.model.cfg.n_layers, model_acts.model.cfg.n_heads)

    return px.imshow(norm_diffs, labels = {"x" : "Heads", "y": "Layers"},title = "Norm Differences (divided by original norm) of ITI and Normal Head Activations", color_continuous_midpoint = 0, color_continuous_scale="RdBu", origin = "lower")

def get_inference_accuracy(filename, threshold=0):
    num_correct = 0
    num_total = 0
    acc = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                if p_true > threshold or p_false > threshold:
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total


def acc_tensor_from_dict(probe_accs_dict, n_layers, n_heads=None):
    """
    Helper method to convert dictionaries with component indices as keys (e.g. (5, 4) for Z dict or 79 for resid dict) to tensors, of shape (n_layers, n_heads) for Z or just (n_layers) for resid.
    """
    if n_heads is not None:
        probe_accs = np.zeros(shape=(n_layers, n_heads))
        for layer in range(n_layers):
            for head in range(n_heads):
                probe_accs[layer, head] = probe_accs_dict[(layer, head)]

    else:
        probe_accs = np.zeros(shape=(n_layers,))
        for layer in range(n_layers):
            probe_accs[layer] = probe_accs_dict[layer]
    return probe_accs


def get_px_fig(act_type, transfer_accs, n_layers, n_heads, title, graph_type=None):
    """
    Helper method to generate a figure showing a quantity (accuracy, cosine sim, whatever) for each layer of a model. If act_type is 
    args:
        act_type: "z" is treated alone, all others are 
    """
    if act_type == "z":
        px_fig = plot_z_probe_accuracies(transfer_accs, n_layers, n_heads=n_heads, title=title)
    else:
        px_fig = plot_resid_probe_accuracies(transfer_accs, n_layers, title=title, graph_type=graph_type)
    return px_fig

from utils.new_probing_utils import ModelActs
def plot_transfer_acc_subplots(train_model_acts, test_model_acts, act_type="z", n_layers=80, n_heads=64, test_only=False, cosine_sim=False):
    """
    A function to plot the transfer accuracies for all the different activations, in a grid of subplots. The rows correspond to probes trained on train_model_acts, and the columns correspond to probes tested on test_model_acts. 

    train_model_acts: dictionary of ModelActs objects that already have trained probes
    test_model_acts: dictionary of ModelActs objects, don't need to have trained probes
    Uses test_only in get_probe_transfer_acc method.
    If cosine_sim is True, then the transfer accs are cosine similarities instead of accuracies, and test_model_acts need to have probes trained.

    Returns a tensor of transfer accuracies, and a plotly figure.
    """
    n_rows = len(train_model_acts)
    n_cols = len(test_model_acts)

    if act_type == "z":
        transfer_acc_tensors = np.zeros(shape=(n_rows, n_cols, n_layers, n_heads))
    else:
        transfer_acc_tensors = np.zeros(shape=(n_rows, n_cols, n_layers))

    fig = make_subplots(rows=n_rows, cols=n_cols)
    for row, (train_name, train_model_act) in enumerate(train_model_acts.items()):
        for col, (test_name, test_model_act) in enumerate(test_model_acts.items()):
            transfer_accs = {}
            print(f"{train_name} -> {test_name}")

            for probe_index in train_model_act.probes[act_type]:
                if cosine_sim:
                    train_probe_coef = train_model_act.probes[act_type][probe_index].coef_.squeeze()
                    assert test_model_act.probes is not None
                    test_probe_coef = test_model_act.probes[act_type][probe_index].coef_.squeeze()
                    transfer_accs[probe_index] = np.dot(train_probe_coef, test_probe_coef)/(np.linalg.norm(train_probe_coef)*np.linalg.norm(test_probe_coef))
                else:
                    transfer_accs[probe_index] = train_model_act.get_probe_transfer_acc(act_type, probe_index, test_model_act, test_only=test_only)
            
            px_fig = get_px_fig(act_type, transfer_accs, n_layers, n_heads, title = f"{act_type} probes from {train_name} tested on {test_name}")
        
            fig.add_trace(
                px_fig['data'][0],  # add the trace from plotly express figure
                row=row+1,
                col=col+1
            )
            if act_type == "z":
                transfer_acc_tensors[row, col] = acc_tensor_from_dict(transfer_accs, n_layers, n_heads)
            else:
                transfer_acc_tensors[row, col] = acc_tensor_from_dict(transfer_accs, n_layers)

    train_names = list(train_model_acts.keys())
    test_names = list(test_model_acts.keys())
    for idx1 in range(1, n_cols+1):
        fig.update_xaxes(title_text=f"{test_names[idx1-1]}", row=n_rows, col=idx1)
    for idx2 in range(1, n_rows+1):
        fig.update_yaxes(title_text=f"{train_names[idx2-1]}", row=idx2, col=1)

    return transfer_acc_tensors, fig
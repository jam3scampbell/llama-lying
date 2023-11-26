# Localizing Lying in Llama

Understanding what regions are important for lying in LLaMa-2-70b-chat using various interpretability methods.

Setup: python 3.10, pip install -r requirements.txt

# Features in this repository
## Prompted Lying

All prompted lying results are held in a seperate folder. 

Start by initially filtering data using LLaMA-7b-chat through the filter_data.py file; the file then allows you to save this in a HuggingFace dataset upload.

Afterwards, one can test and graph the recall of various prompts of interest on randomly selected samples of a certain size (e.g. 300 randomly selected samples) using the test-and-graph-7/13/70b-.py files. This will create a seperate folder that will contain images of bar graphs indicating performance of prompts on various test splits.

One can then generate a .csv file of results and place them on a table using save-results-70b.py and table-results-70b.py to generate the figures in the paper.

## Activation Patching

All our results for activation patching can be replicated using patching.py. One just has to specify the prompt combinations they're using (system prompt, user prompt, and prefix) and a dataloader to perform patching using activation_patching. All meta-data involved in patching is stored in PatchInfo objects, such as the type of system prompt (honest vs liar) and the nature of the activation intervention (caching vs overwriting, including which heads). All model predictions are also stored in these objects and results can be computed using compute_acc. Included in this file are also functions to reproduce all patching figures found in the paper. This code is built on PyTorch hooks and makes extensive use of the HookedModel class in torch_hooks_utils.py.

## ModelActs: Activation Caching At Scale
Activation caching and probing is done in the ModelActs class, defined in the utils/new_probing_utils.py file. ModelActs is a class to handle all preparation for training and testing probes.

First, generating and caching activations can be done natively with ModelActs or separately and then loaded into ModelActs. SmallModelActs, a sublcass of ModelActs, uses HookedTransformers from TransformerLens to generate all activations, but does not support multi-GPU transformers. Use gen_acts.py to generate activations for an given model and dataset, then load them into either ModelActsLargeSimple or ChunkedModelActs objects. ModelActsLargeSimple will keep all activations in memory (not VRAM) at once and thus train/test faster, and ChunkedModelActs will load in activations for particular heads/layers one at a time for better memory efficiency but a significant slowdown.

### Training Probes
Once activations have been generated, use the train_probes() method to automatically train probes on the already sampled activations and labels. The train_probes method takes an activation type, so you can specify any activation type (e.g. "z" for ITI) in act_types to train probes on.

Probes and probe accuracies are stored in the self.probes and self.probe_accs dictionaries (keys are also act_types).

To measure how probing generalizes, call the get_transfer_acc() method from a ModelActs object. get_transfer_acc determines how the calling ModelActs object's probes perform on the test data from a different ModelActs object, returning a list of transfer accuracies for every probe.

### Visualizing Probes
utils/analytics_utils contains functions to visualize the transfer probe accuracies and cosine similarities between probes, using plotly.

## LEACE Results
LEACE results use patching and caching utils from utils/interp_utils.py, and require the latest version of the concept-erasure github repo from EleutherAI. The leace.ipynb notebook contains all code to generate LEACE results, including generating clean activations, performing LEACE on those activations, and patching in corrupted (LEACEd) activations 5 or 10 layers at a time.

## Replicating Paper
Logit attribution and transfer probe results/graphs from our paper can be replicated fully with the workshop_paper_graphs.ipynb notebook. LEACE results can be replicated with the leace.ipynb notebook. All patching results can be replicated easily from patching.py

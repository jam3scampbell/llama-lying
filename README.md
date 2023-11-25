# llama-lying

Understanding what regions are important for lying in LLaMa-2-70b-chat using various interpretability methods.

Setup: python 3.10, pip install -r requirements.txt

# Features in this repository
## Prompted Lying
Filter data using LLaMA-7b-chat through the filter_data.py file. 

Afterwards, one can test and graph the recall of various prompts of interest using the test-and-graph-**b-.py files. One can then generate a .csv file of results and place them on a table using save-results-**b.py and table-results-**b.py.

## ModelActs: Activation Caching
Activation caching and probing is done in the ModelActs class, defined in the utils/new_probing_utils.py file. ModelActs is a class to handle all preparation for training and testing probes.

First, generating and caching activations can be done natively with ModelActs or separately and then loaded into ModelActs. SmallModelActs, a sublcass of ModelActs, uses HookedTransformers from TransformerLens to generate all activations, but does not support multi-GPU transformers. Use gen_acts_and_inference_run.py to generate activations for an given model and dataset, then load them into either ModelActsLargeSimple or ChunkedModelActs objects. ModelActsLargeSimple will keep all activations in memory (not VRAM) at once and thus train/test faster, and ChunkedModelActs will load in activations for particular heads/layers one at a time for better memory efficiency but a significant slowdown.

### Training Probes
Once activations have been generated, use the train_probes() method to automatically train probes on the already sampled activations and labels. The train_probes method takes an activation type, so you can specify any activation type (e.g. "z" for ITI) in act_types to train probes on.

Probes and probe accuracies are stored in the self.probes and self.probe_accs dictionaries (keys are also act_types).

To measure how probing generalizes, call the get_transfer_acc() method from a ModelActs object. get_transfer_acc determines how the calling ModelActs object's probes perform on the test data from a different ModelActs object, returning a list of transfer accuracies for every probe.

### Visualizing Probes
utils/analytics_utils contains functions to visualize the transfer probe accuracies and cosine similarities between probes, using plotly.

## LEACE Results
LEACE results use patching and caching utils from utils/interp_utils.py, and require the latest version of the concept-erasure github repo from EleutherAI. The leace.ipynb notebook contains all code to generate LEACE results, including generating clean activations, performing LEACE on those activations, and patching in corrupted (LEACEd) activations 5 or 10 layers at a time.

## Replicating Paper
Logit attribution and transfer probe results/graphs from our paper can be replicated fully with the workshop_paper_graphs.ipynb notebook. LEACE results can be replicated with the leace.ipynb notebook.

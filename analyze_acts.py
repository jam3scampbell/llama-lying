
#%%

#%%

# import torch
# x = torch.load("data/large_run_1410/activations/formatted/large_run_1410_z_l0_h0.pt")
# x[27] # verify it has things

from datasets import load_dataset
import numpy as np

no_prefix_inclusion = [27, 39, 40, 51, 110, 130, 155, 237, 266, 277, 282, 289, 297, 328, 329, 383, 477, 491, 549, 554, 567, 734, 754, 769, 788, 793, 822, 979, 1054, 1102, 1148, 1178, 1200, 1239, 1271, 1329, 1345, 1355, 1448, 1487, 1641, 1642, 1645, 1764, 1904, 1972, 1987, 2080, 2113, 2117, 2118, 2241, 2288, 2364, 2478, 2492, 2559, 2581, 2588, 2589, 2615, 2695, 2708, 2798, 2837, 2866, 2875, 2879, 3003, 3015, 3094, 3186, 3242, 3270, 3309, 3341, 3406, 3421, 3592, 3598, 3730, 3775, 3840, 3858, 3860, 3898, 3928, 4007, 4011, 4035, 4061, 4093, 4099, 4159, 4233, 4253, 4257, 4283, 4295, 4298, 4422, 4461, 4488, 4591, 4592, 4601, 4618, 4641, 4653, 4757, 4843, 4851, 4948, 4954, 8002, 8017, 8113, 8116, 8137, 8139, 8143, 8262, 8288, 8476, 8526, 8531, 8587, 8612, 8613, 8622, 8651, 8717, 8738, 8739, 8753, 8773, 8813, 8867, 8904, 8979, 8982, 8991, 9004, 9012, 9017, 9066, 9069, 9085, 9144, 9150, 9173, 9207, 9332, 9350, 9359, 9398, 9477, 9490, 9543, 9589, 9670, 9750, 9755, 9819, 9840, 9953, 11271, 11289, 11352, 11388, 11396, 11421, 11445, 11463, 11484, 11582, 11640, 11673, 11689, 11704, 11718, 11758, 11868, 11914, 11963, 11967, 12018, 12091, 12134, 12145, 12153, 12187, 12188, 12203, 12293, 12303, 12305, 12391, 12438, 12470, 12487, 12593, 12603, 12624, 12627, 12661, 12695, 12696, 12713, 12733, 12746, 12760, 12778, 12889, 13081, 13107, 13136, 13137, 13146, 13158, 13173, 13197, 13334, 13440, 13525, 13567, 13579, 13623, 13741, 13780, 13835, 13854, 13887, 13921, 14010, 14103, 14121, 14207, 14231, 14283, 14286, 14325, 14334, 14454, 14493, 14501, 14600, 14719, 14767, 14776, 14825, 14955, 14998, 15068, 15096, 15137, 15200, 15208, 15226, 15246, 15276, 15285, 15303, 15310, 15325, 15393, 15455, 15464, 15481, 15515, 15538, 15580, 15608, 15635, 15644, 15704, 15829, 15965, 15972, 15974, 15977]
dataset = load_dataset("notrichardren/azaria-mitchell", split="combined")
dataset = [row for row in dataset if row['ind'] in no_prefix_inclusion]

# Extract the labels into a list
labels = [row['label'] for row in dataset]

# Convert the list of labels into a numpy array
labels_array = np.array(labels)

# labels_array now contains your labels as a numpy array
print(labels_array)

#%%

# NO PREFIX

no_prefix_exclusion = [5062, 5085, 5120, 5151, 5183, 5204, 5219, 5236, 5301, 5334, 5427, 5517, 5550, 5578, 5660, 5687, 5741, 5867, 5879, 5918, 5939, 5961, 5966, 6014, 6118, 6196, 6205, 6272, 6311, 6418, 6506, 6510, 6621, 6662, 6744, 6747, 6753, 6791, 6803, 6843, 6953, 7026, 7031, 7032, 7150, 7286, 7295, 7405, 7419, 7447, 7460, 7510, 7540, 7564, 7573, 7605, 7638, 7648, 7824, 7864, 7901, 7914, 7951, 7988, 10102, 10108, 10133, 10229, 10326, 10356, 10358, 10448, 10519, 10557, 10566, 10573, 10579, 10600, 10607, 10623, 10654, 10661, 10676, 10719, 10753, 10814, 10888, 10919, 10955, 11056, 11064, 11070, 11150, 11153, 11196, 16049, 16060, 16114, 16122, 16126, 16136, 16139, 16155, 16203, 16262, 16411, 16424, 16461, 16536, 16563, 16611, 16630, 16641, 16757, 16758, 16762, 16768, 16770, 16775, 16807, 16879, 16908, 17043]

from utils.new_probing_utils import ModelActsLargeSimple

liar_acts = ModelActsLargeSimple()
liar_acts.load_acts("data/large_run_100/activations/formatted/large_run_100_liar",  n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
liar_acts.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

honest_acts = ModelActsLargeSimple()
honest_acts.load_acts("data/large_run_100/activations/formatted/large_run_100_honest", n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
honest_acts.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 1
hello_acts = ModelActsLargeSimple()
hello_acts.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1_-1",  n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
hello_acts.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 4
countries_acts = ModelActsLargeSimple()
countries_acts.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1_-1",  n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
countries_acts.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

#%%

print("starting transfer probe accs")
# transfer accs between honest and liar
train_acts = {"Trained on Honest": honest_acts, "Trained on Liar": liar_acts}
test_acts = {"Tested on Honest": honest_acts, "Tested on Liar": liar_acts}
from utils.analytics_utils import plot_transfer_acc_subplots
transfer_accs, fig = plot_transfer_acc_subplots(train_acts, test_acts, test_only=True)

fig.update_layout(
    width=600,
    height=600,
    title_text=f"Transfer Probe Accuracies Between Honest and Liar Models",
)

#%%

print("starting transfer probe accs")
# transfer accs between honest and liar
train_acts = {"Trained on Honest": honest_acts, "Trained on Liar": liar_acts, "Trained on Hello": hello_acts, "Trained on Countries": countries_acts}
test_acts = {"Tested on Honest": honest_acts, "Tested on Liar": liar_acts, "Tested on Hello": hello_acts, "Tested on Countries": countries_acts}
from utils.analytics_utils import plot_transfer_acc_subplots
transfer_accs, fig = plot_transfer_acc_subplots(train_acts, test_acts, test_only=True)

fig.update_layout(
    width=600,
    height=600,
    title_text=f"Transfer Probe Accuracies Between Honest and Liar Models",
)


#%%

transfer_accs, cossim_fig = plot_transfer_acc_subplots(train_acts, test_acts, test_only=True, cosine_sim=True)
cossim_fig.update_layout(
    width=600,
    height=600,
    title_text=f"Cosine Similarities Between Honest and Liar Activation Probes",
    # annotations=[
    #     go.layout.Annotation(
    #         text="Tested on:",
    #         showarrow=False,
    #         xref='paper',
    #         yref='paper',
    #         x=0.5,
    #         y=-0.15,
    #         font=dict(
    #             size=16,
    #             color="black"
    #         ),
    #     )
    # ]
)
cossim_fig.show()

#%%

liar_acts_5 = ModelActsLargeSimple()
liar_acts_5.load_acts("data/large_run_3000/activations/formatted/large_run_100_liar", seq_pos = -5, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
liar_acts_5.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

honest_acts_5 = ModelActsLargeSimple()
honest_acts_5.load_acts("data/large_run_3000/activations/formatted/large_run_100_honest", seq_pos = -5, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
honest_acts_5.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 1
hello_acts_5 = ModelActsLargeSimple()
hello_acts_5.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -5, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
hello_acts_5.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 4
countries_acts_5 = ModelActsLargeSimple()
countries_acts_5.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -5, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
countries_acts_5.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)


#%%

liar_acts_10 = ModelActsLargeSimple()
liar_acts_10.load_acts("data/large_run_3000/activations/formatted/large_run_100_liar", seq_pos = -10, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
liar_acts_10.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

honest_acts_10 = ModelActsLargeSimple()
honest_acts_10.load_acts("data/large_run_3000/activations/formatted/large_run_100_honest", seq_pos = -10, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
honest_acts_10.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 1
hello_acts_10 = ModelActsLargeSimple()
hello_acts_10.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -10, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
hello_acts_10.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 4
countries_acts_10 = ModelActsLargeSimple()
countries_acts_10.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -10, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
countries_acts_10.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

#%%

#%%

liar_acts_20 = ModelActsLargeSimple()
liar_acts_20.load_acts("data/large_run_3000/activations/formatted/large_run_100_liar", seq_pos = -20, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
liar_acts_20.train_probe_10s(act_type="z", verbose=True, train_ratio=.8, in_order=False)

honest_acts_20 = ModelActsLargeSimple()
honest_acts_20.load_acts("data/large_run_3000/activations/formatted/large_run_100_honest", seq_pos = -20, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
honest_acts_20.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 1
hello_acts_20 = ModelActsLargeSimple()
hello_acts_20.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -20, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
hello_acts_20.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)

# sys other 4
countries_acts_20 = ModelActsLargeSimple()
countries_acts_20.load_acts("data/large_run_3000/activations/formatted/large_run_3000_sys_other_1", seq_pos = -20, n_layers = 80, n_heads = 64, labels = labels_array, exclude_points = no_prefix_exclusion)
countries_acts_20.train_probes(act_type="z", verbose=True, train_ratio=.8, in_order=False)



#%%

# YES PREFIX

prefix_inclusion = [27, 39, 40, 51, 110, 130, 155, 237, 266, 277, 282, 289, 297, 328, 329, 383, 477, 491, 549, 554, 567, 734, 754, 769, 788, 793, 822, 979, 1054, 1102, 1148, 1178, 1200, 1239, 1271, 1329, 1345, 1355, 1448, 1487, 1641, 1642, 1645, 1764, 1904, 1972, 1987, 2080, 2113, 2117, 2118, 2241, 2288, 2364, 2478, 2492, 2559, 2581, 2588, 2589, 2615, 2695, 2708, 2798, 2837, 2866, 2875, 2879, 3003, 3015, 3094, 3186, 3242, 3270, 3309, 3341, 3406, 3421, 3592, 3598, 3730, 3775, 3840, 3858, 3860, 3898, 3928, 4007, 4011, 4035, 4061, 4093, 4099, 4159, 4233, 4253, 4257, 4283, 4295, 4298, 4422, 4461, 4488, 4591, 4592, 4601, 4618, 4641, 4653, 4757, 4843, 4851, 4948, 4954, 5062, 7026, 7031, 7032, 7150, 7286, 7295, 7405, 7419, 7447, 7460, 7510, 7540, 7564, 7573, 7605, 7638, 7648, 7824, 7864, 7901, 7914, 7951, 7988, 8002, 8017, 8113, 8116, 8137, 8139, 8143, 8262, 8288, 8476, 8526, 8531, 8587, 8612, 8613, 8622, 8651, 8717, 8738, 8739, 8753, 8773, 8813, 8867, 8904, 8979, 8982, 8991, 9004, 9012, 9017, 9066, 9069, 9085, 9144, 9150, 9173, 9207, 9332, 9350, 9359, 9398, 9477, 9490, 9543, 9589, 9670, 9750, 9755, 9819, 9840, 9953, 11271, 11289, 11352, 11388, 11396, 11421, 11445, 11463, 11484, 11582, 11640, 11673, 11689, 11704, 11718, 11758, 11868]

dataset = load_dataset("notrichardren/azaria-mitchell", split="combined")
dataset = [row for row in dataset if row['ind'] in prefix_inclusion]

# Extract the labels into a list
labels = [row['label'] for row in dataset]

# Convert the list of labels into a numpy array
labels_array = np.array(labels)

# labels_array now contains your labels as a numpy array
print(labels_array)


#%%



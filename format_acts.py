# #%%
# from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

# reformat_acts_for_probing(run_id = 1410, N = 17043, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = "z")

#%%

from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

seq_poses = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
prompt_tag = []

for prompt_tag in prompt_tags:
    for seq_pos in seq_poses:
        reformat_acts_for_probing_fully_batched(run_id = 3000, N = 17044, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = prompt_tag, seq_pos = seq_pos, act_type = "z")

#%%
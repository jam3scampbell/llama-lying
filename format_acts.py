# #%%
# from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

# reformat_acts_for_probing(run_id = 1410, N = 17043, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = "z")

#%%

from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

reformat_acts_for_probing_fully_batched(run_id = 1410, N = 17044, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = "liar", seq_pos = -1, act_type = "z")

#%%
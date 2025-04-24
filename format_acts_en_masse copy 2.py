# #%% REVISION 1
# from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

# reformat_acts_for_probing(run_id = 1410, N = 17043, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = "z")

#%% REVISION 2

from utils.new_probing_utils import reformat_acts_for_probing, reformat_acts_for_probing_batched_across_heads, reformat_acts_for_probing_fully_batched

seq_poses = [-1]
# prompt_tags = ["sys_other_1"]
# prompt_tags = ["sys_other_4"]
# prompt_tags = ["honest"]
prompt_tags = ["liar"]

for prompt_tag in prompt_tags:
    for seq_pos in seq_poses:
        reformat_acts_for_probing_fully_batched(run_id = 100, N = 17044, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = prompt_tag, seq_pos = seq_pos, act_type = "z")

#%%

# prompt_tags = ["honest", "liar"]

# for prompt_tag in prompt_tags:
#     for seq_pos in seq_poses:
#         reformat_acts_for_probing_fully_batched(run_id = 100, N = 17044, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = prompt_tag, seq_pos = seq_pos, act_type = "z")

# prompt_tags = ["honest", "liar"]

# for prompt_tag in prompt_tags:
#     for seq_pos in seq_poses:
#         reformat_acts_for_probing_fully_batched(run_id = 200, N = 17044, d_head = 128, n_layers = 80, n_heads = 64, prompt_tag = prompt_tag, seq_pos = seq_pos, act_type = "z")

#%%

# import multiprocessing
# from utils.new_probing_utils import reformat_acts_for_probing_fully_batched

# def process_batch(args):
#     run_id, N, d_head, n_layers, n_heads, prompt_tag, seq_pos, act_type = args
#     reformat_acts_for_probing_fully_batched(run_id, N, d_head, n_layers, n_heads, prompt_tag, seq_pos, act_type)

# if __name__ == '__main__':
#     seq_poses = [-20, -10, -1]
    
#     # Job 1
#     prompt_tags1 = ["sys_other_1", "sys_other_4"]
#     args1 = [(3000, 17044, 128, 80, 64, prompt_tag, seq_pos, "z") for prompt_tag in prompt_tags1 for seq_pos in seq_poses]
    
#     # Job 2
#     prompt_tags2 = ["honest", "liar"]
#     args2 = [(100, 17044, 128, 80, 64, prompt_tag, seq_pos, "z") for prompt_tag in prompt_tags2 for seq_pos in seq_poses]
    
#     # Job 3
#     prompt_tags3 = ["honest", "liar"]
#     args3 = [(200, 17044, 128, 80, 64, prompt_tag, seq_pos, "z") for prompt_tag in prompt_tags3 for seq_pos in seq_poses]
    
#     # Combine all the arguments into a single list
#     all_args = args1 + args2 + args3
    
#     # Create a pool of processes
#     pool = multiprocessing.Pool(processes=18)
    
#     # Process the batches in parallel
#     pool.map(process_batch, all_args)
    
#     # Close the pool
#     pool.close()
#     pool.join()
# %%



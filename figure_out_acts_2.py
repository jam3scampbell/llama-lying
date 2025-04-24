#%%

import os
import re

def extract_info(filename):
    match = re.match(r"run_(\d+)_(.+)_(-?\d+)_z_(\d+)", filename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3)), int(match.group(4))
    return None, None, None, None

def get_common_and_unique_datapoints(datapoints, selected_runs):
    common_datapoints = None
    unique_datapoints = {(run_id, name): set() for run_id, name in selected_runs.items()}

    for run_id, name in selected_runs.items():
        if common_datapoints is None:
            common_datapoints = datapoints[(run_id, name)].copy()
        else:
            common_datapoints.intersection_update(datapoints[(run_id, name)])

    for run_id, name in selected_runs.items():
        unique_datapoints[(run_id, name)] = datapoints[(run_id, name)] - common_datapoints

    return sorted(common_datapoints) if common_datapoints is not None else [], unique_datapoints

def main():
    base_directory = "data"
    names = [f"sys_other_{i}" for i in range(1, 12)] + ["honest", "liar"]
    run_ids = [3000, 100, 200]
    seq_positions = list(range(-20, 0))

    datapoints = {(run_id, name): set() for run_id in run_ids for name in names}

    for run_id in run_ids:
        directory = os.path.join(base_directory, f"large_run_{run_id}", "activations/unformatted")
        for filename in os.listdir(directory):
            file_run_id, name, seqpos, datapoint = extract_info(filename)
            if file_run_id == run_id and name in names and seqpos in seq_positions:
                datapoints[(run_id, name)].add(datapoint)

    for run_id in run_ids:
        for name in names:
            print(f"Run {run_id}, {name}: {sorted(datapoints[(run_id, name)])}")

    selected_runs = {
        200: "honest",
        200: "liar",
        # 3000: "sys_other_1",
        # 3000: "sys_other_4",
    }

    common_datapoints, unique_datapoints = get_common_and_unique_datapoints(datapoints, selected_runs)
    print(f"\nCommon datapoints between the selected runs: {common_datapoints}")
    for run_id, name in selected_runs.items():
        print(f"Unique datapoints for Run {run_id}, {name}: {sorted(unique_datapoints[(run_id, name)])}")

if __name__ == "__main__":
    main()
# %%

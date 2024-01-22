import pandas as pd
import os
import shutil
import time
from tqdm import tqdm

first_dir = "/mnt/ucsf_dp_server/DP_CP_Projects/Kim.203bc.Cyto/203.B"
second_dir = "/media/hdd1/pancreas"

# move all the .svs files from the first_dir to the second_dir and profile the time taken for each slide in a csv file saved in the second_dir

# get all the .svs files in the first_dir
svs_files = [
    os.path.join(first_dir, f) for f in os.listdir(first_dir) if f.endswith(".svs")
]

# create a dataframe to store the profiling results
df = pd.DataFrame(columns=["slide_name", "time_taken"])

# iterate over the svs_files and move them to the second_dir
for svs_file in tqdm(svs_files, desc="Moving files"):
    start_time = time.time()
    shutil.move(svs_file, second_dir)
    end_time = time.time()
    time_taken = end_time - start_time
    df = df.append(
        {"slide_name": os.path.basename(svs_file), "time_taken": time_taken},
        ignore_index=True,
    )

# save the profiling results
df.to_csv(os.path.join(second_dir, "move_with_profile.csv"), index=False)

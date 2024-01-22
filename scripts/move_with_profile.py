# import pandas as pd
# import os
# import shutil
# import time
# from tqdm import tqdm

# first_dir = "/mnt/ucsf_dp_server/DP_CP_Projects/Kim.203bc.Cyto/203.B"
# second_dir = "/media/hdd1/pancreas"

# # move all the .svs files from the first_dir to the second_dir and profile the time taken for each slide in a csv file saved in the second_dir

# # get all the .svs files in the first_dir
# svs_files = [
#     os.path.join(first_dir, f) for f in os.listdir(first_dir) if f.endswith(".svs")
# ]

# # create a dataframe to store the profiling results
# df = pd.DataFrame(columns=["slide_name", "time_taken"])

# # iterate over the svs_files and move them to the second_dir
# for svs_file in tqdm(svs_files, desc="Moving files"):
#     start_time = time.time()
#     shutil.move(svs_file, second_dir)
#     end_time = time.time()
#     time_taken = end_time - start_time
#     df = df.append(
#         {"slide_name": os.path.basename(svs_file), "time_taken": time_taken},
#         ignore_index=True,
#     )

# # save the profiling results
# df.to_csv(os.path.join(second_dir, "move_with_profile.csv"), index=False)

import os
import subprocess
import time
from tqdm import tqdm
import pandas as pd


def rsync_file(source, destination):
    """
    Function to rsync a single file and capture the time taken.
    """
    start_time = time.time()
    subprocess.run(["rsync", "-av", source, destination])
    end_time = time.time()
    return end_time - start_time


def main():
    source_folder = "/mnt/ucsf_dp_server/DP_CP_Projects/Kim.203bc.Cyto/203.B"
    destination_folder = "/media/hdd1/pancreas"
    log_file = "/media/hdd1/pancreas/moving_time.csv"

    # Prepare a list to store log data
    log_data = []

    # Get list of files in the source folder
    files = [
        f
        for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]

    # Process each file with a progress bar
    for file in tqdm(files, desc="Syncing Files"):
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)

        # Get file size
        file_size = os.path.getsize(source_file)

        # Rsync the file and measure the time taken
        duration = rsync_file(source_file, destination_file)

        # Append log data
        log_data.append([file, duration, file_size])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(
        log_data, columns=["File Name", "Time Taken (s)", "File Size (bytes)"]
    )
    df.to_csv(log_file, index=False)


if __name__ == "__main__":
    main()

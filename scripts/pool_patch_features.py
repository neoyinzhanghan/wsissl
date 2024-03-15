import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

h5_dir = "/media/hdd1/neo/pancreas_SimCLR_2024-03-11"
num_patches_per_h5 = 1000
save_dir = "/media/hdd1/neo/LE_pancreas_LUAD/pancreas"

# find all the h5 files in the h5_dir
h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]

# create a metadata file to store where the features come from
metadata_dict = {
    "idx": [],
    "h5_file": [],
}

idx = 0

# iterate over the h5 files
for h5_file in tqdm(h5_files, desc="Pooling Patch Features"):
    # open the h5 file at the given path
    h5_path = os.path.join(h5_dir, h5_file)

    h5_pack = h5py.File(h5_path, "r")

    # Get the features from the h5 file which has dim (N, 2048, 1, 1)
    features = np.array(h5_pack["features"])

    # first reshape the features to (N, 2048)
    features = features.reshape(features.shape[0], -1)

    # Then randomly sample num_patches_per_h5 patches from the features each feature has dim (2048,)
    # shuffle the features
    np.random.shuffle(features)

    for i in range(num_patches_per_h5):
        # save the feature as a numpy array in the save_dir
        np.save(os.path.join(save_dir, f"{idx}.npy"), features[i])

        # update the metadata
        metadata_dict["idx"].append(idx)
        metadata_dict["h5_file"].append(h5_file)

        idx += 1

    h5_pack.close()

# save the metadata file
metadata_path = os.path.join(save_dir, "metadata.csv")
metadata_df = pd.DataFrame(metadata_dict)
metadata_df.to_csv(metadata_path, index=False)
print(f"Metadata file saved at {metadata_path}")
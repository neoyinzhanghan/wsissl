import os
import random
import shutil
from tqdm import tqdm

subsample_size = 20
data_dir = "/media/hdd2/pancreas_PF"
save_dir = "/media/hdd2/pancreas_PF_SMALL"
symbolic = False

# for all the subfolders in data_dir, randomly pick a subset of subsample_size images and copy them to save_dir
# if symbolic is True, then create symbolic links instead of copying

# if the save_dir does not exist, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# for all the subfolders in data_dir, randomly pick a subset of subsample_size images and copy them to save_dir
folder_list = os.listdir(data_dir)
subsampled_folders = random.sample(folder_list, subsample_size)

for folder in tqdm(subsampled_folders, desc="Copying folders"):
    if symbolic:
        os.symlink(os.path.join(data_dir, folder), os.path.join(save_dir, folder))
    else:  # use shutil
        shutil.copytree(os.path.join(data_dir, folder), os.path.join(save_dir, folder))

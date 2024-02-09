import os
import random
import pandas as pd
from tqdm import tqdm

# path to the current full dataset
dataset_dir = "/media/hdd1/neo/pancreas_SimCLR_data/pancreas_IMAGENET_SYM"

mini_dataset_dir = "/media/hdd1/neo/pancreas_SimCLR_data/pancreas_IMAGENET_SYM_mini"
os.makedirs(mini_dataset_dir, exist_ok=True)

num_train_per_class = 10000
num_val_per_class = 1000
num_test_per_class = 1000

# there should be three folders in the dataset_dir, train, val, and test
# each of these folders should contain a subfolder named patch

# randomly subssample the train, val, and test folders, and copy them to the mini_dataset_dir using the same folder structure and using shutil
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(mini_dataset_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(mini_dataset_dir, folder, "patch"), exist_ok=True)
    images = os.listdir(os.path.join(dataset_dir, folder, "patch"))

    if folder == "train":
        num_per_class = num_train_per_class
    elif folder == "val":
        num_per_class = num_val_per_class
    elif folder == "test":
        num_per_class = num_test_per_class

    # randomly sample the images
    selected_images = random.sample(images, num_per_class)

    for image in tqdm(selected_images, desc=folder):
        os.system(f"cp {os.path.join(dataset_dir, folder, 'patch', image)} {os.path.join(mini_dataset_dir, folder, 'patch', image)}")
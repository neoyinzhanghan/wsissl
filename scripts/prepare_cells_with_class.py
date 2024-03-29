# we want to split the data at a slide level, into train, val, test
# and we want need a metadata file for each of these splits
# we first do an argparse, the input directories can have multiple directories where we will pool the content together
# each input directory containsa bunch of subdirectories, each contains a bunch of .jpg files
# we first pool the subdirectories together, then split them into train, val, test
# we create a metadata CSV file with columns: slide_name, split and save the metadata CSV file in the save_dir
# we create subfolders train, val, test in the save_dir and copy the jpg in the corresponding slide subdirectories into these subfolders

import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import shutil
from imghdr import what


#########################################################
# ARGUMENT PARSING
#########################################################

parser = argparse.ArgumentParser()


#########################################################
group = parser.add_argument_group("Directories")
#########################################################

group.add_argument(
    "--input_dirs",
    type=str,
    nargs="+",
    required=True,
    help="List of input directories, each containing a bunch of subdirectories, each containing a bunch of .jpg files",
)

group.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Directory where we save the metadata CSV file and the train, val, test subdirectories",
)

#########################################################
group = parser.add_argument_group("Splitting parameters")
#########################################################

group.add_argument(
    "--train_prop",
    type=float,
    default=0.8,
    help="Fraction of data to use for training",
)

group.add_argument(
    "--val_prop",
    type=float,
    default=0.1,
    help="Fraction of data to use for validation",
)

group.add_argument(
    "--test_prop",
    type=float,
    default=0.1,
    help="Fraction of data to use for testing",
)

group.add_argument(
    "--max_n_wsis",
    type=int,
    default=None,
    help="Maximum number of WSIs to use",
)

#########################################################
group = parser.add_argument_group("Misc")
#########################################################

group.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed",
)

group.add_argument(
    "--symbolic",
    action="store_true",
    help="Create symbolic links instead of copying the files",
)

args = parser.parse_args()

# if the save_dir does not exist, create it
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# first pool the subdirectories together
pooled_pdrs = []

lsts_to_combined = [
    [
        os.path.join(input_dir, wsi_dir, "cells")
        for wsi_dir in os.listdir(input_dir)
        if os.path.isdir(
            os.path.join(input_dir, wsi_dir)
        )  # check that the folder does not start with name "ERROR"
        and not wsi_dir.startswith("ERROR")
    ]
    for input_dir in args.input_dirs
]

for lst in lsts_to_combined:
    pooled_pdrs += lst


# split the pooled subdirectories into train, val, test
# we use the same seed for reproducibility

# shuffle the pooled subdirectories
shuffled_pooled_pdrs = random.sample(pooled_pdrs, len(pooled_pdrs))

# if max_n_wsis is not None, we only keep the first max_n_wsis elements
if args.max_n_wsis is not None:
    shuffled_pooled_pdrs = shuffled_pooled_pdrs[: args.max_n_wsis]

# split the shuffled pooled subdirectories into train, val, test
train_pdrs = shuffled_pooled_pdrs[: int(args.train_prop * len(shuffled_pooled_pdrs))]
val_pdrs = shuffled_pooled_pdrs[
    int(args.train_prop * len(shuffled_pooled_pdrs)) : int(
        (args.train_prop + args.val_prop) * len(shuffled_pooled_pdrs)
    )
]
test_pdrs = shuffled_pooled_pdrs[
    int((args.train_prop + args.val_prop) * len(shuffled_pooled_pdrs)) :
]

# create the metadata CSV file starting with a pandas dataframe
df = pd.DataFrame(
    {
        "slide_name": [os.path.basename(pdr) for pdr in shuffled_pooled_pdrs],
        "split": ["train"] * len(train_pdrs)
        + ["val"] * len(val_pdrs)
        + ["test"] * len(test_pdrs),
    }
)

# save the metadata CSV file
df.to_csv(os.path.join(args.save_dir, "metadata.csv"), index=False)

# create the train, val, test subdirectories
if not os.path.exists(os.path.join(args.save_dir, "train")):
    os.mkdir(os.path.join(args.save_dir, "train"))
if not os.path.exists(os.path.join(args.save_dir, "val")):
    os.mkdir(os.path.join(args.save_dir, "val"))
if not os.path.exists(os.path.join(args.save_dir, "test")):
    os.mkdir(os.path.join(args.save_dir, "test"))
if not os.path.exists(os.path.join(args.save_dir, "train", "patch")):
    os.mkdir(os.path.join(args.save_dir, "train", "patch"))
if not os.path.exists(os.path.join(args.save_dir, "val", "patch")):
    os.mkdir(os.path.join(args.save_dir, "val", "patch"))
if not os.path.exists(os.path.join(args.save_dir, "test", "patch")):
    os.mkdir(os.path.join(args.save_dir, "test", "patch"))

# copy the jpg in the corresponding slide subdirectories into the train, val, test subdirectories
# the extension of the files is .JPEG

current_index = 0

for pdr in tqdm(train_pdrs, desc="Copying train files"):
    # for all the image files in the current pdr make sure to check they are images in the for loop
    for root, dirs, files in os.walk(pdr):
        for file in files:
            file_path = os.path.join(root, file)
            if what(file_path):
                if args.symbolic:
                    # the name of the symbolic link should be patch_current_index.JPEG
                    symlink_name = os.path.join(
                        args.save_dir,
                        "train",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now create the symbolic link
                    symlink_target = os.path.join(pdr, file)

                    os.symlink(symlink_target, symlink_name)

                    current_index += 1

                else:
                    # the name of the new file should be patch_current_index.JPEG
                    new_name = os.path.join(
                        args.save_dir,
                        "train",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now copy the file
                    shutil.copyfile(
                        os.path.join(pdr, file),
                        new_name,
                    )

                    current_index += 1

for pdr in tqdm(val_pdrs, desc="Copying val files"):
    for root, dirs, files in os.walk(pdr):
        for file in files:
            file_path = os.path.join(root, file)

            if what(file_path):
                if args.symbolic:
                    # the name of the symbolic link should be patch_current_index.JPEG
                    symlink_name = os.path.join(
                        args.save_dir,
                        "val",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now create the symbolic link
                    symlink_target = os.path.join(pdr, file)

                    os.symlink(symlink_target, symlink_name)

                    current_index += 1

                else:
                    # the name of the new file should be patch_current_index.JPEG
                    new_name = os.path.join(
                        args.save_dir,
                        "val",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now copy the file
                    shutil.copyfile(
                        os.path.join(pdr, file),
                        new_name,
                    )

                    current_index += 1

for pdr in tqdm(test_pdrs, desc="Copying test files"):
    for root, dirs, files in os.walk(pdr):
        for file in files:
            file_path = os.path.join(root, file)

            if what(file_path):
                if args.symbolic:
                    # the name of the symbolic link should be patch_current_index.JPEG
                    symlink_name = os.path.join(
                        args.save_dir,
                        "test",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now create the symbolic link
                    symlink_target = os.path.join(pdr, file)

                    os.symlink(symlink_target, symlink_name)

                    current_index += 1

                else:
                    # the name of the new file should be patch_current_index.JPEG
                    new_name = os.path.join(
                        args.save_dir,
                        "test",
                        "patch",
                        "patch_" + str(current_index) + ".JPEG",
                    )

                    # now copy the file
                    shutil.copyfile(
                        os.path.join(pdr, file),
                        new_name,
                    )

                    current_index += 1

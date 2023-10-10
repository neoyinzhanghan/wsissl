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
    "--max_n_data",
    type=int,
    default=None,
    help="Maximum number of data to use",
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
        os.path.join(input_dir, wsi_dir)
        for wsi_dir in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, wsi_dir))
    ]
    for input_dir in args.input_dirs
]

for lst in lsts_to_combined:
    pooled_pdrs += lst


# split the pooled subdirectories into train, val, test
# we use the same seed for reproducibility

# shuffle the pooled subdirectories
shuffled_pooled_pdrs = random.sample(pooled_pdrs, len(pooled_pdrs))

# if max_n_data is not None, we only use the first max_n_data data
if args.max_n_data is not None:
    shuffled_pooled_pdrs = shuffled_pooled_pdrs[: args.max_n_data]

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
        "slide_name": [os.path.basename(pdr) for pdr in pooled_pdrs],
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

# copy the jpg in the corresponding slide subdirectories into the train, val, test subdirectories
# the new file name should be subdirectory_name___file_name (triple underscores)
for pdr in tqdm(train_pdrs, desc="Copying train files"):
    for file in os.listdir(pdr):
        if args.symbolic:
            os.symlink(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "train",
                    os.path.basename(pdr) + "___" + file,
                ),
            )
        else:
            # use shutil.copyfile
            shutil.copyfile(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "train",
                    os.path.basename(pdr) + "___" + file,
                ),
            )

for pdr in tqdm(val_pdrs, desc="Copying val files"):
    for file in os.listdir(pdr):
        if args.symbolic:
            os.symlink(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "val",
                    os.path.basename(pdr) + "___" + file,
                ),
            )
        else:
            shutil.copyfile(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "val",
                    os.path.basename(pdr) + "___" + file,
                ),
            )

for pdr in tqdm(test_pdrs, desc="Copying test files"):
    for file in os.listdir(pdr):
        if args.symbolic:
            os.symlink(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "test",
                    os.path.basename(pdr) + "___" + file,
                ),
            )
        else:
            shutil.copyfile(
                os.path.join(pdr, file),
                os.path.join(
                    args.save_dir,
                    "test",
                    os.path.basename(pdr) + "___" + file,
                ),
            )

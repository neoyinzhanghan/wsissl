import argparse

#######################################################
# ARGUMENT PARSING
#######################################################

parser = argparse.ArgumentParser()

#######################################################
group = parser.add_argument_group("Directories")
#######################################################

group.add_argument(
    "--data_dirs",
    type=str,
    nargs="+",
    required=True,
    help="Each directories should contain a folder of patches for each class",
)

group.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Directory where to save the logs. By default, we look for a DINO subdirectory, within which we name the log directory inside DINO with version_n",
)

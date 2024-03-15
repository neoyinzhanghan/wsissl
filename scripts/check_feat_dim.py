import h5py

h5_path = "/media/hdd1/neo/pancreas_SimCLR_2024-03-11/23.CFNA.9 A1 H&E _154610-patch_features.h5"

# open the h5 file at the given path
h5_file = h5py.File(h5_path, "r")

# print the data structure of the h5 file
print(h5_file)

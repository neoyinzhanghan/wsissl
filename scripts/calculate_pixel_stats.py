import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

root_folder = "/media/hdd1/neo/bma_region_clf_data_full_v2"
img_exts = ["jpg", "png"]

# first iteratively find all the images in the root_folder
images = []
non_images = []
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if Path(file).suffix[1:] in img_exts:
            images.append(os.path.join
                            (root, file))
        else:
            non_images.append(os.path.join(root, file))


# calculate the pixel stats (the RGB mean and std) for the images
# we will use these stats to normalize the images
            
# first calculate the mean
mean = np.zeros(3)
std = np.zeros(3)
num_pixels = 0
for image in tqdm(images, desc="Calculating pixel stats:"):

    img = Image.open(image)
    # if the image is RGBA, convert it to RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")
        
    img = np.array(img)/255.0
    mean += img.mean(axis=(0, 1))
    std += img.std(axis=(0, 1))
    num_pixels += img.shape[0] * img.shape[1]

mean /= len(images)
std /= len(images)

print(f"Mean: {mean}")
print(f"Std: {std}")
print(f"Num pixels: {num_pixels}")

print(non_images)
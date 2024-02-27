import os
import shutil
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

data_dir = "/media/hdd1/neo/bma_region_clf_data_full_v2"
train_ratio = 0.8
val_ratio = 0.1
# Assuming test_ratio is implicitly defined as the remaining percentage.
save_dir = "/media/hdd1/neo/bma_region_clf_data_full_v2_split"

# Prepare directories for the splits
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(save_dir, split), exist_ok=True)

# Function to convert and save image
def convert_and_save_image(src_path, dest_path):
    with Image.open(src_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(dest_path, 'JPEG')

# Initialize lists to store metadata
metadata = {'train': [], 'val': [], 'test': []}

# Process each class directory
for class_name in tqdm(os.listdir(data_dir), desc="Processing classes"):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files

    # Get a list of images and shuffle
    images = os.listdir(class_dir)
    np.random.shuffle(images)

    # Compute split sizes
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Function to process and save split data
    def process_split(images, split_name):
        split_dir = os.path.join(save_dir, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for image_name in tqdm(images, desc=f"Processing {split_name} images"):
            src_path = os.path.join(class_dir, image_name)
            dest_path = os.path.join(split_dir, image_name)
            convert_and_save_image(src_path, dest_path)
            metadata[split_name].append([dest_path, class_name])

    # Process each split
    process_split(train_images, 'train')
    process_split(val_images, 'val')
    process_split(test_images, 'test')

# Save metadata to CSV
for split in ['train', 'val', 'test']:
    df = pd.DataFrame(metadata[split], columns=['image_path', 'label'])
    df.to_csv(os.path.join(save_dir, f'{split}_metadata.csv'), index=False)

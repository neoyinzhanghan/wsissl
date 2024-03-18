import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random


# Function to split slides into train, val, test sets
def split_slides(root_paths, test_size=0.2, val_size=0.25):
    slide_paths = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}

    for class_label, root_path in enumerate(root_paths):
        slide_dirs = os.listdir(root_path)
        # Splitting the slides into train and test
        train_slides, test_slides = train_test_split(
            slide_dirs, test_size=test_size, random_state=42
        )
        # Further split train into train and val
        train_slides, val_slides = train_test_split(
            train_slides, test_size=val_size, random_state=42
        )

        # Assign slide paths and labels based on the split
        for slide in train_slides:
            slide_paths["train"].append(os.path.join(root_path, slide))
            labels["train"].append(class_label)
        for slide in val_slides:
            slide_paths["val"].append(os.path.join(root_path, slide))
            labels["val"].append(class_label)
        for slide in test_slides:
            slide_paths["test"].append(os.path.join(root_path, slide))
            labels["test"].append(class_label)

    return slide_paths, labels


# Function to load regions from given slide paths with a limit on the number of patches per slide
def load_regions(slide_paths, labels, max_num_patches_per_slide=10):
    X = {"train": [], "val": [], "test": []}
    y = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for path, label in zip(slide_paths[split], labels[split]):
            if not os.path.isdir(path):
                continue

            slide_files = os.listdir(path)
            if (
                max_num_patches_per_slide is not None
                and len(slide_files) > max_num_patches_per_slide
            ):
                slide_files = random.sample(slide_files, max_num_patches_per_slide)

            for file in slide_files:
                file_path = os.path.join(path, file)
                img_array = np.load(file_path)
                X[split].append(img_array.flatten())

                # print the shape of the img_array.flatten()
                print(img_array.flatten().shape)

                y[split].append(label)

    for split in ["train", "val", "test"]:
        X[split] = np.array(X[split])
        y[split] = np.array(y[split])

    return X, y


runtime_data = {}
root_paths = [
    "/media/hdd1/neo/SC_pancreas_LUAD/LUAD",
    "/media/hdd1/neo/SC_pancreas_LUAD/pancreas",
]  # Replace with your actual paths


start_time = time.time()
print("Preparing data...")
slide_paths, labels = split_slides(root_paths)
X, y = load_regions(slide_paths, labels)
runtime_data["Data Preparation"] = time.time() - start_time

# Flatten X and convert y to integers
X_train, X_val, X_test = X["train"], X["val"], X["test"]
y_train, y_val, y_test = (
    np.array(labels["train"]),
    np.array(labels["val"]),
    np.array(labels["test"]),
)

start_time = time.time()
print("Fitting Logistic Regression Model...")
model = LogisticRegression(max_iter=1000, solver="lbfgs")

# check the length of the X_train and y_train
print(len(X_train))
print(len(y_train))
import sys

sys.exit()

model.fit(X_train, y_train)
runtime_data["Model Fitting"] = time.time() - start_time

start_time = time.time()
print("Evaluating model...")
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
runtime_data["Model Evaluation"] = time.time() - start_time

# Save runtime data
runtime_df = pd.DataFrame(list(runtime_data.items()), columns=["Operation", "Duration"])
runtime_df.to_csv("runtime_data.csv", index=False)
print("Runtime data saved to 'runtime_data.csv'.")

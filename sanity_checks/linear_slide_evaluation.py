import os
import numpy as np
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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


def load_regions(slide_paths, labels, max_num_patches_per_slide=100):
    X = {"train": [], "val": [], "test": []}
    y = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for path, label in tqdm(
            zip(slide_paths[split], labels[split]),
            desc=f"Loading {split} regions",
        ):
            if not os.path.isdir(path):
                continue

            slide_files = os.listdir(path)
            if (
                max_num_patches_per_slide is not None
                and len(slide_files) > max_num_patches_per_slide
            ):
                slide_files = random.sample(slide_files, max_num_patches_per_slide)

            # Initialize a list to hold all patch features for the current slide
            slide_features = []
            for file in slide_files:
                file_path = os.path.join(path, file)
                img_array = np.load(file_path)
                slide_features.append(img_array.flatten())

            if (
                slide_features
            ):  # Check if there are any features to avoid division by zero
                # Compute the average feature vector across all patches for this slide
                avg_feature = np.mean(slide_features, axis=0)
                X[split].append(avg_feature)
                y[split].append(label)

    for split in ["train", "val", "test"]:
        X[split] = np.array(X[split])
        y[split] = np.array(y[split])

    return X, y


runtime_data = {}
root_paths = [
    "/media/hdd2/neo/SC_pancreas_LUAD_resnet/LUAD",
    "/media/hdd2/neo/SC_pancreas_LUAD_resnet/pancreas",
]  # Replace with your actual paths


start_time = time.time()
print("Preparing data...")
slide_paths, labels = split_slides(root_paths)
X, y = load_regions(slide_paths, labels)
runtime_data["Data Preparation"] = time.time() - start_time

# Flatten X and convert y to integers
X_train, X_val, X_test = X["train"], X["val"], X["test"]
y_train, y_val, y_test = y["train"], y["val"], y["test"]

# print the dimension of the data
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

start_time = time.time()
print("Fitting Logistic Regression Model...")
model = LogisticRegression(max_iter=100, solver="lbfgs")

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

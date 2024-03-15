import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

data_dir = "/media/hdd1/neo/LE_pancreas_LUAD"  # Adjust this to your data directory
run_dir = "/media/hdd1/neo/runs/2024-03-15 LE_pancreas_LUAD"  # Adjust this to your run directory

print("Pooling Data")
# Step 1: Pool data together
features = []
labels = []
for class_dir in tqdm(os.listdir(data_dir), desc="Pooling Classes"):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for file in tqdm(os.listdir(class_path), desc=f"Pooling {class_dir}"):
            if file.endswith(".npy"):
                file_path = os.path.join(class_path, file)
                feature = np.load(file_path)
                features.append(feature)
                labels.append(class_dir)

features = np.array(features)
labels = np.array(labels)

print("Splitting Data")
# Step 2: Split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Saving Data")
# Step 3: Save metadata
metadata = pd.DataFrame(
    {
        "filename": np.concatenate([X_train, X_val, X_test]),
        "split": ["train"] * len(X_train)
        + ["val"] * len(X_val)
        + ["test"] * len(X_test),
        "label": np.concatenate([y_train, y_val, y_test]),
    }
)
metadata.to_csv(os.path.join(run_dir, "split.csv"), index=False)

print("Linear Evaluation Fitting Model")
# Step 4: Linear evaluation
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Linear Evaluation Predicting")
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {accuracy}")

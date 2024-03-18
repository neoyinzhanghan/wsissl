import os
import numpy as np
import xgboost as xgb
import pandas as pd
import random
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
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
    "/media/hdd1/neo/SC_pancreas_LUAD/LUAD",
    "/media/hdd1/neo/SC_pancreas_LUAD/pancreas",
]  # Replace with your actual paths

start_time = time.time()

print("Preparing data...")
slide_paths, labels = split_slides(root_paths)
X, y = load_regions(slide_paths, labels, max_num_patches_per_slide=10)

X_train, X_val, X_test = X["train"], X["val"], X["test"]
y_train, y_val, y_test = y["train"], y["val"], y["test"]

runtime_data["Patches Preparation"] = time.time() - start_time


start_time = time.time()

# Now you have X['train'], X['val'], X['test'] and y['train'], y['val'], y['test']

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define hyperparameter search space
param_dist = {
    "eta": uniform(0.01, 0.3),  # learning rate
    "max_depth": randint(3, 10),  # depth of tree
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.8, 0.2),  # percentage of samples used per tree
    "colsample_bytree": uniform(0.8, 0.2),  # percentage of features used per tree
    # "num_class": len(np.unique(y_train)),
}

runtime_data["Setup Time"] = time.time() - start_time


start_time = time.time()

print("Starting hyperparameter tuning...")
# Initialize XGBClassifier
if len(np.unique(y_train)) > 2:
    xgb_model = xgb.XGBClassifier(objective="multi:softmax", eval_metric="mlogloss")
elif len(np.unique(y_train)) == 2:
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="mlogloss")
else:
    raise ValueError("Number of classes should be greater than 1")

# Randomized Search
n_iter_search = 20
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring="accuracy",
    cv=3,
)

print(f"Searching through {n_iter_search} different random combinations...")
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

runtime_data["Hyperparameter Tuning"] = time.time() - start_time

start_time = time.time()

# Predictions and evaluation on validation set
print("Evaluating model...")
preds = best_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, preds)}")

runtime_data["Validation Evaluation"] = time.time() - start_time

# Optionally: save your model
# best_model.save_model('best_xgb_model.json')

# Assuming random_search is your RandomizedSearchCV object and the search has been completed


start_time = time.time()

print("Saving hyperparameter search results...")
# Extract the results into a DataFrame
results = pd.DataFrame(random_search.cv_results_)

# Select only the columns you're interested in
# For instance, you might want to keep the parameter columns, the mean test score, std test score, etc.
# The column names would depend on the version of scikit-learn but typically include:
# 'param_eta', 'param_max_depth', 'param_min_child_weight', 'param_subsample', 'param_colsample_bytree',
# 'mean_test_score', 'std_test_score', and 'rank_test_score'.
interesting_columns = [
    "param_eta",
    "param_max_depth",
    "param_min_child_weight",
    "param_subsample",
    "param_colsample_bytree",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]
filtered_results = results.loc[:, interesting_columns]

# Save the filtered results to a CSV file
filtered_results.to_csv("hyperparameter_search_results.csv", index=False)

print("Hyperparameter search results saved to 'hyperparameter_search_results.csv'.")

runtime_data["Result Saving"] = time.time() - start_time

# save the runtime data as a csv file in the current directory
runtime_df = pd.DataFrame(runtime_data, index=[0])

runtime_df.to_csv("runtime_data.csv", index=False)
print("Runtime data saved to 'runtime_data.csv'.")

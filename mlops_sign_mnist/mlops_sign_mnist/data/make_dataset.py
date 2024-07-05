import os
import subprocess
import zipfile

import pandas as pd
import torch

# NOTE: Kaggle API must be installed

subprocess.run(["kaggle", "datasets", "download", "-d", "datamunge/sign-language-mnist"])

zip_file = "sign-language-mnist.zip"
extract_path = "data/raw/"
processed_path = "data/processed/"
# Ensure the extraction directory exists
os.makedirs(extract_path, exist_ok=True)
os.makedirs(processed_path, exist_ok=True)

# Open the ZIP file
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    # Extract all the contents
    zip_ref.extractall(extract_path)
# delete zipfile after extracting
os.remove(zip_file)

# transform data
train_df = pd.read_csv("data/raw/sign_mnist_train.csv")
test_df = pd.read_csv("data/raw/sign_mnist_test.csv")

X_train = train_df.drop(columns=["label"]).values.reshape(-1, 1, 28, 28).astype("float32") / 255.0
labels_train = train_df["label"].values
X_test = test_df.drop(columns=["label"]).values.reshape(-1, 1, 28, 28).astype("float32") / 255.0
labels_test = test_df["label"].values

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# save tensors
torch.save(X_train, "data/processed/X_train.pt")
torch.save(labels_train, "data/processed/labels_train.pt")
torch.save(X_test, "data/processed/X_test.pt")
torch.save(labels_test, "data/processed/labels_test.pt")

print("Data saved successfully!")

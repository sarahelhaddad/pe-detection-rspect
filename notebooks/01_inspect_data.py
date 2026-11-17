# Run this in a Kaggle notebook with the RSNA PE competition data attached.
import os
import pandas as pd

# Kaggle mounts competition data in one of these places depending on the session.
CANDIDATES = [
    "/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection",
    "/kaggle/input/rsna-str-pulmonary-embolism-detection",
]
ROOT = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if ROOT is None:
    raise SystemExit(f"Data folder not found. Tried: {CANDIDATES}")
print("Using data folder:", ROOT)
print("Top-level entries:", sorted(os.listdir(ROOT)))

train_csv = f"{ROOT}/train.csv"
print(f"train.csv size: {os.path.getsize(train_csv) / 1e6:.0f} MB")

train = pd.read_csv(train_csv)
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())
print(train.head())

# One row per image, so study-level columns repeat for every image of a study.
# Check the exact column names printed above before relying on these.
if "StudyInstanceUID" in train.columns:
    print("Number of studies:", train["StudyInstanceUID"].nunique())
if "pe_present_on_image" in train.columns:
    print("Image-level PE rate:", train["pe_present_on_image"].mean())
if "negative_exam_for_pe" in train.columns and "StudyInstanceUID" in train.columns:
    study = train.drop_duplicates("StudyInstanceUID")
    print("Study-level negative_exam_for_pe rate:", study["negative_exam_for_pe"].mean())

# Count study folders without listing every image file.
train_dir = f"{ROOT}/train"
if os.path.isdir(train_dir):
    print("Study folders in train/:", len(os.listdir(train_dir)))

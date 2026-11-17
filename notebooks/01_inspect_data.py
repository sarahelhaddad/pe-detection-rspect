# Run this in a Kaggle notebook with the RSNA PE competition data attached.
import os
import pandas as pd

ROOT = "/kaggle/input/rsna-str-pulmonary-embolism-detection"

print("Top-level files:", os.listdir(ROOT))

train = pd.read_csv(f"{ROOT}/train.csv")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())
print(train.head())

# Label overview: one row per image, so study-level columns repeat.
# Check the exact column names printed above before relying on these.
if "StudyInstanceUID" in train.columns:
    print("Number of studies:", train["StudyInstanceUID"].nunique())
if "pe_present_on_image" in train.columns:
    print("Image-level PE rate:", train["pe_present_on_image"].mean())
if "negative_exam_for_pe" in train.columns and "StudyInstanceUID" in train.columns:
    study = train.drop_duplicates("StudyInstanceUID")
    print("Study-level negative_exam_for_pe rate:", study["negative_exam_for_pe"].mean())

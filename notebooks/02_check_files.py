# Check how many image files are actually present on disk for each study.
# Run in a Kaggle notebook after 01_inspect_data.py.
import os
import time
import pandas as pd

ROOT = "/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection"
TRAIN_DIR = f"{ROOT}/train"
OUT = "/kaggle/working/study_file_counts.csv"

labels = pd.read_csv(f"{ROOT}/train.csv", usecols=["StudyInstanceUID", "SOPInstanceUID"])
expected = labels.groupby("StudyInstanceUID").size().rename("expected_images")
print("Studies in train.csv:", len(expected), "| images listed:", int(expected.sum()))

rows, t0 = [], time.time()
studies = sorted(os.listdir(TRAIN_DIR))
for i, study in enumerate(studies):
    n = 0
    sdir = os.path.join(TRAIN_DIR, study)
    for series in os.listdir(sdir):
        n += sum(1 for f in os.listdir(os.path.join(sdir, series)) if f.endswith(".dcm"))
    rows.append((study, n))
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(studies)} studies scanned ({time.time() - t0:.0f}s)", flush=True)

found = pd.DataFrame(rows, columns=["StudyInstanceUID", "found_images"]).set_index("StudyInstanceUID")
df = expected.to_frame().join(found, how="left").fillna({"found_images": 0})
df["found_images"] = df["found_images"].astype(int)
df["complete"] = df["found_images"] >= df["expected_images"]
df.to_csv(OUT)

print("\nDICOM files found on disk:", int(df["found_images"].sum()))
print("Studies with all images present:", int(df["complete"].sum()), "of", len(df))
print("Studies with no images:", int((df["found_images"] == 0).sum()))
print(df["found_images"].describe())
print("Saved per-study counts to", OUT)

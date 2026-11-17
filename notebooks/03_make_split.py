# Build the study-level label table and the train / validation / test split.
# Run in a Kaggle notebook. Writes small text files to /kaggle/working/splits/.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = "/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection"
OUT_DIR = "/kaggle/working/splits"
SEED = 42
TEST_FRAC, VAL_FRAC = 0.15, 0.15
os.makedirs(OUT_DIR, exist_ok=True)

cols = ["StudyInstanceUID", "negative_exam_for_pe", "indeterminate",
        "chronic_pe", "acute_and_chronic_pe"]
df = pd.read_csv(f"{ROOT}/train.csv", usecols=cols)

# Study-level labels should be identical on every row of a study. Verify, then collapse.
varying = df.groupby("StudyInstanceUID")[cols[1:]].nunique().max()
print("Max distinct values per study for each label column (should all be 1):")
print(varying.to_string())
studies = df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")

# Label definition (deviation log: indeterminate and chronic-only studies are excluded).
neg = studies["negative_exam_for_pe"] == 1
indet = studies["indeterminate"] == 1
chronic_only = (studies["chronic_pe"] == 1) & (studies["acute_and_chronic_pe"] == 0)
pos = (studies["negative_exam_for_pe"] == 0) & ~indet & ~chronic_only

print("\nStudy counts before exclusions:", len(studies))
print("  negative:", int(neg.sum()))
print("  indeterminate (excluded):", int(indet.sum()))
print("  chronic-only (excluded):", int(chronic_only.sum()))
print("  PE-positive (acute, or acute + chronic):", int(pos.sum()))

keep = neg | pos
lab = pos[keep].astype(int).rename("pe")
print("\nUsed for modelling:", len(lab), "| PE prevalence:", round(lab.mean(), 4))

ids = lab.index.to_numpy()
trainval_ids, test_ids = train_test_split(ids, test_size=TEST_FRAC, stratify=lab.values, random_state=SEED)
val_rel = VAL_FRAC / (1 - TEST_FRAC)
train_ids, val_ids = train_test_split(
    trainval_ids, test_size=val_rel, stratify=lab.loc[trainval_ids].values, random_state=SEED)

for name, part in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    pd.Series(sorted(part)).to_csv(f"{OUT_DIR}/{name}_studies.txt", index=False, header=False)
    print(f"{name}: {len(part)} studies | PE prevalence {lab.loc[part].mean():.4f}")

lab.to_csv(f"{OUT_DIR}/study_labels.csv")
assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) and not (set(val_ids) & set(test_ids))
print("\nNo study appears in more than one split. Files saved in", OUT_DIR)

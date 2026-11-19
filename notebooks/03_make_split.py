# Build the study-level label table and the train / validation / test split.
# Run in a Kaggle notebook. Writes small text files to /kaggle/working/splits/.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def find_root():
    """Locate the competition folder (Kaggle mounts it at different paths in different sessions)."""
    env = os.environ.get("RSPECT_ROOT")
    if env:
        return env
    base = "/kaggle/input"
    for p in ("/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection",
              "/kaggle/input/rsna-str-pulmonary-embolism-detection"):
        if os.path.isfile(f"{p}/train.csv"):
            return p
    for d1 in sorted(os.listdir(base)) if os.path.isdir(base) else []:      # search up to 3 levels
        for p in (f"{base}/{d1}", *[f"{base}/{d1}/{d2}" for d2 in (sorted(os.listdir(f"{base}/{d1}")) if os.path.isdir(f"{base}/{d1}") else [])]):
            if os.path.isfile(f"{p}/train.csv"):
                return p
    raise SystemExit("Could not find train.csv under /kaggle/input. Check that the competition is attached "
                     "(Input panel) and tell Claude what `os.listdir('/kaggle/input')` shows.")

ROOT = find_root()
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

# Label definition, matching Kim et al. (2,211 positive / 4,911 negative studies):
#   negative = negative_exam_for_pe == 1
#   positive = negative_exam_for_pe == 0 and not indeterminate
#              (this INCLUDES chronic-only studies)
# Deviation log: indeterminate studies are excluded.
neg = studies["negative_exam_for_pe"] == 1
indet = studies["indeterminate"] == 1
chronic_only = (studies["chronic_pe"] == 1) & (studies["acute_and_chronic_pe"] == 0)
pos = (studies["negative_exam_for_pe"] == 0) & ~indet

print("\nStudy counts before exclusions:", len(studies))
print("  negative:", int(neg.sum()))
print("  indeterminate (excluded):", int(indet.sum()))
print("  PE-positive:", int(pos.sum()), f"(of which chronic-only: {int((pos & chronic_only).sum())})")

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

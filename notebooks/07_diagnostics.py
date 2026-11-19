"""Diagnostics: how much does keeping only ~10% of the slices limit the study-level result?

For each split it reports
  * slice-level AUC of the CNN,
  * the share of PE-positive studies in which at least one KEPT slice is PE-positive
    (a study whose PE slices were all skipped cannot be detected from the kept slices),
  * an 'oracle' study-level AUC that uses the TRUE slice labels (score = number of kept PE slices):
    the best any slice-based method could do with these kept slices.
This script does not train or tune anything. Use the validation numbers to decide about any change;
the test numbers are for reporting only.

Usage:  !python pe-detection-rspect/notebooks/07_diagnostics.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SPLITS = os.environ.get("RSPECT_SPLITS", "/kaggle/working/splits")


def find_preds():
    env = os.environ.get("RSPECT_PREDS")
    if env:
        return env
    for pat in ["/kaggle/working/preds", "/kaggle/input/preds", "/kaggle/input/*/preds",
                "/kaggle/input/*/*/preds", "/kaggle/input/*/*/*/preds"]:
        for p in sorted(glob.glob(pat)):
            if os.path.isfile(f"{p}/val_slice_probs.npz"):
                return p
    raise SystemExit("Could not find preds/. Attach the training notebook output as an input.")


preds = find_preds()
print("Using predictions from:", preds)
labels = pd.read_csv(f"{SPLITS}/study_labels.csv", index_col=0)["pe"]
for name in ("val", "test"):
    d = np.load(f"{preds}/{name}_slice_probs.npz")
    df = pd.DataFrame({"study": d["study"], "y": d["y"], "p": d["prob"]})
    g = df.groupby("study", sort=False)
    st = pd.DataFrame({"pos_slices": g["y"].sum(), "n_slices": g["y"].size(), "max_p": g["p"].max()})
    st["label"] = labels.loc[st.index].to_numpy()
    pos = st[st.label == 1]
    print(f"\n=== {name} ({len(st)} studies, {int(st.label.sum())} PE-positive) ===")
    print(f"Slice-level AUC: {roc_auc_score(df.y, df.p):.4f}  | PE slices: {int(df.y.sum())} of {len(df)} ({df.y.mean():.3%})")
    print(f"PE-positive studies with at least one kept PE slice: {(pos.pos_slices > 0).mean():.1%}")
    print("Kept PE slices per PE-positive study (median / mean):", pos.pos_slices.median(), round(pos.pos_slices.mean(), 2))
    print(f"Oracle study AUC (true slice labels, kept slices only): {roc_auc_score(st.label, st.pos_slices):.4f}")
    print(f"CNN study AUC (max slice probability): {roc_auc_score(st.label, st.max_p):.4f}")
    detect = st[(st.label == 0) | (st.pos_slices > 0)]
    print(f"CNN study AUC on studies whose kept slices show the PE: {roc_auc_score(detect.label, detect.max_p):.4f}")

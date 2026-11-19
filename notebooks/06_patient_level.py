"""Step 4-5 of the Kim et al. style pipeline: patient-level Random Forest and final evaluation.

Input : slice probabilities from notebook 05 (preds/{train,val,test}_slice_probs.npz).
Method: for every study, resample its slice probabilities (in body order) to a fixed length
        of 200 (as Kim et al.), then a Random Forest predicts PE vs no PE for the study.

The paper does not say which patients the forest was trained on. Default here (--fit-on val):
the forest is trained on the VALIDATION studies, which the CNN never saw, and the operating
threshold (Youden's J) comes from 5-fold out-of-fold predictions on those same studies.
The TEST studies are then scored once with the final forest and that threshold.
(--fit-on train trains the forest on the training studies instead; their CNN outputs are
over-confident because the CNN was fitted to them, so treat that variant as a comparison only.)

Run this as the FINAL evaluation: do not tune anything after looking at the test results.

Usage:  !python pe-detection-rspect/notebooks/06_patient_level.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

SPLITS = os.environ.get("RSPECT_SPLITS", "/kaggle/working/splits")
OUT = os.environ.get("RSPECT_OUT", "/kaggle/working")
LEN = 200


def find_preds():
    env = os.environ.get("RSPECT_PREDS")
    if env:
        return env
    for pat in ["/kaggle/working/preds", "/kaggle/input/preds", "/kaggle/input/*/preds",
                "/kaggle/input/*/*/preds", "/kaggle/input/*/*/*/preds"]:
        for p in sorted(glob.glob(pat)):
            if os.path.isfile(f"{p}/val_slice_probs.npz"):
                return p
    raise SystemExit("Could not find preds/ with *_slice_probs.npz. Attach the output of the training "
                     "notebook (Add Input -> Notebook Output) and tell Claude what the Input panel shows.")


def study_vectors(path):
    """-> (study ids, matrix [n_studies x LEN] of resampled slice probabilities)"""
    d = np.load(path, allow_pickle=False)
    prob, study = d["prob"], d["study"]
    ids, vecs, start = [], [], 0
    order_change = np.flatnonzero(study[1:] != study[:-1]) + 1
    for seg in np.split(np.arange(len(study)), order_change):
        p = prob[seg]                                   # already in body order (notebook 04 sorted them)
        x = np.linspace(0, len(p) - 1, LEN)
        vecs.append(np.interp(x, np.arange(len(p)), p) if len(p) > 1 else np.full(LEN, p[0]))
        ids.append(study[seg[0]])
    return np.array(ids), np.vstack(vecs)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def hanley_ci(auc, n_pos, n_neg, z=1.96):
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-on", choices=["val", "train"], default="val")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    preds = find_preds()
    print("Using predictions from:", preds)
    labels = pd.read_csv(f"{SPLITS}/study_labels.csv", index_col=0)["pe"]
    data = {}
    for n in ("train", "val", "test"):
        ids, X = study_vectors(f"{preds}/{n}_slice_probs.npz")
        data[n] = (ids, X, labels.loc[ids].to_numpy())
        print(f"{n}: {len(ids)} studies | PE prevalence {data[n][2].mean():.4f}")

    rf = lambda: RandomForestClassifier(n_estimators=a.trees, random_state=a.seed, n_jobs=-1)
    _, Xva, yva = data["val"]
    _, Xte, yte = data["test"]

    if a.fit_on == "val":
        oof = np.zeros(len(yva))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=a.seed).split(Xva, yva):
            oof[te] = rf().fit(Xva[tr], yva[tr]).predict_proba(Xva[te])[:, 1]
        thr = youden_threshold(yva, oof)
        print(f"Validation out-of-fold AUC: {roc_auc_score(yva, oof):.4f} | Youden threshold: {thr:.4f}")
        final = rf().fit(Xva, yva)
    else:
        _, Xtr, ytr = data["train"]
        final = rf().fit(Xtr, ytr)
        pv = final.predict_proba(Xva)[:, 1]
        thr = youden_threshold(yva, pv)
        print(f"Validation AUC: {roc_auc_score(yva, pv):.4f} | Youden threshold: {thr:.4f}")

    pt = final.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pt)
    n_pos, n_neg = int(yte.sum()), int((1 - yte).sum())
    lo, hi = hanley_ci(auc, n_pos, n_neg)
    pred = (pt >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    res = {
        "fit_on": a.fit_on, "threshold": thr, "n_test": int(len(yte)), "test_pe_prevalence": float(yte.mean()),
        "auc": float(auc), "auc_ci95_hanley": [float(lo), float(hi)],
        "sensitivity": [tp / (tp + fn), *wilson(tp, tp + fn)],
        "specificity": [tn / (tn + fp), *wilson(tn, tn + fp)],
        "ppv": [tp / (tp + fp) if tp + fp else float("nan"), *wilson(tp, tp + fp)],
        "npv": [tn / (tn + fn) if tn + fn else float("nan"), *wilson(tn, tn + fn)],
        "accuracy": [(tp + tn) / len(yte), *wilson(tp + tn, len(yte))],
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }
    # Simple baselines for context: study score = max / mean of the slice probabilities
    res["baseline_auc_max_slice"] = float(roc_auc_score(yte, Xte.max(axis=1)))
    res["baseline_auc_mean_slice"] = float(roc_auc_score(yte, Xte.mean(axis=1)))

    print("\n=== TEST RESULTS (final evaluation) ===")
    print(f"Test studies: {len(yte)} ({n_pos} PE-positive, {n_neg} negative)")
    print(f"AUC {auc:.4f} (95% CI {lo:.4f}-{hi:.4f}, Hanley)")
    for k in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
        v = res[k]
        print(f"{k:12s} {v[0]:.4f} (95% CI {v[1]:.4f}-{v[2]:.4f}, Wilson)")
    print("Confusion matrix:", res["confusion"])
    print(f"Baselines (AUC): max slice {res['baseline_auc_max_slice']:.4f} | mean slice {res['baseline_auc_mean_slice']:.4f}")

    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump(res, open(f"{OUT}/results/test_metrics_{a.fit_on}.json", "w"), indent=1)
    pd.DataFrame({"study": data["test"][0], "y": yte, "prob": pt, "pred": pred}).to_csv(
        f"{OUT}/results/test_predictions_{a.fit_on}.csv", index=False)
    print(f"Saved results to {OUT}/results/")


if __name__ == "__main__":
    main()

"""Step 1 of the Kim et al. style 2D pipeline: preprocess CT slices and cache them.

For every study in the split files this script
  1. sorts the study's slices by position along the body axis,
  2. keeps every STEP-th slice (STEP=10 -> about 10% of the slices, as in Kim et al.),
  3. converts pixels to Hounsfield units, applies the lung window (centre 40, width 400),
     scales to 0-1 (stored as uint8 0-255 to save space), and resizes to 224 x 224,
  4. saves one small file per study: /kaggle/working/cache/<StudyInstanceUID>.npz

It is resumable: studies that already have a cache file are skipped.

Usage (in a Kaggle notebook):
    !python pe-detection-rspect/notebooks/04_preprocess.py --limit 20     # quick test + timing
    !python pe-detection-rspect/notebooks/04_preprocess.py                # everything
"""
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import pydicom

ROOT = os.environ.get("RSPECT_ROOT", "/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection")
SPLITS = os.environ.get("RSPECT_SPLITS", "/kaggle/working/splits")
CACHE = os.environ.get("RSPECT_CACHE", "/kaggle/working/cache")
STEP = 10                 # keep every 10th slice (about 10% of each study)
SIZE = 224                # ConvNeXt input size used by Kim et al.
WL, WW = 40, 400          # window centre / width used by Kim et al.
LO, HI = WL - WW / 2, WL + WW / 2


def slice_position(path):
    """Position of a slice along the body axis (falls back to InstanceNumber)."""
    ds = pydicom.dcmread(path, stop_before_pixels=True,
                         specific_tags=["ImagePositionPatient", "InstanceNumber"])
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0))


def load_slice(path):
    """Read one DICOM slice and return a SIZE x SIZE uint8 image (windowed HU)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = arr * slope + intercept
    x = (np.clip(hu, LO, HI) - LO) / WW                      # 0..1
    x = cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.round(x * 255).astype(np.uint8)


def process_study(args):
    study, slices = args                                    # slices: list of (series, sop, label)
    out = os.path.join(CACHE, f"{study}.npz")
    if os.path.exists(out):
        return study, "skipped", 0, 0
    try:
        # Use the series with the most slices if a study has several.
        series_counts = pd.Series([s for s, _, _ in slices]).value_counts()
        main_series = series_counts.index[0]
        items = [(sop, lab) for s, sop, lab in slices if s == main_series]
        paths = [os.path.join(ROOT, "train", study, main_series, f"{sop}.dcm") for sop, _ in items]
        z = np.array([slice_position(p) for p in paths])
        order = np.argsort(z, kind="stable")
        pick = order[STEP // 2::STEP]                       # evenly spaced, in body order
        imgs, labels, zs, fails = [], [], [], 0
        for i in pick:
            try:
                imgs.append(load_slice(paths[i]))
                labels.append(items[i][1])
                zs.append(z[i])
            except Exception:
                fails += 1
        if not imgs:
            return study, "no_images", 0, fails
        tmp = out + ".tmp.npz"
        np.savez(tmp, x=np.stack(imgs), y=np.array(labels, dtype=np.uint8),
                 z=np.array(zs, dtype=np.float32), n_total=len(items))
        os.replace(tmp, out)
        return study, "ok" if len(set(series_counts.index)) == 1 else "ok_multi_series", len(imgs), fails
    except Exception as e:                                  # keep going, report at the end
        return study, f"error: {type(e).__name__}: {e}", 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process this many random studies (test run)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    lab = pd.read_csv(f"{SPLITS}/study_labels.csv", index_col=0)["pe"]
    study_ids = list(lab.index)
    if a.limit:
        rng = np.random.default_rng(0)
        study_ids = list(rng.choice(study_ids, size=min(a.limit, len(study_ids)), replace=False))
    wanted = set(study_ids)

    df = pd.read_csv(f"{ROOT}/train.csv",
                     usecols=["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "pe_present_on_image"])
    df = df[df["StudyInstanceUID"].isin(wanted)]
    jobs = [(s, list(zip(g["SeriesInstanceUID"], g["SOPInstanceUID"], g["pe_present_on_image"].astype(int))))
            for s, g in df.groupby("StudyInstanceUID")]
    print(f"Studies to process: {len(jobs)} | workers: {a.workers} | keep every {STEP}th slice")

    t0, stats, n_img, n_fail = time.time(), {}, 0, 0
    with Pool(a.workers) as pool:
        for i, (study, status, n, fails) in enumerate(pool.imap_unordered(process_study, jobs), 1):
            stats[status] = stats.get(status, 0) + 1
            n_img += n
            n_fail += fails
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} studies | {el:.0f}s elapsed | ~{el / i * (len(jobs) - i) / 60:.0f} min left", flush=True)

    print("\nStatus counts:", stats)
    print("Images cached this run:", n_img, "| slices that failed to decode:", n_fail)
    size = sum(os.path.getsize(os.path.join(CACHE, f)) for f in os.listdir(CACHE)) / 1e9
    print(f"Cache folder size: {size:.2f} GB | total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

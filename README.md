# pe-detection-rspect

PE detection on the public RSNA-STR Pulmonary Embolism CT dataset (RSPECT).

Dataset: https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection
Citation: Colak E, et al. The RSNA Pulmonary Embolism CT Dataset. Radiology: Artificial Intelligence 2021;3(2). doi:10.1148/ryai.2021200254

## Goal

Re-implement and evaluate published PE classification models on RSPECT with our own
train/validation/test split. Two-stage pipelines skip the first (synthetic image) stage.

Planned models:
1. 2D CNN baseline (slice-level classification)
2. 2D CNN with slice aggregation to a study-level prediction (Kim et al. style)
3. 3D CNN on slice windows (PENet style)

## Where things run

- Code lives in this repository (code only, small files).
- The data stays on Kaggle, under `/kaggle/input/rsna-str-pulmonary-embolism-detection/`.
- Never commit data, model weights, API tokens or passwords.

## Rules for fair evaluation

- Split at the study level, never at the slice level.
- Fixed random seed; save the split files and reuse them for every model.
- Choose thresholds on the validation set only; use the test set once at the end.
- Report AUROC, sensitivity, specificity, PPV, NPV with 95% confidence intervals.
- Log every deviation from the original papers.

## Layout

```
notebooks/   Kaggle notebook scripts, run in order (01, 02, ...)
src/         Reusable code (datasets, models, metrics)
splits/      Study-ID lists for train / val / test
```

## Status

- [x] 01 Inspect data and labels
- [x] 02 Define labels and study-level split (Kim et al. label rule: 7,122 studies)
- [x] 03 Preprocessing and caching (10% of slices, 175,071 images, 7,122 studies)
- [x] 04 Slice classifier (ConvNeXt-Tiny) - notebook 05
- [ ] 05 Patient-level Random Forest + test evaluation - notebook 06
- [ ] 06 PENet-style 3D model
- [ ] 07 Final test evaluation

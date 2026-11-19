"""Step 2-3 of the Kim et al. style pipeline: train the ConvNeXt slice classifier.

* ImageNet-pretrained ConvNeXt, one sigmoid output (BCE-with-logits), Adam, lr 1e-4,
  224 x 224 inputs (the grey slice is repeated into 3 channels).
* Trains on the cached slices of the TRAIN studies, picks the best epoch by slice-level
  AUC on the VALIDATION studies. The TEST studies are never used for model selection.
* At the end it saves the predicted probability of every cached slice (train / val / test),
  ordered along the body axis, for the patient-level Random Forest (notebook 06).

Usage (Kaggle notebook with a GPU, the cache from notebook 04 attached as an input):
    !python pe-detection-rspect/notebooks/05_train_convnext.py --limit-studies 40 --epochs 1   # quick test
    !python pe-detection-rspect/notebooks/05_train_convnext.py                                  # full run
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score

SPLITS = os.environ.get("RSPECT_SPLITS", "/kaggle/working/splits")
OUT = os.environ.get("RSPECT_OUT", "/kaggle/working")


def find_cache():
    """Find the folder of per-study .npz files (output of notebook 04) without walking the huge data folders."""
    env = os.environ.get("RSPECT_CACHE")
    if env:
        return env
    patterns = ["/kaggle/working/cache", "/kaggle/input/cache", "/kaggle/input/*/cache",
                "/kaggle/input/*/*/cache", "/kaggle/input/*/*/*/cache"]
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if os.path.isdir(p) and any(f.endswith(".npz") for f in os.listdir(p)[:50]):
                return p
    raise SystemExit("Could not find the cache folder with .npz files. Attach the output of the preprocessing "
                     "notebook (Add Input -> Notebook Output) and tell Claude what the Input panel shows.")


def load_split(cache, name, limit=None):
    ids = [l.strip() for l in open(f"{SPLITS}/{name}_studies.txt") if l.strip()]
    if limit:
        ids = ids[:limit]
    xs, ys, sids = [], [], []
    for s in ids:
        d = np.load(f"{cache}/{s}.npz")
        xs.append(d["x"])
        ys.append(d["y"])
        sids += [s] * len(d["y"])
    X = np.concatenate(xs)
    del xs
    return X, np.concatenate(ys), np.array(sids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=None, help="optional BCE weight for PE-positive slices")
    ap.add_argument("--limit-studies", type=int, default=None, help="use only the first N studies of each split (test run)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline tests only)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = find_cache()
    print(f"Device: {dev} | cache: {cache}")
    os.makedirs(f"{OUT}/ckpt", exist_ok=True)
    os.makedirs(f"{OUT}/preds", exist_ok=True)

    t0 = time.time()
    data = {n: load_split(cache, n, a.limit_studies) for n in ("train", "val", "test")}
    for n, (X, y, s) in data.items():
        print(f"{n}: {len(y)} slices from {len(set(s))} studies | slice PE rate {y.mean():.4f}")
    print(f"Loaded in {time.time() - t0:.0f}s")

    model = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=1).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    pw = torch.tensor([a.pos_weight], device=dev) if a.pos_weight else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def prep(xb):
        x = torch.from_numpy(xb).to(dev).float().div_(255).unsqueeze(1).repeat(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def predict(X, bs=128):
        model.eval()
        out = []
        for i in range(0, len(X), bs):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                out.append(torch.sigmoid(model(prep(X[i:i + bs])).float()).squeeze(1).cpu().numpy())
        return np.concatenate(out)

    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    best_auc, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.default_rng(a.seed + ep).permutation(len(Xtr))
        run_loss, n_b, te = 0.0, 0, time.time()
        for i in range(0, len(order), a.batch_size):
            idx = np.sort(order[i:i + a.batch_size])
            xb, yb = prep(Xtr[idx]), torch.from_numpy(ytr[idx]).float().to(dev).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb).float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            n_b += 1
            if n_b % 200 == 0:
                print(f"  epoch {ep} batch {n_b}/{int(np.ceil(len(order) / a.batch_size))} loss {run_loss / n_b:.4f}", flush=True)
        pv = predict(Xva)
        auc = roc_auc_score(yva, pv) if len(set(yva)) > 1 else float("nan")
        history.append({"epoch": ep, "train_loss": run_loss / max(n_b, 1), "val_slice_auc": auc, "seconds": time.time() - te})
        print(f"Epoch {ep}: train loss {run_loss / max(n_b, 1):.4f} | validation slice AUC {auc:.4f} | {time.time() - te:.0f}s", flush=True)
        if auc > best_auc or ep == 1:
            best_auc = auc
            torch.save(model.state_dict(), f"{OUT}/ckpt/best.pt")
            print("  saved best checkpoint", flush=True)

    model.load_state_dict(torch.load(f"{OUT}/ckpt/best.pt", map_location=dev))
    for name, (X, y, s) in data.items():
        p = predict(X)
        np.savez(f"{OUT}/preds/{name}_slice_probs.npz", prob=p, y=y, study=s)
        print(f"Saved probabilities for {len(p)} {name} slices")
    json.dump({"args": vars(a), "history": history, "best_val_slice_auc": best_auc},
              open(f"{OUT}/preds/train_log.json", "w"), indent=1)
    print(f"Done. Best validation slice AUC: {best_auc:.4f} | total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

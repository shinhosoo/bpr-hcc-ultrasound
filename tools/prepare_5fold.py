#!/usr/bin/env python3
"""
Prepare Benign/Malignant images as a stratified 5-fold CV layout.

For each fold i in [0..k-1]:
    fold i's test  = items in stratum i
    fold i's train+val = rest, with inner-val-ratio held out as val

Output structure:
    <out>/fold_<i>/imagefolder/{train,val,test}/{Benign,Malignant}/
    <out>/fold_<i>/pkl/lesion_{train,val,test}.pkl
"""
import argparse, os, pickle, random, shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {"Benign": 0, "Malignant": 1}


def collect(src):
    samples = []
    for cn, lbl in CLASS_TO_LABEL.items():
        d = src / cn
        if not d.is_dir(): raise FileNotFoundError(d)
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((p, cn, lbl))
    return samples


def kfold_stratified(samples, k, seed):
    """Return list of folds: folds[i] = list of samples in stratum i (test for fold i)."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for cn in CLASS_TO_LABEL:
        items = [s for s in samples if s[1] == cn]
        rng.shuffle(items)
        for idx, item in enumerate(items):
            folds[idx % k].append(item)
    return folds


def inner_val_split(train_val_items, val_ratio, seed):
    """Stratified val carve-out from train_val pool."""
    rng = random.Random(seed)
    tr, va = [], []
    for cn in CLASS_TO_LABEL:
        items = [s for s in train_val_items if s[1] == cn]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        va.extend(items[:n_val])
        tr.extend(items[n_val:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


def write_split(samples, split_name, out_root):
    sd = out_root / "imagefolder" / split_name
    for cn in CLASS_TO_LABEL: (sd / cn).mkdir(parents=True, exist_ok=True)
    recs = []
    for src, cn, lbl in samples:
        dst = sd / cn / src.name
        shutil.copy2(src, dst)
        recs.append({"img_root": str(dst.resolve()), "label": lbl})
    return recs


def write_pkl(recs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f: pickle.dump(recs, f)


def counts(samples):
    return {cn: sum(1 for s in samples if s[1] == cn) for cn in CLASS_TO_LABEL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="image")
    ap.add_argument("--out", default="prepared_binary_lesion_5fold")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--inner-val-ratio", type=float, default=0.15,
                    help="proportion of train+val pool to hold out for validation")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = Path(a.source).resolve()
    out = Path(a.out).resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = collect(src)
    folds = kfold_stratified(samples, a.k, a.seed)

    summary_lines = [
        f"5-fold stratified CV (k={a.k}, inner_val_ratio={a.inner_val_ratio}, seed={a.seed})",
        f"Total samples: {len(samples)}  {counts(samples)}",
        "",
    ]
    for i in range(a.k):
        fold_dir = out / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test = folds[i]
        train_val = [s for j, f in enumerate(folds) if j != i for s in f]
        train, val = inner_val_split(train_val, a.inner_val_ratio, a.seed + i)

        tr_r = write_split(train, "train", fold_dir)
        va_r = write_split(val,   "val",   fold_dir)
        te_r = write_split(test,  "test",  fold_dir)
        write_pkl(tr_r, fold_dir / "pkl" / "lesion_train.pkl")
        write_pkl(va_r, fold_dir / "pkl" / "lesion_val.pkl")
        write_pkl(te_r, fold_dir / "pkl" / "lesion_test.pkl")

        line = (f"fold_{i}: train={len(train)} ({counts(train)})  "
                f"val={len(val)} ({counts(val)})  test={len(test)} ({counts(test)})")
        summary_lines.append(line)
        print(line)

    (out / "README.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()

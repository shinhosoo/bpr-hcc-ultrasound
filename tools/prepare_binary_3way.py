#!/usr/bin/env python3
"""
Prepare Benign/Malignant images as a 3-way split: train / val / test.

train  : training the model
val    : early stopping + hyperparameter tuning
test   : held out, evaluated once at the very end

Output structure:
    <out>/imagefolder/{train,val,test}/{Benign,Malignant}/
    <out>/pkl/lesion_{train,val,test}.pkl
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


def split3(samples, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for cn in CLASS_TO_LABEL:
        items = [s for s in samples if s[1] == cn]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        te.extend(items[:n_test])
        va.extend(items[n_test:n_test + n_val])
        tr.extend(items[n_test + n_val:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def copy_to_imagefolder(samples, split_name, out):
    sd = out / "imagefolder" / split_name
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="image")
    ap.add_argument("--out", default="prepared_binary_lesion_3way")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not 0 < a.val_ratio < 1 or not 0 < a.test_ratio < 1:
        raise SystemExit("ratios must be in (0,1)")
    if a.val_ratio + a.test_ratio >= 1.0:
        raise SystemExit("val_ratio + test_ratio must be < 1")

    src = Path(a.source).resolve()
    out = Path(a.out).resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)

    samples = collect(src)
    tr, va, te = split3(samples, a.val_ratio, a.test_ratio, a.seed)

    tr_r = copy_to_imagefolder(tr, "train", out)
    va_r = copy_to_imagefolder(va, "val",   out)
    te_r = copy_to_imagefolder(te, "test",  out)
    write_pkl(tr_r, out / "pkl" / "lesion_train.pkl")
    write_pkl(va_r, out / "pkl" / "lesion_val.pkl")
    write_pkl(te_r, out / "pkl" / "lesion_test.pkl")

    def counts(rs): return {cn: sum(1 for r in rs if r["label"] == CLASS_TO_LABEL[cn]) for cn in CLASS_TO_LABEL}
    summary = [
        "Binary lesion 3-way split",
        "",
        f"Train: {len(tr_r)}  Val: {len(va_r)}  Test: {len(te_r)}",
        f"Train counts: {counts(tr_r)}",
        f"Val counts:   {counts(va_r)}",
        f"Test counts:  {counts(te_r)}",
    ]
    (out / "README.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

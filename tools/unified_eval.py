#!/usr/bin/env python3
"""
Unified evaluation for binary lesion classification across MedViT, DiffMIC, DiffMICv2.

Each model is expected to save its validation predictions as an .npz file with:
    y_true  : (N,)        ground-truth labels in {0, 1}
    y_score : (N, 2)      softmax/probability output OR
              (N,)        positive-class probability
    paths   : (N,) str    (optional) image file paths in val set order

Usage
-----
python3 tools/unified_eval.py --pred outputs/medvit_binary/predictions_val.npz --name MedViT
python3 tools/unified_eval.py \
    --pred outputs/medvit_binary/predictions_val.npz \
    --pred models/DiffMIC-main/results_lesion_binary/predictions_val.npz \
    --pred models/DiffMICv2-main/logs/lesion_binary/predictions_val.npz \
    --name MedViT --name DiffMIC --name DiffMICv2 \
    --out outputs/comparison.csv
"""

import argparse
import csv
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)


def _to_pos_score(y_score):
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    if y_score.ndim == 2 and y_score.shape[1] == 1:
        return y_score[:, 0]
    raise ValueError(f"Unsupported y_score shape: {y_score.shape}")


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    pos = _to_pos_score(y_score)
    y_pred = (pos >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")     # = Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")     # Precision (PPV)
    npv  = tn / (tn + fn) if (tn + fn) > 0 else float("nan")     # NPV
    auc = roc_auc_score(y_true, pos) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, pos) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "N": int(len(y_true)),
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": balanced_accuracy_score(y_true, y_pred),
        "AUC": auc,
        "AP": ap,
        "Precision": prec,
        "Recall": sens,
        "Specificity": spec,
        "NPV": npv,
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "F1_pos": f1_score(y_true, y_pred, pos_label=1),
        "Sensitivity": sens,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def load_pred(path):
    d = np.load(path, allow_pickle=True)
    if "y_true" not in d or "y_score" not in d:
        raise KeyError(f"{path} must contain 'y_true' and 'y_score'")
    return d["y_true"], d["y_score"]


def fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", action="append", required=True,
                        help="path to predictions .npz (can be repeated)")
    parser.add_argument("--name", action="append", default=None,
                        help="display name for each prediction file (same order, same count)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="positive-class threshold (default 0.5)")
    parser.add_argument("--out", type=str, default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    names = args.name or []
    if len(names) and len(names) != len(args.pred):
        raise SystemExit("--name count must match --pred count")

    rows = []
    for i, p in enumerate(args.pred):
        name = names[i] if i < len(names) else os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        y_true, y_score = load_pred(p)
        m = compute_metrics(y_true, y_score, threshold=args.threshold)
        m["Model"] = name
        m["File"] = p
        rows.append(m)

    cols = ["Model", "N", "ACC", "BACC", "AUC", "AP",
            "Precision", "Recall", "Specificity", "NPV",
            "F1_macro", "F1_pos",
            "TP", "FP", "FN", "TN", "File"]
    header = "  ".join(f"{c:>11}" if c != "Model" and c != "File" else c for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c, "")
            s = fmt(v)
            if c == "Model":
                line.append(f"{s:<12}")
            elif c == "File":
                line.append(s)
            else:
                line.append(f"{s:>11}")
        print("  ".join(line))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"\nCSV written: {args.out}")


def eval_folds(fold_dir, name=None, threshold=0.5, k=5):
    """fold_dir 아래 fold_0..fold_{k-1}/predictions_val.npz 를 읽어 per-fold 결과 반환."""
    name = name or os.path.basename(fold_dir.rstrip("/"))
    results = []
    for i in range(k):
        p = os.path.join(fold_dir, f"fold_{i}", "predictions_val.npz")
        if not os.path.isfile(p):
            print(f"  [warn] missing: {p}")
            continue
        y_true, y_score = load_pred(p)
        m = compute_metrics(y_true, y_score, threshold=threshold)
        results.append(m)
    if not results:
        raise FileNotFoundError(f"No fold predictions found in {fold_dir}")
    return name, results


def print_fold_summary(name, results):
    import math
    metrics = ["ACC", "BACC", "AUC", "AP", "Precision", "Recall",
               "Specificity", "NPV", "F1_macro", "F1_pos"]
    print(f"\n{'='*65}")
    print(f"  {name}  ({len(results)} folds)")
    print(f"{'='*65}")
    print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>8}  per-fold values")
    print(f"  {'-'*61}")
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results if not math.isnan(r[m])]
        if not vals:
            continue
        arr = np.array(vals)
        mn, sd = arr.mean(), arr.std()
        fold_str = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {m:<14} {mn:>8.4f}  {sd:>8.4f}  [{fold_str}]")
        summary[m] = (mn, sd)
    return summary


def main_fold():
    import argparse as _ap
    parser = _ap.ArgumentParser(description="5-fold mean±std evaluation")
    parser.add_argument("--fold-dir", action="append", required=True,
                        help="모델 결과 루트 (fold_0~N 포함). 반복 가능.")
    parser.add_argument("--name", action="append", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None, help="CSV 저장 경로")
    args = parser.parse_args()

    names = args.name or []
    all_summaries = []
    for i, d in enumerate(args.fold_dir):
        nm = names[i] if i < len(names) else None
        name, results = eval_folds(d, name=nm, threshold=args.threshold, k=args.k)
        summary = print_fold_summary(name, results)
        all_summaries.append((name, summary))

    if args.out and all_summaries:
        import csv as _csv
        metrics = list(all_summaries[0][1].keys())
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["Model"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics])
            for name, sm in all_summaries:
                row = [name] + [f"{sm[m][0]:.4f}" for m in metrics] + [f"{sm[m][1]:.4f}" for m in metrics]
                w.writerow(row)
        print(f"\nCSV written: {args.out}")


if __name__ == "__main__":
    import sys
    if "--fold-dir" in sys.argv:
        main_fold()
    else:
        main()

#!/usr/bin/env python3
"""
Aggregate 5-fold predictions per (model, tag): pool all folds' test predictions,
then compute metrics (pooled) AND per-fold metrics (mean ± std).
"""
import argparse, os, glob, csv
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def _pos(s):
    s = np.asarray(s)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(
        ACC=accuracy_score(y, yp),
        BACC=balanced_accuracy_score(y, yp),
        AUC=(roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')),
        AP=(average_precision_score(y, p) if len(np.unique(y)) > 1 else float('nan')),
        F1_macro=f1_score(y, yp, average='macro'),
        Sens=tp / (tp + fn) if tp + fn else float('nan'),
        Spec=tn / (tn + fp) if tn + fp else float('nan'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-root", default=None,
                    help="defaults to <this dir>/results/<tag>")
    ap.add_argument("--out-csv", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.results_root or os.path.join(here, "results", a.tag)
    models = ["medvit", "diffmic", "diffmicv2"]

    print(f"Aggregating 5-fold from {root}")
    rows = []
    for m in models:
        per_fold = []
        ys, ps = [], []
        candidates = sorted(set(
            glob.glob(os.path.join(root, m, "fold_*", f"predictions_test_{m}.npz"))   # new
            + glob.glob(os.path.join(root, "fold_*", f"predictions_test_{m}.npz"))    # legacy
        ))
        if not candidates:
            print(f"  [warn] no predictions for {m}")
            continue
        for f in candidates:
            d = np.load(f, allow_pickle=True)
            y = np.asarray(d["y_true"]).astype(int).ravel()
            p = _pos(d["y_score"]).astype(float)
            per_fold.append(metrics(y, p))
            ys.append(y); ps.append(p)
        if not per_fold:
            continue
        keys = list(per_fold[0].keys())
        agg = {k + "_mean": float(np.mean([r[k] for r in per_fold])) for k in keys}
        agg.update({k + "_std": float(np.std([r[k] for r in per_fold], ddof=1)) for k in keys})
        pooled = metrics(np.concatenate(ys), np.concatenate(ps))
        agg.update({k + "_pooled": v for k, v in pooled.items()})
        agg["model"] = m
        agg["n_folds"] = len(per_fold)
        rows.append(agg)

        print(f"\n[{m}]  n_folds={len(per_fold)}")
        for k in keys:
            mean = agg[k + "_mean"]
            std = agg[k + "_std"]
            print(f"  {k:10s}  fold mean ± std = {mean:.4f} ± {std:.4f}   pooled = {agg[k+'_pooled']:.4f}")

    if rows:
        out_csv = a.out_csv or os.path.join(root, "summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fold_dir 아래 fold_0..fold_{k-1} 의 prediction npz 를 읽어
Precision 포함 전 지표의 5-fold mean ± std 를 출력.

npz 키 이름 자동 감지 (y_true/y_score, labels/probs, labels/scores 등)
파일명 패턴 자동 시도 (predictions_val.npz, predictions_test.npz, *.npz)

Usage:
  python3 eval_folds_precision.py \
      --fold-dir experiments/option_b_5fold/results/diffmicv2_baseline_32/diffmicv2\
      --name diffmicv2_baseline --k 5
"""

import argparse, os, math
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

_TRUE_KEYS  = ["y_true", "labels", "label", "targets", "target"]
_SCORE_KEYS = ["y_score", "probs", "scores", "prob", "score",
               "logits", "y_prob", "y_pred_prob"]

def _find_key(d, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    raise KeyError(f"None of {candidates} found. Available: {list(d.files)}")

def load_pred(path):
    d = np.load(path, allow_pickle=True)
    y_true  = _find_key(d, _TRUE_KEYS).astype(int).ravel()
    y_score = _find_key(d, _SCORE_KEYS)
    # (N,2) → positive class column
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]
    return y_true, y_score.ravel()

def compute_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = tp/(tp+fp) if (tp+fp) > 0 else float("nan")
    sens = tp/(tp+fn) if (tp+fn) > 0 else float("nan")
    spec = tn/(tn+fp) if (tn+fp) > 0 else float("nan")
    npv  = tn/(tn+fn) if (tn+fn) > 0 else float("nan")
    return {
        "ACC":         accuracy_score(y_true, y_pred),
        "BACC":        balanced_accuracy_score(y_true, y_pred),
        "AUC":         roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan"),
        "AP":          average_precision_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan"),
        "Precision":   prec,
        "Recall":      sens,
        "Specificity": spec,
        "NPV":         npv,
        "F1_macro":    f1_score(y_true, y_pred, average="macro"),
        "F1_pos":      f1_score(y_true, y_pred, pos_label=1),
    }

_FNAME_CANDIDATES = ["predictions_val.npz",
                     "predictions_test_diffmic.npz",
                     "predictions_test_diffmicv2.npz",
                     "predictions_test_medvit.npz",
                     "predictions_test.npz",
                     "predictions.npz", "pred.npz"]

def find_npz(fold_path, pred_name=None):
    if pred_name:
        p = os.path.join(fold_path, pred_name)
        return p if os.path.isfile(p) else None
    for fn in _FNAME_CANDIDATES:
        p = os.path.join(fold_path, fn)
        if os.path.isfile(p):
            return p
    for fn in sorted(os.listdir(fold_path)):
        if fn.endswith(".npz"):
            return os.path.join(fold_path, fn)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pred-name", default=None, help="npz 파일명 직접 지정 (예: predictions_test_diffmic.npz)")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.fold_dir.rstrip("/"))
    results = []

    for i in range(args.k):
        fold_path = os.path.join(args.fold_dir, f"fold_{i}")
        npz = find_npz(fold_path, pred_name=args.pred_name)
        if npz is None:
            print(f"  [warn] fold_{i}: npz 없음 — skip")
            continue
        y_true, y_score = load_pred(npz)
        m = compute_metrics(y_true, y_score, thr=args.threshold)
        results.append(m)
        print(f"  fold_{i} ({os.path.basename(npz)}): "
              f"ACC={m['ACC']:.4f}  Precision={m['Precision']:.4f}  AUC={m['AUC']:.4f}")

    if not results:
        print("[error] 유효한 fold 없음"); return

    METRICS = ["ACC","BACC","AUC","AP","Precision","Recall","Specificity","NPV","F1_macro","F1_pos"]
    print(f"\n{'='*65}")
    print(f"  {name}  ({len(results)} folds)")
    print(f"{'='*65}")
    print(f"  {'Metric':<14} {'Mean':>8}  {'Std':>8}  per-fold values")
    print(f"  {'-'*61}")
    for m in METRICS:
        vals = [r[m] for r in results if not math.isnan(r[m])]
        if not vals: continue
        arr = np.array(vals)
        fold_str = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {m:<14} {arr.mean():>8.4f}  {arr.std():>8.4f}  [{fold_str}]")

if __name__ == "__main__":
    main()

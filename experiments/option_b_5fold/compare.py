#!/usr/bin/env python3
"""Paired comparison of two tags across all folds (per model).

각 모델 × 메트릭에 대해 fold별 (tag_b - tag_a) 차이를 계산해서
  mean Δ ± std    + paired t-test p-value (scipy 있으면)
출력. 모델별 fold npz 가 양쪽 tag 모두에 있어야 비교 가능.

레이아웃 두 가지 모두 지원:
  model-first:  <root>/<model>/fold_*/predictions_test_<model>.npz   (현재 표준)
  legacy:       <root>/fold_*/predictions_test_<model>.npz

Usage:
    python3 experiments/option_b_5fold/compare.py --tag-a baseline --tag-b bpr
    python3 experiments/option_b_5fold/compare.py --tag-a baseline --tag-b bpr \\
        --models diffmicv2
"""
import argparse, os, glob
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, confusion_matrix)
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


ALL_MODELS = ["medvit", "medvitv2", "diffmic", "diffmicv2", "diffmicv2_sam"]


def _pos(s):
    s = np.asarray(s)
    return s[:, 1] if s.ndim == 2 and s.shape[1] == 2 else s.ravel()


def m(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    multi_class = len(np.unique(y)) > 1
    return dict(
        ACC=accuracy_score(y, yp),
        BACC=balanced_accuracy_score(y, yp),
        AUC=(roc_auc_score(y, p) if multi_class else float('nan')),
        AP=(average_precision_score(y, p) if multi_class else float('nan')),
        Precision=precision_score(y, yp, pos_label=1, zero_division=0),
        Recall=recall_score(y, yp, pos_label=1, zero_division=0),
        F1=f1_score(y, yp, pos_label=1, zero_division=0),
        F1_macro=f1_score(y, yp, average="macro", zero_division=0),
        Sens=tp / (tp + fn) if tp + fn else float('nan'),
        Spec=tn / (tn + fp) if tn + fp else float('nan'),
        NPV=tn / (tn + fn) if tn + fn else float('nan'),
    )


def per_fold(root, model):
    """Return list of metric dicts, one per fold that has predictions."""
    rs = []
    candidates = sorted(set(
        glob.glob(os.path.join(root, model, "fold_*", f"predictions_test_{model}.npz"))
        + glob.glob(os.path.join(root, "fold_*", f"predictions_test_{model}.npz"))
    ))
    for f in candidates:
        x = np.load(f, allow_pickle=True)
        y = np.asarray(x["y_true"]).astype(int).ravel()
        p = _pos(x["y_score"]).astype(float)
        rs.append(m(y, p))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", required=True, help="baseline tag")
    ap.add_argument("--tag-b", required=True, help="treatment tag")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"비교할 모델 (subset of {ALL_MODELS}). "
                         f"기본: 양쪽 tag 모두에 npz 가 있는 모델 자동 감지")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root_a = os.path.join(here, "results", a.tag_a)
    root_b = os.path.join(here, "results", a.tag_b)

    if a.models:
        models = [x for x in a.models if x in ALL_MODELS]
        bad = [x for x in a.models if x not in ALL_MODELS]
        if bad:
            print(f"[warn] unknown model(s) ignored: {bad}")
    else:
        models = [x for x in ALL_MODELS
                  if per_fold(root_a, x) and per_fold(root_b, x)]
        if not models:
            print(f"[error] {a.tag_a} 와 {a.tag_b} 양쪽에 fold npz 가 있는 모델이 없습니다.")
            print(f"  확인: ls {root_a}/<model>/fold_*/predictions_test_<model>.npz")
            return
        print(f"[info] auto-detected models: {models}")

    if not HAS_SCIPY:
        print("[note] scipy 미설치 — paired t-test p-value 생략 "
              "(설치: pip install scipy)")

    for model in models:
        ra = per_fold(root_a, model)
        rb = per_fold(root_b, model)
        n = min(len(ra), len(rb))
        if n == 0:
            print(f"\n[{model}] no matched folds")
            continue
        print(f"\n[{model}]  paired over n={n} folds  ({a.tag_a} vs {a.tag_b})")
        keys = list(ra[0].keys())
        for k in keys:
            a_vals = np.array([ra[i][k] for i in range(n)], dtype=float)
            b_vals = np.array([rb[i][k] for i in range(n)], dtype=float)
            mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
            if mask.sum() == 0:
                print(f"  Δ{k:14s} = (no valid folds)")
                continue
            d = b_vals[mask] - a_vals[mask]
            mean = d.mean()
            std = d.std(ddof=1) if mask.sum() > 1 else 0.0
            line = f"  Δ{k:14s} = {mean:+.4f} ± {std:.4f}"
            if HAS_SCIPY and mask.sum() > 1:
                t = stats.ttest_rel(b_vals[mask], a_vals[mask])
                marker = " *" if t.pvalue < 0.05 else "  "
                line += f"   t={t.statistic:+6.3f}  p={t.pvalue:.4f}{marker}"
            print(line)


if __name__ == "__main__":
    main()

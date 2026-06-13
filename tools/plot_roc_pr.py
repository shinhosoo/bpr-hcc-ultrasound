#!/usr/bin/env python3
"""
Pooled ROC and PR curve plotter for 5-fold cross-validation results.

각 fold의 predictions_val.npz 를 이어붙여(pooled) 단일 ROC / PR 곡선을 그립니다.
여러 모델을 같은 figure 에 겹쳐 비교할 수 있습니다.

Usage (single model)
--------------------
python3 tools/plot_roc_pr.py \
    --fold-dir experiments/option_b_5fold/results/hsq_bpr/hsq \
    --name "HSQ" \
    --out outputs/roc_pr_hsq.png

Usage (multi-model comparison)
-------------------------------
python3 tools/plot_roc_pr.py \
    --fold-dir experiments/option_b_5fold/results/hsq_bpr/hsq \
    --fold-dir experiments/option_b_5fold/results/baseline/medvit \
    --fold-dir experiments/option_b_5fold/results/baseline/diffmic \
    --name "HSQ" --name "MedViT" --name "DiffMIC" \
    --out outputs/roc_pr_compare.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)


def load_pooled(fold_dir: str, pred_file: str = "predictions_val.npz", k: int = 5):
    """fold_0 .. fold_{k-1}/<pred_file> 을 pooling."""
    ys_true, ys_score = [], []
    for i in range(k):
        p = os.path.join(fold_dir, f"fold_{i}", pred_file)
        if not os.path.isfile(p):
            print(f"  [warn] missing: {p}")
            continue
        d = np.load(p, allow_pickle=True)
        y_true = np.asarray(d["y_true"]).ravel().astype(int)
        y_score = np.asarray(d["y_score"])
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2:
            y_score = y_score[:, 0]
        ys_true.append(y_true)
        ys_score.append(y_score.ravel())
    if not ys_true:
        raise FileNotFoundError(f"No '{pred_file}' found under {fold_dir}")
    return np.concatenate(ys_true), np.concatenate(ys_score)


def main():
    parser = argparse.ArgumentParser(description="Pooled ROC / PR curve (5-fold)")
    parser.add_argument("--fold-dir", action="append", required=True,
                        help="모델 결과 루트 (fold_0~N 포함). 반복 가능.")
    parser.add_argument("--name", action="append", default=None,
                        help="각 --fold-dir 에 대응하는 표시 이름 (같은 순서, 같은 개수)")
    parser.add_argument("--pred-file", type=str, default="predictions_val.npz",
                        help="npz 파일명 (default: predictions_val.npz). "
                             "test 기준: predictions_test_medvitv2.npz 등")
    parser.add_argument("--k", type=int, default=5, help="fold 수 (default 5)")
    parser.add_argument("--out", type=str, default="outputs/roc_pr.png",
                        help="출력 이미지 경로")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 5],
                        metavar=("W", "H"))
    args = parser.parse_args()

    names = args.name or []
    if names and len(names) != len(args.fold_dir):
        raise SystemExit("--name 개수와 --fold-dir 개수가 다릅니다.")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=args.figsize)
    colors = plt.cm.tab10.colors

    for i, fold_dir in enumerate(args.fold_dir):
        label = names[i] if i < len(names) else os.path.basename(fold_dir.rstrip("/\\"))
        color = colors[i % len(colors)]

        y_true, y_score = load_pooled(fold_dir, pred_file=args.pred_file, k=args.k)
        n_pos = y_true.sum()
        n_total = len(y_true)
        print(f"[{label}] pooled N={n_total}  pos={n_pos}  neg={n_total - n_pos}")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label}  AUC = {roc_auc:.3f}")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        order = np.argsort(recall)
        ax_pr.plot(recall[order], precision[order], color=color, lw=2,
                   label=f"{label}  AP = {ap:.3f}")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45, label="Random")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    src = os.path.basename(args.pred_file).replace(".npz", "")
    split_tag = "Test" if "test" in src.lower() else "Val"
    ax_roc.set_title(f"ROC Curve  (Pooled {args.k}-Fold, {split_tag})", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title(f"Precision-Recall Curve  (Pooled {args.k}-Fold, {split_tag})", fontsize=12)
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

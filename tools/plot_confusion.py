#!/usr/bin/env python3
"""
5-fold pooled confusion matrix plotter.

각 fold 의 predictions npz 를 이어붙여(pooled) confusion matrix 를 그립니다.
여러 모델을 같은 figure 에 나란히 비교할 수 있습니다.

AUC 는 pooled 예측값 기준으로 계산되며 테이블의 per-fold mean AUC 와 다를 수 있습니다.
캡션에 "AUC values reflect pooled 5-fold predictions; per-fold mean AUCs are reported in Table X."
를 추가하는 것을 권장합니다.

Usage (single model)
--------------------
python3 tools/plot_confusion.py \
    --fold-dir experiments/option_b_5fold/results/hsq_base_pretrained/hsq \
    --name "Base" \
    --out outputs/cm_base.png

Usage (multi-model comparison)
-------------------------------
python3 tools/plot_confusion.py \
    --fold-dir experiments/option_b_5fold/results/hsq_base_pretrained/hsq \
    --fold-dir experiments/option_b_5fold/results/hsq_base_bpr_with_pretrained/hsq \
    --name "Base" --name "Base+BPR" \
    --out outputs/cm_compare.png

Options
-------
--pred-file   npz 파일명 (default: predictions_val.npz)
              test npz 사용 시: predictions_test_hsq_base.npz 등
--threshold   양성 임계값 (default: 0.5)
--k           fold 수 (default: 5)
--normalize   행 정규화 (recall 기준, default: False)
"""

import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score,
)


def load_pooled(fold_dir: str, pred_file: str, k: int = 5):
    """fold_0 .. fold_{k-1}/<pred_file> 을 pooling."""
    ys_true, ys_score = [], []
    for i in range(k):
        p = os.path.join(fold_dir, f"fold_{i}", pred_file)
        if not os.path.isfile(p):
            print(f"  [warn] missing: {p}")
            continue
        d = np.load(p, allow_pickle=True)
        y_true  = np.asarray(d["y_true"]).ravel().astype(int)
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


def _metrics_str(y_true, y_pred, y_score):
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    f1    = f1_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return (f"ACC={acc:.3f}  BACC={bacc:.3f}  AUC={auc:.3f}\n"
            f"F1={f1:.3f}  Sens={sens:.3f}  Spec={spec:.3f}\n"
            f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")


def plot_cm(ax, cm, title, normalize=False, cmap="Blues"):
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=(1.0 if normalize else None))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Negative (0)", "Positive (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=9)

    thresh = cm_plot.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = f"{cm_plot[i, j]:{fmt}}"
            if not normalize:
                val += f"\n({cm[i,j]/cm.sum()*100:.1f}%)"
            ax.text(j, i, val, ha="center", va="center", fontsize=11,
                    color="white" if cm_plot[i, j] > thresh else "black")

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)


def main():
    parser = argparse.ArgumentParser(description="Pooled confusion matrix (5-fold)")
    parser.add_argument("--fold-dir",   action="append", required=True)
    parser.add_argument("--name",       action="append", default=None)
    parser.add_argument("--pred-file",  type=str, default="predictions_val.npz",
                        help="npz 파일명 (default: predictions_val.npz)")
    parser.add_argument("--k",          type=int, default=5)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--normalize",  action="store_true",
                        help="행 정규화 (recall 기준)")
    parser.add_argument("--out",        type=str, default="outputs/confusion.png")
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    names = args.name or []
    if names and len(names) != len(args.fold_dir):
        raise SystemExit("--name 개수와 --fold-dir 개수가 다릅니다.")

    n = len(args.fold_dir)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4.8 * rows + 0.6 * rows),
                             squeeze=False)

    for i, fold_dir in enumerate(args.fold_dir):
        label = names[i] if i < len(names) else os.path.basename(fold_dir.rstrip("/\\"))
        ax    = axes[i // cols][i % cols]

        y_true, y_score = load_pooled(fold_dir, args.pred_file, k=args.k)
        y_pred = (y_score >= args.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        n_pos = y_true.sum(); n_total = len(y_true)
        print(f"\n[{label}] pooled N={n_total}  pos={n_pos}  neg={n_total-n_pos}")
        print(_metrics_str(y_true, y_pred, y_score))

        title = f"{label}  (thr={args.threshold})"
        plot_cm(ax, cm, title, normalize=args.normalize)

        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else float("nan")
        spec = tn/(tn+fp) if (tn+fp)>0 else float("nan")
        auc  = roc_auc_score(y_true, y_score) if len(np.unique(y_true))>1 else float("nan")
        ax.set_xlabel(
            f"Predicted label\n"
            f"AUC(pooled)={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}",
            fontsize=9,
        )

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    src = os.path.basename(args.pred_file).replace(".npz", "")
    fig.suptitle(f"Confusion Matrix — {src}  (Pooled {args.k}-Fold)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\n[plot_confusion] saved → {args.out}")


if __name__ == "__main__":
    main()

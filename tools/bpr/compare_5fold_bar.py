#!/usr/bin/env python3
"""5-fold CV 모델별 mean ± std 메트릭 비교 바 차트.

각 fold 의 predictions_val.npz 를 모아 (y_true, y_score) 에서 메트릭 계산,
모델별로 fold 수만큼 sample 을 모아 mean / std 를 산출.

기대하는 디렉토리 구조:
    <results-root>/
        fold_0/<model>/predictions_val.npz
        fold_1/<model>/predictions_val.npz
        ...

Usage
-----
python3 tools/bpr/compare_5fold_bar.py \
    --results-root experiments/option_b_5fold/results/bpr_dual_adv_balanced \
    --models medvit diffmic diffmicv2 \
    --labels MedViT DiffMIC DiffMICv2 \
    --out experiments/option_b_5fold/results/bpr_dual_adv_balanced/compare_5fold.png \
    --title "5-fold CV — bpr_dual_adv_balanced"

옵션:
    --metrics  ACC Precision Recall F1 AUC AP   (기본)
    --k        5                                (fold 수)
    --pred-name predictions_val.npz             (npz 파일명)
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

# headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    roc_auc_score,
)



def _pos_score(y_score):
    s = np.asarray(y_score)
    if s.ndim == 1:
        return s
    if s.ndim == 2 and s.shape[1] == 2:
        return s[:, 1]
    if s.ndim == 2 and s.shape[1] == 1:
        return s[:, 0]
    raise ValueError(f"Unsupported y_score shape: {s.shape}")


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).ravel()
    pos = _pos_score(y_score).astype(float)
    y_pred = (pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")              # Recall
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    can_auc = len(np.unique(y_true)) > 1
    return {
        "ACC":         accuracy_score(y_true, y_pred),
        "Precision":   prec,
        "Recall":      sens,
        "Specificity": spec,
        "F1":          f1_score(y_true, y_pred, average="binary",
                                zero_division=0),
        "AUC":         roc_auc_score(y_true, pos) if can_auc else float("nan"),
        "AP":          average_precision_score(y_true, pos) if can_auc else float("nan"),
    }



def _load_npz(path):
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    if "y_true" not in d or "y_score" not in d:
        return None, None
    return np.asarray(d["y_true"]).ravel(), np.asarray(d["y_score"])


def collect_fold_metrics(results_root, model, k, pred_name, metrics):
    """모델 1개에 대해 fold_{0..k-1} 의 메트릭 dict 리스트 반환."""
    rows = []
    missing_folds = []
    for i in range(k):
        p = os.path.join(results_root, f"fold_{i}", model, pred_name)
        y, s = _load_npz(p)
        if y is None:
            missing_folds.append(i)
            continue
        m = compute_metrics(y, s)
        m["_fold"] = i
        rows.append(m)
    return rows, missing_folds


def aggregate(rows, metrics):
    """fold 메트릭 list → {metric: (mean, std, n)} dict."""
    out = {}
    for k in metrics:
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])],
                        dtype=float)
        if vals.size == 0:
            out[k] = (float("nan"), float("nan"), 0)
        else:
            out[k] = (float(vals.mean()), float(vals.std(ddof=0)), int(vals.size))
    return out



DEFAULT_COLORS = {
    "medvit":        "#1f77b4",
    "diffmic":       "#ff7f0e",
    "diffmicv2":     "#2ca02c",
    "medvitv2":      "#9467bd",
    "diffmicv2_sam": "#8c564b",
    "translive":     "#e377c2",
}


def plot_grouped_bars(per_model_agg, model_labels, model_keys, metrics,
                      title, out_path, colors=None, ylim=(0.0, 1.0)):
    n_models = len(model_keys)
    n_metrics = len(metrics)
    x = np.arange(n_metrics, dtype=float)
    width = min(0.8 / max(n_models, 1), 0.28)
    fig, ax = plt.subplots(figsize=(1.7 * n_metrics + 2.0, 5.5))

    for mi, (mkey, mlabel) in enumerate(zip(model_keys, model_labels)):
        agg = per_model_agg[mkey]
        means = [agg[k][0] for k in metrics]
        stds  = [agg[k][1] for k in metrics]
        offset = (mi - (n_models - 1) / 2.0) * width
        color = (colors or {}).get(mkey, DEFAULT_COLORS.get(mkey.lower(), None))
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=mlabel, color=color, edgecolor="black",
                      linewidth=0.6, error_kw={"elinewidth": 1.0, "ecolor": "black"})
        for xi, m, s in zip(x + offset, means, stds):
            if not np.isfinite(m):
                continue
            ax.text(xi, m + (s if np.isfinite(s) else 0) + 0.012,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_title(f"{title}  (mean ± std)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def dump_csv(per_model_agg, model_labels, model_keys, metrics, out_path,
             per_fold=None):
    """mean ± std summary + (옵션) per-fold long-format."""
    summary_path = os.path.splitext(out_path)[0] + ".csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{k}_mean" for k in metrics] +
                   [f"{k}_std" for k in metrics] +
                   [f"{k}_n" for k in metrics])
        for mkey, mlabel in zip(model_keys, model_labels):
            agg = per_model_agg[mkey]
            means = [f"{agg[k][0]:.6f}" if np.isfinite(agg[k][0]) else "nan" for k in metrics]
            stds  = [f"{agg[k][1]:.6f}" if np.isfinite(agg[k][1]) else "nan" for k in metrics]
            ns    = [str(agg[k][2]) for k in metrics]
            w.writerow([mlabel] + means + stds + ns)
    print(f"[csv] summary → {summary_path}")

    if per_fold is not None:
        long_path = os.path.splitext(out_path)[0] + "_perfold.csv"
        with open(long_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "fold"] + metrics)
            for mkey, mlabel in zip(model_keys, model_labels):
                for row in per_fold[mkey]:
                    w.writerow([mlabel, row["_fold"]] +
                               [f"{row[k]:.6f}" if np.isfinite(row.get(k, float('nan'))) else "nan"
                                for k in metrics])
        print(f"[csv] per-fold → {long_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="experiments/option_b_5fold/results/<TAG>/")
    ap.add_argument("--models", nargs="+",
                    default=["medvit", "diffmic", "diffmicv2"],
                    help="비교할 모델 디렉토리명 (fold_<i>/<model>/predictions_val.npz)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend 표시명 (개수 = --models 와 동일). 생략 시 모델명 그대로 사용")
    ap.add_argument("--metrics", nargs="+",
                    default=["ACC", "Precision", "Recall", "F1", "AUC", "AP"])
    ap.add_argument("--k", type=int, default=5, help="fold 수 (기본 5)")
    ap.add_argument("--pred-name", default="predictions_val.npz",
                    help="각 fold/model 폴더 안의 npz 파일명")
    ap.add_argument("--out", required=True, help="저장할 PNG 경로")
    ap.add_argument("--title", default=None,
                    help="플롯 제목 (생략 시 results-root 의 마지막 dirname 으로 자동 생성)")
    ap.add_argument("--ylim", nargs=2, type=float, default=(0.0, 1.05),
                    help="y축 범위 (기본 0~1.05)")
    args = ap.parse_args()

    if args.labels is not None and len(args.labels) != len(args.models):
        sys.exit("[ERR] --labels 개수가 --models 개수와 다름")
    labels = args.labels or args.models

    if args.title is None:
        tag = os.path.basename(os.path.normpath(args.results_root))
        args.title = f"{args.k}-fold CV — {tag}"

    per_model_agg = {}
    per_model_fold = {}
    for mkey in args.models:
        rows, missing = collect_fold_metrics(
            args.results_root, mkey, args.k, args.pred_name, args.metrics)
        if missing:
            print(f"[warn] {mkey}: missing folds {missing}", file=sys.stderr)
        if not rows:
            print(f"[warn] {mkey}: 0 folds found — bar 가 비어 보일 수 있음", file=sys.stderr)
        per_model_agg[mkey] = aggregate(rows, args.metrics)
        per_model_fold[mkey] = rows

    print(f"\n{args.title}")
    header = ["model"] + args.metrics
    print("  " + " | ".join(f"{h:>11s}" for h in header))
    for mkey, mlabel in zip(args.models, labels):
        agg = per_model_agg[mkey]
        cells = [mlabel] + [
            f"{agg[k][0]:.3f}±{agg[k][1]:.3f}" if np.isfinite(agg[k][0]) else "    nan    "
            for k in args.metrics
        ]
        print("  " + " | ".join(f"{c:>11s}" for c in cells))

    plot_grouped_bars(per_model_agg, labels, args.models, args.metrics,
                      args.title, args.out, ylim=tuple(args.ylim))
    dump_csv(per_model_agg, labels, args.models, args.metrics, args.out,
             per_fold=per_model_fold)
    print(f"[plot] saved → {args.out}")


if __name__ == "__main__":
    main()
